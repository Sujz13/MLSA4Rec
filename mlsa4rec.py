import torch
from torch import nn
from mamba_ssm import Mamba
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.model.layers import ItemToInterestAggregation
import copy
from torch.nn import GLU
import math
import torch.nn.functional as F

class MLSA4Rec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(MLSA4Rec, self).__init__(config, dataset)

        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.k_interests = config["k_interests"]
        self.n_heads = config["n_heads"]

        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]

        self.mamba = Mamba(
            d_model=self.hidden_size,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )
        self.models = nn.ModuleList([Mamba_attention(
            self.mamba,
            self.hidden_size,
            self.dropout_prob,
            self.n_heads,
            self.k_interests,
            self.attn_dropout_prob) for _ in range(1)])
        self.mamba_layers = nn.ModuleList([
            MambaLayer(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                num_layers=self.num_layers,
            ) for _ in range(self.num_layers)  # self.num_layers
        ])

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)
        item_emb = self.dropout(item_emb)
        item_emb = self.LayerNorm(item_emb)

        item_emb = self.models[0](item_emb)

        for i in range(self.num_layers):
            item_emb = self.mamba_layers[i](item_emb)

        seq_output = self.gather_indexes(item_emb, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores


class Mamba_attention(nn.Module):
    def __init__(self, Mamba, hidden_size, dropout_prob, n_heads, k_interests, attn_dropout_prob):
        super().__init__()
        self.mambas = nn.ModuleList([
            copy.deepcopy(Mamba) for _ in range(1)  # self.num_layers
        ])
        self.hidden_size = hidden_size
        self.mlp1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.mlp2 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.mlp3 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout3 = nn.Dropout(dropout_prob)

        self.LayerNorm1 = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.LayerNorm3 = nn.LayerNorm(self.hidden_size, eps=1e-12)

        self.act1 = nn.GELU()  # nn.LeakyReLU()
        self.act2 = nn.GELU()
        self.act3 = nn.GELU()
        self.trm_encoder = LightMultiHeadAttention(
            n_heads=n_heads,
            k_interests=k_interests,
            seq_len=self.hidden_size,
            hidden_size=self.hidden_size,
            hidden_dropout_prob=dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            layer_norm_eps=1e-12
        )

    def forward(self, item_emb):
        item_emb0 = item_emb
        item_emb = self.LayerNorm1(self.dropout1(self.mambas[0](item_emb)) + item_emb)
        item_emb1 = self.trm_encoder(item_emb)
        item_emb2 = self.LayerNorm2(torch.mul(self.act1(self.mlp1(item_emb1)), item_emb))
        item_emb = self.LayerNorm3(self.dropout3(
            self.act2(self.mlp2(torch.cat([item_emb1, item_emb2], dim=-1)))) + self.act3(self.mlp3(item_emb0)))
        return item_emb


class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.mamba(input_tensor)
        hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class LightMultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            k_interests,
            hidden_size,
            seq_len,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
    ):
        super(LightMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # initialization for low-rank decomposed self-attention
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attpooling_key = ItemToInterestAggregation(
            seq_len, hidden_size, k_interests
        )
        self.attpooling_value = ItemToInterestAggregation(
            seq_len, hidden_size, k_interests
        )

        # initialization for decoupled position encoding
        self.attn_scale_factor = 2
        self.pos_q_linear = nn.Linear(hidden_size, self.all_head_size)
        self.pos_k_linear = nn.Linear(hidden_size, self.all_head_size)
        self.pos_scaling = (
                float(self.attention_head_size * self.attn_scale_factor) ** -0.5
        )
        self.pos_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):  # transfor to multihead
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        # linear map
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        # low-rank decomposed self-attention: relation of items
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(self.attpooling_key(mixed_key_layer))
        value_layer = self.transpose_for_scores(
            self.attpooling_value(mixed_value_layer)
        )

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-2)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer_item = torch.matmul(attention_probs, value_layer)

        # decoupled position encoding: relation of positions
        context_layer = context_layer_item

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = context_layer  # self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


