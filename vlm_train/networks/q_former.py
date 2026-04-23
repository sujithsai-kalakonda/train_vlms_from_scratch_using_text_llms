import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig


def create_attention_mask(num_queries, seq_len, mode="uni_modal"):
    total_len = num_queries + seq_len
    mask = torch.zeros(total_len, total_len)

    if mode == "uni_modal":
        mask[:num_queries, :num_queries] = 1

    elif mode == "multi_modal":
        mask[:num_queries, :num_queries] = 1
        mask[:num_queries, num_queries:] = 1
        mask[num_queries:, :num_queries] = 1
        mask[num_queries:, num_queries:] = 1

    elif mode == "multi_modal_causal":
        mask[:num_queries, :num_queries] = 1
        mask[:num_queries, num_queries:] = 1
        mask[num_queries:, :num_queries] = 1

        causal = torch.tril(torch.ones(seq_len, seq_len))
        mask[num_queries:, num_queries:] = causal

    return mask


class CrossAttensionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, queries, visual_features):
        attended, _ = self.cross_attn(queries, visual_features, visual_features)
        queries = self.norm(queries + attended)
        queries = self.norm2(queries + self.ffn(queries))
        return queries


class QFormer(nn.Module):
    def __init__(self, num_queries=32, cross_attn_freq=2):
        super().__init__()
        config = DistilBertConfig()
        self.distilbert = DistilBertModel(config)

        hidden_size = config.dim
        num_heads = config.n_heads
        num_layers = config.n_layers

        self.query_embeddings = nn.Parameter(torch.randn(1, num_queries, hidden_size))
        self.num_queries = num_queries
        self.cross_attn_freq = cross_attn_freq

        self.cross_attn_blocks = nn.ModuleList(
            [
                CrossAttensionBlock(hidden_size, num_heads)
                for _ in range(num_layers // cross_attn_freq)
            ]
        )

    def encode_image(self, visual_features):
        batch_size = visual_features.shape[0]
        queries = self.query_embeddings.expand(batch_size, -1, -1)

        cross_idx = 0
        for i, layer in enumerate(self.distilbert.transformer.layer):
            out = layer(queries, attn_mask=None)

            queries = out[0] if isinstance(out, tuple) else out

            if (i + 1) % self.cross_attn_freq == 0:
                queries = self.cross_attn_blocks[cross_idx](queries, visual_features)
                cross_idx += 1

        return queries

    def forward(self, visual_features, input_ids, attention_mask=None):
        batch_size = visual_features.shape[0]
        queries = self.query_embeddings.expand(batch_size, -1, -1)

        text_embeddings = self.distilbert.embeddings(input_ids)
        x = torch.cat([queries, text_embeddings], dim=1)

        seq_len = input_ids.shape[1]
        attn_mask = create_attention_mask(self.num_queries, seq_len, mode="multi_modal")
        attn_mask = attn_mask.unsqueeze(0).to(x.device)

        cross_idx = 0
        for i, layer in enumerate(self.distilbert.transformer.layer):
            out = layer(x, attn_mask=attn_mask)
            x = out[0] if isinstance(out, tuple) else out

            if (i + 1) % self.cross_attn_freq == 0:
                x[:, : self.num_queries] = self.cross_attn_blocks[cross_idx](
                    x[:, : self.num_queries], visual_features
                )
                cross_idx += 1

        return x[:, : self.num_queries]
