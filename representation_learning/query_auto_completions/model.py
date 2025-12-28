"""
Simplified Query Auto-Completion Model
Uses CNN+Transformer for prefix/candidate encoding (IE module)
Optionally uses pretrained ByT5 embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel


class CNNLocalEncoder(nn.Module):
    """Multi-scale CNN for local pattern extraction"""

    def __init__(self, embed_dim=128, num_filters=64, filter_sizes=[3, 4, 5]):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, num_filters, fs, padding=fs // 2)
                for fs in filter_sizes
            ]
        )
        self.layer_norm = nn.LayerNorm(num_filters * len(filter_sizes))
        self._init_weights()

    def _init_weights(self):
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(conv.bias)

    def forward(self, x):
        x = x.transpose(1, 2)
        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        pooled = [
            (
                F.max_pool1d(out, out.size(2)).squeeze(2)
                if out.size(2) > 1
                else out.squeeze(2)
            )
            for out in conv_outs
        ]
        out = torch.cat(pooled, dim=1)
        return self.layer_norm(out)


class PrefixEncoder(nn.Module):
    """CNN + Transformer encoder for prefix"""

    def __init__(self, embed_dim=128, num_filters=64, num_heads=4, num_layers=2):
        super().__init__()
        self.cnn = CNNLocalEncoder(embed_dim, num_filters)
        cnn_out_dim = num_filters * 3

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cnn_out_dim,
                nhead=num_heads,
                dim_feedforward=cnn_out_dim * 4,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
                layer_norm_eps=1e-6,
                norm_first=True,
            ),
            num_layers=num_layers,
        )
        self.proj = nn.Linear(cnn_out_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight, gain=0.5)
        nn.init.zeros_(self.proj.bias)

    def forward(self, prefix_embed):
        cnn_out = self.cnn(prefix_embed).unsqueeze(1)
        transformer_out = self.transformer(cnn_out).squeeze(1)
        return self.layer_norm(self.proj(transformer_out))


class CandidateEncoder(nn.Module):
    """Transformer encoder for candidate (no CNN)"""

    def __init__(self, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
                layer_norm_eps=1e-6,
                norm_first=True,
            ),
            num_layers=num_layers,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, candidate_embed):
        transformer_out = self.transformer(candidate_embed)
        pooled = torch.max(transformer_out, dim=1)[0]
        return self.layer_norm(pooled)


class QueryCompletionModel(nn.Module):
    """Query auto-completion: CNN+Transformer for prefix, Transformer for candidate"""

    def __init__(
        self,
        vocab_size=10000,
        embed_dim=128,
        num_filters=64,
        num_heads=4,
        num_transformer_layers=2,
        use_pretrained_embeddings=False,
        pretrained_model_name="google/byt5-small",
    ):
        super().__init__()
        self.use_pretrained_embeddings = use_pretrained_embeddings

        if use_pretrained_embeddings:
            # Load pretrained ByT5 and use its embeddings
            print(f"Loading pretrained embeddings from {pretrained_model_name}...")
            byt5_model = T5EncoderModel.from_pretrained(pretrained_model_name)
            pretrained_embed_dim = byt5_model.config.d_model

            # Share the pretrained embedding for both prefix and candidate
            self.shared_embedding = byt5_model.shared
            self.shared_embedding.requires_grad_(True)  # Fine-tune embeddings

            # Project to target embed_dim if different
            if pretrained_embed_dim != embed_dim:
                self.embed_proj = nn.Linear(pretrained_embed_dim, embed_dim)
                nn.init.xavier_uniform_(self.embed_proj.weight, gain=0.5)
            else:
                self.embed_proj = nn.Identity()

            print(
                f"  ✓ Using pretrained embeddings: {pretrained_embed_dim}D → {embed_dim}D"
            )
        else:
            # Use separate learned embeddings (original behavior)
            self.prefix_embedding = nn.Embedding(vocab_size, embed_dim)
            self.candidate_embedding = nn.Embedding(vocab_size, embed_dim)
            self._init_embeddings()

        self.prefix_encoder = PrefixEncoder(
            embed_dim, num_filters, num_heads, num_transformer_layers
        )
        self.candidate_encoder = CandidateEncoder(
            embed_dim, num_heads, num_transformer_layers
        )

        self.match_predictor = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 1),
        )
        self._init_predictor()

    def _init_embeddings(self):
        if not self.use_pretrained_embeddings:
            nn.init.normal_(self.prefix_embedding.weight, std=0.02)
            nn.init.normal_(self.candidate_embedding.weight, std=0.02)

    def _init_predictor(self):
        for module in self.match_predictor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)

    def forward(self, prefix_ids, candidate_ids):
        if self.use_pretrained_embeddings:
            # Use shared pretrained embeddings for both
            prefix_embed = self.embed_proj(self.shared_embedding(prefix_ids))
            candidate_embed = self.embed_proj(self.shared_embedding(candidate_ids))
        else:
            # Use separate learned embeddings
            prefix_embed = self.prefix_embedding(prefix_ids)
            candidate_embed = self.candidate_embedding(candidate_ids)

        prefix_intention = self.prefix_encoder(prefix_embed)
        candidate_intention = self.candidate_encoder(candidate_embed)
        combined = torch.cat([prefix_intention, candidate_intention], dim=-1)
        logits = self.match_predictor(combined)
        return torch.sigmoid(logits)
