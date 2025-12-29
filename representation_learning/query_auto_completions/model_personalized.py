"""
Personalized Query Auto-Completion Model (SIN Architecture)

Paper: "Search Intention Network for Personalized Query Auto-Completion in E-Commerce"

New modules added on top of base model:
- HistoricalIntentionReformulationEncoder (Section 4.4)
- SearchIntentEvolutionInferencer (Section 4.5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel

# Base encoders from model.py:
# - PrefixEncoder: CNN + Transformer (handles IE problem - incomplete/ambiguous prefix)
# - CandidateEncoder: Transformer only (complete queries)
from model import PrefixEncoder, CandidateEncoder


class HistoricalIntentionReformulationEncoder(nn.Module):
    """
    Section 4.4: Attention-based Historical Intention Reformulation Encoder

    Steps:
    1. Reformulation: s_i = t_i - t_{i-1}
    2. Fusion: r_i = ReLU(ψ_ts(t_i || s_i))
    3. Context-level Transformer
    4. Candidate-to-history attention: h = Σ α * c
    """

    def __init__(self, embed_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        self.reformulation_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
        )

        self.context_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            ),
            num_layers=num_layers,
        )

        self.attention_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, history_encodings, history_mask, candidate_encoding):
        """
        Args:
            history_encodings: [B, H, D] - encoded history items
            history_mask: [B, H] - 1=valid, 0=pad
            candidate_encoding: [B, D] - encoded candidate (attention key)
        Returns:
            h: [B, D] - weighted historical intention
        """
        B, H, D = history_encodings.shape
        device = history_encodings.device

        if H == 0 or history_mask.sum() == 0:
            return torch.zeros(B, self.embed_dim, device=device)

        # Per-sample mask: which samples have at least one valid history item
        has_history = history_mask.sum(dim=1) > 0  # [B]

        # For samples with no history, we'll return zeros
        # But we still need to process samples WITH history through transformer
        output = torch.zeros(B, self.embed_dim, device=device)

        if not has_history.any():
            return output

        # Only process samples that have history
        valid_idx = has_history.nonzero(as_tuple=True)[0]
        hist_enc_valid = history_encodings[valid_idx]  # [N, H, D]
        hist_mask_valid = history_mask[valid_idx]  # [N, H]
        cand_enc_valid = candidate_encoding[valid_idx]  # [N, D]

        # Reformulation: s_i = t_i - t_{i-1}
        t = hist_enc_valid
        t_shifted = F.pad(t[:, :-1, :], (0, 0, 1, 0))
        s = t - t_shifted

        # Fusion: r_i = ReLU(ψ(t_i || s_i))
        r = self.reformulation_fusion(torch.cat([t, s], dim=-1))

        # Context transformer
        attn_mask = ~hist_mask_valid.bool()
        c = self.layer_norm(self.context_transformer(r, src_key_padding_mask=attn_mask))

        # Candidate-to-history attention
        f = self.attention_proj(c)
        scores = torch.bmm(f, cand_enc_valid.unsqueeze(2)).squeeze(-1)
        scores = scores.masked_fill(~hist_mask_valid.bool(), float("-inf"))

        alpha = F.softmax(scores, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)  # Safety for edge cases

        h_valid = (alpha.unsqueeze(-1) * c).sum(dim=1)  # [N, D]

        # Scatter back to full batch
        output[valid_idx] = h_valid

        return output


class SearchIntentEvolutionInferencer(nn.Module):
    """
    Section 4.5: Search Intent Evolution Inferencer

    Evolution layer: e = ReLU(h̃ - p || h̃ * p || cosine(h̃, p))
    """

    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.evolution_dim = embed_dim * 2 + 1

    def forward(self, prefix_encoding, history_encoding):
        """
        Args:
            prefix_encoding: [B, D] - current prefix
            history_encoding: [B, D] - aggregated history
        Returns:
            e: [B, D*2+1] - evolution representation
        """
        p, h = prefix_encoding, history_encoding

        diff = h - p
        prod = h * p

        # Handle zero vectors: F.normalize(zeros) = NaN
        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(p, dim=-1)
        cosine = (h_norm * p_norm).sum(-1, keepdim=True)
        cosine = torch.nan_to_num(cosine, nan=0.0)

        return F.relu(torch.cat([diff, prod, cosine], dim=-1))


class PersonalizedQueryCompletionModel(nn.Module):
    """
    Full SIN model: CTR = Predict(h || p || e || q)

    Components:
    - CandidateEncoder (Transformer) -> q
    - PrefixEncoder (CNN+Transformer) -> p
    - HistoricalIntentionReformulationEncoder -> h
    - SearchIntentEvolutionInferencer -> e
    """

    def __init__(
        self,
        vocab_size=10000,
        embed_dim=128,
        num_filters=64,
        num_heads=4,
        num_transformer_layers=2,
        max_history_len=10,
        use_pretrained_embeddings=False,
        pretrained_model_name="google/byt5-small",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_history_len = max_history_len

        # Embeddings
        if use_pretrained_embeddings:
            byt5 = T5EncoderModel.from_pretrained(pretrained_model_name)
            self.shared_embedding = byt5.shared
            self.shared_embedding.requires_grad_(True)
            pretrained_dim = byt5.config.d_model
            self.embed_proj = (
                nn.Linear(pretrained_dim, embed_dim)
                if pretrained_dim != embed_dim
                else nn.Identity()
            )
        else:
            self.shared_embedding = nn.Embedding(vocab_size, embed_dim)
            self.embed_proj = nn.Identity()

        # Encoders
        self.candidate_encoder = CandidateEncoder(
            embed_dim, num_heads, num_transformer_layers
        )
        self.prefix_encoder = PrefixEncoder(
            embed_dim, num_filters, num_heads, num_transformer_layers
        )
        self.history_item_encoder = CandidateEncoder(
            embed_dim, num_heads, num_transformer_layers
        )

        # Personalization modules
        self.history_reformulation = HistoricalIntentionReformulationEncoder(
            embed_dim, num_heads, num_transformer_layers
        )
        self.intent_evolution = SearchIntentEvolutionInferencer(embed_dim)

        # Predictor: h || p || e || q -> D*5+1
        self.predictor = nn.Sequential(
            nn.LayerNorm(embed_dim * 5 + 1),
            nn.Linear(embed_dim * 5 + 1, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 1),
        )

    def _embed(self, ids):
        return self.embed_proj(self.shared_embedding(ids))

    def _encode_history(self, history_ids, history_mask):
        B, H, S = history_ids.shape
        if H == 0:
            return torch.zeros(B, 0, self.embed_dim, device=history_ids.device)
        flat = self._embed(history_ids.view(-1, S))
        return self.history_item_encoder(flat).view(B, H, -1)

    def forward(self, prefix_ids, candidate_ids, history_ids=None, history_mask=None):
        B, device = prefix_ids.shape[0], prefix_ids.device

        q = self.candidate_encoder(self._embed(candidate_ids))
        p = self.prefix_encoder(self._embed(prefix_ids))

        if (
            history_ids is not None
            and history_mask is not None
            and history_mask.sum() > 0
        ):
            hist_enc = self._encode_history(history_ids, history_mask)
            h = self.history_reformulation(hist_enc, history_mask, q)
            e = self.intent_evolution(p, h)
        else:
            h = torch.zeros(B, self.embed_dim, device=device)
            e = torch.zeros(B, self.embed_dim * 2 + 1, device=device)

        return torch.sigmoid(self.predictor(torch.cat([h, p, e, q], dim=-1)))
