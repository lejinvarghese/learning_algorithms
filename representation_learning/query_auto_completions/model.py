"""
Search Intention Network (SIN) Implementation
Addresses Intention Equivocality (IE) and Intention Transfer (IT) problems
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CNNLocalEncoder(nn.Module):
    """CNN for extracting local dependencies from prefixes"""

    def __init__(self, embed_dim=128, num_filters=64, filter_sizes=[3, 4, 5]):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, num_filters, kernel_size=fs, padding=fs // 2)
                for fs in filter_sizes
            ]
        )

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        # Pool over sequence dimension (dim=2)
        pooled = []
        for conv_out in conv_outs:
            if conv_out.size(2) > 1:
                pooled_out = F.max_pool1d(
                    conv_out, kernel_size=conv_out.size(2)
                ).squeeze(2)
            else:
                pooled_out = conv_out.squeeze(2)
            pooled.append(pooled_out)
        return torch.cat(pooled, dim=1)  # (batch, num_filters * len(filter_sizes))


class IntentionEquivocalityModule(nn.Module):
    """CNN + Transformer for ambiguous prefix intention extraction"""

    def __init__(self, embed_dim=128, num_filters=64, num_heads=4, num_layers=2):
        super().__init__()
        self.cnn_encoder = CNNLocalEncoder(embed_dim, num_filters)
        cnn_out_dim = num_filters * 3

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_out_dim,
            nhead=num_heads,
            dim_feedforward=cnn_out_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(cnn_out_dim, embed_dim)

    def forward(self, prefix_embed):
        cnn_features = self.cnn_encoder(prefix_embed)
        cnn_features = cnn_features.unsqueeze(1)
        transformer_out = self.transformer(cnn_features)
        return self.output_proj(transformer_out.squeeze(1))


class MultiViewSequenceEncoder(nn.Module):
    """Transformer encoder for behavior sequences"""

    def __init__(self, embed_dim=128, num_heads=4, num_layers=2, max_seq_len=50):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim))

    def forward(self, sequence_embed, mask=None):
        seq_len = sequence_embed.size(1)
        x = sequence_embed + self.pos_encoding[:seq_len, :].unsqueeze(0)
        attn_mask = ~mask.bool() if mask is not None else None
        encoded = self.transformer(x, src_key_padding_mask=attn_mask)

        if mask is not None:
            lengths = mask.sum(dim=1) - 1
            sequence_repr = encoded[torch.arange(encoded.size(0)), lengths]
        else:
            sequence_repr = encoded[:, -1, :]
        return sequence_repr, encoded


class CandidateToHistoryAttention(nn.Module):
    """Attention with time-decaying weights"""

    def __init__(self, embed_dim=128):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(embed_dim)

    def forward(self, candidate_embed, history_embeds, time_decay_weights=None):
        query = self.query_proj(candidate_embed).unsqueeze(1)
        keys = self.key_proj(history_embeds)
        values = self.value_proj(history_embeds)
        scores = torch.bmm(query, keys.transpose(1, 2)) / self.scale

        if time_decay_weights is not None:
            scores = scores + time_decay_weights.unsqueeze(1).log()

        attn_weights = F.softmax(scores, dim=-1)
        return torch.bmm(attn_weights, values).squeeze(1), attn_weights.squeeze(1)


class IntentionTransferModule(nn.Module):
    """Balances current vs historical intentions"""

    def __init__(self, embed_dim=128):
        super().__init__()
        self.distance_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, current_intention, historical_intention):
        concat = torch.cat([current_intention, historical_intention], dim=-1)
        transfer_score = self.distance_proj(concat)
        weighted = (
            transfer_score * current_intention
            + (1 - transfer_score) * historical_intention
        )
        return weighted, transfer_score


class SearchIntentionNetwork(nn.Module):
    """Main SIN model combining all components"""

    def __init__(
        self,
        vocab_size=10000,
        embed_dim=128,
        num_filters=64,
        num_heads=4,
        num_transformer_layers=2,
        num_behaviors=3,  # searches, clicks, purchases
        max_seq_len=50,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Embedding layers with proper initialization
        self.prefix_embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.prefix_embedding.weight, mean=0.0, std=0.02)

        # Candidate query embedding (for CTR prediction)
        self.candidate_embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.candidate_embedding.weight, mean=0.0, std=0.02)

        self.behavior_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, embed_dim) for _ in range(num_behaviors)]
        )
        for emb in self.behavior_embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

        # Core modules
        self.ie_module = IntentionEquivocalityModule(
            embed_dim, num_filters, num_heads, num_transformer_layers
        )
        self.sequence_encoders = nn.ModuleList(
            [
                MultiViewSequenceEncoder(
                    embed_dim, num_heads, num_transformer_layers, max_seq_len
                )
                for _ in range(num_behaviors)
            ]
        )
        self.history_attention = CandidateToHistoryAttention(embed_dim)
        self.it_module = IntentionTransferModule(embed_dim)

        # Output projection for CTR prediction
        # Input: [candidate_intention, current_intention, final_intention]
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, 1),
            nn.Sigmoid(),
        )
        # Initialize output layer to prevent saturation
        for module in self.output_layer:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        prefix_ids,
        candidate_ids,
        behavior_sequences,
        behavior_masks=None,
        time_decays=None,
    ):
        # Extract current intention from prefix (IE)
        prefix_embed = self.prefix_embedding(prefix_ids)
        current_intention = self.ie_module(prefix_embed)

        # Extract candidate query representation
        candidate_embed = self.candidate_embedding(candidate_ids)
        candidate_intention = self.ie_module(
            candidate_embed
        )  # Use same IE module for candidate

        # Extract historical intentions (Multi-view)
        historical_intentions = []
        sequence_representations = []
        for i, (seq, encoder) in enumerate(
            zip(behavior_sequences, self.sequence_encoders)
        ):
            seq_embed = self.behavior_embeddings[i](seq)
            mask = behavior_masks[i] if behavior_masks else None
            seq_repr, encoded_seq = encoder(seq_embed, mask)
            historical_intentions.append(seq_repr)
            sequence_representations.append(encoded_seq)

        historical_intention = torch.stack(historical_intentions).mean(dim=0)

        # Candidate-to-history attention with time decay
        # Use candidate intention to attend over history (per paper)
        if len(sequence_representations) > 0:
            attended_history, attn_weights = self.history_attention(
                candidate_intention,  # Use candidate, not current prefix
                sequence_representations[0],
                time_decays[0] if time_decays else None,
            )
        else:
            attended_history = historical_intention
            attn_weights = None

        # Intention transfer: balance candidate vs historical intentions
        final_intention, transfer_score = self.it_module(
            candidate_intention, attended_history
        )

        # CTR prediction: combine candidate, current prefix intention, and historical context
        combined = torch.cat(
            [candidate_intention, current_intention, final_intention], dim=-1
        )
        ctr_score = self.output_layer(combined)

        return ctr_score, {
            "current_intention": current_intention,
            "historical_intention": historical_intention,
            "attended_history": attended_history,
            "final_intention": final_intention,
            "transfer_score": transfer_score,
            "attention_weights": attn_weights,
        }
