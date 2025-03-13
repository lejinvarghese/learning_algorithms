import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, AutoModel

class EmbeddingMoEConfig(PretrainedConfig):
    model_type = "embedding_moe"
    def __init__(self, output_dim=128, num_experts=2, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.dropout_rate = dropout_rate


# Expert class using pre-trained BERT
class EmbeddingExpert(nn.Module):
    def __init__(self, model_name, output_dim, dropout_rate=0.1):
        super().__init__()
        self.base = AutoModel.from_pretrained(model_name)
        self.layer_norm = nn.LayerNorm(self.base.config.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        for param in self.base.parameters():
            param.requires_grad = False

        # Projection layer to get the final embedding
        self.projection = nn.Linear(self.base.config.hidden_size, output_dim)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def mean_pooling(self, model_output, attention_mask):
        # Mean pooling - take attention mask into account for averaging
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.mean_pooling(outputs, attention_mask)
        pooled_output = self.layer_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        embedding = self.projection(pooled_output)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding


# Gating Network
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, dropout_rate=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.dropout(x)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = torch.clamp(x, min=-10, max=10)
        x = self.softmax(x)
        return x


# Mixture of Experts for sentence embeddings using BERT
class EmbeddingMoE(PreTrainedModel):
    config_class = EmbeddingMoEConfig

    def __init__(self, config):
        super().__init__(config)
        output_dim = config.output_dim if hasattr(config, "output_dim") else 128
        num_experts = config.num_experts if hasattr(config, "num_experts") else 2

        self.expert1 = EmbeddingExpert("bert-base-uncased", output_dim)
        self.expert2 = EmbeddingExpert("bert-base-uncased", output_dim)
        self.gating = GatingNetwork(output_dim, 256, num_experts)

    def forward(self, input_ids, attention_mask):
        # Get embeddings from both experts
        expert1_output = self.expert1(input_ids, attention_mask)
        expert2_output = self.expert2(input_ids, attention_mask)

        # Average the output as input to gating
        gating_input = (expert1_output + expert2_output) / 2

        # Get gating weights
        gating_output = self.gating(gating_input)

        # Combine expert outputs
        mixed_output = (
            gating_output[:, 0].unsqueeze(1) * expert1_output
            + gating_output[:, 1].unsqueeze(1) * expert2_output
        )

        # Normalize the embedding to unit length
        embedding = torch.nn.functional.normalize(mixed_output, p=2, dim=1)

        return embedding

    def encode_sentence(self, input_ids, attention_mask):
        """Helper method to get the embedding for a single sentence"""
        with torch.no_grad():
            return self.forward(input_ids, attention_mask)
