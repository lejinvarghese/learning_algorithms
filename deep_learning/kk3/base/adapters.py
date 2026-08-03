"""Weight adaptation utilities for transferring pretrained weights to K3."""
import torch
import torch.nn as nn
import numpy as np


def rescale_weights(weight: torch.Tensor, target_std: float = 0.02, target_mean: float = 0.0) -> torch.Tensor:
    """
    Rescale weights using z-score normalization to match target statistics.

    This prevents magnitude mismatches when transferring weights from models
    with different initialization scales.

    Args:
        weight: Weight tensor to rescale
        target_std: Target standard deviation (K3 uses 0.02 for most weights)
        target_mean: Target mean (usually 0.0)

    Returns:
        Rescaled weight tensor
    """
    # Compute current statistics
    current_mean = weight.mean()
    current_std = weight.std()

    if current_std < 1e-6:
        # Avoid division by zero for constant tensors
        return weight

    # Z-score normalization: (x - μ) / σ
    normalized = (weight - current_mean) / current_std

    # Rescale to target distribution: x' = σ_target * z + μ_target
    rescaled = normalized * target_std + target_mean

    return rescaled


def adapt_embeddings(source_embed: nn.Embedding, target_vocab_size: int = 256, target_dim: int = None) -> torch.Tensor:
    """
    Adapt BPE embeddings to byte-level embeddings.

    Strategy: For byte-level vocab (0-255), we can either:
    1. Use the first 256 embeddings if source has them
    2. Initialize from source embedding distribution

    We use approach #2 for better coverage.
    If target_dim < source_dim, we take the first K dimensions (preserves most variance).
    """
    source_weights = source_embed.weight.data
    source_vocab, source_dim = source_weights.shape

    # Sample from the distribution of source embeddings
    mean = source_weights.mean(dim=0)
    std = source_weights.std(dim=0)

    # Generate byte-level embeddings with same statistics
    byte_embeddings = torch.randn(target_vocab_size, source_dim) * std + mean

    # Adapt dimension if needed
    if target_dim is not None and target_dim != source_dim:
        if target_dim < source_dim:
            # Take first K dimensions (preserves most variance in practice)
            byte_embeddings = byte_embeddings[:, :target_dim]
        else:
            # Pad with noise
            pad_size = target_dim - source_dim
            padding = torch.randn(target_vocab_size, pad_size) * std[:pad_size].mean()
            byte_embeddings = torch.cat([byte_embeddings, padding], dim=1)

    # Rescale to K3's embedding initialization scale (std=0.02)
    byte_embeddings = rescale_weights(byte_embeddings, target_std=0.02, target_mean=0.0)

    return byte_embeddings


def adapt_layer_norm(source_norm: nn.Module, target_dim: int) -> torch.Tensor:
    """
    Adapt layer norm weights to target dimension.

    If dimensions match: direct copy
    If source smaller: interpolate/repeat
    If source larger: truncate/average
    """
    source_weight = source_norm.weight.data
    source_dim = source_weight.shape[0]

    if source_dim == target_dim:
        # Direct copy
        return source_weight.clone()
    elif source_dim < target_dim:
        # Interpolate to match target
        scale = target_dim / source_dim
        indices = (torch.arange(target_dim).float() / scale).long()
        return source_weight[indices]
    else:
        # Average chunks to reduce dimension
        chunk_size = source_dim // target_dim
        chunks = source_weight[:target_dim * chunk_size].view(target_dim, chunk_size)
        return chunks.mean(dim=1)


def clone_ffn_to_experts(
    ffn_up: nn.Linear,
    ffn_down: nn.Linear,
    num_experts: int,
    noise_std: float = 0.01
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Clone a dense FFN layer to initialize multiple MoE experts.

    Each expert starts with the same weights but with small gaussian noise
    added to encourage specialization during training.
    """
    up_weight = ffn_up.weight.data
    down_weight = ffn_down.weight.data

    # Clone to all experts with noise
    expert_up_weights = []
    expert_down_weights = []

    for _ in range(num_experts):
        noise_up = torch.randn_like(up_weight) * noise_std
        noise_down = torch.randn_like(down_weight) * noise_std

        expert_up_weights.append(up_weight + noise_up)
        expert_down_weights.append(down_weight + noise_down)

    # Stack into [num_experts, ...] tensors
    up_stacked = torch.stack(expert_up_weights, dim=0)
    down_stacked = torch.stack(expert_down_weights, dim=0)

    return up_stacked, down_stacked


def adapt_ffn_to_moe_expert(
    source_ffn_up: nn.Linear,
    source_ffn_down: nn.Linear,
    target_expert_hidden: int,
    target_model_dim: int,
    noise_std: float = 0.01,
    random_dims: bool = True,
    seed: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Adapt a dense FFN to a single MoE expert with potentially different dimensions.

    Handles mismatches in both hidden size and model dimension.
    Truncates when source > target, pads when source < target.
    """
    up_weight = source_ffn_up.weight.data  # [ffn_hidden, model_dim]
    down_weight = source_ffn_down.weight.data  # [model_dim, ffn_hidden]

    source_hidden = up_weight.shape[0]
    source_model_dim = up_weight.shape[1]

    # Adapt model dimension first (input/output of FFN)
    if source_model_dim > target_model_dim:
        # Truncate columns (input dimension)
        up_weight = up_weight[:, :target_model_dim]
        down_weight = down_weight[:target_model_dim, :]
    elif source_model_dim < target_model_dim:
        # Pad columns
        pad_size = target_model_dim - source_model_dim
        noise_up = torch.randn(source_hidden, pad_size) * 0.02
        noise_down = torch.randn(pad_size, source_hidden) * 0.02
        up_weight = torch.cat([up_weight, noise_up], dim=1)
        down_weight = torch.cat([down_weight, noise_down], dim=0)

    # Now adapt hidden dimension (FFN intermediate size)
    if source_hidden > target_expert_hidden:
        if random_dims:
            # Randomly sample different dimensions for each expert (creates diversity!)
            if seed is not None:
                torch.manual_seed(seed)
            indices = torch.randperm(source_hidden)[:target_expert_hidden]
            adapted_up = up_weight[indices, :].clone()
            adapted_down = down_weight[:, indices].clone()
        else:
            # Take first dims (old behavior - all experts identical)
            adapted_up = up_weight[:target_expert_hidden, :].clone()
            adapted_down = down_weight[:, :target_expert_hidden].clone()
    elif source_hidden < target_expert_hidden:
        # Pad rows
        pad_size = target_expert_hidden - source_hidden
        noise_up = torch.randn(pad_size, target_model_dim) * 0.02
        noise_down = torch.randn(target_model_dim, pad_size) * 0.02
        adapted_up = torch.cat([up_weight, noise_up], dim=0)
        adapted_down = torch.cat([down_weight, noise_down], dim=1)
    else:
        adapted_up = up_weight.clone()
        adapted_down = down_weight.clone()

    # Add small noise for diversity
    adapted_up = adapted_up + torch.randn_like(adapted_up) * noise_std
    adapted_down = adapted_down + torch.randn_like(adapted_down) * noise_std

    # Rescale to match K3's linear layer initialization (Xavier/Kaiming scale)
    # Target std ≈ sqrt(2 / fan_in) for typical linear layers
    fan_in = adapted_up.shape[1]
    target_std = (2.0 / fan_in) ** 0.5
    adapted_up = rescale_weights(adapted_up, target_std=target_std, target_mean=0.0)
    adapted_down = rescale_weights(adapted_down, target_std=target_std, target_mean=0.0)

    return adapted_up, adapted_down


def interpolate_attention_layers(source_layers: list, target_count: int) -> list:
    """
    Sample layers from source model to match target layer count.

    If source has more layers: sample evenly spaced
    If source has fewer: repeat with noise
    """
    source_count = len(source_layers)

    if source_count == target_count:
        return list(range(source_count))
    elif source_count > target_count:
        # Sample evenly
        indices = np.linspace(0, source_count - 1, target_count).astype(int)
        return indices.tolist()
    else:
        # Repeat pattern
        repeats = (target_count + source_count - 1) // source_count
        indices = (list(range(source_count)) * repeats)[:target_count]
        return indices
