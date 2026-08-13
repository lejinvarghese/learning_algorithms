"""Transfer pretrained weights from small foundation models to K3."""
import click
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoImageProcessor, SiglipVisionModel, CLIPVisionModel
from pathlib import Path

from k3 import K3Config, K3Model
from .adapters import (
    adapt_embeddings,
    adapt_layer_norm,
    clone_ffn_to_experts,
    adapt_ffn_to_moe_expert,
    interpolate_attention_layers,
    rescale_weights,
)


AVAILABLE_SOURCES = {
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
}

VISION_ENCODERS = {
    "siglip-so400m": "google/siglip-so400m-patch14-384",
    "clip-vit-base": "openai/clip-vit-base-patch32",
}

AUDIO_ENCODERS = {
    "whisper-tiny": "openai/whisper-tiny",
    "whisper-base": "openai/whisper-base",
}


def available_sources():
    """Return dict of available source models."""
    return AVAILABLE_SOURCES.copy()


def transfer_embeddings(source_model, k3_model, cfg: K3Config):
    """Transfer embedding weights."""
    click.echo("  📦 Transferring embeddings...")

    # Get source embeddings (model.embed_tokens for Llama/Qwen style)
    if hasattr(source_model, 'model') and hasattr(source_model.model, 'embed_tokens'):
        source_embed = source_model.model.embed_tokens
    elif hasattr(source_model, 'embed_tokens'):
        source_embed = source_model.embed_tokens
    else:
        click.secho("    ⚠ Could not find source embeddings, skipping", fg="yellow")
        return False

    # Adapt to byte-level vocab with dimension handling
    adapted_weights = adapt_embeddings(source_embed, target_vocab_size=cfg.vocab_size, target_dim=cfg.hidden_dim)

    source_dim = source_embed.weight.data.shape[1]
    if source_dim != cfg.hidden_dim:
        click.secho(
            f"    📐 Dimension adapted: {source_dim}d → {cfg.hidden_dim}d (truncated)",
            fg="cyan"
        )

    k3_model.embed.weight.data.copy_(adapted_weights)
    if cfg.tie_embeddings:
        k3_model.lm_head.weight.data.copy_(adapted_weights)

    click.secho(f"    ✓ Embeddings: {cfg.vocab_size} × {cfg.hidden_dim}", fg="green")
    return True


def transfer_layer_norms(source_model, k3_model, cfg: K3Config):
    """Transfer layer norm weights where dimensions match."""
    click.echo("  📦 Transferring layer norms...")

    transferred = 0

    # Try to get source layers
    if hasattr(source_model, 'model') and hasattr(source_model.model, 'layers'):
        source_layers = source_model.model.layers
    elif hasattr(source_model, 'layers'):
        source_layers = source_model.layers
    else:
        click.secho("    ⚠ Could not find source layers, skipping", fg="yellow")
        return 0

    # Map source layers to K3 blocks
    source_count = len(source_layers)
    k3_layer_count = sum(len(block.layers) for block in k3_model.blocks)
    layer_indices = interpolate_attention_layers(list(range(source_count)), k3_layer_count)

    k3_layer_idx = 0
    for block in k3_model.blocks:
        for sub in block.layers:
            if k3_layer_idx >= len(layer_indices):
                break

            source_idx = layer_indices[k3_layer_idx]
            source_layer = source_layers[source_idx]

            # Transfer input layer norm (pre-attention) - adapt dimensions
            if hasattr(source_layer, 'input_layernorm'):
                source_norm = source_layer.input_layernorm
                if hasattr(source_norm, 'weight'):
                    source_dim = source_norm.weight.shape[0]
                    # AttnResGate has a layer norm
                    if hasattr(sub['gate'], 'norm') and hasattr(sub['gate'].norm, 'weight'):
                        if source_dim >= cfg.hidden_dim:
                            # Truncate
                            sub['gate'].norm.weight.data.copy_(source_norm.weight.data[:cfg.hidden_dim])
                        else:
                            # Pad
                            sub['gate'].norm.weight.data[:source_dim].copy_(source_norm.weight.data)
                        transferred += 1

            # Transfer post-attention layer norm - adapt dimensions
            if hasattr(source_layer, 'post_attention_layernorm'):
                source_norm = source_layer.post_attention_layernorm
                if hasattr(source_norm, 'weight'):
                    source_dim = source_norm.weight.shape[0]
                    # MoE has pre_norm
                    if hasattr(sub['moe'], 'pre_norm') and hasattr(sub['moe'].pre_norm, 'weight'):
                        if source_dim >= cfg.hidden_dim:
                            # Truncate
                            sub['moe'].pre_norm.weight.data.copy_(source_norm.weight.data[:cfg.hidden_dim])
                        else:
                            # Pad
                            sub['moe'].pre_norm.weight.data[:source_dim].copy_(source_norm.weight.data)
                        transferred += 1

            k3_layer_idx += 1

    # Transfer final norm - adapt dimensions
    if hasattr(source_model, 'model') and hasattr(source_model.model, 'norm'):
        source_final_norm = source_model.model.norm
        if hasattr(source_final_norm, 'weight'):
            source_dim = source_final_norm.weight.shape[0]
            if source_dim >= cfg.hidden_dim:
                # Truncate
                k3_model.final_norm.weight.data.copy_(source_final_norm.weight.data[:cfg.hidden_dim])
            else:
                # Pad
                k3_model.final_norm.weight.data[:source_dim].copy_(source_final_norm.weight.data)
            transferred += 1

    click.secho(f"    ✓ Layer norms: {transferred} transferred", fg="green")
    return transferred


def transfer_vision_encoder(vision_model, k3_model, cfg: K3Config):
    """Transfer vision encoder weights (CLIP or SigLIP) to MoonViT."""
    if k3_model.vision is None:
        click.secho("    ⚠ K3 has no vision encoder, skipping", fg="yellow")
        return 0

    click.echo("  📦 Transferring vision encoder → MoonViT...")

    transferred = 0

    # Detect model type and get embeddings
    if hasattr(vision_model, 'vision_model'):
        # SigLIP
        vision = vision_model.vision_model
        embeddings = vision.embeddings
    elif hasattr(vision_model, 'embeddings'):
        # CLIP
        vision = vision_model
        embeddings = vision.embeddings
    else:
        click.secho("    ⚠ Unknown vision model structure", fg="yellow")
        return 0

    # Transfer patch embedding (conv layer) - with dimension adaptation
    if hasattr(embeddings, 'patch_embedding'):
        source_patch_embed = embeddings.patch_embedding
        target_patch_embed = k3_model.vision.patch_embed

        source_shape = source_patch_embed.weight.shape  # [out_channels, in_channels, H, W]
        target_shape = target_patch_embed.weight.shape

        click.echo(f"    📐 Patch embed: {source_shape} → {target_shape}")

        # Adapt dimensions: truncate/pad channels if needed
        if source_shape[0] >= target_shape[0]:  # Output channels
            # Truncate output channels
            adapted_weight = source_patch_embed.weight.data[:target_shape[0], :, :, :]
        else:
            # Pad output channels
            adapted_weight = torch.zeros(target_shape[0], source_shape[1], source_shape[2], source_shape[3])
            adapted_weight[:source_shape[0], :, :, :] = source_patch_embed.weight.data

        # Adapt spatial dimensions if needed
        if adapted_weight.shape[2:] != target_shape[2:]:
            # Different patch sizes - interpolate
            adapted_weight = torch.nn.functional.interpolate(
                adapted_weight, size=target_shape[2:], mode='bilinear', align_corners=False
            )

        # Rescale to match K3's conv initialization
        final_weight = rescale_weights(adapted_weight[:, :target_shape[1], :, :], target_std=0.02)
        target_patch_embed.weight.data.copy_(final_weight)
        transferred += 1
        click.secho(f"    ✓ Patch embedding transferred (adapted + rescaled)", fg="green")

    # Transfer MLP/FFN layers from vision encoder
    if hasattr(vision, 'encoder') and hasattr(vision.encoder, 'layers'):
        source_layers = vision.encoder.layers

        # Just grab FFN weights from middle layers and use them for projector
        # Simpler approach: take dense layers directly
        click.echo(f"    📦 Transferring MLP layers from vision encoder...")

        mid_layer_idx = len(source_layers) // 2
        if hasattr(source_layers[mid_layer_idx], 'mlp'):
            src_mlp = source_layers[mid_layer_idx].mlp

            # CLIP/SigLIP MLP has fc1 and fc2
            if hasattr(src_mlp, 'fc1') and hasattr(src_mlp, 'fc2'):
                fc1_weight = src_mlp.fc1.weight.data  # [mlp_hidden, vision_hidden]
                fc2_weight = src_mlp.fc2.weight.data  # [vision_hidden, mlp_hidden]

                # Adapt to K3's projector dimensions
                # K3 projector: [vit_hidden*4, vit_hidden*4] -> GELU -> [vit_hidden*4, hidden_dim]
                proj_layer1 = k3_model.vision.projector[0]  # First linear
                proj_layer2 = k3_model.vision.projector[2]  # Second linear (after GELU)

                target_dim1 = proj_layer1.weight.shape  # [vit_hidden*4, vit_hidden*4]
                target_dim2 = proj_layer2.weight.shape  # [hidden_dim, vit_hidden*4]

                # Truncate/adapt fc1 -> projector layer 1
                if fc1_weight.shape[0] >= target_dim1[0] and fc1_weight.shape[1] >= target_dim1[1]:
                    truncated = fc1_weight[:target_dim1[0], :target_dim1[1]]
                    fan_in = target_dim1[1]
                    rescaled = rescale_weights(truncated, target_std=(2.0 / fan_in) ** 0.5)
                    proj_layer1.weight.data.copy_(rescaled)
                    transferred += 1
                    click.secho(f"    ✓ Projector layer 1: {fc1_weight.shape} → {target_dim1} (rescaled)", fg="green")

                # Truncate/adapt fc2 -> projector layer 2
                if fc2_weight.shape[0] >= target_dim2[0] and fc2_weight.shape[1] >= target_dim2[1]:
                    truncated = fc2_weight[:target_dim2[0], :target_dim2[1]]
                    fan_in = target_dim2[1]
                    rescaled = rescale_weights(truncated, target_std=(2.0 / fan_in) ** 0.5)
                    proj_layer2.weight.data.copy_(rescaled)
                    transferred += 1
                    click.secho(f"    ✓ Projector layer 2: {fc2_weight.shape} → {target_dim2} (rescaled)", fg="green")

    click.secho(f"    ✓ Vision encoder: {transferred} components transferred", fg="green")
    return transferred


def transfer_moe_from_ffn(source_model, k3_model, cfg: K3Config):
    """Initialize MoE experts from source FFN layers."""
    click.echo("  📦 Transferring FFN → MoE experts...")

    # Try to get source layers
    if hasattr(source_model, 'model') and hasattr(source_model.model, 'layers'):
        source_layers = source_model.model.layers
    elif hasattr(source_model, 'layers'):
        source_layers = source_model.layers
    else:
        click.secho("    ⚠ Could not find source layers, skipping", fg="yellow")
        return 0

    if len(source_layers) == 0:
        click.secho("    ⚠ No source layers found", fg="yellow")
        return 0

    # Get a representative FFN layer
    source_layer = source_layers[len(source_layers) // 2]  # Use middle layer

    # Find FFN components (Llama/Qwen use gate_proj, up_proj, down_proj)
    if hasattr(source_layer, 'mlp'):
        ffn = source_layer.mlp
        if hasattr(ffn, 'up_proj') and hasattr(ffn, 'down_proj'):
            ffn_up = ffn.up_proj
            ffn_down = ffn.down_proj
        else:
            click.secho("    ⚠ Could not find up_proj/down_proj in FFN", fg="yellow")
            return 0
    else:
        click.secho("    ⚠ Could not find mlp in source layer", fg="yellow")
        return 0

    transferred = 0
    k3_layer_idx = 0
    source_count = len(source_layers)
    k3_layer_count = sum(len(block.layers) for block in k3_model.blocks)
    layer_indices = interpolate_attention_layers(list(range(source_count)), k3_layer_count)

    for block in k3_model.blocks:
        for sub in block.layers:
            if k3_layer_idx >= len(layer_indices):
                break

            source_idx = layer_indices[k3_layer_idx]
            source_ffn_layer = source_layers[source_idx]

            # Get this layer's FFN
            if hasattr(source_ffn_layer, 'mlp'):
                current_ffn = source_ffn_layer.mlp
                if hasattr(current_ffn, 'up_proj') and hasattr(current_ffn, 'down_proj'):
                    current_up = current_ffn.up_proj
                    current_down = current_ffn.down_proj
                else:
                    k3_layer_idx += 1
                    continue
            else:
                k3_layer_idx += 1
                continue

            # Initialize shared experts from this FFN
            moe = sub['moe']
            if hasattr(moe, 'shared_experts') and moe.shared_experts is not None:
                for i, expert in enumerate(moe.shared_experts):
                    adapted_up, adapted_down = adapt_ffn_to_moe_expert(
                        current_up, current_down,
                        cfg.shared_moe_hidden,
                        cfg.hidden_dim,  # Pass target model dimension
                        noise_std=0.01 * (i + 1),  # Increasing noise for diversity
                        random_dims=True,
                        seed=42 + k3_layer_idx * 1000 + i  # Unique seed per expert
                    )
                    # K3 experts are SiTUGLU with w_up, w_down (and w_gate)
                    if hasattr(expert, 'w_up') and hasattr(expert, 'w_down'):
                        expert.w_up.weight.data.copy_(adapted_up)
                        expert.w_down.weight.data.copy_(adapted_down)
                        transferred += 1

            # Initialize routed experts from same FFN (each with different random dims!)
            # Routed experts run in latent space, so both input and hidden are latent_dim
            if hasattr(moe, 'routed_experts') and moe.routed_experts is not None:
                for i, expert in enumerate(moe.routed_experts):
                    adapted_up, adapted_down = adapt_ffn_to_moe_expert(
                        current_up, current_down,
                        cfg.moe_hidden_per_expert,
                        cfg.latent_dim,  # Routed experts use latent space
                        noise_std=0.02 * (1 + i / cfg.num_routed_experts),  # Progressive noise
                        random_dims=True,
                        seed=42 + k3_layer_idx * 1000 + 100 + i  # Unique seed per routed expert
                    )
                    # K3 experts are SiTUGLU with w_up, w_down (and w_gate)
                    if hasattr(expert, 'w_up') and hasattr(expert, 'w_down'):
                        expert.w_up.weight.data.copy_(adapted_up)
                        expert.w_down.weight.data.copy_(adapted_down)
                        transferred += 1

            k3_layer_idx += 1

    click.secho(f"    ✓ MoE experts: {transferred} experts initialized", fg="green")
    return transferred


def transfer_from_pretrained(
    source_name: str,
    k3_config: K3Config,
    output_path: str = "checkpoints/k3_pretrained_init.pt",
    vision_encoder: str = "siglip-so400m",
):
    """
    Transfer weights from a pretrained model to K3.

    Args:
        source_name: Name of source model (e.g., 'smollm2-360m')
        k3_config: K3 configuration
        output_path: Where to save the initialized checkpoint
        vision_encoder: Vision encoder to use (e.g., 'siglip-so400m')

    Returns:
        Initialized K3Model
    """
    if source_name not in AVAILABLE_SOURCES:
        raise ValueError(f"Unknown source: {source_name}. Available: {list(AVAILABLE_SOURCES.keys())}")

    model_id = AVAILABLE_SOURCES[source_name]

    click.secho(f"\n🔄 Transferring weights from {source_name}", fg="cyan", bold=True)
    click.echo(f"   Language: {model_id}")
    if k3_config.use_vision and vision_encoder:
        click.echo(f"   Vision:   {VISION_ENCODERS.get(vision_encoder, vision_encoder)}")
    click.echo(f"   Target:   K3 ({k3_config.hidden_dim}d, {sum(len(b.layers) for b in K3Model(k3_config).blocks)} layers)")

    # Load source language model
    click.echo("\n📥 Loading source language model...")
    try:
        source_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        click.secho(f"✗ Failed to load source model: {e}", fg="red")
        return None

    click.secho("   ✓ Language model loaded", fg="green")

    # Load vision encoder if needed
    vision_model = None
    if k3_config.use_vision and vision_encoder in VISION_ENCODERS:
        click.echo("\n📥 Loading vision encoder...")
        try:
            vision_id = VISION_ENCODERS[vision_encoder]
            if 'clip' in vision_encoder:
                vision_model = CLIPVisionModel.from_pretrained(vision_id)
            else:
                vision_model = SiglipVisionModel.from_pretrained(vision_id)
            click.secho("   ✓ Vision encoder loaded", fg="green")
        except Exception as e:
            click.secho(f"⚠ Failed to load vision encoder: {e}", fg="yellow")
            vision_model = None

    # Create K3 model
    click.echo("\n🏗️  Creating K3 model...")
    k3_model = K3Model(k3_config)
    click.secho("   ✓ K3 model created", fg="green")

    # Transfer components
    click.echo("\n📦 Transferring components...")

    stats = {
        'embeddings': transfer_embeddings(source_model, k3_model, k3_config),
        'layer_norms': transfer_layer_norms(source_model, k3_model, k3_config),
        'moe_experts': transfer_moe_from_ffn(source_model, k3_model, k3_config),
        'vision': transfer_vision_encoder(vision_model, k3_model, k3_config) if vision_model else 0,
    }

    # Summary
    click.echo("\n" + "="*60)
    click.secho("📊 Transfer Summary:", fg="cyan", bold=True)
    click.echo(f"   Embeddings:   {'✓' if stats['embeddings'] else '✗'}")
    click.echo(f"   Layer Norms:  {stats['layer_norms']} transferred")
    click.echo(f"   MoE Experts:  {stats['moe_experts']} initialized")
    click.echo("   Attention:    ✗ (left random as requested)")
    click.echo(f"   Vision:       {'✓ ' + str(stats['vision']) + ' components' if stats['vision'] > 0 else '✗ (none transferred)'}")

    total_params = sum(p.numel() for p in k3_model.parameters())
    click.echo(f"\n   Total K3 params: {total_params:,}")

    # Save checkpoint
    click.echo(f"\n💾 Saving to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': k3_model.state_dict(),
        'config': k3_config,
        'source_model': source_name,
        'transfer_stats': stats,
        'epoch': 0,
    }

    torch.save(checkpoint, output_path)
    click.secho(f"✓ Saved checkpoint", fg="green")
    click.echo("="*60)

    return k3_model


@click.command()
@click.option("--source", type=click.Choice(list(AVAILABLE_SOURCES.keys())),
              default="smollm2-360m", help="Source language model to transfer from")
@click.option("--vision", type=click.Choice(list(VISION_ENCODERS.keys())),
              default="clip-vit-base", help="Vision encoder to transfer from")
@click.option("--output", type=str, default="checkpoints/k3_pretrained_init.pt",
              help="Output checkpoint path")
@click.option("--scale", type=float, default=1.0, help="K3 config scale multiplier")
def main(source, vision, output, scale):
    """Transfer pretrained weights to K3."""
    cfg = K3Config.scaled(scale, use_vision=True, use_gradient_checkpointing=True, vit_num_frames=4)

    click.echo("\n🎯 K3 Configuration:")
    click.echo(f"   Hidden dim:      {cfg.hidden_dim}")
    click.echo(f"   Num blocks:      {cfg.num_blocks}")
    click.echo(f"   Layers/block:    {cfg.layers_per_block}")
    click.echo(f"   MoE experts:     {cfg.num_routed_experts} routed + {cfg.num_shared_experts} shared")
    click.echo(f"   Vision:          {cfg.use_vision} (ViT: {cfg.vit_hidden}d, {cfg.vit_layers} layers)")
    click.echo(f"   MTP:             {cfg.use_mtp}")

    transfer_from_pretrained(source, cfg, output, vision_encoder=vision)


if __name__ == "__main__":
    main()
