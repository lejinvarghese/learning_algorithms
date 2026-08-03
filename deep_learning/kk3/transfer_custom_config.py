#!/usr/bin/env python3
"""Transfer weights with custom K3 config matching train.py exactly."""
import torch
from k3 import K3Config, K3Model
from base.transfer_weights import transfer_embeddings, transfer_layer_norms, transfer_moe_from_ffn, transfer_vision_encoder
from transformers import AutoModel
import click

# Exact config from train.py
cfg = K3Config(
    vocab_size=163840,
    hidden_dim=192,
    num_blocks=4,
    layers_per_block=4,
    num_routed_experts=128,
    num_experts_active=4,  # Will be overridden at runtime via --active-experts
    num_shared_experts=1,
    moe_hidden_per_expert=48,
    shared_moe_hidden=96,
    use_vision=True,
    use_gradient_checkpointing=True,
    vit_num_frames=4,
)

click.secho(f"🎯 Creating K3 model with custom config", fg="cyan")
click.secho(f"   hidden_dim={cfg.hidden_dim}, routed_experts={cfg.num_routed_experts}", fg="cyan")

model = K3Model(cfg)
total_params = sum(p.numel() for p in model.parameters())
click.secho(f"   Total params: {total_params/1e6:.1f}M\n", fg="green")

# Load source models
click.secho("📥 Loading source models...", fg="cyan")
source_lang = AutoModel.from_pretrained("HuggingFaceTB/SmolLM2-360M")
source_vision = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
click.secho("   ✓ Models loaded\n", fg="green")

# Transfer components
click.secho("📦 Transferring weights with diverse expert initialization...", fg="cyan")
transfer_embeddings(source_lang, model, cfg)
transfer_layer_norms(source_lang, model, cfg)
transfer_moe_from_ffn(source_lang, model, cfg)
transfer_vision_encoder(source_vision, model, cfg)

# Save checkpoint
output_path = "checkpoints/k3_pretrained_init.pt"
click.secho(f"\n💾 Saving to {output_path}...", fg="cyan")
torch.save({
    "config": cfg,
    "model_state_dict": model.state_dict(),
    "epoch": 0,
}, output_path)
click.secho("✓ Checkpoint saved!\n", fg="green")

counts = model.param_counts()
click.secho(f"📊 Final: {counts['total']/1e6:.1f}M total, {counts['activated_approx']/1e6:.1f}M active", fg="cyan")
