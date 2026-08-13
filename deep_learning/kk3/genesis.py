#!/usr/bin/env python3
"""Transfer weights with custom K3 config matching train.py exactly."""
import torch
from k3 import K3Config, K3Model
from base.transfer_weights import transfer_embeddings, transfer_layer_norms, transfer_moe_from_ffn, transfer_vision_encoder
from base.transfer_audio import transfer_audio_encoder
from transformers import AutoModel, WhisperModel
import click

# Balanced config: expert_size = hidden_dim = 64
cfg = K3Config(
    vocab_size=163840,
    hidden_dim=64,
    num_blocks=4,
    layers_per_block=4,
    num_routed_experts=512,
    num_experts_active=8,
    num_shared_experts=1,
    moe_hidden_per_expert=64,
    shared_moe_hidden=128,
    use_vision=True,
    use_audio=True,
    use_gradient_checkpointing=True,
    vit_num_frames=4,
)

click.secho(f"🎯 Creating K3 model with custom config", fg="cyan")
click.secho(f"   hidden_dim={cfg.hidden_dim}, routed_experts={cfg.num_routed_experts}", fg="cyan")

model = K3Model(cfg)
total_params = sum(p.numel() for p in model.parameters())
click.secho(f"   Total params: {total_params/1e6:.1f}M\n", fg="green")

# Load ancestral models
click.secho("📥 Loading ancestral models...", fg="cyan")
source_lang = AutoModel.from_pretrained("HuggingFaceTB/SmolLM2-360M")
source_vision = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
source_audio = WhisperModel.from_pretrained("openai/whisper-base")
click.secho("   ✓ Language (SmolLM2), Vision (CLIP), Audio (Whisper) loaded\n", fg="green")

# Transfer components (evolutionary crossbreeding)
click.secho("📦 Transferring weights with diverse expert initialization...", fg="cyan")
transfer_embeddings(source_lang, model, cfg)
transfer_layer_norms(source_lang, model, cfg)
transfer_moe_from_ffn(source_lang, model, cfg)
transfer_vision_encoder(source_vision, model, cfg)
transfer_audio_encoder(source_audio, model, cfg)

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
