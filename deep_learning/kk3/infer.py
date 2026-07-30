#!/usr/bin/env python3
"""
Inference script for K3 model - supports text completion and image captioning.

Usage:
    # Text completion
    python infer.py --checkpoint checkpoints/k3_epoch3.pt --text "hello world"

    # Image captioning
    python infer.py --checkpoint checkpoints/k3_epoch3.pt --image path/to/image.jpg
"""
import click
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np

from k3 import K3Model


def load_checkpoint(ckpt_path: str, device: str):
    """Load model from checkpoint."""
    # weights_only=False is safe for your own checkpoints
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = K3Model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    click.secho(f"Loaded checkpoint from epoch {ckpt['epoch']}", fg="green")
    click.secho(f"Eval metrics: loss={ckpt['eval_metrics']['loss']:.4f}, "
                f"accuracy={ckpt['eval_metrics']['accuracy']:.2%}", fg="cyan")
    return model, cfg


def encode_text(text: str, seq_len: int = 128) -> torch.Tensor:
    """Encode text as UTF-8 bytes."""
    raw = text.encode("utf-8")[:seq_len]
    ids = torch.zeros(seq_len, dtype=torch.long)
    ids[:len(raw)] = torch.tensor(list(raw), dtype=torch.long)
    return ids


def decode_text(ids: torch.Tensor) -> str:
    """Decode UTF-8 bytes back to text."""
    # Stop at first zero (padding)
    ids = ids.cpu().numpy()
    end = np.where(ids == 0)[0]
    if len(end) > 0:
        ids = ids[:end[0]]
    try:
        return bytes(ids).decode("utf-8", errors="ignore")
    except:
        return "[decode error]"


def load_image(image_path: str, size: int = 112, num_frames: int = 4) -> torch.Tensor:
    """Load and preprocess image."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    # Repeat to create "frames" (static image)
    frames = img_tensor.unsqueeze(0).repeat(num_frames, 1, 1, 1)
    return frames.unsqueeze(0)  # Add batch dim


@torch.no_grad()
def generate(
    model,
    cfg,
    prompt_ids: torch.Tensor,
    images: torch.Tensor = None,
    max_new_tokens: int = 64,
    temperature: float = 0.1,
    device: str = "cuda"
):
    """Generate text continuation."""
    prompt_ids = prompt_ids.to(device).unsqueeze(0)  # (1, seq_len)
    has_visual = torch.tensor(1.0 if images is not None else 0.0, device=device)

    if images is not None:
        images = images.to(device)

    generated = prompt_ids.clone()
    prompt_len = (prompt_ids != 0).sum().item()

    for i in range(max_new_tokens):
        # Forward pass
        logits, _ = model(generated, images=images, has_visual=has_visual)

        # Sample next token
        next_token_logits = logits[0, prompt_len + i - 1] / temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()

        # Stop at padding or invalid tokens
        if next_token == 0 or next_token >= cfg.vocab_size:
            break

        # Add to sequence
        if prompt_len + i < generated.shape[1]:
            generated[0, prompt_len + i] = next_token
        else:
            break

    return generated[0]


@click.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True), help="Path to checkpoint")
@click.option("--text", type=str, help="Text prompt for completion")
@click.option("--image", type=click.Path(exists=True), help="Image path for captioning")
@click.option("--max-tokens", type=int, default=64, help="Max tokens to generate")
@click.option("--temperature", type=float, default=0.8, help="Sampling temperature")
@click.option("--device", type=str, default=None, help="Device (auto-detect if not specified)")
def main(checkpoint, text, image, max_tokens, temperature, device):
    """Run inference with K3 model."""
    if text is None and image is None:
        click.secho("Error: Must provide either --text or --image", fg="red")
        return

    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    click.secho(f"Using device: {device}", fg="cyan")

    # Load model
    model, cfg = load_checkpoint(checkpoint, device)

    # Prepare inputs
    if image:
        # Image captioning mode
        click.secho(f"\n📷 Image captioning: {image}", fg="yellow")
        images = load_image(image, size=cfg.vit_patch_size * 8, num_frames=cfg.vit_num_frames)

        # Start with empty or a prompt like "a photo of"
        prompt = text if text else ""
        prompt_ids = encode_text(prompt, seq_len=128)

        click.secho(f"Prompt: '{prompt}'", fg="cyan")
    else:
        # Text completion mode
        click.secho(f"\n📝 Text completion", fg="yellow")
        images = None
        prompt_ids = encode_text(text, seq_len=128)
        click.secho(f"Prompt: '{text}'", fg="cyan")

    # Generate
    click.secho("Generating...", fg="yellow")
    output_ids = generate(
        model, cfg, prompt_ids, images,
        max_new_tokens=max_tokens,
        temperature=temperature,
        device=device
    )

    # Decode and display
    output_text = decode_text(output_ids)
    click.secho(f"\n✨ Generated:", fg="green", bold=True)
    click.secho(output_text, fg="white")


if __name__ == "__main__":
    main()
