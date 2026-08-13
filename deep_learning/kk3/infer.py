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
from k3.tokenizer import get_k3_tokenizer


def load_checkpoint(ckpt_path: str, device: str):
    """Load model from checkpoint."""
    # weights_only=False is safe for your own checkpoints
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = K3Model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load tokenizer
    tokenizer = get_k3_tokenizer()

    click.secho(f"Loaded checkpoint from epoch {ckpt['epoch']}", fg="green")
    return model, cfg, tokenizer


def encode_text(text: str, tokenizer, seq_len: int = 128) -> torch.Tensor:
    """Encode text using K3 tokenizer."""
    tokens = tokenizer.encode(text, add_special_tokens=False)[:seq_len]
    ids = torch.zeros(seq_len, dtype=torch.long)
    ids[:len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    return ids


def decode_text(ids: torch.Tensor, tokenizer) -> str:
    """Decode tokens using K3 tokenizer."""
    # Stop at first zero (padding) or special tokens
    ids = ids.cpu().numpy()
    # Remove padding
    end = np.where(ids == 0)[0]
    if len(end) > 0:
        ids = ids[:end[0]]

    return tokenizer.decode(ids.tolist(), skip_special_tokens=True)


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
    tokenizer,
    prompt_ids: torch.Tensor,
    images: torch.Tensor = None,
    max_new_tokens: int = 64,
    temperature: float = 0.2,
    device: str = "cuda",
):
    """Generate text continuation.

    Note: This uses naive autoregressive generation without KV/state caching.
    Each iteration reprocesses the full sequence. For K3's hybrid attention:
    - KDA could cache recurrent state S
    - Gated MLA could cache K/V matrices
    But this requires model architecture changes (not implemented yet).
    """
    prompt_ids = prompt_ids.to(device).unsqueeze(0)  # (1, seq_len)
    has_visual = torch.tensor(1.0 if images is not None else 0.0, device=device)

    if images is not None:
        images = images.to(device)

    generated = prompt_ids.clone()
    prompt_len = (prompt_ids != 0).sum().item()

    for i in range(max_new_tokens):
        current_len = prompt_len + i

        # Only process up to current length (minor optimization)
        current_seq = generated[:, :current_len]

        # Forward pass - still reprocesses everything (needs caching for real speedup)
        logits, _, _ = model(current_seq, images=images, has_visual=has_visual)

        # Sample next token (greedy if temp very low, otherwise sample)
        next_token_logits = logits[0, -1]  # Last position
        if temperature < 0.01:
            next_token = next_token_logits.argmax().item()
        else:
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

        # Stop at padding, invalid tokens, or EOS
        if next_token == 0 or next_token >= cfg.vocab_size:
            break
        if next_token == tokenizer.eos_token_id:
            break

        # Add to sequence
        if current_len < generated.shape[1]:
            generated[0, current_len] = next_token
        else:
            break

    return generated[0]


@click.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True), help="Path to checkpoint")
@click.option("--text", type=str, help="Text prompt for completion")
@click.option("--image", type=click.Path(exists=True), help="Image path for captioning")
@click.option("--max-tokens", type=int, default=64, help="Max tokens to generate")
@click.option("--temperature", type=float, default=0.01, help="Sampling temperature")
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

    # Load model and tokenizer
    model, cfg, tokenizer = load_checkpoint(checkpoint, device)

    # Prepare inputs
    if image:
        # Image captioning mode
        click.secho(f"\n📷 Image captioning: {image}", fg="yellow")
        images = load_image(image, size=cfg.vit_patch_size * 8, num_frames=cfg.vit_num_frames)

        # Start with empty or a prompt like "a photo of"
        prompt = text if text else ""
        prompt_ids = encode_text(prompt, tokenizer, seq_len=128)

        click.secho(f"Prompt: '{prompt}'", fg="cyan")
    else:
        # Text completion mode
        click.secho(f"\n📝 Text completion", fg="yellow")
        images = None
        prompt_ids = encode_text(text, tokenizer, seq_len=128)
        click.secho(f"Prompt: '{text}'", fg="cyan")

    # Generate
    click.secho("Generating...", fg="yellow")
    output_ids = generate(
        model, cfg, tokenizer, prompt_ids, images, max_new_tokens=max_tokens, temperature=temperature, device=device
    )

    # Decode and display
    output_text = decode_text(output_ids, tokenizer)
    click.secho(f"\n✨ Generated:", fg="green", bold=True)
    click.secho(output_text, fg="white")


if __name__ == "__main__":
    main()
