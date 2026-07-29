"""Smoke test: builds K3-Micro, prints param counts, trains a few steps on
the tiny procedural (caption, image) dataset in k3micro/data.py. Sized to fit
a 6GB GPU at the default config.

    python train.py                  # default micro config
    python train.py --mult 0.5       # scale everything down further
    python train.py --no-vision      # drop the vision tower to save memory
"""
import click
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from k3micro import K3MicroConfig, K3MicroModel
from k3micro.data import ToyMultimodalDataset


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@click.command()
@click.option("--mult", type=float, default=1.0, show_default=True, help="scale factor over micro defaults")
@click.option("--no-vision", is_flag=True, help="drop the vision tower to save memory")
@click.option("--steps", type=int, default=5, show_default=True)
@click.option("--batch-size", type=int, default=4, show_default=True)
@click.option("--seq-len", type=int, default=32, show_default=True)
def main(mult: float, no_vision: bool, steps: int, batch_size: int, seq_len: int):
    cfg = K3MicroConfig.scaled(mult, use_vision=not no_vision)
    device = pick_device()
    model = K3MicroModel(cfg).to(device)

    counts = model.param_counts()
    click.secho(f"device={device}", fg="cyan")
    click.secho(
        f"config: hidden={cfg.hidden_dim} layers={cfg.num_layers} "
        f"routed_experts={cfg.num_routed_experts} active={cfg.num_experts_active} "
        f"vocab={cfg.vocab_size} vision={cfg.use_vision}",
        fg="cyan",
    )
    click.secho(f"total params:     {counts['total'] / 1e6:8.2f} M", fg="yellow")
    click.secho(
        f"activated approx: {counts['activated_approx'] / 1e6:8.2f} M  "
        f"(sparse MoE => far fewer FLOPs/token than total params)",
        fg="yellow",
    )

    dataset = ToyMultimodalDataset(n_samples=32, seq_len=seq_len, image_size=cfg.vit_patch_size * 8)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    step = 0
    first_loss = None
    while step < steps:
        for ids, images in loader:
            if step >= steps:
                break
            ids = ids.to(device)
            images = images.to(device) if cfg.use_vision else None

            logits, mtp_logits = model(ids, images=images)
            targets = ids.roll(-1, dims=1)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size), targets[:, :-1].reshape(-1))
            if mtp_logits is not None:
                mtp_targets = ids.roll(-2, dims=1)
                loss = loss + 0.1 * F.cross_entropy(
                    mtp_logits[:, :-2].reshape(-1, cfg.vocab_size), mtp_targets[:, :-2].reshape(-1)
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            loss_val = loss.item()
            first_loss = first_loss if first_loss is not None else loss_val
            color = "green" if loss_val <= first_loss else "red"
            click.secho(f"step {step}: loss={loss_val:.4f}", fg=color)
            step += 1

    if device == "cuda":
        click.secho(f"peak CUDA memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", fg="magenta")


if __name__ == "__main__":
    main()
