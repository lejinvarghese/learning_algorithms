# Builds K3, prints parameter counts, and trains for a few epochs on the toy dataset in
# k3/data.py, evaluating on a held-out split after each epoch.
import click
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from k3 import K3Config, K3Model
from k3.data import ToyMultimodalDataset
from k3.eval import evaluate


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@click.command()
@click.option("--mult", type=float, default=1.0, show_default=True, help="scale factor over the default config")
@click.option("--no-vision", is_flag=True, help="drop the vision tower")
@click.option("--epochs", type=int, default=3, show_default=True)
@click.option("--batch-size", type=int, default=4, show_default=True)
@click.option("--seq-len", type=int, default=32, show_default=True)
def main(mult: float, no_vision: bool, epochs: int, batch_size: int, seq_len: int):
    cfg = K3Config.scaled(mult, use_vision=not no_vision)
    device = pick_device()
    model = K3Model(cfg).to(device)

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

    image_size = cfg.vit_patch_size * 8
    train_data = ToyMultimodalDataset(n_samples=32, seq_len=seq_len, image_size=image_size, seed=0)
    test_data = ToyMultimodalDataset(n_samples=16, seq_len=seq_len, image_size=image_size, seed=1)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    first_loss, global_step = None, 0
    for epoch in range(1, epochs + 1):
        for step, (ids, images) in enumerate(train_loader, start=1):
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
            global_step += 1
            click.secho(
                f"epoch {epoch}/{epochs} step {step}/{len(train_loader)} (global {global_step}): "
                f"loss={loss_val:.4f}",
                fg="green" if loss_val <= first_loss else "red",
            )

        metrics = evaluate(model, test_loader, cfg.vocab_size, device)
        click.secho(
            f"epoch {epoch}/{epochs} eval: loss={metrics['loss']:.4f} accuracy={metrics['accuracy']:.2%}",
            fg="blue",
        )

    if device == "cuda":
        click.secho(f"peak CUDA memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", fg="magenta")


if __name__ == "__main__":
    main()
