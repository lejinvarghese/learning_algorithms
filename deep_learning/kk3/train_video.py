import click
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader

from k3 import K3Config, K3Model
from k3.eval import evaluate
from k3.video import RealVideoDataset


def _deepspeed_plugin(cpu_offload: bool):
    if not cpu_offload:
        return None
    if not torch.cuda.is_available():
        click.secho("--cpu-offload requires CUDA; no CUDA device found, ignoring.", fg="yellow")
        return None
    from accelerate.utils import DeepSpeedPlugin

    return DeepSpeedPlugin(zero_stage=2, offload_optimizer_device="cpu")


@click.command()
@click.option("--mult", type=float, default=1.0, show_default=True, help="scale factor over the default config")
@click.option("--epochs", type=int, default=3, show_default=True)
@click.option("--num-frames", type=int, default=8, show_default=True, help="frames sampled per clip")
@click.option("--seq-len", type=int, default=32, show_default=True)
@click.option("--grad-checkpoint", is_flag=True, help="recompute activations on backward to cut peak memory")
@click.option("--mixed-precision", type=click.Choice(["no", "fp16", "bf16"]), default="no", show_default=True)
@click.option("--cpu-offload", is_flag=True, help="ZeRO-Offload optimizer state to CPU RAM (needs CUDA + deepspeed)")
def main(mult, epochs, num_frames, seq_len, grad_checkpoint, mixed_precision, cpu_offload):
    accelerator = Accelerator(mixed_precision=mixed_precision, deepspeed_plugin=_deepspeed_plugin(cpu_offload))

    cfg = K3Config.scaled(mult, vit_num_frames=num_frames, use_gradient_checkpointing=grad_checkpoint)
    model = K3Model(cfg)
    counts = model.param_counts()

    frame_size = cfg.vit_patch_size * 8
    if accelerator.is_main_process:
        click.secho("downloading/caching test clips (first run only)...", fg="cyan")
    train_data = RealVideoDataset(num_frames=num_frames, frame_size=frame_size, seq_len=seq_len, split="train")
    test_data = RealVideoDataset(num_frames=num_frames, frame_size=frame_size, seq_len=seq_len, split="test")
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1)
    opt = AdamW(model.parameters(), lr=3e-4)

    model, opt, train_loader, test_loader = accelerator.prepare(model, opt, train_loader, test_loader)

    if accelerator.is_main_process:
        click.secho(f"device={accelerator.device} mixed_precision={mixed_precision}", fg="cyan")
        click.secho(
            f"config: hidden={cfg.hidden_dim} layers={cfg.num_layers} vit_layers={cfg.vit_layers} "
            f"vit_num_frames={cfg.vit_num_frames} vit_temporal_pool={cfg.vit_temporal_pool}",
            fg="cyan",
        )
        click.secho(f"total params:     {counts['total'] / 1e6:8.2f} M", fg="yellow")
        click.secho(
            f"activated approx: {counts['activated_approx'] / 1e6:8.2f} M  "
            f"(sparse MoE => far fewer FLOPs/token than total params)",
            fg="yellow",
        )

    first_loss, global_step = None, 0
    for epoch in range(1, epochs + 1):
        for step, (ids, images) in enumerate(train_loader, start=1):
            logits, mtp_logits = model(ids, images=images)
            targets = ids.roll(-1, dims=1)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size), targets[:, :-1].reshape(-1))
            if mtp_logits is not None:
                mtp_targets = ids.roll(-2, dims=1)
                loss = loss + 0.1 * F.cross_entropy(
                    mtp_logits[:, :-2].reshape(-1, cfg.vocab_size), mtp_targets[:, :-2].reshape(-1)
                )

            accelerator.backward(loss)
            opt.step()
            opt.zero_grad()

            if accelerator.is_main_process:
                loss_val = loss.item()
                first_loss = first_loss if first_loss is not None else loss_val
                global_step += 1
                click.secho(
                    f"epoch {epoch}/{epochs} step {step}/{len(train_loader)} (global {global_step}): "
                    f"loss={loss_val:.4f}",
                    fg="green" if loss_val <= first_loss else "red",
                )

        metrics = evaluate(model, test_loader, cfg.vocab_size, accelerator.device)
        if accelerator.is_main_process:
            click.secho(
                f"epoch {epoch}/{epochs} eval: loss={metrics['loss']:.4f} accuracy={metrics['accuracy']:.2%}",
                fg="blue",
            )

    if accelerator.is_main_process and accelerator.device.type == "cuda":
        click.secho(f"peak CUDA memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", fg="magenta")


if __name__ == "__main__":
    main()
