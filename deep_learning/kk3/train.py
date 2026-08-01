import click
import torch
import torch.nn.functional as F
from pathlib import Path
from accelerate import Accelerator
from torch.utils.data import ConcatDataset, DataLoader

from k3 import K3Config, K3Model
from k3.eval import evaluate
from k3.hf_data import HFImageCaptionDataset, HFTextDataset

from optimizer import create_k3_optimizer


def _datasets(n_train, n_eval, seq_len, frame_size, num_frames):

    train_sets = [
        HFTextDataset("train", n_train, seq_len, frame_size, num_frames),
        HFImageCaptionDataset("train", n_train, seq_len, frame_size, num_frames),
    ]
    test_sets = [
        HFTextDataset("val", n_eval, seq_len, frame_size, num_frames),
        HFImageCaptionDataset("val", n_eval, seq_len, frame_size, num_frames),
    ]

    return ConcatDataset(train_sets), ConcatDataset(test_sets)


def _deepspeed_plugin(cpu_offload: bool):
    if not cpu_offload:
        return None
    if not torch.cuda.is_available():
        click.secho("--cpu-offload requires CUDA; no CUDA device found, ignoring.", fg="yellow")
        return None
    from accelerate.utils import DeepSpeedPlugin

    return DeepSpeedPlugin(zero_stage=2, offload_optimizer_device="cpu")


@click.command()
@click.option("--epochs", type=int, default=3, show_default=True, help="number of training epochs")
@click.option("--batch-size", type=int, default=4, show_default=True, help="training batch size")
@click.option("--n-train", type=int, default=100_000, show_default=True, help="samples per source, training split")
def main(epochs, batch_size, n_train):
    # Hardcoded configuration
    mult = 1.0
    vision = False
    seq_len = 32
    num_frames = 4
    n_eval = 1_000
    grad_checkpoint = True
    grad_accum = 1
    mixed_precision = "bf16"
    cpu_offload = True
    grad_clip = 1.0
    muon_lr = 0.005
    warmup_ratio = 0.01
    min_lr_ratio = 0.1

    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision=mixed_precision,
        deepspeed_plugin=_deepspeed_plugin(cpu_offload),
    )

    cfg = K3Config.scaled(
        mult, use_vision=vision, use_gradient_checkpointing=grad_checkpoint, vit_num_frames=num_frames
    )
    model = K3Model(cfg)
    counts = model.param_counts()

    frame_size = cfg.vit_patch_size * 8
    train_data, test_data = _datasets(n_train, n_eval, seq_len, frame_size, num_frames)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    opt = create_k3_optimizer(model, cfg, muon_lr=muon_lr)

    # Cosine annealing with warmup using PyTorch's built-in schedulers
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    # Warmup phase: linear ramp from 0 to peak LR
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1e-10, end_factor=1.0, total_iters=warmup_steps
    )
    # Cosine decay phase: from peak LR to min_lr_ratio * peak LR
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps - warmup_steps, eta_min=muon_lr * min_lr_ratio
    )
    # Chain them together
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )

    if accelerator.is_main_process:
        opt_name = "K3 Optimizer (per-head Muon + Muon + Adam)"
        if cpu_offload:
            click.secho(f"{opt_name} with DeepSpeed ZeRO-2", fg="green")
        else:
            click.secho(opt_name, fg="green")
        click.secho(f"LR: {muon_lr}, warmup {warmup_steps}/{total_steps} steps ({warmup_ratio:.0%}), cosine decay to {min_lr_ratio:.0%}", fg="cyan")

    model, opt, train_loader, test_loader = accelerator.prepare(model, opt, train_loader, test_loader)

    # Verify CPU offload and optimizer configuration
    if accelerator.is_main_process:
        if cpu_offload and hasattr(accelerator.state, "deepspeed_plugin"):
            # Check if DeepSpeed is actually offloading
            if hasattr(model, "optimizer") and hasattr(model.optimizer, "optimizer_swapper"):
                click.secho("✓ DeepSpeed ZeRO-2 CPU offload active", fg="green")
            elif hasattr(model, "config") and hasattr(model.config, "zero_optimization"):
                zero_cfg = model.config.zero_optimization
                if zero_cfg.get("offload_optimizer", {}).get("device") == "cpu":
                    click.secho("✓ DeepSpeed configured for CPU offload", fg="green")
                    # Check optimizer state location
                    import psutil

                    process = psutil.Process()
                    cpu_mem_gb = process.memory_info().rss / 1e9
                    click.secho(f"  CPU RAM usage: {cpu_mem_gb:.2f} GB", fg="cyan")
            else:
                click.secho("⚠ DeepSpeed enabled but offload status unclear", fg="yellow")

        # Show GPU memory before training starts
        if torch.cuda.is_available():
            gpu_mem_allocated = torch.cuda.memory_allocated() / 1e9
            gpu_mem_reserved = torch.cuda.memory_reserved() / 1e9
            click.secho(
                f"GPU memory: {gpu_mem_allocated:.2f} GB allocated, {gpu_mem_reserved:.2f} GB reserved", fg="cyan"
            )

    if accelerator.is_main_process:
        click.secho(f"device={accelerator.device} mixed_precision={mixed_precision}", fg="cyan")
        click.secho(
            f"config: hidden={cfg.hidden_dim} layers={cfg.num_layers} "
            f"routed_experts={cfg.num_routed_experts} active={cfg.num_experts_active} "
            f"vocab={cfg.vocab_size} vision={cfg.use_vision}",
            fg="cyan",
        )
        click.secho(f"train samples: {len(train_data)}  eval samples: {len(test_data)}", fg="cyan")
        click.secho(f"total params:     {counts['total'] / 1e6:8.2f} M", fg="yellow")
        click.secho(
            f"activated approx: {counts['activated_approx'] / 1e6:8.2f} M  "
            f"(sparse MoE => far fewer FLOPs/token than total params)",
            fg="yellow",
        )

    first_loss, global_step = None, 0
    for epoch in range(1, epochs + 1):
        for step, (ids, images, has_visual) in enumerate(train_loader, start=1):
            with accelerator.accumulate(model):
                logits, mtp_logits = model(ids, images=images, has_visual=has_visual)
                targets = ids.roll(-1, dims=1)
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size), targets[:, :-1].reshape(-1))
                if mtp_logits is not None:
                    mtp_targets = ids.roll(-2, dims=1)
                    loss = loss + 0.1 * F.cross_entropy(
                        mtp_logits[:, :-2].reshape(-1, cfg.vocab_size), mtp_targets[:, :-2].reshape(-1)
                    )

                accelerator.backward(loss)
                if grad_clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                scheduler.step()
                opt.zero_grad()

            if accelerator.is_main_process:
                loss_val = loss.item()
                first_loss = first_loss if first_loss is not None else loss_val
                global_step += 1

                # Show memory usage after first step to verify offload
                if global_step == 1 and cpu_offload:
                    import psutil

                    process = psutil.Process()
                    cpu_mem_gb = process.memory_info().rss / 1e9
                    if torch.cuda.is_available():
                        gpu_mem_gb = torch.cuda.memory_allocated() / 1e9
                        click.secho(
                            f"After first step - GPU: {gpu_mem_gb:.2f} GB, CPU RAM: {cpu_mem_gb:.2f} GB "
                            f"(optimizer state should be on CPU)",
                            fg="magenta",
                        )

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

            # Save checkpoint
            ckpt_dir = Path("checkpoints")
            ckpt_dir.mkdir(exist_ok=True)
            ckpt_path = ckpt_dir / f"k3_epoch{epoch}.pt"
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": unwrapped_model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "config": cfg,
                    "eval_metrics": metrics,
                },
                ckpt_path,
            )
            click.secho(f"Saved checkpoint: {ckpt_path}", fg="magenta")

    if accelerator.is_main_process and accelerator.device.type == "cuda":
        click.secho(f"peak CUDA memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", fg="magenta")


if __name__ == "__main__":
    main()
