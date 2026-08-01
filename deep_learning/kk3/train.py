import click
import torch
import torch.nn.functional as F
from pathlib import Path
from accelerate import Accelerator
from torch.utils.data import ConcatDataset, DataLoader

from k3 import K3Config, K3Model
from k3.eval import evaluate
from k3.hf_data import HFImageCaptionDataset, HFTextDataset

from optimizer import K3Optimizer


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
@click.option("--mult", type=float, default=1.0, show_default=True, help="scale factor over the default config")
@click.option("--no-vision", is_flag=True, help="drop the vision tower")
@click.option("--epochs", type=int, default=3, show_default=True)
@click.option("--batch-size", type=int, default=4, show_default=True)
@click.option("--seq-len", type=int, default=32, show_default=True)
@click.option("--num-frames", type=int, default=4, show_default=True, help="frames per video clip / per image")
@click.option("--n-train", type=int, default=100_000, show_default=True, help="samples per source, training split")
@click.option("--n-eval", type=int, default=1_000, show_default=True, help="samples per source, eval split")
@click.option("--grad-checkpoint/--no-grad-checkpoint", default=True, show_default=True)
@click.option("--grad-accum", type=int, default=1, show_default=True, help="steps to accumulate before an update")
@click.option("--mixed-precision", type=click.Choice(["no", "fp16", "bf16"]), default="no", show_default=True)
@click.option("--cpu-offload/--no-cpu-offload", default=True, show_default=True, help="use DeepSpeed ZeRO-2 (gradient sharding)")
@click.option("--grad-clip", type=float, default=1.0, show_default=True, help="gradient clipping max norm")
@click.option("--muon-lr", type=float, default=0.005, show_default=True, help="Muon optimizer learning rate")
@click.option("--warmup-ratio", type=float, default=0.1, show_default=True, help="warmup as fraction of total steps")
@click.option("--min-lr-ratio", type=float, default=0.1, show_default=True, help="minimum LR as fraction of peak LR")
def main(
    mult,
    no_vision,
    epochs,
    batch_size,
    seq_len,
    num_frames,
    n_train,
    n_eval,
    grad_checkpoint,
    grad_accum,
    mixed_precision,
    cpu_offload,
    grad_clip,
    muon_lr,
    warmup_ratio,
    min_lr_ratio,
):
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision=mixed_precision,
        deepspeed_plugin=_deepspeed_plugin(cpu_offload),
    )

    cfg = K3Config.scaled(
        mult, use_vision=not no_vision, use_gradient_checkpointing=grad_checkpoint, vit_num_frames=num_frames
    )
    model = K3Model(cfg)
    counts = model.param_counts()

    frame_size = cfg.vit_patch_size * 8
    train_data, test_data = _datasets(n_train, n_eval, seq_len, frame_size, num_frames)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # K3 optimizer: per-head Muon for Q/K/V, standard Muon for other 2D+, Adam for 1D
    per_head_muon_params = []
    muon_params = []
    adam_params = []

    for name, param in model.named_parameters():
        # Q, K, V projections in KimiDeltaAttention and GatedMLA get per-head Muon
        if param.ndim >= 2 and any(x in name for x in ['q_lin', 'k_lin', 'v_lin', 'q_up', 'k_up', 'v_up']):
            per_head_muon_params.append((name, param))
        # Other 2D+ parameters (excluding embeddings) get standard Muon
        elif param.ndim >= 2 and "embed" not in name.lower():
            muon_params.append(param)
        # 1D parameters (norms, biases) and embeddings get Adam
        else:
            adam_params.append(param)

    # Create parameter groups for K3Optimizer
    param_groups = []

    # Per-head Muon for Q, K, V projections
    if per_head_muon_params:
        for name, param in per_head_muon_params:
            param_groups.append({
                'params': [param],
                'optimizer_type': 'per_head_muon',
                'num_heads': cfg.num_heads,
                'lr': muon_lr,
                'momentum': 0.95,
            })

    # Standard Muon for other matrix parameters
    if muon_params:
        param_groups.append({
            'params': muon_params,
            'optimizer_type': 'muon',
            'lr': muon_lr,
            'momentum': 0.95,
        })

    # Adam for 1D parameters and embeddings
    if adam_params:
        param_groups.append({
            'params': adam_params,
            'optimizer_type': 'adam',
            'lr': muon_lr / 10,  # Adam uses 10x lower LR
            'betas': (0.9, 0.95),
            'eps': 1e-10,
        })

    opt = K3Optimizer(param_groups)

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
        n_per_head = len(per_head_muon_params)
        n_muon = len(muon_params)
        n_adam = len(adam_params)
        if cpu_offload:
            click.secho(f"Using K3 Optimizer: {n_per_head} per-head Muon, {n_muon} Muon, {n_adam} Adam params with DeepSpeed ZeRO-2", fg="green")
        else:
            click.secho(f"Using K3 Optimizer: {n_per_head} per-head Muon, {n_muon} Muon, {n_adam} Adam params", fg="green")
        click.secho(f"LR schedule: warmup {warmup_steps}/{total_steps} steps ({warmup_ratio:.0%}), cosine decay to {min_lr_ratio:.0%} of peak", fg="cyan")

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
