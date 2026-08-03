import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import click
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import ConcatDataset, DataLoader
from accelerate import Accelerator

from k3 import K3Config, K3Model
from k3.eval import evaluate
from k3.data import HFImageCaptionDataset, HFTextDataset

from optimizer import create_k3_optimizer


def get_datasets(n_train, n_eval, seq_len, frame_size, num_frames):

    train_sets = [
        HFTextDataset("train", n_train, seq_len, frame_size, num_frames),
        HFImageCaptionDataset("train", n_train, seq_len, frame_size, num_frames),
    ]
    test_sets = [
        HFTextDataset("val", n_eval, seq_len, frame_size, num_frames),
        HFImageCaptionDataset("val", n_eval, seq_len, frame_size, num_frames),
    ]

    return ConcatDataset(train_sets), ConcatDataset(test_sets)


@click.command()
@click.option("--epochs", type=int, default=2, show_default=True, help="number of training epochs")
@click.option("--batch-size", type=int, default=12, show_default=True, help="training batch size")
@click.option("--n-train", type=int, default=100_000, show_default=True, help="samples per source, training split")
@click.option("--n-eval", type=int, default=1_000, show_default=True, help="samples per source, evaluation split")
@click.option("--resume", type=click.Path(exists=True), default=None, help="resume from checkpoint")
@click.option("--adam", is_flag=True, help="use Adam instead of Muon (10-20x faster optimizer)")
@click.option("--active-experts", type=int, default=4, help="number of active experts (default: 4, balanced for 6GB GPU)")
def main(epochs, batch_size, n_train, n_eval, resume, adam, active_experts):

    # Removed DeepSpeed - adds overhead for small models, CPU offloading slows down optimizer steps
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="no",  # fp32 required for Muon's Newton-Schulz stability
        log_with="wandb",  # Built-in wandb integration
    )

    # Load from checkpoint or create new
    start_epoch = 1
    ckpt = None
    if resume:
        if accelerator.is_main_process:
            click.secho(f"Loading checkpoint: {resume}", fg="cyan")
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        model = K3Model(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        if accelerator.is_main_process:
            click.secho(f"Resumed from epoch {ckpt.get('epoch', 0)}", fg="green")
    else:
        # Sparse config: 70M total, ~50M active with 4 experts (3→4 for better GPU util without OOM)
        cfg = K3Config(
            vocab_size=163840,
            hidden_dim=192,
            num_blocks=4,
            layers_per_block=4,
            num_routed_experts=128,
            num_experts_active=active_experts,
            num_shared_experts=1,
            moe_hidden_per_expert=48,
            shared_moe_hidden=96,
            use_vision=True,
            use_gradient_checkpointing=True,
            vit_num_frames=4,
        )
        model = K3Model(cfg)

    counts = model.param_counts()

    frame_size = cfg.vit_patch_size * 8
    train_data, test_data = get_datasets(
        n_train=n_train, n_eval=n_eval, seq_len=64, frame_size=frame_size, num_frames=4
    )
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=12, pin_memory=True, prefetch_factor=8, persistent_workers=True
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=12, pin_memory=True, prefetch_factor=8)

    # Choose optimizer
    if adam:
        # Fused AdamW: CUDA-optimized, 2-3× faster than standard
        # LR at 8e-4: sweet spot between stability (5e-4 too slow) and spikes (1e-3 too high)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=8e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused=torch.cuda.is_available(),  # Use fused kernels on CUDA
        )
        if accelerator.is_main_process:
            click.secho("Using fused AdamW optimizer (fast mode, modern hyperparams)", fg="green")
    else:
        opt = create_k3_optimizer(model, cfg, muon_lr=0.001)
        if accelerator.is_main_process:
            click.secho("Using K3 optimizer (Muon - slow but high quality)", fg="yellow")

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.3)  # 30% warmup for MoE router stability

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1e-10, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps - warmup_steps, eta_min=8e-4 * 0.1  # 10% of peak LR
    )
    # Chain them together
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )

    model, opt, train_loader, test_loader = accelerator.prepare(model, opt, train_loader, test_loader)

    # Resume optimizer state if available
    if ckpt is not None and "optimizer_state_dict" in ckpt:
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        if accelerator.is_main_process:
            click.secho("Optimizer state restored", fg="green")

    # Initialize wandb tracker
    accelerator.init_trackers(
        project_name="k3-training",
        config={
            "model": "K3",
            "total_params": counts['total'],
            "active_params": counts['activated_approx'],
            "vocab_size": cfg.vocab_size,
            "num_experts": cfg.num_routed_experts,
            "active_experts": cfg.num_experts_active,
            "batch_size": batch_size,
            "optimizer": "AdamW" if adam else "K3-Muon",
            "learning_rate": 1e-3,
            "epochs": epochs,
        }
    )

    if accelerator.is_main_process:
        click.secho(
            f"K3 {counts['total'] / 1e6:.1f}M params ({counts['activated_approx'] / 1e6:.1f}M active) | "
            f"{len(train_data)} train, {len(test_data)} eval | {accelerator.device}",
            fg="cyan",
        )

    first_loss, global_step = None, 0
    grad_norm_history = []  # For adaptive gradient clipping

    for epoch in range(start_epoch, epochs + 1):
        for step, (ids, images, has_visual) in enumerate(train_loader, start=1):
            with accelerator.accumulate(model):
                logits, mtp_logits, aux_losses = model(ids, images=images, has_visual=has_visual)
                targets = ids.roll(-1, dims=1)
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size), targets[:, :-1].reshape(-1))
                if mtp_logits is not None:
                    mtp_targets = ids.roll(-2, dims=1)
                    loss = loss + 0.1 * F.cross_entropy(
                        mtp_logits[:, :-2].reshape(-1, cfg.vocab_size), mtp_targets[:, :-2].reshape(-1)
                    )

                # Add auxiliary losses for MoE stability
                # Router z-loss: increased from 1e-3 to 1e-2 to combat logit explosion
                loss = loss + 1e-2 * aux_losses["router_z_loss"] + 1e-2 * aux_losses["load_balance_loss"]

                # Check for loss spikes
                loss_val = loss.item()
                if torch.isnan(loss).any() or torch.isinf(loss).any() or loss_val > 5000.0:
                    if accelerator.is_main_process:
                        click.secho(f"⚠ Skipping step {step}: loss spike {loss_val:.2f}", fg="yellow")
                    opt.zero_grad()
                    continue

                accelerator.backward(loss)

                # Gradient norm monitoring and adaptive clipping
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
                grad_norm_history.append(total_norm.item())
                if len(grad_norm_history) > 100:
                    grad_norm_history.pop(0)

                clip_value = np.percentile(grad_norm_history, 95) if len(grad_norm_history) > 20 else 1.0
                accelerator.clip_grad_norm_(model.parameters(), max(clip_value, 0.5))

                opt.step()
                scheduler.step()
                opt.zero_grad()

            if accelerator.is_main_process:
                first_loss = first_loss if first_loss is not None else loss_val
                global_step += 1

                # Log to wandb
                accelerator.log({
                    "train/loss": loss_val,
                    "train/grad_norm": total_norm.item(),
                    "train/grad_clip": clip_value,
                    "train/router_z_loss": aux_losses["router_z_loss"].item(),
                    "train/load_balance_loss": aux_losses["load_balance_loss"].item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                }, step=global_step)

                # Console log every 10 steps
                if step % 10 == 0:
                    router_z = aux_losses["router_z_loss"].item()
                    load_bal = aux_losses["load_balance_loss"].item()
                    click.secho(
                        f"epoch {epoch}/{epochs} step {step}/{len(train_loader)}: "
                        f"loss={loss_val:.4f} grad_norm={total_norm:.2f} clip={clip_value:.2f} "
                        f"router_z={router_z:.2e} load_bal={load_bal:.2e}",
                        fg="green" if loss_val <= first_loss else "red",
                    )
                else:
                    click.secho(
                        f"epoch {epoch}/{epochs} step {step}/{len(train_loader)}: loss={loss_val:.4f}",
                        fg="green" if loss_val <= first_loss else "red",
                    )

        metrics = evaluate(model, test_loader, cfg.vocab_size, accelerator.device)
        if accelerator.is_main_process:
            # Log eval metrics
            accelerator.log({
                "eval/loss": metrics['loss'],
                "eval/accuracy": metrics['accuracy'],
                "epoch": epoch,
            }, step=global_step)

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

    # Clean up wandb
    accelerator.end_training()


if __name__ == "__main__":
    main()
