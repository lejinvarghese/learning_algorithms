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


def get_datasets(n_train, n_eval, seq_len, frame_size, num_frames, use_audio=False, use_video=False):
    train_sets = [
        HFTextDataset("train", n_train, seq_len, frame_size, num_frames),
        HFImageCaptionDataset("train", n_train, seq_len, frame_size, num_frames),
    ]
    test_sets = [
        HFTextDataset("val", n_eval, seq_len, frame_size, num_frames),
        HFImageCaptionDataset("val", n_eval, seq_len, frame_size, num_frames),
    ]

    # Add audio dataset if enabled (smaller audio length to save memory)
    if use_audio:
        from k3.audio_data import HFAudioDataset
        # Pass num_frames and frame_size to match vision dataset tensor shapes
        train_sets.append(HFAudioDataset("train", n_train, seq_len, max_audio_len=1500,
                                         num_frames=num_frames, frame_size=frame_size))
        test_sets.append(HFAudioDataset("val", n_eval, seq_len, max_audio_len=1500,
                                        num_frames=num_frames, frame_size=frame_size))

    # Add video dataset if enabled
    if use_video:
        from k3.video_data import HFVideoDataset
        train_sets.append(HFVideoDataset("train", n_train, seq_len,
                                         num_frames=num_frames, frame_size=frame_size))
        test_sets.append(HFVideoDataset("val", n_eval, seq_len,
                                        num_frames=num_frames, frame_size=frame_size))

    return ConcatDataset(train_sets), ConcatDataset(test_sets)


@click.command()
@click.option("--epochs", type=int, default=2, show_default=True, help="number of training epochs")
@click.option("--batch-size", type=int, default=8, show_default=True, help="training batch size")
@click.option("--grad-accum", type=int, default=4, show_default=True, help="gradient accumulation steps")
@click.option("--n-train", type=int, default=10_000, show_default=True, help="samples per source, training split (reduced for memory)")
@click.option("--n-eval", type=int, default=100, show_default=True, help="samples per source, evaluation split (small - eval only uses first 10 batches)")
@click.option("--resume", type=click.Path(exists=True), default=None, help="resume from checkpoint")
@click.option("--adam", is_flag=True, help="use Adam instead of Muon (10-20x faster optimizer)")
@click.option(
    "--active-experts", type=int, default=8, help="number of active experts"
)
@click.option(
    "--total-experts", type=int, default=512, help="total number of routed experts"
)
@click.option(
    "--use-audio", is_flag=True, help="enable audio encoder and load audio dataset"
)
@click.option(
    "--use-video", is_flag=True, help="enable video dataset (lv12/MultiModalDataset openvid config)"
)
@click.option(
    "--expert-size", type=int, default=64, help="hidden dim per routed expert"
)
@click.option(
    "--hidden-dim", type=int, default=64, help="model hidden dimension"
)
@click.option(
    "--vocab-size", type=int, default=163840, help="vocabulary size (must match tokenizer)"
)
def main(epochs, batch_size, n_train, n_eval, resume, adam, active_experts, total_experts, use_audio, use_video, expert_size, hidden_dim, vocab_size, grad_accum):

    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision="no",
        log_with="wandb",
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
        # Sparse expert config matching paper's design: many experts, few active
        # 128 experts @ 3.1% activation (4/128) - closer to paper's 2.0% (18/896)
        cfg = K3Config(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_blocks=4,
            layers_per_block=4,
            num_routed_experts=total_experts,
            num_experts_active=active_experts,
            num_shared_experts=1,
            moe_hidden_per_expert=expert_size,
            shared_moe_hidden=expert_size * 2,
            use_vision=True,
            use_audio=use_audio,
            use_gradient_checkpointing=True,
            vit_num_frames=4,
        )
        model = K3Model(cfg)

    counts = model.param_counts()

    frame_size = cfg.vit_patch_size * 8
    train_data, test_data = get_datasets(
        n_train=n_train, n_eval=n_eval, seq_len=64, frame_size=frame_size, num_frames=4,
        use_audio=use_audio, use_video=use_video
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=12,
        pin_memory=True,
        prefetch_factor=8,
        persistent_workers=True,
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=4, pin_memory=True, prefetch_factor=2)

    # Choose optimizer
    if adam:
        # Fused AdamW: CUDA-optimized, 2-3× faster than standard
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=5e-4,  # Higher LR for training from scratch with large model
            betas=(0.9, 0.95),
            weight_decay=0.01,
            fused=torch.cuda.is_available(),
        )
        if accelerator.is_main_process:
            click.secho("Using fused AdamW optimizer (fast mode, modern hyperparams)", fg="green")
    else:
        opt = create_k3_optimizer(model, cfg, muon_lr=0.001)
        if accelerator.is_main_process:
            click.secho("Using K3 optimizer (Muon - slow but high quality)", fg="yellow")

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.3)  # 30% warmup - MoE needs gentle ramp with low peak LR

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1e-10, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps - warmup_steps, eta_min=1.5e-4 * 0.1  # 10% of peak LR
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
            "total_params": counts["total"],
            "active_params": counts["activated_approx"],
            "vocab_size": cfg.vocab_size,
            "num_experts": cfg.num_routed_experts,
            "active_experts": cfg.num_experts_active,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum,
            "effective_batch_size": batch_size * grad_accum,
            "optimizer": "AdamW" if adam else "K3-Muon",
            "learning_rate": 5e-4 if adam else 1e-3,
            "epochs": epochs,
        },
    )

    if accelerator.is_main_process:
        click.secho(
            f"K3 {counts['total'] / 1e6:.1f}M params ({counts['activated_approx'] / 1e6:.1f}M active) | "
            f"{len(train_data)} train, {len(test_data)} eval | {accelerator.device}",
            fg="cyan",
        )

    first_loss = None
    global_step = 0
    grad_norm_history = []  # For adaptive gradient clipping

    for epoch in range(start_epoch, epochs + 1):
        for step, (ids, images, has_visual, audio_mel, has_audio) in enumerate(train_loader, start=1):
            global_step += 1
            with accelerator.accumulate(model):
                logits, mtp_logits, aux_losses = model(
                    ids, images=images, has_visual=has_visual, audio_mel=audio_mel, has_audio=has_audio
                )

                # Logits include prepended vision/audio tokens, need to extract text portion
                # logits shape: [B, text_len + prepended_tokens, vocab_size]
                # ids shape: [B, text_len]
                text_len = ids.shape[1]
                num_prepended = logits.shape[1] - text_len

                # Extract only text logits (skip prepended tokens)
                text_logits = logits[:, num_prepended:, :]

                targets = ids.roll(-1, dims=1)
                # Mask out padding tokens (0s) from loss - they're not real targets
                mask = (targets[:, :-1] != 0).reshape(-1)
                loss = F.cross_entropy(
                    text_logits[:, :-1].reshape(-1, cfg.vocab_size)[mask],
                    targets[:, :-1].reshape(-1)[mask]
                ) if mask.any() else torch.tensor(0.0, device=ids.device)

                if mtp_logits is not None:
                    mtp_targets = ids.roll(-2, dims=1)
                    loss = loss + 0.1 * F.cross_entropy(
                        mtp_logits[:, :-2].reshape(-1, cfg.vocab_size), mtp_targets[:, :-2].reshape(-1)
                    )

                # Add auxiliary losses for MoE stability
                # Reduced coefficients: 1e-2 was too constraining, causing routing collapse
                loss = loss + 1e-3 * aux_losses["router_z_loss"] + 1e-3 * aux_losses["load_balance_loss"]

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
                if first_loss is None:
                    first_loss = loss_val

                # Log to wandb (no modality separation - user found it not useful)
                accelerator.log(
                    {
                        "train/loss": loss_val,
                        "train/grad_norm": total_norm.item(),
                        "train/grad_clip": clip_value,
                        "train/router_z_loss": aux_losses["router_z_loss"].item(),
                        "train/load_balance_loss": aux_losses["load_balance_loss"].item(),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )

                # Simple console logging
                click.secho(
                    f"[step {global_step}] epoch {epoch}/{epochs} step {step}/{len(train_loader)}: loss={loss_val:.4f}",
                    fg="green" if loss_val <= first_loss else "red",
                )

            # Evaluate every 20 steps (frequent checks, limited batches to save memory)
            if global_step % 20 == 0:
                metrics = evaluate(model, test_loader, cfg.vocab_size, accelerator.device, max_batches=10)
                if accelerator.is_main_process:
                    accelerator.log({
                        "eval/loss": metrics['loss'],
                        "eval/accuracy": metrics['accuracy'],
                        "eval/top5_accuracy": metrics['top5_accuracy'],
                    }, step=global_step)
                    click.secho(
                        f"[step {global_step}] eval: loss={metrics['loss']:.4f} acc={metrics['accuracy']:.2%} top5={metrics['top5_accuracy']:.2%}",
                        fg="blue",
                    )
                # Clear CUDA cache after eval
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Save checkpoint every 500 steps
            if global_step % 500 == 0 and accelerator.is_main_process:
                ckpt_dir = Path("checkpoints")
                ckpt_dir.mkdir(exist_ok=True)
                ckpt_path = ckpt_dir / f"k3_step{global_step}.pt"
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": unwrapped_model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "config": cfg,
                    },
                    ckpt_path,
                )
                click.secho(f"💾 Checkpoint saved: {ckpt_path}", fg="magenta")

                # Keep only latest checkpoint to save disk space
                old_ckpts = sorted(ckpt_dir.glob("k3_step*.pt"), key=lambda p: p.stat().st_mtime)[:-1]
                for old_ckpt in old_ckpts:
                    old_ckpt.unlink()
                    click.secho(f"🗑️  Removed old checkpoint: {old_ckpt.name}", fg="yellow")

        # End of epoch - just log it
        if accelerator.is_main_process:
            click.secho(f"✓ Epoch {epoch}/{epochs} complete", fg="cyan")

    if accelerator.is_main_process and accelerator.device.type == "cuda":
        click.secho(f"peak CUDA memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", fg="magenta")

    # Clean up wandb
    accelerator.end_training()


if __name__ == "__main__":
    main()
