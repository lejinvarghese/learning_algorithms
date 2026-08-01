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
@click.option("--epochs", type=int, default=3, show_default=True, help="number of training epochs")
@click.option("--batch-size", type=int, default=16, show_default=True, help="training batch size")
@click.option("--n-train", type=int, default=100_000, show_default=True, help="samples per source, training split")
def main(epochs, batch_size, n_train):
    # Hardcoded configuration
    mult = 1.0
    vision = True
    seq_len = 32
    num_frames = 4
    n_eval = 1_000
    grad_checkpoint = True
    grad_accum = 1
    mixed_precision = "no"
    grad_clip = 1.0
    muon_lr = 0.005
    warmup_ratio = 0.01
    min_lr_ratio = 0.1

    from accelerate.utils import DeepSpeedPlugin

    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision=mixed_precision,
        deepspeed_plugin=DeepSpeedPlugin(zero_stage=2, offload_optimizer_device="cpu") if torch.cuda.is_available() else None,
    )

    cfg = K3Config.scaled(
        mult, use_vision=vision, use_gradient_checkpointing=grad_checkpoint, vit_num_frames=num_frames
    )
    model = K3Model(cfg)
    counts = model.param_counts()

    frame_size = cfg.vit_patch_size * 8
    train_data, test_data = get_datasets(n_train, n_eval, seq_len, frame_size, num_frames)
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

    model, opt, train_loader, test_loader = accelerator.prepare(model, opt, train_loader, test_loader)

    if accelerator.is_main_process:
        click.secho(f"K3 {counts['total'] / 1e6:.1f}M params ({counts['activated_approx'] / 1e6:.1f}M active) | "
                    f"{len(train_data)} train, {len(test_data)} eval | {accelerator.device}", fg="cyan")

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
                click.secho(f"epoch {epoch}/{epochs} step {step}/{len(train_loader)}: loss={loss_val:.4f}",
                           fg="green" if loss_val <= first_loss else "red")

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
