import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader

from k3 import K3Config, K3Model
from k3.data import ToyMultimodalDataset
from k3.eval import evaluate
from k3.hf_data import HFImageCaptionDataset, HFTextDataset
from k3.video import RealVideoDataset

from muon import SingleDeviceMuon as Muon


def _datasets(toy, n_train, n_eval, seq_len, frame_size, num_frames):
    if toy:
        train_sets = [
            ToyMultimodalDataset(32, seq_len, frame_size, num_frames, seed=0),
            RealVideoDataset(num_frames, frame_size, seq_len, split="train"),
        ]
        test_sets = [
            ToyMultimodalDataset(16, seq_len, frame_size, num_frames, seed=1),
            RealVideoDataset(num_frames, frame_size, seq_len, split="test"),
        ]
        return ConcatDataset(train_sets), ConcatDataset(test_sets)

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

    # DeepSpeed ZeRO-2 for gradient sharding
    # Note: Muon doesn't support optimizer offload, so just using ZeRO-2
    ds_config = {
        "zero_optimization": {
            "stage": 2,
        },
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
    }

    return DeepSpeedPlugin(hf_ds_config=ds_config)


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
@click.option("--toy", is_flag=True, help="offline synthetic data instead of streaming from Hugging Face")
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
    toy,
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
    train_data, test_data = _datasets(toy, n_train, n_eval, seq_len, frame_size, num_frames)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Hybrid optimizer: Muon for 2D+ params (conv/linear), AdamW for 1D (embed/norm)
    muon_params = []
    adamw_params = []
    for name, param in model.named_parameters():
        if param.ndim >= 2 and "embed" not in name.lower():
            muon_params.append(param)
        else:
            adamw_params.append(param)

    muon_opt = Muon(muon_params, lr=muon_lr, momentum=0.95)
    adamw_opt = AdamW(adamw_params, lr=muon_lr / 10)  # AdamW uses 10x lower LR

    class HybridOptimizer:
        def __init__(self, muon, adamw):
            self.muon = muon
            self.adamw = adamw

        def step(self):
            self.muon.step()
            self.adamw.step()

        def zero_grad(self):
            self.muon.zero_grad()
            self.adamw.zero_grad()

        def state_dict(self):
            return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

        def load_state_dict(self, state_dict):
            self.muon.load_state_dict(state_dict["muon"])
            self.adamw.load_state_dict(state_dict["adamw"])

    opt = HybridOptimizer(muon_opt, adamw_opt)
    if accelerator.is_main_process:
        if cpu_offload:
            click.secho(f"Using Muon ({len(muon_params)} params) + AdamW ({len(adamw_params)} params) with DeepSpeed ZeRO-2", fg="green")
        else:
            click.secho(f"Using Muon ({len(muon_params)} params) + AdamW ({len(adamw_params)} params)", fg="green")

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
