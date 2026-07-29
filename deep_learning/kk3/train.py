"""Smoke test: builds K3-Micro, prints param counts, runs a few optimizer
steps on synthetic data. Sized to fit a 6GB GPU at the default config.

    python train.py                  # default micro config
    python train.py --mult 0.5       # scale everything down further
    python train.py --no-vision      # drop the vision tower to save memory
"""
import argparse

import torch
import torch.nn.functional as F

from k3micro import K3MicroConfig, K3MicroModel


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mult", type=float, default=1.0, help="scale factor over micro defaults")
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=128)
    args = ap.parse_args()

    cfg = K3MicroConfig.scaled(args.mult, use_vision=not args.no_vision)
    device = pick_device()
    model = K3MicroModel(cfg).to(device)

    counts = model.param_counts()
    print(f"device={device}")
    print(f"config: hidden={cfg.hidden_dim} layers={cfg.num_layers} "
          f"routed_experts={cfg.num_routed_experts} active={cfg.num_experts_active} "
          f"vocab={cfg.vocab_size} vision={cfg.use_vision}")
    print(f"total params:     {counts['total'] / 1e6:8.2f} M")
    print(f"activated approx: {counts['activated_approx'] / 1e6:8.2f} M  "
          f"(sparse MoE => far fewer FLOPs/token than total params)")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for step in range(args.steps):
        ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device)
        images = torch.randn(args.batch_size, 3, cfg.vit_patch_size * 8, cfg.vit_patch_size * 8, device=device) \
            if cfg.use_vision else None

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
        print(f"step {step}: loss={loss.item():.4f}")

    if device == "cuda":
        print(f"peak CUDA memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
