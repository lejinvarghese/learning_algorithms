# K3-Micro

A shape-faithful, fractionally-scaled replica of [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3): same
architectural mechanisms, ~1000x smaller, built to fit a 6GB GPU.

## What's replicated (see `k3micro/layers.py`, `k3micro/model.py`)

| K3 mechanism | Report ref | File |
|---|---|---|
| Kimi Delta Attention (delta-rule + channel-wise lower-bounded decay + full-rank gate) | §2.1.1, Eq. 1-6 | `layers.py: KimiDeltaAttention` |
| Gated MLA (latent q/kv compression, NoPE) | §2.1.2, Eq. 7 | `layers.py: GatedMLA` |
| Hybrid attention, 3 KDA : 1 MLA per block, final layer forced global | §2.1 | `model.py: build_layer_pattern` |
| Attention Residuals (depth-wise attention pool over prior block outputs) | §2.2, Eq. 8-10 | `layers.py: AttnResGate` |
| Stable LatentMoE: shared + latent-space routed experts | §2.3, Eq. 11 | `layers.py: StableLatentMoE` |
| SiTU-GLU (dual-softcap GLU) | §2.3.2, Eq. 12 | `layers.py: SiTUGLU` |
| Quantile Balancing (aux-loss-free routing bias) | §2.3.3, Eq. 13-14 | `layers.py: QuantileBalancingRouter` |
| Native vision (ViT -> pixel-shuffle -> projector into shared embedding space) | §2.4 | `model.py: MoonViTMicro` |
| Multi-Token Prediction head | §4.1.4 | `model.py: MTPHead` |

What's simplified, and why it doesn't change the shape:
- **KDA recurrence** is a plain sequential loop, not the chunkwise UT-transform parallel kernel —
  same math (Eq. 1), just no throughput optimization needed at micro sequence lengths.
- **MoE dispatch** is a dense masked loop over experts, not a scatter/gather kernel — fine when
  `num_routed_experts` is tens, not ~900.
- **Quantile Balancing** uses an exact `torch.quantile` instead of the cross-rank histogram
  approximation — the histogram exists only to make the quantile affordable across GPUs at
  ~10^3 experts; single-process exact quantile computes the identical update (Eq. 14).
- **Vision** is single-image, flat-patch ViT — MoonViT-3D's factorized intra-frame/inter-frame
  attention for video is dropped for concision; the pixel-shuffle token-reduction trick is kept.

## Scaling

Absolute widths (hidden dim, #experts, #layers, ViT size) scale via a multiplier; K3's
architectural *ratios* (KDA:MLA = 3:1, 2 shared experts, latent-MoE width = 0.5x hidden, 1 MTP
layer) are fixed constants, matching the real model's shape regardless of scale:

```python
from k3micro import K3MicroConfig, K3MicroModel

cfg = K3MicroConfig.scaled(1.0)     # default micro config, fits comfortably in 6GB
cfg = K3MicroConfig.scaled(0.5)     # smaller still
cfg = K3MicroConfig.scaled(2.0)     # if you have more headroom
model = K3MicroModel(cfg)
```

Or override any field directly: `K3MicroConfig(hidden_dim=192, num_routed_experts=16, ...)`.

## Run

Trains on `k3micro/data.py`'s tiny procedural dataset: colored-block images paired with
byte-tokenized captions describing them (e.g. "a red square in the top-left") -- no download,
pure torch, just enough signal to confirm the vision splice and text loss are both learning.

```bash
pip install -r requirements.txt
python train.py                # smoke-test: builds the model, prints param counts, trains a few steps
python train.py --mult 0.5     # scale down further
python train.py --no-vision    # drop the vision tower to save memory
python train.py --help         # full option list (click-based CLI)
```
