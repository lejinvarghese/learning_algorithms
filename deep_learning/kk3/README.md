# K3

A shape-faithful replica of [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3), scaled down
by roughly three orders of magnitude: the same architectural mechanisms, sized to train and run
comfortably on modest hardware.

## What's replicated (see `k3/layers.py`, `k3/model.py`)

| Kimi K3 mechanism | Implementation |
|---|---|
| Kimi Delta Attention: delta-rule linear attention with a bounded forget gate and a full-rank output gate | `layers.py: KimiDeltaAttention` |
| Gated MLA: latent q/kv compression, no positional encoding | `layers.py: GatedMLA` |
| Hybrid attention, KDA layers closed out by a Gated MLA layer per block, final layer forced global | `model.py: build_layer_pattern` |
| Attention Residuals: depth-wise attention pool over prior block outputs | `layers.py: AttnResGate` |
| Stable LatentMoE: shared + latent-space routed experts | `layers.py: StableLatentMoE` |
| SiTU-GLU: dual-softcap gated linear unit | `layers.py: SiTUGLU` |
| Quantile Balancing: auxiliary-loss-free routing bias | `layers.py: QuantileBalancingRouter` |
| Native vision: ViT, token merging, projector into the shared embedding space | `model.py: MoonViT` |
| Multi-token prediction head | `model.py: MTPHead` |

What's simplified, and why it doesn't change the shape:
- **KDA's recurrence** is a plain sequential loop rather than a chunked parallel kernel — same
  math, just without the throughput optimization that only matters at long sequence lengths.
- **Expert dispatch** is a dense masked loop rather than a scatter/gather kernel — fine when the
  expert pool is tens, not hundreds, of experts.
- **Quantile Balancing** computes an exact quantile rather than approximating one from a
  cross-rank histogram — the histogram exists only to make the quantile affordable when sharding
  a huge expert pool across many devices; computed directly, it's the identical update.
- **Vision** is single-image, flat-patch ViT — factorized frame-to-frame attention for video is
  left out, though the token-merging step that keeps long multimodal sequences affordable is kept.

## Scaling

Absolute widths (hidden dim, expert count, layer count, vision tower size) scale via a
multiplier; Kimi K3's architectural ratios (KDA-to-MLA mix, shared expert count, latent-MoE
width, one multi-token-prediction layer) are fixed constants, so the shape holds at any scale:

```python
from k3 import K3Config, K3Model

cfg = K3Config.scaled(1.0)     # the defaults
cfg = K3Config.scaled(0.5)     # smaller
cfg = K3Config.scaled(2.0)     # larger
model = K3Model(cfg)
```

Or override any field directly: `K3Config(hidden_dim=192, num_routed_experts=16, ...)`.

## Run

Trains on `k3/data.py`'s tiny procedural dataset: colored-block images paired with byte-tokenized
captions describing them (e.g. "a red square in the top-left") — no download required, just
enough signal to confirm the vision splice and text loss are both learning. Each epoch logs
per-step training loss, then evaluates on a held-out split (`k3/eval.py: evaluate`) and logs
loss and next-token accuracy on unseen samples.

```bash
pip install -r requirements.txt
python train.py                # builds the model, prints param counts, trains for a few epochs
python train.py --epochs 10    # train longer
python train.py --mult 0.5     # scale down further
python train.py --no-vision    # drop the vision tower
python train.py --help         # full option list
```
