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
| Native vision, images and video: factorized spatial/temporal attention, temporal pooling, token merging | `model.py: MoonViT` |
| Multi-token prediction head | `model.py: MTPHead` |

Simplifications, and why they don't change the shape:
- **KDA's recurrence** is a plain sequential loop rather than a chunked parallel kernel — same
  math, just without the throughput optimization that only matters at long sequence lengths.
- **Expert dispatch** is a dense masked loop rather than a scatter/gather kernel — fine when the
  expert pool is tens, not hundreds, of experts.
- **Quantile Balancing** computes an exact quantile rather than approximating one from a
  cross-rank histogram — the histogram exists only to make the quantile affordable when sharding
  a huge expert pool across many devices; computed directly, it's the identical update.
- No audio — Kimi K3 itself has no audio modality (text + native vision only).

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

## Data

`train.py` trains on [lv12/MultiModalDataset](https://huggingface.co/datasets/lv12/MultiModalDataset) 
streamed from Hugging Face (`k3/hf_data.py`):
- **fineweb** config: 37.5K high-quality web text samples (>8,192 tokens each)
- **coco** config: 37.5K COCO 2017 image-caption pairs (512x512 resolution)
- **audioset** config: 10K audio samples with human-annotated labels (audio encoder not yet implemented)

Every sample — text-only or image-paired — is shaped identically (`ids`, `frames: [num_frames, 3, H, W]`, 
`has_visual`) so they batch together in one `DataLoader`; text-only samples carry an all-zero frame tensor 
and `has_visual=0`, which zeroes out the vision contribution in `K3Model.forward` for that sample. 
Audio support requires implementing an audio encoder to process the audioset config.

`--toy` switches to fully offline data instead: `k3/data.py`'s procedural colored-block images
and `k3/video.py`'s four small cached real video clips — no network dependency, useful for a
quick sanity check or CI.

## Run

```bash
uv sync                                # install dependencies (or pip install -r requirements.txt)
python train.py                        # streams text + image from lv12/MultiModalDataset
python train.py --n-train 10000        # larger dataset (10K samples per modality)
python train.py --n-train 30000 --epochs 10 --warmup-ratio 0.1  # full dataset with cosine LR decay
python train.py --toy                  # fully offline procedural data instead
python train.py --mult 0.5             # scale the model down further
python train.py --no-vision            # drop the vision tower (text only)
python train.py --grad-checkpoint      # recompute activations on backward (lower peak memory)
python train.py --mixed-precision bf16 # fp16/bf16 training via Accelerate
python train.py --cpu-offload          # DeepSpeed ZeRO-2 optimizer offload (needs CUDA)
python train.py --help                 # full option list
```

**Learning rate schedule**: Training uses cosine annealing with warmup by default:
- `--warmup-ratio 0.1`: Warmup for 10% of total steps (default)
- `--min-lr-ratio 0.1`: Decay to 10% of peak LR (default)
- `--muon-lr 0.005`: Peak learning rate (default 0.005 for Muon, 0.0005 for AdamW)

Each epoch logs per-step training loss, then evaluates on a held-out split (`k3/eval.py:
evaluate`) and logs loss and next-token accuracy on unseen samples.

`--grad-checkpoint` and `--cpu-offload` default **on**; both degrade gracefully where they don't
apply (checkpointing costs a bit of recompute even when memory isn't tight; `--cpu-offload` is a
silent no-op without CUDA). Training runs via [Accelerate](https://huggingface.co/docs/accelerate),
so `python train.py` is enough for CPU, MPS, or a single GPU — `accelerate launch` is only needed
for multi-GPU.

Checkpoints are saved after each epoch to `checkpoints/k3_epoch{N}.pt`, containing model weights,
optimizer state, config, and eval metrics.

## Inference

After training, use `infer.py` to generate text or captions:

**Text completion:**
```bash
python infer.py --checkpoint checkpoints/k3_epoch3.pt --text "hello world"
```

**Image captioning:**
```bash
python infer.py --checkpoint checkpoints/k3_epoch3.pt --image photo.jpg
```

**Image captioning with prompt:**
```bash
python infer.py --checkpoint checkpoints/k3_epoch3.pt --image photo.jpg --text "a photo of"
```

**Options:**
- `--max-tokens`: Maximum tokens to generate (default: 64)
- `--temperature`: Sampling temperature, higher = more random (default: 0.8)
- `--device`: Force device (auto-detects CUDA/MPS/CPU if not specified)

The model generates byte-level sequences (UTF-8 encoding), so output is character-by-character
rather than tokenized words. For image captioning, vision embeddings are added to text embeddings
as continuous features (not discrete tokens), conditioning the generation on the image.
