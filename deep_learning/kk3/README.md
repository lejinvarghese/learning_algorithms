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
- Audio encoder added (Whisper-style, 80 mel bins, 6 transformer layers) — extends beyond Kimi K3's text+vision.

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

`train.py` trains on [lv12/MultiModalDataset](https://huggingface.co/datasets/lv12/MultiModalDataset):
- **fineweb**: Text-only (high-quality web content)
- **coco**: Image-caption pairs
- **audioset**: Audio clips with captions (Whisper-style encoder)
- **openvid**: Video clips (4 frames/video, spatial-temporal attention)

Unified 5-item format: `(ids, frames, has_visual, audio_mel, has_audio)`. All modalities batch together; 
unused modalities zeroed via mask flags.

`--toy` switches to fully offline data instead: `k3/data.py`'s procedural colored-block images
and `k3/video.py`'s four small cached real video clips — no network dependency, useful for a
quick sanity check or CI.

## Genesis: Evolutionary Initialization

Create K3 from ancestral models (SmolLM2-360M + CLIP):

```bash
uv run python genesis.py
# → checkpoints/k3_pretrained_init.pt (74M total, 24M active)
```

**Transferred:**
- Embeddings (960d→64d adapted)
- Layer norms
- MoE experts (dense FFN→512 diverse experts)
- Vision encoder (CLIP→MoonViT)

Attention initialized randomly (KDA/MLA incompatible with standard attention).

## Essential Commands

```bash
# 1. Create pretrained init
uv run python genesis.py

# 2. Quick test (all modalities, pretrained init)
./scripts/quick_test.sh

# 3. Full training
uv run python train.py \
    --adam \
    --use-audio \
    --use-video \
    --n-train 5000 \
    --batch-size 1 \
    --grad-accum 4 \
    --epochs 2 \
    --resume checkpoints/k3_pretrained_init.pt
```

**Current architecture (balanced config):**
- hidden_dim: 64 (small activations)
- expert_size: 64 (no bottleneck)
- total_experts: 512
- active_experts: 8 (1.5% sparsity)
- **Result: 74M total, 24M active (33% active)**

**Key flags:**
- `--adam`: AdamW optimizer (faster than Muon)
- `--use-audio/--use-video`: Enable modalities
- `--grad-accum`: Gradient accumulation (effective batch = batch × accum)
- `--resume`: Start from checkpoint

**Hardcoded configuration:**
- Optimizer: K3 (per-head Muon for Q/K/V, Muon for 2D+, Adam for 1D)
- Learning rate: 0.001 with 20% warmup, cosine decay to 10%
- Adaptive gradient clipping (95th percentile, min 0.5)
- MoE auxiliary losses: router z-loss (1e-3) + load balancing (1e-2)
- Loss spike detection: skip batches with NaN/Inf/loss>5000
- DataLoader: 12 workers, prefetch_factor=8, persistent workers
- Gradient checkpointing enabled
- No mixed precision (fp32 for Newton-Schulz stability)
- No DeepSpeed (removed - adds overhead for small models)

Training runs via [Accelerate](https://huggingface.co/docs/accelerate). Checkpoints saved to 
`checkpoints/k3_epoch{N}.pt` after each epoch with model weights, optimizer state, config, and metrics.

## Inference

```bash
python infer.py --checkpoint checkpoints/k3_epoch1.pt --text "hello world"
python infer.py --checkpoint checkpoints/k3_epoch1.pt --image dog.png
python infer.py --checkpoint checkpoints/k3_epoch1.pt --image photo.jpg --text "a photo of"
python infer.py --checkpoint checkpoints/k3_epoch1.pt --text "once upon a time" --max-tokens 128
```

**Options:**
- `--checkpoint`: Path to checkpoint (required)
- `--text`: Text prompt for completion
- `--image`: Image path for captioning
- `--max-tokens`: Maximum tokens to generate (default: 64)
- `--temperature`: Sampling temperature (default: 0.8)
- `--device`: Device override (auto-detects if not specified)

The model uses Kimi K3's tiktoken-based BPE tokenizer (163,840 tokens). Multimodal inputs (images, video, audio) 
are encoded as continuous token sequences prepended to text, preserving temporal structure for video.

## Publish to HuggingFace

```bash
python convert_to_hf.py --checkpoint checkpoints/k3_epoch5.pt --output my_model
huggingface-cli login
huggingface-cli upload username/model-name ./my_model
```
