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
uv sync                                      # install dependencies
python train.py                              # default: 5 epochs, batch 16, 100K samples/modality
python train.py --epochs 10                  # train for 10 epochs
python train.py --batch-size 32              # larger batch size
python train.py --n-train 10000              # 10K samples per modality
python train.py --n-eval 5000                # 5K eval samples
python train.py --epochs 10 --n-train 50000  # combine options
```

**Available options:**
- `--epochs`: Number of training epochs (default: 5)
- `--batch-size`: Training batch size (default: 16)
- `--n-train`: Training samples per modality (default: 100,000)
- `--n-eval`: Evaluation samples per modality (default: 1,000)

**Hardcoded configuration:**
- Optimizer: K3 (per-head Muon for Q/K/V, Muon for 2D+, Adam for 1D)
- Learning rate: 0.001 with 20% warmup, cosine decay to 10%
- Adaptive gradient clipping (95th percentile, min 0.5)
- MoE auxiliary losses: router z-loss (1e-3) + load balancing (1e-2)
- Loss spike detection: skip batches with NaN/Inf/loss>100
- DeepSpeed ZeRO-2 with CPU offload (auto-enabled on CUDA)
- Gradient checkpointing enabled
- No mixed precision (fp32 for Newton-Schulz stability)

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

The model generates byte-level UTF-8 sequences. For image captioning, vision embeddings condition 
the generation as continuous features (not discrete tokens).

## Publish to HuggingFace

```bash
python convert_to_hf.py --checkpoint checkpoints/k3_epoch5.pt --output my_model
huggingface-cli login
huggingface-cli upload username/model-name ./my_model
```
