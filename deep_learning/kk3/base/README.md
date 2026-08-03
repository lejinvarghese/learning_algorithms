# Pretrained Weight Transfer

Initialize K3 from pretrained small foundation models instead of training from scratch.

## Quick Start

```bash
# Transfer from SmolLM2-360M (default, recommended)
uv run python -m base.transfer_weights

# Or specify source and output
uv run python -m base.transfer_weights --source smollm2-360m --output checkpoints/k3_init.pt

# Then train from this checkpoint
python train.py --resume checkpoints/k3_init.pt
```

## Available Source Models

**Language:**
- `smollm2-135m` - HuggingFace SmolLM2-135M (135M params, 2T tokens)
- `smollm2-360m` - HuggingFace SmolLM2-360M (360M params, 4T tokens) **[recommended]**
- `qwen2.5-0.5b` - Qwen2.5-0.5B (494M params, multilingual)

**Vision:**
- `siglip-so400m` - Google SigLIP SO400M (93M params, used in SmolVLM) **[default]**

## What Gets Transferred

✅ **Embeddings** - Adapted from BPE vocab to byte-level (256 tokens)
- Uses embedding distribution statistics from source model

✅ **Layer Norms** - Direct copy where dimensions match
- Input/output norms for each layer
- Final model normalization

✅ **MoE Experts** - Initialized from dense FFN layers
- Each expert starts with same FFN weights + gaussian noise
- Encourages specialization during fine-tuning

✅ **Vision Encoder** - SigLIP → MoonViT transfer
- Patch embedding (conv layer)
- Transformer blocks (attention, FFN, norms)
- Adapted from google/siglip-so400m-patch14-384

❌ **Attention** - Left random (K3 uses KDA/MLA, incompatible with standard attention)

## Expected Benefits

- **2-5x faster convergence** vs random init
- **Better final performance** with same compute budget
- **Data efficiency** from 2-4T token pretraining

## Architecture Mapping

Source → K3:
```
Source Model (Qwen/SmolLM)     K3 Model
├─ embed_tokens               → embed (vocab adapted 50K→256)
├─ layers[i]
│  ├─ input_layernorm         → blocks[j].layers[k].gate.norm
│  ├─ self_attn               → [random - KDA/MLA incompatible]
│  ├─ post_attention_layernorm→ blocks[j].layers[k].moe.pre_norm
│  └─ mlp (up/down)           → routed_experts + shared_experts (cloned with noise)
└─ norm                       → final_norm
```

## Custom Config

```python
from k3 import K3Config
from base import transfer_from_pretrained

cfg = K3Config(
    hidden_dim=512,      # Match or adapt from source
    num_blocks=6,
    layers_per_block=4,
)

model = transfer_from_pretrained("smollm2-360m", cfg, "my_init.pt")
```

## Implementation

- `adapters.py` - Weight adaptation utilities
- `transfer_weights.py` - Main transfer logic
- `__init__.py` - Public API

See source files for detailed adaptation strategies.
