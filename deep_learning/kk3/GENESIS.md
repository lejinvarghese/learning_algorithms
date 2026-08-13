# Genesis: Evolutionary Model Initialization

## Ancestral Species

K3 inherits genetic material from two pretrained foundation models:

### Language Ancestor: SmolLM2-360M
- **Genome**: 360M parameters trained on 4T tokens
- **Lineage**: HuggingFaceTB/SmolLM2-360M
- **Inherited traits**:
  - Token embeddings (960d → 64d adapted)
  - Layer normalization patterns
  - FFN knowledge (dense → 512 sparse experts)

### Vision Ancestor: CLIP ViT-Base
- **Genome**: 93M parameters trained on 400M image-text pairs
- **Lineage**: openai/clip-vit-base-patch32
- **Inherited traits**:
  - Patch embedding (spatial feature extraction)
  - Vision MLP layers (adapted to projector)

## Evolutionary Mechanisms

### 1. **Crossbreeding** (Heterogeneous Transfer)
Combines genetic material from two unrelated species:
```
Language DNA (SmolLM2) + Vision DNA (CLIP) → Multimodal K3
```

### 2. **Mutation** (Adaptive Noise)
Each MoE expert receives unique mutations for diversity:
- **Shared experts**: 1-2% Gaussian noise
- **Routed experts**: 2-4% progressive noise (increases with expert index)
- **Seed**: `42 + layer_idx * 1000 + expert_idx` (deterministic diversity)

**Why**: Identical clones would collapse to same function. Mutation forces specialization.

### 3. **Speciation** (Dimension Adaptation)
Adapts ancestor traits to new environment:
- **Embeddings**: 960d → 64d (PCA-like truncation, keeps high-variance dims)
- **Vision patch**: 768×3×32×32 → 128×3×14×14 (interpolated + rescaled)
- **FFN cloning**: Dense 960×3840 → 512 experts of 64×64 each

**Why**: K3 lives in a resource-constrained niche (6GB GPU).

### 4. **Genetic Drift** (Random Initialization)
Some traits can't transfer (architectural incompatibility):
- **Attention**: Left random (KDA/MLA ≠ standard attention)
- **Audio encoder**: Random init (no pretrained ancestor)
- **Video temporal**: Random init (novel mechanism)

**Why**: KDA uses delta-rule linear attention; standard models use softmax. No common ancestor.

## Inheritance Table

| Component | Ancestor | Transfer Method | Diversity Mechanism |
|-----------|----------|-----------------|---------------------|
| **Embeddings** | SmolLM2 | Truncate high-variance dims | Statistical sampling |
| **Layer Norms** | SmolLM2 | Direct copy | None (stable trait) |
| **Shared Experts** | SmolLM2 FFN | Clone + 1-2% noise | Gaussian mutation |
| **Routed Experts** | SmolLM2 FFN | Clone + 2-4% noise | Progressive mutation |
| **Vision Patch** | CLIP | Interpolate + rescale | None |
| **Vision Projector** | CLIP MLP | Truncate + rescale | None |
| **Attention** | — | Random | Novel trait |
| **Audio Encoder** | — | Random | Novel trait |
| **Video Temporal** | — | Random | Novel trait |

## Research Precedent

**MoE Upcycling** (Komatsuzaki et al., 2022):
- Dense model → Sparse MoE by cloning FFN to experts
- 2-5x faster convergence vs random init
- Enables scaling without starting from scratch

**Heterogeneous Transfer** (Ilharco et al., 2022 - CLIP):
- Vision + Language models can share knowledge despite different architectures
- Statistical rescaling prevents magnitude mismatches

**Why Audio/Video Are Random**:
- **Audio**: No widely-adopted pretrained encoder with our architecture (Whisper uses different tokenization)
- **Video**: Temporal factorization is novel to K3/MoonViT
- **Tradeoff**: Could use Whisper → K3 audio adapter, but adds complexity. Random init works fine with enough data.

## Fitness Comparison

**Random Init** (baseline):
- Embeddings: Random Gaussian
- Experts: Random Gaussian
- Vision: Random CNN
- Convergence: ~5-10K steps to see signal

**Genesis Init** (this approach):
- Embeddings: SmolLM2 knowledge
- Experts: Dense FFN knowledge + diversity
- Vision: CLIP spatial features
- **Expected convergence: ~1-2K steps** (2-5x faster)

## Running Genesis

```bash
# Create new species from ancestors
uv run python genesis.py

# Output: checkpoints/k3_pretrained_init.pt
# - 74M total params
# - 24M active (33% sparsity)
# - Ready for training
```

## Evolutionary Analogy

Think of K3 as a **hybrid species**:
- **Language skills** from SmolLM2 (like tool use in corvids)
- **Vision** from CLIP (like hawk eyesight)
- **Novel traits** (audio, video, sparse routing) evolved in isolation
- **Mutations** ensure expert diversity (genetic variation in population)

Just as hybrid vigor (heterosis) makes offspring stronger than parents, heterogeneous transfer + MoE specialization gives K3 better multimodal capabilities than either ancestor alone.
