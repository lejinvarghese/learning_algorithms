"""
Config for K3-Micro: a shape-faithful, fractionally-scaled replica of Kimi K3.

Every field is annotated with the real Kimi K3 value it derives from (Table 1
of the tech report). Architectural *ratios* (KDA:MLA = 3:1, shared experts=2,
latent-MoE dim = 0.5x hidden, MTP=1 layer) are treated as the "shape" and kept
fixed; absolute *widths* (hidden dim, #experts, #layers, vocab) are what you
scale via `K3MicroConfig.scaled(mult)` to fit your GPU budget.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class K3MicroConfig:
    # --- token / sequence ---
    vocab_size: int = 32_000          # K3: 160_000 (shrunk for a demo tokenizer)
    max_seq_len: int = 1024           # K3: trains up to 1M via progressive extension

    # --- backbone width/depth (K3: hidden=7168, 93 layers, 8 AttnRes blocks) ---
    hidden_dim: int = 256
    num_blocks: int = 4                # AttnRes blocks (K3: 8, block size 12)
    layers_per_block: int = 4          # must be a multiple of (kda_mla_ratio + 1)
    kda_mla_ratio: int = 3             # 3 KDA layers : 1 Gated MLA layer (K3: 69:24 ~ 3:1)

    # --- attention (K3: 96 heads) ---
    num_heads: int = 8
    conv_kernel_size: int = 4          # causal ShortConv kernel for KDA q/k/v
    kda_decay_rank: int = 8            # low-rank projection rank for KDA decay logits z_t^h
    kda_gmin: float = -5.0             # lower-bounded log-decay floor (Eq. 5)
    mla_latent_dim: int = 128          # MLA q/kv compression width (K3 KV latent dim)

    # --- Stable LatentMoE (K3: 896 routed / 16 active / 2 shared, latent=0.5*hidden) ---
    latent_moe_ratio: float = 0.5      # latent width l = ratio * hidden_dim (K3: 3584/7168)
    num_routed_experts: int = 32       # K3: 896
    num_experts_active: int = 4        # K3: 16 (sparsity 56x; infeasible to preserve exactly
                                        # at micro scale, kept qualitatively sparse instead)
    num_shared_experts: int = 2        # K3: 2 (fixed by design, not scaled)
    moe_hidden_per_expert: int = 128   # K3: 3072 (~0.43 * hidden)
    shared_moe_hidden: int = 256       # shared-expert FFN hidden width
    situglu_beta1: float = 4.0         # SiTU-GLU gate-branch cap (Eq. 12)
    situglu_beta2: float = 25.0        # SiTU-GLU up-branch cap (Eq. 12)

    # --- MTP (K3: exactly 1 layer, mirrors a backbone block; not scaled) ---
    use_mtp: bool = True

    # --- native vision, MoonViT-micro (K3: 401M params, 27 layers, patch 14, 12 heads) ---
    use_vision: bool = True
    vit_layers: int = 4
    vit_hidden: int = 128
    vit_heads: int = 4
    vit_patch_size: int = 14           # kept equal to K3 (input granularity, not model scale)
    vit_mlp_ratio: float = 4.0

    tie_embeddings: bool = True

    def __post_init__(self):
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"
        assert self.layers_per_block % (self.kda_mla_ratio + 1) == 0, (
            "layers_per_block must be a multiple of (kda_mla_ratio + 1) to tile whole 3:1 groups"
        )
        assert self.num_experts_active <= self.num_routed_experts
        if self.use_vision:
            assert self.vit_hidden % self.vit_heads == 0

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def latent_dim(self) -> int:
        return max(8, int(round(self.hidden_dim * self.latent_moe_ratio)))

    @property
    def num_layers(self) -> int:
        return self.num_blocks * self.layers_per_block

    @classmethod
    def scaled(cls, mult: float = 1.0, **overrides) -> "K3MicroConfig":
        """Scale absolute widths by `mult` from the micro defaults, keeping K3's
        architectural ratios fixed. mult=1.0 reproduces the defaults above;
        mult=2.0 doubles hidden/experts/etc; mult=0.5 halves them. Pass explicit
        field overrides (e.g. vocab_size=...) to fine-tune after scaling."""
        base = cls()

        def m(v, minimum=1):
            return max(minimum, round(v * mult))

        hidden = max(32, (m(base.hidden_dim) // 8) * 8)
        heads = base.num_heads
        while hidden % heads != 0 and heads > 1:
            heads -= 1
        vit_hidden = max(16, (m(base.vit_hidden) // 4) * 4)
        vit_heads = base.vit_heads
        while vit_hidden % vit_heads != 0 and vit_heads > 1:
            vit_heads -= 1

        cfg = dict(
            hidden_dim=hidden,
            num_heads=heads,
            num_blocks=m(base.num_blocks),
            layers_per_block=base.layers_per_block,  # ratio-defining, not scaled
            kda_decay_rank=m(base.kda_decay_rank, minimum=2),
            mla_latent_dim=m(base.mla_latent_dim, minimum=16),
            num_routed_experts=m(base.num_routed_experts, minimum=base.num_shared_experts + 2),
            num_experts_active=max(1, min(m(base.num_experts_active), m(base.num_routed_experts))),
            moe_hidden_per_expert=m(base.moe_hidden_per_expert, minimum=16),
            shared_moe_hidden=m(base.shared_moe_hidden, minimum=16),
            vit_layers=m(base.vit_layers, minimum=1),
            vit_hidden=vit_hidden,
            vit_heads=vit_heads,
        )
        cfg.update(overrides)
        return cls(**cfg)
