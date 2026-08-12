# Config for K3: architectural ratios (KDA:MLA mix, expert counts, latent widths) match Kimi
# K3's shape; absolute sizes scale via K3Config.scaled(mult) to whatever budget you're working with.
from dataclasses import dataclass
from transformers import PretrainedConfig


@dataclass
class K3Config:
    # --- token / sequence ---
    vocab_size: int = 163_840  # Kimi K3: 163,840 (tiktoken BPE)
    max_seq_len: int = 1024  # Kimi K3: extends up to 1M tokens via progressive training

    # --- backbone width/depth (Kimi K3: hidden=7168, 93 layers, 8 attention-residual blocks) ---
    hidden_dim: int = 256
    num_blocks: int = 4  # attention-residual blocks (Kimi K3: 8, block size 12)
    layers_per_block: int = 4  # must be a multiple of (kda_mla_ratio + 1)
    kda_mla_ratio: int = 3  # KDA layers per Gated MLA layer (Kimi K3: ~3:1)

    # --- attention (Kimi K3: 96 heads) ---
    num_heads: int = 8
    conv_kernel_size: int = 4  # causal short-conv kernel applied to KDA's q/k/v
    kda_decay_rank: int = 8  # low-rank projection rank for KDA's decay logits
    kda_gmin: float = -5.0  # lower bound on KDA's log-decay
    mla_latent_dim: int = 128  # MLA q/kv compression width

    # --- Stable LatentMoE (Kimi K3: 896 routed / 16 active / 2 shared, latent = 0.5x hidden) ---
    latent_moe_ratio: float = 0.5  # latent width = ratio * hidden_dim
    num_routed_experts: int = 32
    num_experts_active: int = 4  # kept sparse; Kimi K3's exact sparsity ratio isn't
    # reachable once the expert pool itself is this small
    num_shared_experts: int = 2  # fixed by design, not scaled
    moe_hidden_per_expert: int = 128
    shared_moe_hidden: int = 256  # shared-expert FFN hidden width
    situglu_beta1: float = 4.0  # SiTU-GLU gate-branch softcap
    situglu_beta2: float = 25.0  # SiTU-GLU up-branch softcap

    # --- multi-token prediction (Kimi K3: exactly one extra layer, not scaled) ---
    use_mtp: bool = True

    # --- native vision (Kimi K3's vision tower: 401M params, 27 layers, patch 14, 12 heads) ---
    use_vision: bool = True
    vit_layers: int = 4
    vit_hidden: int = 128
    vit_heads: int = 4
    vit_patch_size: int = 14  # kept equal to Kimi K3: input granularity, not model scale
    vit_mlp_ratio: float = 4.0
    vit_num_frames: int = 8
    vit_temporal_pool: int = 2

    # --- audio encoder (Kimi-Audio style: Whisper encoder, continuous acoustic features) ---
    use_audio: bool = False  # disabled by default (K3 doesn't natively support audio)
    audio_n_mels: int = 80  # Whisper standard: 80 mel bins
    audio_hidden: int = 256  # audio encoder hidden dim (separate from main hidden_dim)
    audio_heads: int = 4
    audio_layers: int = 6  # lighter than vision (Whisper tiny uses 4-6 layers)
    audio_max_frames: int = 3000  # max ~30s at 100fps → 50fps after conv

    tie_embeddings: bool = True
    use_gradient_checkpointing: bool = False

    def __post_init__(self):
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"
        assert (
            self.layers_per_block % (self.kda_mla_ratio + 1) == 0
        ), "layers_per_block must be a multiple of (kda_mla_ratio + 1) to tile whole groups"
        assert self.num_experts_active <= self.num_routed_experts
        if self.use_vision:
            assert self.vit_hidden % self.vit_heads == 0
        if self.use_audio:
            assert self.audio_hidden % self.audio_heads == 0

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
    def scaled(cls, mult: float = 1.0, **overrides) -> "K3Config":
        # Scales every width by `mult` from the defaults while keeping the ratios above fixed;
        # pass field overrides to fine-tune further.
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
            layers_per_block=base.layers_per_block,  # defines the ratio, not scaled
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


class K3HFConfig(PretrainedConfig):
    model_type = "k3"

    def __init__(self, vocab_size=163840, max_seq_len=1024, hidden_dim=256, num_blocks=4, layers_per_block=4,
                 kda_mla_ratio=3, num_heads=8, conv_kernel_size=4, kda_decay_rank=8, kda_gmin=-5.0,
                 mla_latent_dim=128, latent_moe_ratio=0.5, num_routed_experts=32, num_experts_active=4,
                 num_shared_experts=2, moe_hidden_per_expert=128, shared_moe_hidden=256, situglu_beta1=4.0,
                 situglu_beta2=25.0, use_mtp=True, use_vision=True, vit_layers=4, vit_hidden=128, vit_heads=4,
                 vit_patch_size=14, vit_mlp_ratio=4.0, vit_num_frames=8, vit_temporal_pool=2,
                 use_audio=False, audio_n_mels=80, audio_hidden=256, audio_heads=4, audio_layers=6, audio_max_frames=3000,
                 tie_embeddings=True, use_gradient_checkpointing=False, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.layers_per_block = layers_per_block
        self.kda_mla_ratio = kda_mla_ratio
        self.num_heads = num_heads
        self.conv_kernel_size = conv_kernel_size
        self.kda_decay_rank = kda_decay_rank
        self.kda_gmin = kda_gmin
        self.mla_latent_dim = mla_latent_dim
        self.latent_moe_ratio = latent_moe_ratio
        self.num_routed_experts = num_routed_experts
        self.num_experts_active = num_experts_active
        self.num_shared_experts = num_shared_experts
        self.moe_hidden_per_expert = moe_hidden_per_expert
        self.shared_moe_hidden = shared_moe_hidden
        self.situglu_beta1 = situglu_beta1
        self.situglu_beta2 = situglu_beta2
        self.use_mtp = use_mtp
        self.use_vision = use_vision
        self.vit_layers = vit_layers
        self.vit_hidden = vit_hidden
        self.vit_heads = vit_heads
        self.vit_patch_size = vit_patch_size
        self.vit_mlp_ratio = vit_mlp_ratio
        self.vit_num_frames = vit_num_frames
        self.vit_temporal_pool = vit_temporal_pool
        self.use_audio = use_audio
        self.audio_n_mels = audio_n_mels
        self.audio_hidden = audio_hidden
        self.audio_heads = audio_heads
        self.audio_layers = audio_layers
        self.audio_max_frames = audio_max_frames
        self.tie_embeddings = tie_embeddings
        self.use_gradient_checkpointing = use_gradient_checkpointing

    @property
    def head_dim(self):
        return self.hidden_dim // self.num_heads

    @property
    def latent_dim(self):
        return max(8, int(round(self.hidden_dim * self.latent_moe_ratio)))

    @property
    def num_layers(self):
        return self.num_blocks * self.layers_per_block
