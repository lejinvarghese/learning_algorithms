# Assembles K3 from the layers in layers.py: hybrid KDA/MLA blocks under attention residuals,
# a native vision pathway, and an optional multi-token-prediction head.
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import K3Config, K3HFConfig
from .layers import AttnResGate, GatedMLA, KimiDeltaAttention, RMSNorm, StableLatentMoE


def build_layer_pattern(cfg: K3Config) -> List[List[str]]:
    # KDA layers closed out by a Gated MLA layer per block; the very last layer of the whole
    # backbone is forced to MLA so global attention always closes the network.
    group = ["kda"] * cfg.kda_mla_ratio + ["mla"]
    per_block = (group * (cfg.layers_per_block // len(group)))[: cfg.layers_per_block]
    pattern = [list(per_block) for _ in range(cfg.num_blocks)]
    pattern[-1][-1] = "mla"
    return pattern


class K3Block(nn.Module):
    def __init__(self, cfg: K3Config, layer_types: List[str]):
        super().__init__()
        self.checkpoint_attn = cfg.use_gradient_checkpointing
        self.layers = nn.ModuleList()
        for lt in layer_types:
            attn = KimiDeltaAttention(cfg) if lt == "kda" else GatedMLA(cfg)
            self.layers.append(
                nn.ModuleDict(
                    {
                        "gate": AttnResGate(cfg.hidden_dim),
                        "attn": attn,
                        "moe": StableLatentMoE(cfg),
                    }
                )
            )

    def forward(self, block_reps: List[torch.Tensor]):
        partial = torch.zeros_like(block_reps[0])
        total_router_z_loss = 0.0
        total_load_balance_loss = 0.0

        for i, sub in enumerate(self.layers):
            ctx = block_reps + ([partial] if i > 0 else [])
            h = sub["gate"](ctx)
            if self.checkpoint_attn and self.training:
                a = checkpoint(sub["attn"], h, use_reentrant=False) + h
            else:
                a = sub["attn"](h) + h
            m, router_z_loss, load_balance_loss = sub["moe"](a)
            m = m + a
            partial = partial + m

            # Accumulate auxiliary losses
            total_router_z_loss = total_router_z_loss + router_z_loss
            total_load_balance_loss = total_load_balance_loss + load_balance_loss

        return partial, total_router_z_loss, total_load_balance_loss


# One extra block-shaped layer that fuses the final hidden state with the next token's
# embedding to predict the token after that.
class MTPHead(nn.Module):
    def __init__(self, cfg: K3Config, embed: nn.Embedding):
        super().__init__()
        self.embed = embed
        self.fuse_norm_h = RMSNorm(cfg.hidden_dim)
        self.fuse_norm_e = RMSNorm(cfg.hidden_dim)
        self.fuse = nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.gate = AttnResGate(cfg.hidden_dim)
        self.attn = GatedMLA(cfg)
        self.moe = StableLatentMoE(cfg)
        self.norm = RMSNorm(cfg.hidden_dim)

    def forward(self, hidden: torch.Tensor, input_ids: torch.Tensor):
        next_emb = torch.zeros_like(hidden)
        next_emb[:, :-1] = self.embed(input_ids)[:, 1:]
        fused = self.fuse(torch.cat([self.fuse_norm_h(hidden), self.fuse_norm_e(next_emb)], dim=-1))
        h = self.gate([fused, hidden])
        a = self.attn(h) + h
        m, router_z_loss, load_balance_loss = self.moe(a)
        m = m + a
        return self.norm(m), router_z_loss, load_balance_loss


def _vit_block(cfg: K3Config) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=cfg.vit_hidden,
        nhead=cfg.vit_heads,
        dim_feedforward=int(cfg.vit_hidden * cfg.vit_mlp_ratio),
        batch_first=True,
        norm_first=True,
        bias=False,
    )


class MoonViT(nn.Module):
    def __init__(self, cfg: K3Config):
        super().__init__()
        self.patch_size = cfg.vit_patch_size
        self.temporal_pool = cfg.vit_temporal_pool
        self.patch_embed = nn.Conv2d(3, cfg.vit_hidden, cfg.vit_patch_size, stride=cfg.vit_patch_size, bias=False)
        self.spatial_pos = nn.Parameter(torch.randn(1, 4096, cfg.vit_hidden) * 0.02)
        self.temporal_pos = nn.Parameter(torch.randn(1, cfg.vit_num_frames, cfg.vit_hidden) * 0.02)

        self.layers = nn.ModuleList([nn.ModuleList([_vit_block(cfg), _vit_block(cfg)]) for _ in range(cfg.vit_layers)])
        self.final_norm = RMSNorm(cfg.vit_hidden)
        self.projector = nn.Sequential(
            nn.Linear(cfg.vit_hidden * 4, cfg.vit_hidden * 4, bias=False),
            nn.GELU(),
            nn.Linear(cfg.vit_hidden * 4, cfg.hidden_dim, bias=False),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        B, F, C, H, W = frames.shape
        x = self.patch_embed(frames.reshape(B * F, C, H, W)).flatten(2).transpose(1, 2)
        P = x.shape[1]
        x = x + self.spatial_pos[:, :P]
        x = (x.view(B, F, P, -1) + self.temporal_pos[:, :F].unsqueeze(2)).view(B * F, P, -1)

        for spatial_attn, temporal_attn in self.layers:
            x = spatial_attn(x)
            if F > 1:
                x = x.view(B, F, P, -1).transpose(1, 2).reshape(B * P, F, -1)
                x = temporal_attn(x)
                x = x.view(B, P, F, -1).transpose(1, 2).reshape(B * F, P, -1)

        x = self.final_norm(x).view(B, F, P, -1)
        if F > 1:
            kept = (F // self.temporal_pool) * self.temporal_pool
            x = x[:, :kept].view(B, kept // self.temporal_pool, self.temporal_pool, P, -1).mean(2)
        F = x.shape[1]

        n = P - (P % 4)
        x = x[:, :, :n].reshape(B * F, n // 4, 4 * x.shape[-1])
        return self.projector(x).view(B, F * (n // 4), -1)


class K3Model(nn.Module):
    def __init__(self, cfg: K3Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.embed_norm = RMSNorm(cfg.hidden_dim)

        self.blocks = nn.ModuleList([K3Block(cfg, pattern) for pattern in build_layer_pattern(cfg)])
        self.final_norm = RMSNorm(cfg.hidden_dim)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.mtp = MTPHead(cfg, self.embed) if cfg.use_mtp else None
        self.vision = MoonViT(cfg) if cfg.use_vision else None
        self.audio = None
        if cfg.use_audio:
            from k3.audio import K3AudioEncoder
            self.audio = K3AudioEncoder(cfg)

        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        has_visual: Optional[torch.Tensor] = None,
        audio_mel: Optional[torch.Tensor] = None,
        has_audio: Optional[torch.Tensor] = None,
    ):
        x = self.embed_norm(self.embed(input_ids))

        # Add vision features
        if images is not None and self.vision is not None:
            vis = self.vision(images).mean(1, keepdim=True)
            if has_visual is not None:
                vis = vis * has_visual.view(-1, 1, 1)
            x = x + vis

        # Add audio features
        if audio_mel is not None and self.audio is not None:
            aud = self.audio(audio_mel).mean(1, keepdim=True)
            if has_audio is not None:
                aud = aud * has_audio.view(-1, 1, 1)
            x = x + aud

        block_reps = [x]
        total_router_z_loss = 0.0
        total_load_balance_loss = 0.0

        for block in self.blocks:
            block_out, router_z_loss, load_balance_loss = block(block_reps)
            block_reps.append(block_out)
            total_router_z_loss = total_router_z_loss + router_z_loss
            total_load_balance_loss = total_load_balance_loss + load_balance_loss

        h = self.final_norm(block_reps[-1])
        logits = self.lm_head(h)

        # Handle MTP head if present
        if self.mtp is not None:
            mtp_out, mtp_router_z_loss, mtp_load_balance_loss = self.mtp(h, input_ids)
            mtp_logits = self.lm_head(mtp_out)
            total_router_z_loss = total_router_z_loss + mtp_router_z_loss
            total_load_balance_loss = total_load_balance_loss + mtp_load_balance_loss
        else:
            mtp_logits = None

        # Return auxiliary losses for training stability
        aux_losses = {
            "router_z_loss": total_router_z_loss,
            "load_balance_loss": total_load_balance_loss,
        }
        return logits, mtp_logits, aux_losses

    def param_counts(self):
        # Total parameters vs. an approximation of what's active per token once only a
        # fraction of each layer's routed experts fire.
        total = sum(p.numel() for p in self.parameters())
        moe_params = sum(
            p.numel() for block in self.blocks for sub in block.layers for p in sub["moe"].routed_experts.parameters()
        )
        active_frac = self.cfg.num_experts_active / max(1, self.cfg.num_routed_experts)
        activated = total - moe_params + int(moe_params * active_frac)
        return {"total": total, "activated_approx": activated}


class K3PreTrainedModel(PreTrainedModel):
    config_class = K3HFConfig
    base_model_prefix = "k3"
    supports_gradient_checkpointing = True
    _no_split_modules = ["K3Block"]

    def _init_weights(self, _):
        pass


class K3ForCausalLM(K3PreTrainedModel):
    def __init__(self, config: K3HFConfig):
        super().__init__(config)
        k3_cfg = K3Config(
            vocab_size=config.vocab_size, max_seq_len=config.max_seq_len, hidden_dim=config.hidden_dim,
            num_blocks=config.num_blocks, layers_per_block=config.layers_per_block,
            kda_mla_ratio=config.kda_mla_ratio, num_heads=config.num_heads,
            conv_kernel_size=config.conv_kernel_size, kda_decay_rank=config.kda_decay_rank,
            kda_gmin=config.kda_gmin, mla_latent_dim=config.mla_latent_dim,
            latent_moe_ratio=config.latent_moe_ratio, num_routed_experts=config.num_routed_experts,
            num_experts_active=config.num_experts_active, num_shared_experts=config.num_shared_experts,
            moe_hidden_per_expert=config.moe_hidden_per_expert, shared_moe_hidden=config.shared_moe_hidden,
            situglu_beta1=config.situglu_beta1, situglu_beta2=config.situglu_beta2, use_mtp=config.use_mtp,
            use_vision=config.use_vision, vit_layers=config.vit_layers, vit_hidden=config.vit_hidden,
            vit_heads=config.vit_heads, vit_patch_size=config.vit_patch_size,
            vit_mlp_ratio=config.vit_mlp_ratio, vit_num_frames=config.vit_num_frames,
            vit_temporal_pool=config.vit_temporal_pool, tie_embeddings=config.tie_embeddings,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
        )
        self.model = K3Model(k3_cfg)

    def forward(self, input_ids, images=None, has_visual=None, labels=None, **_):
        logits, mtp_logits, aux_losses = self.model(input_ids, images, has_visual)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, self.config.vocab_size), labels[:, 1:].reshape(-1),
            )
            if mtp_logits is not None:
                loss = loss + 0.1 * torch.nn.functional.cross_entropy(
                    mtp_logits[:, :-2].reshape(-1, self.config.vocab_size), labels[:, 2:].reshape(-1),
                )
            loss = loss + 1e-3 * aux_losses["router_z_loss"] + 1e-2 * aux_losses["load_balance_loss"]
        return CausalLMOutputWithPast(loss=loss, logits=logits)
