"""
Building blocks of Kimi K3, reproduced at micro scale. Equation numbers refer
to the Kimi K3 tech report, Section 2. Kernels are written as plain PyTorch
(sequential recurrence for KDA, dense expert loop for MoE, exact quantile for
Quantile Balancing) rather than the paper's fused/chunkwise/distributed
kernels -- those optimizations target multi-trillion-parameter, multi-GPU
training and are irrelevant at micro scale; the math they compute is the same.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


class ShortConv(nn.Module):
    """Causal depthwise conv used to locally mix q/k/v before KDA (§2.1.1)."""

    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D)
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv(x)
        return x.transpose(1, 2)


class SiTUGLU(nn.Module):
    """Sigmoid Tanh Unit GLU (Eq. 12): softcaps both GLU branches to avoid the
    unbounded activation growth of SwiGLU at extreme MoE sparsity."""

    def __init__(self, dim_in: int, dim_hidden: int, beta1: float = 4.0, beta2: float = 25.0):
        super().__init__()
        self.w_gate = nn.Linear(dim_in, dim_hidden, bias=False)
        self.w_up = nn.Linear(dim_in, dim_hidden, bias=False)
        self.w_down = nn.Linear(dim_hidden, dim_in, bias=False)
        self.beta1, self.beta2 = beta1, beta2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.w_gate(x)
        gate = self.beta1 * torch.tanh(z / self.beta1) * torch.sigmoid(z)
        up = self.beta2 * torch.tanh(self.w_up(x) / self.beta2)
        return self.w_down(gate * up)


def kda_recurrence(q, k, v, alpha, beta):
    """Sequential form of Eq. 1: S_t = (I - b_t k_t k_t^T) Diag(a_t) S_{t-1} + b_t k_t v_t^T,
    o_t = S_t^T q_t. q,k: (B,T,H,Dk); v: (B,T,H,Dv); alpha: (B,T,H,Dk) in (0,1); beta: (B,T,H) in (0,1).
    A production kernel would use the chunkwise UT-transform parallel form (Kimi Linear);
    at micro sequence lengths a direct per-step loop is simpler and numerically exact."""
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]
    S = q.new_zeros(B, H, Dk, Dv)
    outs = []
    for t in range(T):
        k_t, v_t, a_t, b_t = k[:, t], v[:, t], alpha[:, t], beta[:, t]
        S = S * a_t.unsqueeze(-1)                                   # Diag(alpha) S_{t-1}
        kS = torch.einsum("bhk,bhkv->bhv", k_t, S)                  # k_t^T S
        S = S - b_t.unsqueeze(-1).unsqueeze(-1) * torch.einsum("bhk,bhv->bhkv", k_t, kS)
        S = S + b_t.unsqueeze(-1).unsqueeze(-1) * torch.einsum("bhk,bhv->bhkv", k_t, v_t)
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], S))
    return torch.stack(outs, dim=1)  # (B, T, H, Dv)


class KimiDeltaAttention(nn.Module):
    """KDA (§2.1.1): delta-rule linear attention with a channel-wise, lower-
    bounded forget gate and a full-rank output gate."""

    def __init__(self, cfg):
        super().__init__()
        d, H, Dk, Dv, r = cfg.hidden_dim, cfg.num_heads, cfg.head_dim, cfg.head_dim, cfg.kda_decay_rank
        self.H, self.Dk, self.Dv = H, Dk, Dv

        self.q_lin, self.q_conv = nn.Linear(d, H * Dk, bias=False), ShortConv(H * Dk, cfg.conv_kernel_size)
        self.k_lin, self.k_conv = nn.Linear(d, H * Dk, bias=False), ShortConv(H * Dk, cfg.conv_kernel_size)
        self.v_lin, self.v_conv = nn.Linear(d, H * Dv, bias=False), ShortConv(H * Dv, cfg.conv_kernel_size)

        self.beta_lin = nn.Linear(d, H, bias=True)                  # Eq. 2: beta_t^h = Sigmoid(W_beta x_t)

        self.alpha_down = nn.Linear(d, H * r, bias=False)           # low-rank decay logit (Eq. 2)
        self.alpha_up = nn.Parameter(torch.randn(H, r, Dk) * (r ** -0.5))
        self.alpha_bias = nn.Parameter(torch.zeros(H, Dk))
        self.log_scale = nn.Parameter(torch.zeros(H))               # A_h, per-head log-scale (Eq. 5)
        self.gmin = cfg.kda_gmin

        self.out_gate = nn.Linear(d, H * Dv, bias=False)            # full-rank output gate (Eq. 6)
        self.out_norm = RMSNorm(Dv)                                 # head-wise RMSNorm before gating
        self.out_proj = nn.Linear(H * Dv, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, Dk, Dv = self.H, self.Dk, self.Dv

        q = l2norm(F.silu(self.q_conv(self.q_lin(x))).view(B, T, H, Dk))
        k = l2norm(F.silu(self.k_conv(self.k_lin(x))).view(B, T, H, Dk))
        v = F.silu(self.v_conv(self.v_lin(x))).view(B, T, H, Dv)

        beta = torch.sigmoid(self.beta_lin(x))                       # (B,T,H)

        z = torch.einsum("bthr,hrk->bthk", self.alpha_down(x).view(B, T, H, -1), self.alpha_up)
        z = z + self.alpha_bias
        g = self.gmin * torch.sigmoid(torch.exp(self.log_scale).view(1, 1, H, 1) * z)  # Eq. 5
        alpha = torch.exp(g)

        o = kda_recurrence(q, k, v, alpha, beta)
        o = self.out_norm(o)
        gate = torch.sigmoid(self.out_gate(x)).view(B, T, H, Dv)
        y = (gate * o).reshape(B, T, H * Dv)
        return self.out_proj(y)


class GatedMLA(nn.Module):
    """Gated Multi-Head Latent Attention (§2.1.2): low-rank q/kv compression,
    NoPE (no positional encoding -- position comes from the interleaved KDA
    layers instead), full-rank output gate (Eq. 7)."""

    def __init__(self, cfg):
        super().__init__()
        d, H, Dh, lat = cfg.hidden_dim, cfg.num_heads, cfg.head_dim, cfg.mla_latent_dim
        self.H, self.Dh = H, Dh

        self.q_down = nn.Linear(d, lat, bias=False)
        self.q_up = nn.Linear(lat, H * Dh, bias=False)
        self.kv_down = nn.Linear(d, lat, bias=False)
        self.k_up = nn.Linear(lat, H * Dh, bias=False)
        self.v_up = nn.Linear(lat, H * Dh, bias=False)

        self.out_gate = nn.Linear(d, H * Dh, bias=False)
        self.out_proj = nn.Linear(H * Dh, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, Dh = self.H, self.Dh

        q = self.q_up(self.q_down(x)).view(B, T, H, Dh).transpose(1, 2)
        c_kv = self.kv_down(x)
        k = self.k_up(c_kv).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_up(c_kv).view(B, T, H, Dh).transpose(1, 2)

        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # NoPE: no rotary/pos bias
        o = o.transpose(1, 2).reshape(B, T, H * Dh)

        gate = torch.sigmoid(self.out_gate(x))
        return self.out_proj(gate * o)


class QuantileBalancingRouter(nn.Module):
    """Quantile Balancing (§2.3.3, Eq. 13-14): auxiliary-loss-free routing
    with a bias term set exactly, per-batch, to the quantile that yields the
    target per-expert load. The paper approximates this quantile via a
    cross-rank histogram for distributed training at ~10^3 experts; at micro
    scale a single-process exact `torch.quantile` computes the same update."""

    def __init__(self, dim: int, num_experts: int, k: int):
        super().__init__()
        self.w_r = nn.Linear(dim, num_experts, bias=False)
        self.register_buffer("bias", torch.zeros(num_experts))
        self.k = k
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor):  # x: (N, d)
        s = torch.sigmoid(self.w_r(x))                  # (N, n_experts), Eq. 13
        biased = s + self.bias
        top_val, top_idx = biased.topk(self.k + 1, dim=-1)
        routed_idx = top_idx[:, : self.k]
        cutoff = top_val[:, self.k]                      # alpha_i^(t): the (k+1)-th margin

        routed_raw = torch.gather(s, -1, routed_idx)
        weights = routed_raw / routed_raw.sum(-1, keepdim=True).clamp_min(1e-9)

        if self.training:
            with torch.no_grad():
                margins = s - cutoff.unsqueeze(-1)                     # (N, n_experts)
                q = 1.0 - self.k / self.num_experts
                bias_hat = -torch.quantile(margins, q, dim=0)          # Eq. 14
                self.bias.copy_(bias_hat - bias_hat.mean())            # takes effect next call

        return routed_idx, weights


class StableLatentMoE(nn.Module):
    """Stable LatentMoE (§2.3, Eq. 11): shared experts run full-width on x;
    routed experts run in a compact latent space of width `latent_dim`,
    letting the routed pool scale to many more experts per unit of compute.
    An RMSNorm before the up-projection (§2.3.1) stabilizes the aggregate
    scale before it's combined with the shared-expert path."""

    def __init__(self, cfg):
        super().__init__()
        d, l = cfg.hidden_dim, cfg.latent_dim
        self.down = nn.Linear(d, l, bias=False)
        self.up = nn.Linear(l, d, bias=False)
        self.pre_up_norm = RMSNorm(l)

        self.router = QuantileBalancingRouter(d, cfg.num_routed_experts, cfg.num_experts_active)
        self.routed_experts = nn.ModuleList(
            [SiTUGLU(l, cfg.moe_hidden_per_expert, cfg.situglu_beta1, cfg.situglu_beta2)
             for _ in range(cfg.num_routed_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [SiTUGLU(d, cfg.shared_moe_hidden, cfg.situglu_beta1, cfg.situglu_beta2)
             for _ in range(cfg.num_shared_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        flat = x.reshape(-1, d)
        z = self.down(flat)

        idx, weights = self.router(flat)                 # (N,k), (N,k)
        u = flat.new_zeros(flat.shape[0], z.shape[-1])
        for e, expert in enumerate(self.routed_experts):
            hit = idx == e
            token_mask = hit.any(-1)
            if not token_mask.any():
                continue
            tok_w = (weights * hit).sum(-1, keepdim=True)[token_mask]
            u[token_mask] += tok_w * expert(z[token_mask])

        y = sum(expert(flat) for expert in self.shared_experts)
        y = y + self.up(self.pre_up_norm(u))
        return y.view(B, T, d)


class AttnResGate(nn.Module):
    """Attention Residuals (§2.2, Eq. 8-9): each layer replaces the standard
    residual stream with an attention pool, using a learned per-layer
    pseudo-query, over the representations of all preceding blocks (+ the
    running partial sum within the current block)."""

    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(dim) * dim ** -0.5)
        self.norm = RMSNorm(dim)

    def forward(self, reps: list) -> torch.Tensor:
        stacked = torch.stack(reps, dim=-2)                       # (B, T, N, d)
        logits = torch.einsum("d,btnd->btn", self.q, self.norm(stacked))
        alpha = torch.softmax(logits, dim=-1)                     # softmax(q . RMSNorm(k)) == Eq. 9
        return torch.einsum("btn,btnd->btd", alpha, stacked)
