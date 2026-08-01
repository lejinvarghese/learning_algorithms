"""
Per-head Muon optimizer for attention projections, as described in Kimi K3 paper.

For Q, K, V attention projections, we partition the momentum matrices along the head dimension
and orthogonalize each head's block separately, rather than treating all heads as a single coupled block.
"""
import torch
import torch.nn as nn


def zeropower_via_newtonschulz5(G, steps: int):
    """Newton-Schulz iteration for orthogonalization (from muon library)"""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    """Standard Muon update (from muon library)"""
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1))**0.5
    return update


def per_head_muon_update(grad, momentum, num_heads, beta=0.95, ns_steps=5, nesterov=True):
    """
    Per-head Muon update for Q, K, V projections.

    Instead of orthogonalizing the full projection matrix, partition along head dimension
    and orthogonalize each head separately. This equalizes update scales across heads.

    Args:
        grad: Gradient of shape (out_features, in_features) where out_features = num_heads * head_dim
        momentum: Momentum buffer of same shape
        num_heads: Number of attention heads
        beta: Momentum coefficient
        ns_steps: Newton-Schulz iteration steps
        nesterov: Use Nesterov momentum

    Returns:
        Orthogonalized update of same shape as grad
    """
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum

    # Partition along head dimension: (num_heads * head_dim, in_dim) -> (num_heads, head_dim, in_dim)
    out_features, in_features = update.shape
    head_dim = out_features // num_heads
    assert out_features == num_heads * head_dim, f"out_features {out_features} must be divisible by num_heads {num_heads}"

    # Reshape to expose head dimension
    update_per_head = update.view(num_heads, head_dim, in_features)

    # Orthogonalize each head separately
    orthogonalized = []
    for h in range(num_heads):
        head_update = update_per_head[h]  # (head_dim, in_dim)
        orth = zeropower_via_newtonschulz5(head_update, steps=ns_steps)
        orth *= max(1, orth.size(-2) / orth.size(-1))**0.5
        orthogonalized.append(orth)

    # Stack back: (num_heads, head_dim, in_dim) -> (num_heads * head_dim, in_dim)
    return torch.stack(orthogonalized, dim=0).view(out_features, in_features)


def adam_update(grad, buf1, buf2, step, betas, eps):
    """Adam update (from muon library)"""
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class K3Optimizer(torch.optim.Optimizer):
    """
    Kimi K3 optimizer combining:
    - Per-head Muon for Q, K, V attention projections
    - Standard Muon for other matrix parameters
    - Adam for embeddings, norms, and other non-matrix parameters

    Parameter groups must have 'optimizer_type' field set to one of:
    - 'per_head_muon': Per-head Muon for attention projections (requires 'num_heads')
    - 'muon': Standard Muon for other 2D+ parameters
    - 'adam': Adam for 1D parameters and embeddings
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert 'optimizer_type' in group, "Each param group must have 'optimizer_type' field"
            opt_type = group['optimizer_type']

            if opt_type == 'per_head_muon':
                assert 'num_heads' in group, "Per-head Muon requires 'num_heads' field"
                group['lr'] = group.get('lr', 0.02)
                group['momentum'] = group.get('momentum', 0.95)
                group['weight_decay'] = group.get('weight_decay', 0)
            elif opt_type == 'muon':
                group['lr'] = group.get('lr', 0.02)
                group['momentum'] = group.get('momentum', 0.95)
                group['weight_decay'] = group.get('weight_decay', 0)
            elif opt_type == 'adam':
                group['lr'] = group.get('lr', 3e-4)
                group['betas'] = group.get('betas', (0.9, 0.95))
                group['eps'] = group.get('eps', 1e-10)
                group['weight_decay'] = group.get('weight_decay', 0)
            else:
                raise ValueError(f"Unknown optimizer_type: {opt_type}")

        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            opt_type = group['optimizer_type']

            if opt_type == 'per_head_muon':
                # Per-head Muon for Q, K, V projections
                for p in group['params']:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state['momentum_buffer'] = torch.zeros_like(p)

                    update = per_head_muon_update(
                        p.grad,
                        state['momentum_buffer'],
                        num_heads=group['num_heads'],
                        beta=group['momentum']
                    )
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                    p.add_(update.reshape(p.shape), alpha=-group['lr'])

            elif opt_type == 'muon':
                # Standard Muon for other matrix parameters
                for p in group['params']:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state['momentum_buffer'] = torch.zeros_like(p)

                    update = muon_update(p.grad, state['momentum_buffer'], beta=group['momentum'])
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                    p.add_(update.reshape(p.shape), alpha=-group['lr'])

            elif opt_type == 'adam':
                # Adam for embeddings, norms, and other non-matrix params
                for p in group['params']:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                        state['step'] = 0
                    state['step'] += 1

                    update = adam_update(
                        p.grad,
                        state['exp_avg'],
                        state['exp_avg_sq'],
                        state['step'],
                        group['betas'],
                        group['eps']
                    )
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                    p.add_(update, alpha=-group['lr'])

        return loss


def create_k3_optimizer(model, cfg, muon_lr=0.005):
    """
    Create K3Optimizer for a model with proper parameter grouping.

    Args:
        model: K3Model instance
        cfg: K3Config instance
        muon_lr: Learning rate for Muon (Adam uses 10x lower)

    Returns:
        K3Optimizer instance
    """
    per_head_muon_params = []
    muon_params = []
    adam_params = []

    for name, param in model.named_parameters():
        # Q, K, V projections in KimiDeltaAttention and GatedMLA get per-head Muon
        if param.ndim >= 2 and any(x in name for x in ['q_lin', 'k_lin', 'v_lin', 'q_up', 'k_up', 'v_up']):
            per_head_muon_params.append((name, param))
        # Other 2D+ parameters (excluding embeddings) get standard Muon
        elif param.ndim >= 2 and "embed" not in name.lower():
            muon_params.append(param)
        # 1D parameters (norms, biases) and embeddings get Adam
        else:
            adam_params.append(param)

    param_groups = []

    # Per-head Muon for Q, K, V projections
    if per_head_muon_params:
        for name, param in per_head_muon_params:
            param_groups.append({
                'params': [param],
                'optimizer_type': 'per_head_muon',
                'num_heads': cfg.num_heads,
                'lr': muon_lr,
                'momentum': 0.95,
            })

    # Standard Muon for other matrix parameters
    if muon_params:
        param_groups.append({
            'params': muon_params,
            'optimizer_type': 'muon',
            'lr': muon_lr,
            'momentum': 0.95,
        })

    # Adam for 1D parameters and embeddings
    if adam_params:
        param_groups.append({
            'params': adam_params,
            'optimizer_type': 'adam',
            'lr': muon_lr / 10,
            'betas': (0.9, 0.95),
            'eps': 1e-10,
        })

    return K3Optimizer(param_groups)
