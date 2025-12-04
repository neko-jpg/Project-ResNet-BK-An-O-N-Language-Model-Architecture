import torch
import torch.optim as optim
import math

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power of G
    (i.e., the orthogonal matrix UV^T where G = USV^T).
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

class Muon(optim.Optimizer):
    """
    Muon - Momentum Orthogonal Optimizer

    A novel optimizer that updates parameters using an orthogonalized update step
    derived via Newton-Schulz iteration. Designed for massive models.

    This implementation automatically delegates 1D parameters (biases, norms)
    and embeddings to an internal AdamW optimizer, applying Muon only to
    2D+ weights (linear layers, etc).
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 adamw_lr=1e-4, adamw_betas=(0.9, 0.95), adamw_eps=1e-8, adamw_wd=0.01):

        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)

        # Split params
        muon_params = []
        adamw_params = []

        for p in params:
            if p.requires_grad:
                if p.ndim >= 2 and p.size(0) < 10000 and p.size(1) < 10000:
                    # Use Muon for standard linear layers
                    # Skip massive embeddings (like vocab projection) if they are too huge?
                    # For now, apply to all 2D >=.
                    muon_params.append(p)
                else:
                    # 1D params (biases, norms) or huge embeddings go to AdamW
                    adamw_params.append(p)

        super().__init__(muon_params, defaults)

        self.adamw = optim.AdamW(adamw_params, lr=adamw_lr, betas=adamw_betas, eps=adamw_eps, weight_decay=adamw_wd)
        self.muon_params = muon_params
        self.adamw_params = adamw_params

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # Step AdamW
        self.adamw.step()

        # Step Muon
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                # Init state
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)

                buf = state['momentum_buffer']
                grad = p.grad

                # Momentum
                buf.mul_(momentum).add_(grad)

                if nesterov:
                    g = grad.add(buf, alpha=momentum)
                else:
                    g = buf

                # Orthogonalize update
                if g.ndim >= 2:
                    g_ortho = zeropower_via_newtonschulz5(g, steps=ns_steps)
                else:
                    # Fallback for unexpected shapes (shouldn't happen due to filtering)
                    g_ortho = g / (g.norm() + 1e-8)

                # Update
                # Scale by RMS of param to keep update scale consistent?
                # Muon paper suggests: param -= lr * g_ortho * max(1, param.rms / g_ortho.rms) ?
                # Simplified: param -= lr * g_ortho

                # Ideally, we scale the update by the spectral radius, but NewtonSchulz returns unitary-like.
                # So we just apply LR.

                p.data.add_(g_ortho, alpha=-lr)

        return loss
