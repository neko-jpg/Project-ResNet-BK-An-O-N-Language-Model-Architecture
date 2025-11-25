# src/optimizers/riemannian_adam.py

import torch
from torch.optim.optimizer import Optimizer

# ##############################################################################
# # HYPERBOLIC UTILS
# ##############################################################################

def mobius_add(x, y, dim=-1, eps=1e-8):
    """
    Möbius addition in the Poincaré ball model.
    x_new = x [+] y
    """
    x_norm_sq = torch.sum(x * x, dim=dim, keepdim=True).clamp_min(eps)
    y_norm_sq = torch.sum(y * y, dim=dim, keepdim=True).clamp_min(eps)
    xy_dot = torch.sum(x * y, dim=dim, keepdim=True)

    denominator = 1 + 2 * xy_dot + x_norm_sq * y_norm_sq
    numerator1 = (1 + 2 * xy_dot + y_norm_sq) * x
    numerator2 = (1 - x_norm_sq) * y

    return (numerator1 + numerator2) / denominator.clamp_min(eps)


def poincare_exp_map(x, v, dim=-1, eps=1e-8):
    """
    Exponential map in the Poincaré ball model.
    Moves point x along the geodesic in the direction of tangent vector v.
    """
    # The update vector `v` is assumed to be the result of -lr * m_hat,
    # which already lives in the tangent space T_x M.
    return mobius_add(x, v, dim=dim, eps=eps)


def poincare_vector_transport(v, x, y, dim=-1, eps=1e-8):
    """
    Vector transport of vector `v` from tangent space at `x` to tangent space at `y`.
    This implementation uses the projection-based approximation, which is efficient.
    transport(v) = (lambda_x / lambda_y) * v
    where lambda_p = 2 / (1 - ||p||^2) is the conformal factor.
    """
    x_norm_sq = x.norm(dim=dim, keepdim=True).pow(2)
    y_norm_sq = y.norm(dim=dim, keepdim=True).pow(2)

    # Ratio is (lambda_x / lambda_y) = (1 - ||y||^2) / (1 - ||x||^2)
    transport_ratio = (1 - y_norm_sq) / (1 - x_norm_sq).clamp_min(eps)

    return v * transport_ratio


def project_to_manifold(p, max_norm=1.0 - 1e-5, dim=-1, eps=1e-8):
    """
    Projects a point back into the Poincaré ball if its norm exceeds max_norm.
    """
    p_norm = p.norm(dim=dim, keepdim=True).clamp_min(eps)
    # If norm is greater than max_norm, scale it down, otherwise keep it.
    scale = (p_norm > max_norm).float() * (max_norm / p_norm) + (p_norm <= max_norm).float()
    p.data.mul_(scale)


# ##############################################################################
# # RiemannianAdam IMPLEMENTATION
# ##############################################################################

class RiemannianAdam(Optimizer):
    """
    Implements the Riemannian Adam algorithm for optimization on the Poincaré ball.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(RiemannianAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('RiemannianAdam does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                is_hyperbolic = hasattr(p, 'is_hyperbolic') and p.is_hyperbolic

                if is_hyperbolic:
                    # --- Riemannian Update ---
                    old_p_data = p.data.clone()
                    p_norm_sq = old_p_data.norm(dim=-1, keepdim=True).pow(2)

                    # 1. Scale gradient to Riemannian gradient at p_t
                    riemannian_grad = grad * ((1 - p_norm_sq).pow(2) / 4.0)

                    # Update biased first moment estimate (m_t)
                    exp_avg.mul_(beta1).add_(riemannian_grad, alpha=1 - beta1)

                    # Update biased second raw moment estimate (v_t)
                    exp_avg_sq.mul_(beta2).addcmul_(riemannian_grad, riemannian_grad, value=1 - beta2)

                    # Compute update vector
                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                    update_vec = (exp_avg / bias_correction1) / denom

                    # 2. Set step size and direction in the tangent space at p_t
                    update_vec.mul_(-group['lr'])

                    # 3. Perform exponential map to get p_{t+1}
                    new_p_data = poincare_exp_map(old_p_data, update_vec)
                    p.data.copy_(new_p_data)

                    # 4. Project back to manifold for stability
                    project_to_manifold(p.data)

                    # 5. Transport momentum m_t from T_{p_t}M to T_{p_{t+1}}M for the next step
                    exp_avg_transported = poincare_vector_transport(exp_avg, old_p_data, p.data)
                    exp_avg.copy_(exp_avg_transported)

                else:
                    # --- Standard Euclidean Adam Update ---
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                    step_size = group['lr'] / bias_correction1
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
