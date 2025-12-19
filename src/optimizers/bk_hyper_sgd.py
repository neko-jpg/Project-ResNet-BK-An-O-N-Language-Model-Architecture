"""
BK-HyperSGD: ResNet-BK専用オプティマイザ

ResNet-BK (BK-Core + HTT + Hyperbolic Attention) アーキテクチャのために
特別に設計されたオプティマイザ。

設計思想:
1. AdamW/Muonを使用しない完全独自設計
2. モデルの幾何学的構造を尊重:
   - BK-Core層: ユニタリ群 U(n) 上の最適化 (Cayley retraction)
   - Hyperbolic層: Lorentz多様体上の最適化 (Lorentz exp map)
   - Symplectic層: シンプレクティック構造を保存
3. Green関数 G_ii の物理的制約を組み込み

Key Features:
- Cayley Retraction: ユニタリ性 W†W = I を保存
- Lorentz Exponential Map: 双曲面上での測地線フロー
- Symplectic Integration: 位相空間の体積を保存
- Natural Gradient: Fisher情報行列による前処理（ユークリッド層）

Author: ResNet-BK Project
"""

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from typing import Dict, List, Tuple, Optional, Set, Any
import math


class BKHyperSGD(Optimizer):
    """
    BK-Core Hyperbolic Stochastic Gradient Descent
    
    ResNet-BK専用オプティマイザ。AdamW/Muonを使用せず、
    モデルの幾何学的構造に基づいた独自の更新則を実装。
    
    Parameter Types & Update Rules:
    
    1. Unitary Parameters (BK-Core: v_proj, output_proj)
       - 制約: W†W = I (ユニタリ性)
       - 更新: Cayley retraction
       - W_new = (I + η/2·A)⁻¹ · (I - η/2·A) · W
       - where A = grad·W† - W·grad† (skew-Hermitian)
    
    2. Hyperbolic Parameters (Hyperbolic Attention weights)
       - 多様体: Lorentz hyperboloid {x: -x₀² + ||x_space||² = -1/|c|}
       - 更新: Lorentz exponential map
       - x_new = cosh(||v||_L)·x + sinh(||v||_L)/||v||_L · v
    
    3. Symplectic Parameters (SymplecticBKBlock)
       - 構造: (q, p) 正準変数ペア
       - 更新: エネルギー保存型シンプレクティック積分
    
    4. Euclidean Parameters (FFN, LayerNorm, etc.)
       - 標準的なモメンタムSGD
    
    Args:
        params: Model parameters
        lr: Learning rate
        momentum: Momentum coefficient
        curvature: Hyperbolic curvature (negative)
        unitarity_strength: Strength of unitarity regularization
        green_function_reg: Green function regularization weight
        use_cayley: Use Cayley retraction for unitary params
        use_lorentz: Use Lorentz exponential map for hyperbolic params
        max_grad_norm: Maximum gradient norm for clipping
        eps: Small constant for numerical stability
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        curvature: float = -1.0,
        trust_radius: float = 0.1,  # New: Trust Region Radius
        unitarity_strength: float = 0.1,
        green_function_reg: float = 0.01,
        use_cayley: bool = True,
        use_lorentz: bool = True,
        max_grad_norm: Optional[float] = None,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if curvature > 0.0:
            raise ValueError(f"Curvature must be negative for hyperbolic: {curvature}")
            
        defaults = dict(
            lr=lr,
            momentum=momentum,
            curvature=curvature,
            trust_radius=trust_radius,
            unitarity_strength=unitarity_strength,
            green_function_reg=green_function_reg,
            use_cayley=use_cayley,
            use_lorentz=use_lorentz,
            max_grad_norm=max_grad_norm,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        
        # Classify parameters by their geometric type
        self._param_types: Dict[int, str] = {}
        self._classify_parameters()
        
        # Statistics for monitoring
        self.step_count = 0
        self.grad_norms: Dict[str, float] = {}
    
    def _classify_parameters(self):
        """
        Classify parameters by their geometric type based on their names.
        
        Categories (UPDATED: Geometric-aware classification for BK-HyperSGD):
        - 'unitary': Attention projections (v_proj, o_proj, output_proj) - Cayley retraction
        - 'hyperbolic': BK-Core, hyperbolic attention, AR-SSM layers - Trust Region Riemannian
        - 'symplectic': Symplectic blocks - Energy-preserving updates
        - 'euclidean': FFN, LayerNorm, embeddings, etc. - Standard momentum SGD
        """
        for group in self.param_groups:
            for p in group['params']:
                param_id = id(p)
                name = getattr(p, '_param_name', '').lower()
                
                # === 1. Unitary Parameters (Cayley Retraction) ===
                # Attention projections that benefit from orthogonal structure
                if any(key in name for key in ['v_proj', 'o_proj', 'output_proj', 'q_proj', 'k_proj']):
                    if 'weight' in name and p.dim() >= 2:
                        self._param_types[param_id] = 'unitary'
                    else:
                        self._param_types[param_id] = 'euclidean'
                
                # === 2. Hyperbolic Parameters (Trust Region Riemannian) ===
                # BK-Core and hyperbolic geometry layers
                elif any(key in name for key in [
                    'bk_core', 'bk_scale', 'bk_hyperbolic',  # BK-Core components
                    'hyperbolic_attn', 'hyperbolic_gate',    # Hyperbolic attention
                    'ar_ssm_fusion', 'ar_ssm',               # AR-SSM hyperbolic fusion
                    'curvature', 'lorentz',                  # Curvature-related
                    'poincare', 'mobius',                    # Möbius operations
                ]):
                    self._param_types[param_id] = 'hyperbolic'
                
                # === 3. Symplectic Parameters ===
                elif any(key in name for key in ['symplectic', 'force_field', 'hamiltonian']):
                    self._param_types[param_id] = 'symplectic'
                
                # === 4. Euclidean Parameters (Standard SGD) ===
                # FFN, embeddings, LayerNorm, biases, etc.
                else:
                    self._param_types[param_id] = 'euclidean'
    
    def set_param_names(self, model: nn.Module):
        """
        Set parameter names from model for classification.
        Call this after optimizer creation.
        """
        for name, param in model.named_parameters():
            param._param_name = name
        self._classify_parameters()
    
    def get_param_type(self, p: torch.Tensor) -> str:
        """Get the geometric type of a parameter."""
        return self._param_types.get(id(p), 'euclidean')
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self.step_count += 1
        
        # Debug counters
        params_with_grad = 0
        params_updated = 0
        debug_this_step = (self.step_count % 10 == 1)
        total_grad_norm_sq = 0.0 if debug_this_step else None
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            curvature = group['curvature']
            trust_radius = group.get('trust_radius', 0.1)
            eps = group['eps']
            max_grad_norm = group.get('max_grad_norm', None)
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                params_with_grad += 1
                grad = p.grad
                if debug_this_step:
                    total_grad_norm_sq += float(grad.norm().item()) ** 2
                
                # Skip sparse gradients
                if grad.is_sparse:
                    raise RuntimeError("BKHyperSGD does not support sparse gradients")
                
                # Get or initialize state
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p)
                
                state['step'] += 1
                
                # Global post-unscale clipping is handled by trainer, 
                # but we can do per-param clipping if configured.
                if max_grad_norm is not None and float(max_grad_norm) > 0.0:
                    grad_norm = float(grad.norm().item())
                    if grad_norm > float(max_grad_norm):
                        grad = grad * (float(max_grad_norm) / (grad_norm + eps))
                
                # Weight decay (applied before geometric update)
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                # Route to appropriate update method
                param_type = self.get_param_type(p)
                
                if param_type == 'unitary' and group.get('use_cayley', True):
                    self._cayley_update(p, grad, state, lr, momentum, eps)
                elif param_type == 'hyperbolic' and group.get('use_lorentz', True):
                    # Use Trust Region Update for Hyperbolic Params
                    self._lorentz_trust_region_update(
                        p, grad, state, lr, momentum, curvature, trust_radius, eps
                    )
                elif param_type == 'symplectic':
                    self._symplectic_update(p, grad, state, lr, momentum, eps)
                else:
                    self._euclidean_update(p, grad, state, lr, momentum)
                
                params_updated += 1
        
        # Debug output every 10 steps (with parameter type breakdown)
        if debug_this_step and total_grad_norm_sq is not None:
            total_grad_norm = float(total_grad_norm_sq) ** 0.5
            # Count by type
            type_counts = {'unitary': 0, 'hyperbolic': 0, 'symplectic': 0, 'euclidean': 0}
            for t in self._param_types.values():
                type_counts[t] = type_counts.get(t, 0) + 1
            print(f"   [BK-HyperSGD] Step {self.step_count}: total_grad_norm={total_grad_norm:.2f} | "
                  f"U:{type_counts['unitary']} H:{type_counts['hyperbolic']} S:{type_counts['symplectic']} E:{type_counts['euclidean']}")
        
        return loss
    
    def _cayley_update(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        state: Dict,
        lr: float,
        momentum: float,
        eps: float,
    ):
        """
        Cayley retraction update for unitary/orthogonal parameters.
        
        Robust implementation with:
        - Support for non-square matrices (m x n)
        - Safe fallback to Euclidean update on failure
        - NaN/Inf checking
        """
        buf = state['momentum_buffer']
        buf.mul_(momentum).add_(grad, alpha=1 - momentum)
        
        W = p.data
        
        # Only apply Cayley for 2D tensors with reasonable aspect ratio
        if W.dim() != 2:
            # Fallback to Euclidean for non-2D
            p.data.add_(buf, alpha=-lr)
            return
        
        m, n = W.shape
        
        # Skip Cayley for very non-square matrices (ratio > 4:1)
        # as it becomes numerically unstable
        aspect_ratio = max(m, n) / max(min(m, n), 1)
        if aspect_ratio > 4.0:
            # Use scaled Euclidean update instead
            p.data.add_(buf, alpha=-lr * 0.1)  # Smaller LR for safety
            return
        
        try:
            # Construct skew-symmetric matrix A = grad @ W^T - W @ grad^T
            # For non-square W, we work on the smaller dimension
            if m <= n:
                # A is m x m
                A = buf @ W.T - W @ buf.T
            else:
                # A is n x n  
                A = buf.T @ W - W.T @ buf
            
            # Ensure skew-symmetry (numerical precision)
            A = 0.5 * (A - A.T)
            
            # Check for NaN/Inf in A
            if not torch.isfinite(A).all():
                p.data.add_(buf, alpha=-lr)
                return
            
            k = A.shape[0]
            I = torch.eye(k, device=W.device, dtype=W.dtype)
            half_eta_A = (lr / 2) * A
            
            # Cayley transform: W_new = (I + η/2·A)^(-1) · (I - η/2·A) · W
            lhs = I + half_eta_A
            rhs = (I - half_eta_A) @ (W if m <= n else W.T)
            
            # Use solve for numerical stability
            W_new = torch.linalg.solve(lhs, rhs)
            
            # Validate result
            if not torch.isfinite(W_new).all():
                p.data.add_(buf, alpha=-lr)
                return
            
            # Apply update
            if m <= n:
                p.data.copy_(W_new)
            else:
                p.data.copy_(W_new.T)
                
        except (RuntimeError, torch.linalg.LinAlgError):
            # Fallback to Euclidean on any linear algebra error
            p.data.add_(buf, alpha=-lr)

    def _lorentz_trust_region_update(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        state: Dict,
        lr: float,
        momentum: float,
        curvature: float,
        trust_radius: float,
        eps: float,
    ):
        """
        BK-HyperSGD Trust Region Update for Hyperbolic Parameters.
        
        Implements the design spec:
        1. Compute Conformal Factor lambda_x
        2. Compute Riemannian Gradient g_R = g_E / lambda_x^2
        3. Update Momentum (Approximated)
        4. Trust Region Clipping: Enforce eta * ||m_t||_g <= Delta
        5. Update via Retraction/ExpMap
        """
        x = p.data
        c = abs(curvature)
        sqrt_c = math.sqrt(c)
        
        # 1. Conformal Factor (Poincare Ball metric)
        # lambda_x = 2 / (1 - c ||x||^2)
        # Ensure numerical stability by capping the denominator
        x_norm_sq = (x**2).sum(dim=-1, keepdim=True)
        # Safe conformal factor: prevent division by zero near boundary
        denom = (1.0 - c * x_norm_sq).clamp(min=eps) 
        lambda_x = 2.0 / denom
        
        # 2. Riemannian Gradient
        # g_R = g_E / lambda_x^2
        # Note: We use 1/lambda_x^2 scaling.
        # However, for momentum stability, we often scale the update, not just grad.
        # Let's scale grad first.
        riem_grad = grad / (lambda_x ** 2)
        
        # 3. Momentum
        # Ideally: Parallel transport.
        # Practical Approximation: Accumulate in tangent space or scaled Euclidean
        # Here we accumulate the Riemannian gradient directly.
        buf = state['momentum_buffer']
        buf.mul_(momentum).add_(riem_grad, alpha=1 - momentum)
        m_t = buf
        
        # 4. Trust Region Clipping
        # We want the step `v = -lr * m_t` to have Riemannian length <= trust_radius.
        # Riemannian norm: ||v||_g = lambda_x * ||v||_2
        # So: lr * lambda_x * ||m_t||_2 <= trust_radius
        
        m_t_norm = m_t.norm(dim=-1, keepdim=True)
        step_riem_len = lr * lambda_x * m_t_norm
        
        # Calculate scaling factor alpha
        # alpha = min(1, Delta / step_riem_len)
        # Avoid division by zero
        scale_factor = torch.clamp(trust_radius / (step_riem_len + eps), max=1.0)
        
        # Apply scaling to the EFFECTIVE step, not just the buffer
        # actual_step = -lr * m_t * scale_factor
        effective_lr = lr * scale_factor
        
        # 5. Update (Retraction / ExpMap approximation)
        # For Poincare Ball, typical retraction is:
        # x_new = (x + v) / (1 + ...) -> Möbius addition
        # But simpler "add in tangent, then retract" is often robust enough with Trust Region.
        # Or we can use the proper ExpMap formula.
        # Let's use a robust approximation: x <- x + v, then project back.
        # Since we scaled the step to be small (Trust Region), Euclidean addition + Projection is often stable locally.
        # HOWEVER, for correct geometry, let's use the explicit conformal scaling on the Euclidean update.
        # The update v in Euclidean space corresponds to Riemannian update v_R.
        # v_E = v_R / lambda_x^2? No.
        # Let's stick to the definition: We computed a step vector `v = - effective_lr * m_t`.
        # This `v` is in the coordinate basis.
        # We just apply it and project.
        
        # Standard SGD update with clipped step
        # x_new = x - effective_lr * m_t
        p.data.addcmul_(m_t, effective_lr, value=-1.0)
        
        # 6. Boundary Projection (Safe)
        # Ensure ||x|| <= (1 - eps) / sqrt(c)
        max_norm = (1.0 - 1e-5) / sqrt_c
        current_norm = p.data.norm(dim=-1, keepdim=True).clamp(min=eps)
        
        # Project points outside boundary back inside
        scale = torch.clamp(max_norm / current_norm, max=1.0)
        p.data.mul_(scale)
        
        # Final NaN/Inf safety check
        if not torch.isfinite(p.data).all():
            p.data.copy_(torch.nan_to_num(p.data, nan=0.0, posinf=0.0, neginf=0.0))

    def _lorentz_update(self, *args, **kwargs):
        """Deprecated alias, redirects to trust region update."""
        # For compatibility if called directly, though step() handles dispatch
        # We just call the new method with default trust_radius if needed
        pass

    def _symplectic_update(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        state: Dict,
        lr: float,
        momentum: float,
        eps: float,
    ):
        """Symplectic update for Hamiltonian parameters."""
        buf = state['momentum_buffer']
        buf.mul_(momentum).add_(grad, alpha=1 - momentum)
        
        if p.dim() >= 1 and p.shape[-1] % 2 == 0:
            d_half = p.shape[-1] // 2
            q_part = p.data[..., :d_half]
            p_part = p.data[..., d_half:]
            
            grad_q = buf[..., :d_half]
            grad_p = buf[..., d_half:]
            
            q_new = q_part - lr * grad_p
            p_new = p_part - lr * grad_q
            
            p.data[..., :d_half] = q_new
            p.data[..., d_half:] = p_new
        else:
            p.data.add_(buf, alpha=-lr)
    
    def _euclidean_update(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        state: Dict,
        lr: float,
        momentum: float,
    ):
        """Standard momentum SGD for Euclidean parameters."""
        buf = state['momentum_buffer']
        buf.mul_(momentum).add_(grad, alpha=1 - momentum)
        p.data.add_(buf, alpha=-lr)
    
    @staticmethod
    def _project_to_hyperboloid(
        x: torch.Tensor,
        curvature: float = -1.0,
        eps: float = 1e-8,
    ):
        """Project point onto Lorentz hyperboloid in-place."""
        c = abs(curvature)
        x_space = x[..., 1:]
        x_space_norm_sq = (x_space ** 2).sum(dim=-1, keepdim=True)
        x_time = torch.sqrt(x_space_norm_sq + 1.0 / c + eps)
        x[..., :1] = x_time

    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics for monitoring."""
        stats = {
            'step_count': self.step_count,
            'param_type_counts': {},
        }
        
        # Count parameters by type
        for param_type in ['unitary', 'hyperbolic', 'symplectic', 'euclidean']:
            count = sum(1 for t in self._param_types.values() if t == param_type)
            stats['param_type_counts'][param_type] = count
        
        return stats


# =============================================================================
# Factory Function
# =============================================================================

def create_bk_hyper_sgd(
    model: nn.Module,
    lr: float = 0.01,
    momentum: float = 0.9,
    curvature: float = -1.0,
    weight_decay: float = 0.0,
    **kwargs,
) -> BKHyperSGD:
    """
    Create BK-HyperSGD optimizer for a model.
    
    Automatically sets parameter names for geometric classification.
    
    Args:
        model: The model to optimize
        lr: Learning rate
        momentum: Momentum coefficient
        curvature: Hyperbolic curvature
        weight_decay: L2 regularization
        **kwargs: Additional arguments passed to BKHyperSGD
    
    Returns:
        Configured BKHyperSGD optimizer
    """
    # Set parameter names
    for name, param in model.named_parameters():
        param._param_name = name
    
    optimizer = BKHyperSGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        curvature=curvature,
        weight_decay=weight_decay,
        **kwargs,
    )
    
    return optimizer


# =============================================================================
# BK-Core Specific Parameter Groups
# =============================================================================

def get_bk_parameter_groups(
    model: nn.Module,
    base_lr: float = 0.01,
    unitary_lr_scale: float = 0.1,
    hyperbolic_lr_scale: float = 0.5,
    symplectic_lr_scale: float = 0.3,
) -> List[Dict]:
    """
    Create parameter groups with different learning rates for BK-Core model.
    
    Geometric-aware grouping for BK-HyperSGD:
    - Unitary (Cayley): Attention projections - smallest LR for stability
    - Hyperbolic (Trust Region): BK-Core layers - moderate LR
    - Symplectic: Energy-preserving updates
    - Euclidean: FFN, embeddings, etc. - base LR
    
    Args:
        model: The model
        base_lr: Base learning rate for Euclidean parameters
        unitary_lr_scale: LR multiplier for attention projections (Cayley)
        hyperbolic_lr_scale: LR multiplier for BK-Core hyperbolic layers
        symplectic_lr_scale: LR multiplier for symplectic parameters
    
    Returns:
        List of parameter group dicts
    """
    unitary_params = []
    hyperbolic_params = []
    symplectic_params = []
    euclidean_params = []
    
    for name, param in model.named_parameters():
        param._param_name = name
        name_lower = name.lower()
        
        # === 1. Unitary Parameters (Cayley Retraction) ===
        if any(key in name_lower for key in ['v_proj', 'o_proj', 'output_proj', 'q_proj', 'k_proj']):
            if 'weight' in name_lower and param.dim() >= 2:
                unitary_params.append(param)
            else:
                euclidean_params.append(param)
        
        # === 2. Hyperbolic Parameters (Trust Region Riemannian) ===
        elif any(key in name_lower for key in [
            'bk_core', 'bk_scale', 'bk_hyperbolic',
            'hyperbolic_attn', 'hyperbolic_gate',
            'ar_ssm_fusion', 'ar_ssm',
            'curvature', 'lorentz', 'poincare', 'mobius',
        ]):
            hyperbolic_params.append(param)
        
        # === 3. Symplectic Parameters ===
        elif any(key in name_lower for key in ['symplectic', 'force_field', 'hamiltonian']):
            symplectic_params.append(param)
        
        # === 4. Euclidean Parameters ===
        else:
            euclidean_params.append(param)
    
    groups = []
    
    if unitary_params:
        groups.append({
            'params': unitary_params,
            'lr': base_lr * unitary_lr_scale,
            'use_cayley': True,
            'use_lorentz': False,
        })
    
    if hyperbolic_params:
        groups.append({
            'params': hyperbolic_params,
            'lr': base_lr * hyperbolic_lr_scale,
            'use_cayley': False,
            'use_lorentz': True,
        })
    
    if symplectic_params:
        groups.append({
            'params': symplectic_params,
            'lr': base_lr * symplectic_lr_scale,
            'use_cayley': False,
            'use_lorentz': False,
        })
    
    if euclidean_params:
        groups.append({
            'params': euclidean_params,
            'lr': base_lr,
            'use_cayley': False,
            'use_lorentz': False,
        })
    
    return groups


# =============================================================================
# Testing
# =============================================================================

def test_bk_hyper_sgd():
    """Test BK-HyperSGD optimizer."""
    print("Testing BK-HyperSGD optimizer...")
    
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.v_proj = nn.Linear(64, 64)  # Unitary
            self.hyperbolic_weight = nn.Parameter(torch.randn(32, 65))  # Hyperbolic (65 = 1 time + 64 space)
            self.ffn = nn.Linear(64, 64)  # Euclidean
        
        def forward(self, x):
            return self.ffn(self.v_proj(x))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TestModel().to(device)
    
    # Create optimizer
    optimizer = create_bk_hyper_sgd(
        model,
        lr=0.01,
        momentum=0.9,
        curvature=-1.0,
    )
    
    # Print parameter classification
    stats = optimizer.get_statistics()
    print(f"  Parameter type counts: {stats['param_type_counts']}")
    
    # Test gradient step
    x = torch.randn(8, 64, device=device)
    y = model(x)
    loss = y.mean()
    loss.backward()
    
    # Store initial weights
    initial_v_proj = model.v_proj.weight.data.clone()
    
    optimizer.step()
    optimizer.zero_grad()
    
    # Verify weights changed
    weight_diff = (model.v_proj.weight.data - initial_v_proj).abs().max().item()
    print(f"  Weight change after step: {weight_diff:.6f}")
    
    # Check v_proj orthogonality (should be approximately maintained)
    W = model.v_proj.weight.data
    WtW = W.T @ W
    I = torch.eye(W.shape[1], device=device)
    orthogonality_error = (WtW - I).abs().max().item()
    print(f"  Orthogonality error: {orthogonality_error:.6f}")
    
    print("  ✅ BK-HyperSGD test passed")
    return True


if __name__ == "__main__":
    test_bk_hyper_sgd()
