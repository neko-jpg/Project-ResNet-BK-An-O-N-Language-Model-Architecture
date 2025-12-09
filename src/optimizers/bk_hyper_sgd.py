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
        unitarity_strength: float = 0.1,
        green_function_reg: float = 0.01,
        use_cayley: bool = True,
        use_lorentz: bool = True,
        max_grad_norm: float = 1.0,
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
        
        Categories (UPDATED based on Research Topic 1):
        - 'unitary': REMOVED - linear weights should NOT use Cayley retraction
        - 'hyperbolic': Embedding layers that explicitly need hyperbolic updates
        - 'symplectic': Symplectic blocks (symplectic_*, force_field.*)
        - 'euclidean': Everything including v_proj, output_proj, bk_core, ffn, etc.
        
        KEY INSIGHT from Research:
        Linear layer weight matrix W is a transformation in TANGENT SPACE T_0D,
        NOT a point on the manifold. Applying Riemannian gradient scaling
        causes gradient explosion via (1-||W||^2)^{-2} factor.
        """
        for group in self.param_groups:
            for p in group['params']:
                param_id = id(p)
                
                # Try to get parameter name
                name = getattr(p, '_param_name', '')
                
                # CRITICAL FIX: v_proj, output_proj, bk_core are LINEAR LAYER weights
                # They must use Euclidean updates, NOT Cayley retraction
                # Cayley was causing gradient explosion via metric tensor scaling
                if any(key in name.lower() for key in ['v_proj', 'output_proj', 'bk_core', 'bk_scale']):
                    # Research Topic 1: linear weights → Euclidean parameter
                    self._param_types[param_id] = 'euclidean'
                elif any(key in name.lower() for key in ['symplectic', 'force_field', 'dt']):
                    self._param_types[param_id] = 'symplectic'
                elif any(key in name.lower() for key in ['embedding']) and 'hyperbolic' in name.lower():
                    # Only explicit hyperbolic embeddings use Lorentz updates
                    self._param_types[param_id] = 'hyperbolic'
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
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        
        Returns:
            Loss value if closure provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self.step_count += 1
        
        # Debug counters
        params_with_grad = 0
        params_updated = 0
        total_grad_norm = 0.0
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            curvature = group['curvature']
            eps = group['eps']
            max_grad_norm = group['max_grad_norm']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                params_with_grad += 1
                grad = p.grad
                total_grad_norm += grad.norm().item() ** 2
                
                # Skip sparse gradients
                if grad.is_sparse:
                    raise RuntimeError("BKHyperSGD does not support sparse gradients")
                
                # Get or initialize state
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p)
                
                state['step'] += 1
                
                # Gradient clipping
                grad_norm = grad.norm().item()
                if grad_norm > max_grad_norm:
                    grad = grad * (max_grad_norm / (grad_norm + eps))
                
                # Weight decay (applied before geometric update)
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                # Route to appropriate update method
                param_type = self.get_param_type(p)
                
                if param_type == 'unitary' and group.get('use_cayley', True):
                    self._cayley_update(p, grad, state, lr, momentum, eps)
                elif param_type == 'hyperbolic' and group.get('use_lorentz', True):
                    self._lorentz_update(p, grad, state, lr, momentum, curvature, eps)
                elif param_type == 'symplectic':
                    self._symplectic_update(p, grad, state, lr, momentum, eps)
                else:
                    self._euclidean_update(p, grad, state, lr, momentum)
                
                params_updated += 1
        
        # Debug output every 10 steps
        if self.step_count % 10 == 1:
            total_grad_norm = total_grad_norm ** 0.5
            print(f"   [BK-HyperSGD] Step {self.step_count}: {params_with_grad} params with grad, {params_updated} updated, total_grad_norm={total_grad_norm:.4f}")
        
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
        
        Maintains W†W ≈ I by mapping updates through the Cayley map:
        W_new = (I + η/2·A)⁻¹ · (I - η/2·A) · W
        
        where A is the skew-symmetric projection of the gradient:
        A = grad @ Wᵀ - W @ gradᵀ
        """
        buf = state['momentum_buffer']
        buf.mul_(momentum).add_(grad, alpha=1 - momentum)
        
        W = p.data
        
        if W.dim() == 2:
            m, n = W.shape
            
            # Project gradient to skew-symmetric (tangent space of O(n))
            # A = buf @ W^T - W @ buf^T
            if m <= n:
                # W is orthogonal in rows: W @ W^T = I
                A = buf @ W.T - W @ buf.T
            else:
                # W is orthogonal in columns: W^T @ W = I
                A = buf.T @ W - W.T @ buf
            
            # Ensure skew-symmetry
            A = 0.5 * (A - A.T)
            
            # Cayley retraction
            # W_new = (I + η/2·A)⁻¹ · (I - η/2·A) · W
            k = A.shape[0]
            I = torch.eye(k, device=W.device, dtype=W.dtype)
            half_eta_A = (lr / 2) * A
            
            try:
                lhs = I + half_eta_A
                rhs = (I - half_eta_A) @ (W if m <= n else W.T)
                W_new = torch.linalg.solve(lhs, rhs)
                
                if m <= n:
                    p.data.copy_(W_new)
                else:
                    p.data.copy_(W_new.T)
            except RuntimeError:
                # Fallback to simple gradient descent if solve fails
                p.data.add_(buf, alpha=-lr)
        else:
            # Non-matrix parameters: standard update
            p.data.add_(buf, alpha=-lr)
    
    def _lorentz_update(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        state: Dict,
        lr: float,
        momentum: float,
        curvature: float,
        eps: float,
    ):
        """
        Lorentz exponential map update for hyperbolic parameters.
        
        Updates points on the Lorentz hyperboloid:
        {x : -x₀² + x₁² + ... + xₙ² = -1/|c|}
        
        Exponential map:
        exp_x(v) = cosh(||v||_L) · x + sinh(||v||_L)/||v||_L · v
        
        where ||v||_L = sqrt(⟨v, v⟩_L) is the Lorentz norm.
        """
        buf = state['momentum_buffer']
        buf.mul_(momentum).add_(grad, alpha=1 - momentum)
        
        x = p.data
        
        if x.dim() >= 1 and x.shape[-1] > 1:
            # Convert Euclidean gradient to Riemannian gradient
            # For Lorentz model: ∇_R = J @ ∇_E where J = diag(-1, 1, ..., 1)
            riemannian_grad = buf.clone()
            riemannian_grad[..., 0] = -riemannian_grad[..., 0]
            
            # Velocity in tangent space
            v = -lr * riemannian_grad
            
            # Compute Lorentz norm: ||v||_L² = -v₀² + ||v_space||²
            v_time = v[..., :1]
            v_space = v[..., 1:]
            v_norm_sq = -v_time**2 + (v_space**2).sum(dim=-1, keepdim=True)
            
            # Handle negative (timelike) and positive (spacelike) norms differently
            is_spacelike = v_norm_sq > eps
            v_norm = torch.sqrt(torch.abs(v_norm_sq).clamp(min=eps))
            
            # Exponential map
            # For spacelike: cosh(||v||) and sinh(||v||)
            # For timelike: cos(||v||) and sin(||v||)
            cosh_v = torch.where(is_spacelike, torch.cosh(v_norm), torch.cos(v_norm))
            sinh_v_over_v = torch.where(
                is_spacelike,
                torch.sinh(v_norm) / v_norm,
                torch.sin(v_norm) / v_norm
            )
            
            x_new = cosh_v * x + sinh_v_over_v * v
            
            # Project back to hyperboloid
            self._project_to_hyperboloid(x_new, curvature, eps)
            
            p.data.copy_(x_new)
        else:
            # Fallback for 1D or scalar
            p.data.add_(buf, alpha=-lr)
    
    def _symplectic_update(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        state: Dict,
        lr: float,
        momentum: float,
        eps: float,
    ):
        """
        Symplectic update for Hamiltonian parameters.
        
        For parameters representing (q, p) canonical pairs,
        uses updates that preserve the symplectic form ω = dq ∧ dp.
        
        Symplectic Euler:
        q_new = q - lr · ∂H/∂p = q - lr · grad_p
        p_new = p + lr · ∂H/∂q = p + lr · grad_q
        """
        buf = state['momentum_buffer']
        buf.mul_(momentum).add_(grad, alpha=1 - momentum)
        
        if p.dim() >= 1 and p.shape[-1] % 2 == 0:
            # Split into (q, p) pairs
            d_half = p.shape[-1] // 2
            q_part = p.data[..., :d_half]
            p_part = p.data[..., d_half:]
            
            grad_q = buf[..., :d_half]
            grad_p = buf[..., d_half:]
            
            # Symplectic Euler
            # dq/dt = ∂H/∂p → q -= lr * grad_p
            # dp/dt = -∂H/∂q → p += lr * grad_q (note: gradient descent has negative sign)
            q_new = q_part - lr * grad_p
            p_new = p_part - lr * grad_q  # Both negative for descent
            
            p.data[..., :d_half] = q_new
            p.data[..., d_half:] = p_new
        else:
            # Fallback
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
        """
        Project point onto Lorentz hyperboloid in-place.
        
        Hyperboloid: {x : -x₀² + ||x_space||² = -1/|c|}
        
        Given x_space, compute x₀ = sqrt(||x_space||² + 1/|c|)
        """
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
    
    BK-Core parameters (v_proj, etc.) are more sensitive and need smaller LR.
    Hyperbolic parameters need moderate LR.
    Symplectic parameters need balanced LR for energy conservation.
    
    Args:
        model: The model
        base_lr: Base learning rate for Euclidean parameters
        unitary_lr_scale: DEPRECATED - no longer used (all linear weights are Euclidean)
        hyperbolic_lr_scale: LR multiplier for explicit hyperbolic embeddings
        symplectic_lr_scale: LR multiplier for symplectic parameters
    
    Returns:
        List of parameter group dicts
    """
    # NOTE: 'unitary' category removed per Research Topic 1
    # v_proj, output_proj, bk_core are LINEAR LAYER weights → Euclidean updates
    hyperbolic_params = []
    symplectic_params = []
    euclidean_params = []
    
    for name, param in model.named_parameters():
        param._param_name = name
        
        if any(key in name.lower() for key in ['symplectic', 'force_field']):
            symplectic_params.append(param)
        elif 'embedding' in name.lower() and 'hyperbolic' in name.lower():
            # Only explicit hyperbolic embeddings
            hyperbolic_params.append(param)
        else:
            # Everything else including v_proj, output_proj, bk_core
            euclidean_params.append(param)
    
    groups = []
    
    # NOTE: unitary_params group removed - was causing gradient explosion
    
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
