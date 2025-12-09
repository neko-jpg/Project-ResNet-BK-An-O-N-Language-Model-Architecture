"""
Hyperbolic Modules for ResNet-BK

Research-based implementations for numerical stability in hyperbolic deep learning.
Based on "ResNet-BK 技術監査報告書" (2025-12-09).

Key Principles:
1. Linear layer weights are TANGENT SPACE parameters (Euclidean nn.Parameter)
2. bfloat16 requires larger epsilon (0.01 vs 1e-8) for Poincaré ball boundary
3. BitNet quantization must happen in tangent space, not on manifold
4. Tangent Space Attention is preferred over Fully Hyperbolic Attention
5. Symplectic Green's Initialization (SGI) enables learning vs strict orthogonal

Author: ResNet-BK Project
"""

from __future__ import annotations

import math
from typing import Optional, Dict, Any

import torch
from torch import nn
from torch.nn import functional as F

# Try to import geoopt, fallback to manual implementation if not available
try:
    from geoopt.manifolds import PoincareBall
    _GEOOPT_AVAILABLE = True
except ImportError:
    _GEOOPT_AVAILABLE = False
    PoincareBall = None


# =============================================================================
# 1. Common Utilities (bfloat16-aware)
# =============================================================================

def safe_atanh_bf16(x: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """
    bfloat16-safe atanh.
    
    bfloat16 machine epsilon is ~0.0078, so we need margin > eps/2
    to avoid rounding to 1.0 which causes atanh(1) = inf.
    
    Args:
        x: Input tensor
        eps: Safety margin from boundary (default 0.01 > bf16 epsilon)
    
    Returns:
        atanh(clamp(x, -1+eps, 1-eps))
    """
    x_clamped = torch.clamp(x, min=-1.0 + eps, max=1.0 - eps)
    return torch.atanh(x_clamped)


def safe_project_disk_bf16(
    x: torch.Tensor,
    c: float = 1.0,
    eps: float = 1e-2,
) -> torch.Tensor:
    """
    Project points to be strictly inside the Poincaré ball.
    
    Args:
        x: [..., dim] points
        c: Negative curvature parameter (Poincaré ball radius = 1/sqrt(c))
        eps: bfloat16-aware safety margin
    
    Returns:
        Projected points with ||x|| < (1-eps)/sqrt(c)
    """
    norm = x.norm(dim=-1, keepdim=True)
    maxnorm = (1.0 - eps) / math.sqrt(c)
    
    # Avoid division by zero
    norm_safe = norm.clamp(min=1e-12)
    
    # Only scale if exceeding boundary
    projected = x / norm_safe * maxnorm
    cond = norm > maxnorm
    return torch.where(cond, projected, x)


def register_grad_clamp_hook(
    tensor: torch.Tensor,
    clip_value: float = 1000.0,
) -> None:
    """
    Register gradient clipping hook to prevent explosion.
    
    Args:
        tensor: Parameter to protect
        clip_value: Max absolute gradient value
    """
    if tensor.requires_grad:
        tensor.register_hook(
            lambda grad: grad.clamp_(min=-clip_value, max=clip_value)
        )


# =============================================================================
# 2. Manual logmap0/expmap0 (for when geoopt is not available)
# =============================================================================

def _logmap0_manual(x: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """Log map from Poincaré ball to tangent space at origin."""
    sqrt_c = math.sqrt(c)
    x_norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
    
    # Clamp to avoid atanh(1)
    x_norm_clamped = x_norm.clamp(max=(1.0 / sqrt_c) - eps)
    arg = (sqrt_c * x_norm_clamped).clamp(max=0.99)
    
    # atanh(x) = 0.5 * log((1+x)/(1-x))
    atanh_val = 0.5 * torch.log((1.0 + arg + eps) / (1.0 - arg + eps))
    
    # Scale factor: atanh(sqrt(c)*||x||) / (sqrt(c)*||x||)
    scale = atanh_val / (sqrt_c * x_norm)
    return x * scale


def _expmap0_manual(v: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """Exp map from tangent space at origin to Poincaré ball."""
    sqrt_c = math.sqrt(c)
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)
    
    # Clamp tanh argument to avoid extreme values
    tanh_arg = (sqrt_c * v_norm).clamp(max=15.0)
    
    # Scale factor: tanh(sqrt(c)*||v||) / (sqrt(c)*||v||)
    scale = torch.tanh(tanh_arg) / (sqrt_c * v_norm)
    return v * scale


# =============================================================================
# 3. ReliableMobiusLinear (Topic 1 Solution)
# =============================================================================

class ReliableMobiusLinear(nn.Module):
    """
    Topic 1 Solution: Linear layer with weights in tangent space (Euclidean).
    
    The weight matrix W is a transformation T_0D -> T_0D in tangent space,
    NOT a point on the Poincaré ball. This avoids Riemannian gradient
    scaling that causes explosion when ||W|| -> 1.
    
    Forward:
        1. logmap0: Poincaré -> Tangent (float32 for stability)
        2. F.linear: Standard Euclidean linear transformation
        3. expmap0: Tangent -> Poincaré
    
    Args:
        in_features: Input dimension
        out_features: Output dimension  
        c: Poincaré ball curvature (default 1.0)
        precision: 'fp32' or 'bf16' - affects logmap/expmap precision
        bias: Whether to include bias term
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        c: float = 1.0,
        precision: str = "fp32",
        bias: bool = True,
        use_safe_projection: bool = True,
    ) -> None:
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.c = float(c)
        self.precision = precision
        self.use_safe_projection = use_safe_projection
        
        # geoopt PoincareBall if available
        if _GEOOPT_AVAILABLE:
            self.ball = PoincareBall(c=self.c)
        else:
            self.ball = None
        
        # KEY: Weight is Euclidean parameter, NOT ManifoldParameter
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        # Standard tangent space initialization (Kaiming)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5.0))
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def _to_tangent(self, x: torch.Tensor) -> torch.Tensor:
        """Poincaré -> Tangent space (float32 for stability)."""
        if self.use_safe_projection and x.dtype == torch.bfloat16:
            x = safe_project_disk_bf16(x, c=self.c)
        
        if self.precision == "bf16" and x.dtype == torch.bfloat16:
            x_fp32 = x.float()
            if self.ball is not None:
                v = self.ball.logmap0(x_fp32)
            else:
                v = _logmap0_manual(x_fp32, c=self.c)
            return v.to(torch.bfloat16)
        else:
            if self.ball is not None:
                return self.ball.logmap0(x)
            else:
                return _logmap0_manual(x, c=self.c)
    
    def _from_tangent(self, v: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Tangent space -> Poincaré (float32 for stability)."""
        if self.precision == "bf16" and v.dtype == torch.bfloat16:
            v_fp32 = v.float()
            if self.ball is not None:
                y = self.ball.expmap0(v_fp32)
            else:
                y = _expmap0_manual(v_fp32, c=self.c)
            return y.to(dtype)
        else:
            if self.ball is not None:
                y = self.ball.expmap0(v)
            else:
                y = _expmap0_manual(v, c=self.c)
            return y.to(dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., in_features] on Poincaré ball
        
        Returns:
            [..., out_features] on Poincaré ball
        """
        orig_dtype = x.dtype
        
        # 1. Manifold -> Tangent
        x_tan = self._to_tangent(x)
        
        # 2. Linear (Euclidean operation in tangent space)
        out_tan = F.linear(x_tan, self.weight, self.bias)
        
        # 3. Tangent -> Manifold
        out = self._from_tangent(out_tan, dtype=orig_dtype)
        return out


# =============================================================================
# 4. LorentzWrapper (Topic 2 Strategy B)
# =============================================================================

class LorentzWrapper(nn.Module):
    """
    Topic 2 Strategy B: Wrap operations in Lorentz model for stability.
    
    Poincaré ball has boundary singularity at ||x|| = 1.
    Lorentz hyperboloid has no such singularity - coordinates can grow
    large without numerical issues.
    
    Usage:
        core = SomeLorentzModule(...)
        layer = LorentzWrapper(core, c=1.0)
        y = layer(x_poincare)
    """
    
    def __init__(
        self,
        module: nn.Module,
        c: float = 1.0,
        eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.module = module
        self.c = float(c)
        self.eps = eps
    
    def poincare_to_lorentz(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., n] on Poincaré ball
        returns z: [..., n+1] on Lorentz hyperboloid
        """
        xn = x.norm(dim=-1, keepdim=True).clamp(max=1.0 - self.eps)
        xn2 = xn ** 2
        
        # Standard mapping (c=1)
        factor = 1.0 / (1.0 - xn2)
        z0 = (1.0 + xn2) * factor
        z_spatial = 2.0 * x * factor
        return torch.cat((z0, z_spatial), dim=-1)
    
    def lorentz_to_poincare(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [..., n+1] on Lorentz hyperboloid
        returns x: [..., n] on Poincaré ball
        """
        z0 = z[..., 0:1]
        z_spatial = z[..., 1:]
        
        # x = z_spatial / (z0 + 1)
        # z0 >= 1 so denominator >= 2, numerically stable
        denom = z0 + 1.0
        return z_spatial / denom
    
    def forward(self, x_poincare: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # 1. Poincaré -> Lorentz
        z = self.poincare_to_lorentz(x_poincare)
        
        # 2. Lorentz space operation
        z_out = self.module(z, *args, **kwargs)
        
        # 3. Lorentz -> Poincaré
        x_out = self.lorentz_to_poincare(z_out)
        return x_out


# =============================================================================
# 5. BitLinearHyperbolic (Topic 3 Solution)
# =============================================================================

class BitLinearHyperbolic(nn.Module):
    """
    Topic 3 Solution: BitNet b1.58 with quantization in tangent space.
    
    Flow:
        x_h (Poincaré) -> logmap0 -> activation_quant (8bit) 
                       -> weight_quant (1.58bit) -> F.linear -> expmap0 -> y_h
    
    The key insight is that quantization ({-1, 0, 1} grid) only makes sense
    in flat (Euclidean) space. On curved manifolds, the grid distorts.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        c: float = 1.0,
        bias: bool = True,
        activation_bits: int = 8,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = float(c)
        self.activation_bits = activation_bits
        
        if _GEOOPT_AVAILABLE:
            self.ball = PoincareBall(c=self.c)
        else:
            self.ball = None
        
        # Weight is tangent space (Euclidean) parameter
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5.0))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def weight_quantization(self, w: torch.Tensor) -> torch.Tensor:
        """
        BitNet b1.58: AbsMean-based {-1, 0, 1} quantization with STE.
        """
        gamma = w.abs().mean().clamp(min=1e-5)
        w_scaled = w / gamma
        w_ternary = w_scaled.round().clamp(-1, 1)
        w_quant = w_ternary * gamma
        
        # Straight-Through Estimator: forward uses w_quant, backward uses w
        return (w_quant - w).detach() + w
    
    def activation_quantization(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-token symmetric int8 quantization with STE.
        """
        qmax = float(2 ** (self.activation_bits - 1) - 1)  # 8bit -> 127
        max_abs = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        scale = qmax / max_abs
        
        x_int = (x * scale).round().clamp(-qmax, qmax)
        x_quant = x_int / scale
        
        # STE
        return (x_quant - x).detach() + x
    
    def forward(self, x_h: torch.Tensor) -> torch.Tensor:
        """
        x_h: [..., in_features] on hyperbolic manifold
        Returns: [..., out_features] on hyperbolic manifold
        """
        orig_dtype = x_h.dtype
        
        # 1. Manifold -> Tangent (FP32)
        x_fp32 = x_h.float()
        if self.ball is not None:
            x_tangent = self.ball.logmap0(x_fp32)
        else:
            x_tangent = _logmap0_manual(x_fp32, c=self.c)
        
        # 2. Activation Quantization (tangent space)
        x_quant = self.activation_quantization(x_tangent)
        
        # 3. Weight Quantization (tangent space)
        w_quant = self.weight_quantization(self.weight)
        
        # 4. BitLinear (Euclidean operation)
        y_tangent = F.linear(x_quant, w_quant, self.bias)
        
        # 5. Tangent -> Manifold
        if self.ball is not None:
            y_h = self.ball.expmap0(y_tangent)
        else:
            y_h = _expmap0_manual(y_tangent, c=self.c)
        
        return y_h.to(orig_dtype)


# =============================================================================
# 6. HyperbolicTangentAttention (Topic 4 Solution)
# =============================================================================

class HyperbolicTangentAttention(nn.Module):
    """
    Topic 4 Solution: Tangent Space Attention.
    
    Instead of computing hyperbolic distances (which require acosh),
    we map Q, K, V to tangent space and use standard scaled dot-product
    attention. This is FlashAttention compatible and numerically stable.
    
    Forward:
        1. logmap0: Q_h, K_h, V_h -> Q_t, K_t, V_t (tangent space)
        2. MultiheadAttention in tangent space
        3. Residual connection in tangent space (ResNet-BK style)
        4. expmap0: output back to hyperbolic
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        c: float = 1.0,
        dropout: float = 0.0,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.c = float(c)
        self.embed_dim = embed_dim
        
        if _GEOOPT_AVAILABLE:
            self.ball = PoincareBall(c=self.c)
        else:
            self.ball = None
        
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,
        )
    
    def forward(
        self,
        query_h: torch.Tensor,
        key_h: torch.Tensor,
        value_h: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        query_h, key_h, value_h: (batch, seq, embed_dim) on hyperbolic manifold
        
        Returns:
            output_h: Same shape, on hyperbolic manifold
        """
        orig_dtype = query_h.dtype
        
        # 1. Poincaré -> Tangent (FP32 for logmap0)
        if self.ball is not None:
            q_t = self.ball.logmap0(query_h.float())
            k_t = self.ball.logmap0(key_h.float())
            v_t = self.ball.logmap0(value_h.float())
        else:
            q_t = _logmap0_manual(query_h.float(), c=self.c)
            k_t = _logmap0_manual(key_h.float(), c=self.c)
            v_t = _logmap0_manual(value_h.float(), c=self.c)
        
        # 2. Scaled Dot-Product Attention (Euclidean)
        attn_output_t, _ = self.mha(
            q_t.to(orig_dtype),
            k_t.to(orig_dtype),
            v_t.to(orig_dtype),
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        
        # 3. ResNet-style Residual in Tangent Space
        output_t = q_t + attn_output_t.float()
        
        # 4. Tangent -> Manifold
        if self.ball is not None:
            output_h = self.ball.expmap0(output_t)
        else:
            output_h = _expmap0_manual(output_t, c=self.c)
        
        return output_h.to(orig_dtype)


# =============================================================================
# 7. Symplectic Green's Initialization (Topic 5 Solution)
# =============================================================================

def symplectic_greens_init_(
    tensor: torch.Tensor,
    dt: float = 0.01,
    min_decay: float = 1e-4,
    max_decay: float = 1e-2,
) -> None:
    """
    Symplectic Green's Initialization (SGI).
    
    Creates: W = Q @ D
        Q: Random orthogonal matrix (energy-conserving rotation)
        D: diag(exp(-γ * dt)) decay factors
    
    This puts eigenvalues slightly inside the unit circle (|λ| < 1 but close to 1),
    allowing long-range information flow while enabling forgetting/learning.
    
    Strict orthogonal (|λ| = 1 exactly) blocks learning because there's no
    decay - the system becomes a perfect oscillator with no damping.
    
    Args:
        tensor: Weight tensor (must be square for full SGI)
        dt: Time step for decay (larger = more decay)
        min_decay: Minimum decay rate γ
        max_decay: Maximum decay rate γ
    """
    if tensor.ndim != 2:
        nn.init.orthogonal_(tensor)
        return
    
    rows, cols = tensor.shape
    if rows != cols:
        # Non-square: just use orthogonal
        nn.init.orthogonal_(tensor)
        return
    
    with torch.no_grad():
        # 1. Random orthogonal Q
        Q = torch.empty(rows, cols, device=tensor.device, dtype=tensor.dtype)
        nn.init.orthogonal_(Q)
        
        # 2. Green's-style decay factors
        # γ ~ Uniform(min_decay, max_decay)
        gamma = torch.rand(rows, device=tensor.device, dtype=tensor.dtype)
        gamma = gamma * (max_decay - min_decay) + min_decay
        
        # D = diag(exp(-γ * dt))
        decay_factors = torch.exp(-gamma * dt)
        D = torch.diag(decay_factors)
        
        # 3. W = Q @ D
        W_init = Q @ D
        tensor.copy_(W_init)


# =============================================================================
# Testing
# =============================================================================

def test_hyperbolic_modules():
    """Quick smoke test for all modules."""
    print("Testing hyperbolic modules...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test safe_atanh_bf16
    x = torch.tensor([0.99, -0.99, 0.5], dtype=torch.bfloat16, device=device)
    y = safe_atanh_bf16(x)
    assert not torch.isnan(y).any(), "safe_atanh_bf16 produced NaN"
    print("  ✓ safe_atanh_bf16")
    
    # Test safe_project_disk_bf16
    x = torch.randn(4, 64, dtype=torch.bfloat16, device=device) * 2  # Some outside ball
    y = safe_project_disk_bf16(x)
    norms = y.norm(dim=-1)
    assert (norms < 1.0).all(), "safe_project_disk_bf16 failed to project"
    print("  ✓ safe_project_disk_bf16")
    
    # Test ReliableMobiusLinear
    layer = ReliableMobiusLinear(64, 64, precision="bf16").to(device)
    x = torch.randn(2, 10, 64, device=device).bfloat16() * 0.1  # Small values in ball
    y = layer(x)
    assert not torch.isnan(y).any(), "ReliableMobiusLinear produced NaN"
    print("  ✓ ReliableMobiusLinear")
    
    # Test BitLinearHyperbolic
    layer = BitLinearHyperbolic(64, 64).to(device)
    x = torch.randn(2, 10, 64, device=device) * 0.1
    y = layer(x)
    assert not torch.isnan(y).any(), "BitLinearHyperbolic produced NaN"
    print("  ✓ BitLinearHyperbolic")
    
    # Test HyperbolicTangentAttention
    attn = HyperbolicTangentAttention(64, num_heads=4).to(device)
    x = torch.randn(2, 10, 64, device=device) * 0.1
    y = attn(x, x, x)
    assert not torch.isnan(y).any(), "HyperbolicTangentAttention produced NaN"
    print("  ✓ HyperbolicTangentAttention")
    
    # Test symplectic_greens_init_
    W = torch.empty(64, 64, device=device)
    symplectic_greens_init_(W)
    # Check eigenvalues are inside unit circle
    eigvals = torch.linalg.eigvals(W).abs()
    assert (eigvals < 1.0).all(), "SGI eigenvalues should be < 1"
    assert (eigvals > 0.9).all(), "SGI eigenvalues should be close to 1"
    print("  ✓ symplectic_greens_init_")
    
    print("\n✅ All hyperbolic modules tests passed!")
    return True


if __name__ == "__main__":
    test_hyperbolic_modules()
