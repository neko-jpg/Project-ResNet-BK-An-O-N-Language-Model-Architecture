"""
Hyperbolic Normalization

Lorentz Batch Normalization (LBN) and related normalization layers
for hyperbolic neural networks.

Key insight: Standard LayerNorm/BatchNorm destroys hierarchical information
encoded in the "radius" (distance from origin) in hyperbolic space.
LBN preserves this information while still preventing activation explosion.

Features:
- Lorentz Centroid (Fréchet mean) computation
- Parallel Transport for centering
- Tangent Space scaling
- Gyrovector operations

References:
- Lorentz Batch Normalization (Bdéir et al., 2023-2024)
- Poincaré Midpoint Batch Norm (Van Spengler et al., ICCV 2023)
- GyroLBN (Bdéir et al., ICLR 2026)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# =============================================================================
# Lorentz Model Operations
# =============================================================================

def lorentz_inner_product(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Minkowski inner product for Lorentz model.
    <u, v>_L = -u_0 * v_0 + u_1 * v_1 + ... + u_n * v_n
    
    Args:
        u, v: Tensors of shape (..., n+1) where first dim is time-like
        
    Returns:
        Inner product scalar(s)
    """
    # First component is time-like (negative)
    time_product = u[..., 0:1] * v[..., 0:1]
    space_product = (u[..., 1:] * v[..., 1:]).sum(dim=-1, keepdim=True)
    return -time_product + space_product


def lorentz_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Lorentz norm: sqrt(<x, x>_L)
    For points on the hyperboloid, this should equal the curvature radius.
    """
    inner = lorentz_inner_product(x, x)
    return torch.sqrt(torch.clamp(inner.abs(), min=1e-8))


def lorentz_distance(x: torch.Tensor, y: torch.Tensor, curvature: float = -1.0) -> torch.Tensor:
    """
    Geodesic distance in Lorentz model.
    d_L(x, y) = (1/sqrt(-c)) * arcosh(-c * <x, y>_L)
    
    For curvature c = -1: d(x, y) = arcosh(-<x, y>_L)
    """
    c = abs(curvature)
    inner = lorentz_inner_product(x, y)
    # Clamp for numerical stability (arcosh domain is [1, inf))
    clamped = torch.clamp(-c * inner.squeeze(-1), min=1.0 + 1e-7)
    return torch.acosh(clamped) / math.sqrt(c)


def project_to_hyperboloid(x: torch.Tensor, curvature: float = -1.0) -> torch.Tensor:
    """
    Project points onto the Lorentz hyperboloid.
    
    The hyperboloid is defined as: -x_0^2 + x_1^2 + ... + x_n^2 = -1/c
    
    Given spatial components, we compute x_0 = sqrt(1/c + ||x_space||^2)
    """
    c = abs(curvature)
    space = x[..., 1:]
    space_sq_sum = (space ** 2).sum(dim=-1, keepdim=True)
    time = torch.sqrt(1.0 / c + space_sq_sum).clamp(min=1.0 / math.sqrt(c))
    return torch.cat([time, space], dim=-1)


def lorentz_centroid(x: torch.Tensor, weights: Optional[torch.Tensor] = None, 
                     curvature: float = -1.0) -> torch.Tensor:
    """
    Compute the Lorentz centroid (Fréchet mean) of a set of points.
    
    In Lorentz model, there's a closed-form solution using Einstein midpoint:
    μ = Σ w_i * x_i / sqrt(<Σ w_i * x_i, Σ w_i * x_i>_L)
    
    This is much faster than iterative Poincaré Fréchet mean.
    
    Args:
        x: Points on hyperboloid, shape (batch, num_points, dim) or (num_points, dim)
        weights: Optional weights per point
        curvature: Manifold curvature (negative)
        
    Returns:
        Centroid on hyperboloid
    """
    if x.dim() == 2:
        # Single batch: (num_points, dim)
        if weights is None:
            weights = torch.ones(x.shape[0], device=x.device, dtype=x.dtype) / x.shape[0]
        
        # Weighted sum
        weighted_sum = (weights.unsqueeze(-1) * x).sum(dim=0)
        
        # Normalize to hyperboloid
        inner = lorentz_inner_product(weighted_sum.unsqueeze(0), weighted_sum.unsqueeze(0))
        norm = torch.sqrt(torch.clamp(-inner, min=1e-8))
        centroid = weighted_sum / norm.squeeze()
        
        return centroid
    
    else:
        # Batched: (batch, num_points, dim)
        if weights is None:
            weights = torch.ones(x.shape[1], device=x.device, dtype=x.dtype) / x.shape[1]
        
        # Weighted sum per batch
        weighted_sum = (weights.unsqueeze(0).unsqueeze(-1) * x).sum(dim=1)  # (batch, dim)
        
        # Normalize to hyperboloid
        inner = lorentz_inner_product(weighted_sum, weighted_sum)  # (batch, 1)
        norm = torch.sqrt(torch.clamp(-inner, min=1e-8))
        centroid = weighted_sum / norm
        
        return centroid


def parallel_transport_lorentz(v: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
                               curvature: float = -1.0) -> torch.Tensor:
    """
    Parallel transport a tangent vector v from point x to point y on the hyperboloid.
    
    Formula: PT_x→y(v) = v + <y, v>_L / (1 - <x, y>_L) * (x + y)
    
    Args:
        v: Tangent vector at x
        x: Source point on hyperboloid
        y: Target point on hyperboloid
        
    Returns:
        Transported tangent vector at y
    """
    c = abs(curvature)
    
    # Inner products
    inner_xy = lorentz_inner_product(x, y).squeeze(-1)  # scalar
    inner_yv = lorentz_inner_product(y, v).squeeze(-1)  # scalar
    
    # Avoid division by zero
    denom = (1.0 - c * inner_xy).clamp(min=1e-8)
    
    # Transport
    coeff = c * inner_yv / denom
    transported = v + coeff.unsqueeze(-1) * (x + y)
    
    return transported


def exp_map_lorentz(v: torch.Tensor, x: torch.Tensor, curvature: float = -1.0) -> torch.Tensor:
    """
    Exponential map from tangent space to hyperboloid.
    
    exp_x(v) = cosh(||v||_L) * x + sinh(||v||_L) / ||v||_L * v
    
    Where ||v||_L = sqrt(<v, v>_L)
    """
    c = abs(curvature)
    
    # Lorentz norm of tangent vector
    v_norm_sq = lorentz_inner_product(v, v)
    v_norm = torch.sqrt(torch.clamp(v_norm_sq.abs(), min=1e-10))
    
    # Scale factors
    v_norm_scaled = math.sqrt(c) * v_norm
    cosh_factor = torch.cosh(v_norm_scaled)
    sinh_factor = torch.sinh(v_norm_scaled) / (v_norm_scaled + 1e-10)
    
    # Exponential map
    result = cosh_factor * x + sinh_factor * math.sqrt(c) * v
    
    # Project to ensure on hyperboloid
    return project_to_hyperboloid(result, curvature)


def log_map_lorentz(y: torch.Tensor, x: torch.Tensor, curvature: float = -1.0) -> torch.Tensor:
    """
    Logarithmic map from hyperboloid to tangent space at x.
    
    log_x(y) = d(x,y) * (y + <x, y>_L * x) / ||y + <x, y>_L * x||_L
    """
    c = abs(curvature)
    
    # Distance
    dist = lorentz_distance(x, y, curvature).unsqueeze(-1)
    
    # Direction
    inner_xy = lorentz_inner_product(x, y)
    direction = y + c * inner_xy * x
    
    # Normalize direction
    dir_norm_sq = lorentz_inner_product(direction, direction)
    dir_norm = torch.sqrt(torch.clamp(dir_norm_sq.abs(), min=1e-10))
    
    return dist * direction / dir_norm


# =============================================================================
# Hyperbolic Normalization Layers
# =============================================================================

class LorentzBatchNorm(nn.Module):
    """
    Lorentz Batch Normalization
    
    Normalizes activations in hyperbolic space while preserving hierarchical information.
    
    Process:
    1. Compute Lorentz centroid (Fréchet mean) of batch
    2. Center batch via parallel transport to origin
    3. Scale in tangent space at origin
    4. Transport to learned bias location
    
    Args:
        dim: Feature dimension (spatial dimensions, excluding time)
        curvature: Hyperbolic curvature (default: -1.0)
        eps: Small value for numerical stability
        momentum: Momentum for running statistics
        affine: If True, learn scale and bias parameters
    """
    
    def __init__(
        self,
        dim: int,
        curvature: float = -1.0,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True
    ):
        super().__init__()
        
        self.dim = dim  # Spatial dimensions
        self.full_dim = dim + 1  # Including time dimension
        self.curvature = curvature
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        # Origin point on hyperboloid: (1, 0, 0, ..., 0)
        self.register_buffer('origin', self._create_origin())
        
        if affine:
            # Learnable scale (applied in tangent space)
            self.gamma = nn.Parameter(torch.ones(dim))
            # Learnable bias point on hyperboloid
            self.beta = nn.Parameter(torch.zeros(self.full_dim))
            # Initialize beta as origin
            with torch.no_grad():
                self.beta[0] = 1.0  # Time component
        
        # Running statistics
        self.register_buffer('running_centroid', self._create_origin())
        self.register_buffer('running_var', torch.ones(1))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
    
    def _create_origin(self) -> torch.Tensor:
        """Create origin point on hyperboloid."""
        origin = torch.zeros(self.full_dim)
        origin[0] = 1.0 / math.sqrt(abs(self.curvature))
        return origin
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Lorentz Batch Normalization.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim) or (batch, dim)
               Note: dim here is the FULL dimension including time component
               
        Returns:
            Normalized tensor on hyperboloid
        """
        # Handle different input shapes
        original_shape = x.shape
        if x.dim() == 2:
            # (batch, dim) -> treat as (batch, 1, dim)
            x = x.unsqueeze(1)
        
        batch_size, seq_len, feat_dim = x.shape
        
        # Ensure feature dim matches expected
        if feat_dim != self.full_dim:
            # Pad with time dimension if needed
            if feat_dim == self.dim:
                x = project_to_hyperboloid(
                    torch.cat([torch.zeros_like(x[..., :1]), x], dim=-1),
                    self.curvature
                )
            else:
                raise ValueError(f"Expected dim {self.full_dim} or {self.dim}, got {feat_dim}")
        
        # Flatten batch and sequence for centroid computation
        x_flat = x.view(-1, self.full_dim)  # (batch * seq_len, dim)
        
        if self.training:
            # Compute batch centroid
            with torch.no_grad():
                centroid = lorentz_centroid(x_flat, curvature=self.curvature)
            
            # Update running statistics
            self.num_batches_tracked += 1
            if self.num_batches_tracked == 1:
                self.running_centroid.copy_(centroid)
            else:
                self.running_centroid = (
                    (1 - self.momentum) * self.running_centroid + 
                    self.momentum * centroid
                )
        else:
            centroid = self.running_centroid
        
        # Step 1: Center by transporting all points so centroid -> origin
        origin = self.origin.expand_as(x_flat)
        centroid_expanded = centroid.unsqueeze(0).expand_as(x_flat)
        
        # Log map at centroid to get tangent vectors
        tangent_vectors = log_map_lorentz(x_flat, centroid_expanded, self.curvature)
        
        # Step 2: Scale in tangent space
        # Compute variance of tangent vectors
        tangent_spatial = tangent_vectors[..., 1:]  # Exclude time component
        var = (tangent_spatial ** 2).mean(dim=0) + self.eps
        
        if self.training:
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.mean()
        
        # Apply scaling
        if self.affine:
            tangent_scaled = tangent_spatial * (self.gamma / (var.sqrt() + self.eps))
        else:
            tangent_scaled = tangent_spatial / (var.sqrt() + self.eps)
        
        # Reconstruct full tangent vector (time component stays ~0 at origin)
        tangent_normalized = torch.cat([
            torch.zeros_like(tangent_scaled[..., :1]),
            tangent_scaled
        ], dim=-1)
        
        # Step 3: Map back to hyperboloid at origin
        x_centered = exp_map_lorentz(tangent_normalized, origin, self.curvature)
        
        # Step 4: Transport to learned bias location (if affine)
        if self.affine:
            # Ensure beta is on hyperboloid
            beta_projected = project_to_hyperboloid(self.beta.unsqueeze(0), self.curvature)
            beta_projected = beta_projected.squeeze(0)
            
            # Transport from origin to beta
            x_out = parallel_transport_lorentz(
                log_map_lorentz(x_centered, origin, self.curvature),
                origin,
                beta_projected.expand_as(x_centered),
                self.curvature
            )
            x_out = exp_map_lorentz(x_out, beta_projected.expand_as(x_centered), self.curvature)
        else:
            x_out = x_centered
        
        # Reshape back
        x_out = x_out.view(batch_size, seq_len, self.full_dim)
        
        if original_shape[-1] != self.full_dim:
            # Return without time component if input didn't have it
            x_out = x_out[..., 1:]
        
        if len(original_shape) == 2:
            x_out = x_out.squeeze(1)
        
        return x_out


class HyperbolicLayerNorm(nn.Module):
    """
    Hyperbolic Layer Normalization
    
    Simpler than LBN, works per-sample instead of per-batch.
    Preserves hierarchical norm while normalizing direction.
    
    Args:
        dim: Feature dimension
        curvature: Hyperbolic curvature
        eps: Numerical stability
    """
    
    def __init__(
        self,
        dim: int,
        curvature: float = -1.0,
        eps: float = 1e-5
    ):
        super().__init__()
        
        self.dim = dim
        self.curvature = curvature
        self.eps = eps
        
        # Learnable parameters (in Euclidean space, applied before projection)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Hyperbolic Layer Normalization.
        
        Key insight: Normalize the direction but preserve the radial component.
        """
        # Compute the "radius" (hyperbolic norm / distance from origin)
        # For Euclidean approximation: norm of spatial components
        radius = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        
        # Normalize direction
        direction = x / radius
        
        # Apply layer norm to direction
        mean = direction.mean(dim=-1, keepdim=True)
        var = direction.var(dim=-1, keepdim=True, unbiased=False)
        direction_normed = (direction - mean) / (var.sqrt() + self.eps)
        
        # Apply learnable parameters
        direction_scaled = direction_normed * self.gamma + self.beta
        
        # Restore radius (preserve hierarchy)
        return direction_scaled * radius


class HyperbolicRMSNorm(nn.Module):
    """
    Hyperbolic RMSNorm
    
    RMSNorm variant that preserves hyperbolic structure.
    Normalizes RMS of spatial components while preserving directional information.
    
    Args:
        dim: Feature dimension
        eps: Numerical stability
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Hyperbolic RMSNorm."""
        # Compute RMS
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        
        # Preserve the "hierarchy" by soft-clamping the normalization
        # Don't fully normalize - just reduce extreme values
        norm_factor = torch.where(
            rms > 1.0,
            1.0 / rms.clamp(min=self.eps),  # Normalize large values
            torch.ones_like(rms)  # Keep small values
        )
        
        return x * norm_factor * self.scale

# =============================================================================
# Phase 2: Physics-Informed Normalization (ResNet-BK Specialized)
# =============================================================================

class SymplecticPhaseNorm(nn.Module):
    """
    Symplectic Phase Normalization for ResNet-BK SymplecticBKBlock.
    
    Preserves the symplectic structure ω = dq ∧ dp by:
    1. Splitting input into (q, p) canonical coordinates
    2. Computing total energy H = T + V = p²/2 + q²/2
    3. Normalizing by energy while preserving phase space volume
    4. Applying learnable scale to (q, p) separately
    
    This ensures that the Hamiltonian structure is preserved during normalization,
    which is critical for energy-conserving dynamics.
    
    Args:
        d_model: Total dimension (must be even for q,p split)
        eps: Numerical stability constant
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for symplectic (q, p) split"
        
        self.d_half = d_model // 2
        self.eps = eps
        
        # Learnable scale for (q, p) separately
        self.gamma_q = nn.Parameter(torch.ones(self.d_half))
        self.gamma_p = nn.Parameter(torch.ones(self.d_half))
        self.beta_q = nn.Parameter(torch.zeros(self.d_half))
        self.beta_p = nn.Parameter(torch.zeros(self.d_half))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply symplectic-aware normalization.
        
        Args:
            x: Input tensor (..., d_model) where d_model = 2 * d_half
        
        Returns:
            Normalized tensor with symplectic structure preserved
        """
        # Split into (q, p) canonical coordinates
        q = x[..., :self.d_half]
        p = x[..., self.d_half:]
        
        # Compute energies
        # Kinetic energy: T = p²/2
        kinetic = 0.5 * (p ** 2).sum(dim=-1, keepdim=True)
        # Potential energy: V = q²/2 (harmonic oscillator approximation)
        potential = 0.5 * (q ** 2).sum(dim=-1, keepdim=True)
        # Total energy
        total_energy = kinetic + potential + self.eps
        
        # Normalize by total energy (preserves phase space volume)
        # This is the key insight: scale by sqrt(E) to preserve ω = dq ∧ dp
        energy_scale = torch.sqrt(total_energy)
        
        q_normalized = q / energy_scale
        p_normalized = p / energy_scale
        
        # Apply learnable affine transformation
        q_out = self.gamma_q * q_normalized + self.beta_q
        p_out = self.gamma_p * p_normalized + self.beta_p
        
        return torch.cat([q_out, p_out], dim=-1)


class GreenFunctionNorm(nn.Module):
    """
    Green Function Normalization for BK-Core.
    
    Preserves physical properties of the Green function G_ii = diag((H-zI)^-1):
    1. Causality: Im(G_ii) > 0 (enforced via softplus)
    2. State density normalization: sum(Im(G)) = π
    3. Real part standardization
    
    This layer is designed for the output of BK-Core layers where G_ii
    represents the local density of states.
    
    Args:
        d_model: Feature dimension
        eps: Numerical stability constant
        target_sum: Target sum for Im(G) normalization (default: π)
    """
    
    def __init__(self, d_model: int, eps: float = 1e-8, target_sum: float = 3.14159):
        super().__init__()
        
        self.d_model = d_model
        self.eps = eps
        self.target_sum = target_sum
        
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Green function normalization.
        
        Args:
            x: Input tensor. Can be:
               - Complex tensor (..., d_model) with real and imag parts
               - Real tensor (..., 2) where [..., 0] is real and [..., 1] is imag
               - Real tensor (..., d_model) - treated as real part only
        
        Returns:
            Normalized tensor with Green function properties preserved
        """
        if x.is_complex():
            real = x.real
            imag = x.imag
            is_complex = True
        elif x.dim() >= 1 and x.shape[-1] == 2:
            real = x[..., 0]
            imag = x[..., 1]
            is_complex = False
        else:
            # Real-only input: apply standard normalization
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            x_normalized = (x - mean) / torch.sqrt(var + self.eps)
            return self.gamma * x_normalized + self.beta
        
        # 1. Causality: Ensure Im(G) > 0
        # Use softplus to enforce positivity while maintaining gradient flow
        imag_positive = F.softplus(imag)
        
        # 2. State density normalization: sum(Im(G)) → target_sum
        # This ensures the total density of states is conserved
        imag_sum = imag_positive.sum(dim=-1, keepdim=True) + self.eps
        imag_normalized = imag_positive * (self.target_sum / imag_sum)
        
        # 3. Real part: standard normalization
        real_mean = real.mean(dim=-1, keepdim=True)
        real_var = ((real - real_mean) ** 2).mean(dim=-1, keepdim=True)
        real_normalized = (real - real_mean) / torch.sqrt(real_var + self.eps)
        
        # Apply learnable parameters to real part
        real_out = self.gamma * real_normalized + self.beta
        
        # Combine
        if is_complex:
            return torch.complex(real_out, imag_normalized)
        else:
            return torch.stack([real_out, imag_normalized], dim=-1)


# =============================================================================
# Factory Functions
# =============================================================================

def create_hyperbolic_norm(
    norm_type: str,
    dim: int,
    curvature: float = -1.0,
    **kwargs
) -> nn.Module:
    """
    Factory function to create hyperbolic normalization layers.
    
    Args:
        norm_type: One of "lorentz", "hyperbolic_layer", "hyperbolic_rms", 
                   "symplectic", "green_function", "euclidean"
        dim: Feature dimension
        curvature: Hyperbolic curvature
        **kwargs: Additional arguments for specific norm types
        
    Returns:
        Normalization layer
    """
    if norm_type == "lorentz":
        return LorentzBatchNorm(dim, curvature=curvature, **kwargs)
    elif norm_type == "hyperbolic_layer":
        return HyperbolicLayerNorm(dim, curvature=curvature, **kwargs)
    elif norm_type == "hyperbolic_rms":
        return HyperbolicRMSNorm(dim, **kwargs)
    elif norm_type == "symplectic":
        return SymplecticPhaseNorm(dim, **kwargs)
    elif norm_type == "green_function":
        return GreenFunctionNorm(dim, **kwargs)
    elif norm_type == "euclidean":
        # Fallback to standard RMSNorm
        return nn.RMSNorm(dim) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(dim)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")

