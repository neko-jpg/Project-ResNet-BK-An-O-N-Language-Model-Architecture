"""
Hyperbolic Mixture of Experts (HMoE) - Moonshot #8

Replaces learned router with hyperbolic distance-based expert assignment.
Experts are assigned based on Poincaré distance to expert centroids.

Theory (from research docs):
- Traditional MoE uses a learned router network to assign tokens to experts
- In hyperbolic space, semantic similarity = geometric distance
- Expert selection based on hyperbolic distance eliminates router parameters
- Voronoi partitioning in hyperbolic space for natural specialization

Expected: Router-free MoE with emergent semantic specialization

Reference: docs/research/物理概念による深層学習革新リサーチ.md, Section 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# =============================================================================
# Triton Kernel: Batched Hyperbolic Distance for Expert Routing
# =============================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def hyperbolic_expert_distance_kernel(
        x_ptr,        # Input embeddings [B, N, D]
        centroids_ptr, # Expert centroids [E, D]
        output_ptr,    # Output distances [B, N, E]
        B, N, D, E,    # Dimensions
        stride_xb, stride_xn, stride_xd,
        stride_cb, stride_cd,
        stride_ob, stride_on, stride_oe,
        BLOCK_E: tl.constexpr,
    ):
        """
        Compute Poincaré distance from each token to each expert centroid.
        """
        batch_idx = tl.program_id(0)
        token_idx = tl.program_id(1)
        
        # Load input token
        x_offset = batch_idx * stride_xb + token_idx * stride_xn
        
        # Compute ||x||^2
        x_norm_sq = 0.0
        for d in range(D):
            x_d = tl.load(x_ptr + x_offset + d * stride_xd)
            x_norm_sq += x_d * x_d
        
        # Clamp to Poincaré ball interior
        x_norm_sq = tl.minimum(x_norm_sq, 0.99)
        
        # Compute distance to each expert centroid
        for e in range(E):
            c_offset = e * stride_cb
            
            # Compute ||x - c||^2 and ||c||^2
            diff_sq = 0.0
            c_norm_sq = 0.0
            for d in range(D):
                x_d = tl.load(x_ptr + x_offset + d * stride_xd)
                c_d = tl.load(centroids_ptr + c_offset + d * stride_cd)
                diff_sq += (x_d - c_d) * (x_d - c_d)
                c_norm_sq += c_d * c_d
            
            c_norm_sq = tl.minimum(c_norm_sq, 0.99)
            
            # Poincaré distance formula
            # d(x, c) = arccosh(1 + 2 * ||x-c||^2 / ((1-||x||^2)(1-||c||^2)))
            denom = (1.0 - x_norm_sq) * (1.0 - c_norm_sq) + 1e-7
            arg = 1.0 + 2.0 * diff_sq / denom
            arg = tl.maximum(arg, 1.0 + 1e-7)  # Ensure valid arccosh input
            
            # arccosh approximation: arccosh(x) = log(x + sqrt(x^2 - 1))
            dist = tl.log(arg + tl.sqrt(arg * arg - 1.0 + 1e-7))
            
            # Store distance
            out_offset = batch_idx * stride_ob + token_idx * stride_on + e * stride_oe
            tl.store(output_ptr + out_offset, dist)


# =============================================================================
# Hyperbolic MoE Module
# =============================================================================

class HyperbolicMoE(nn.Module):
    """
    Mixture of Experts with hyperbolic distance-based routing.
    
    Instead of a learned router, tokens are assigned to experts
    based on their Poincaré distance to expert centroids.
    This provides:
    - No additional router parameters
    - Natural semantic clustering
    - Load balancing through centroid optimization
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        expert_dim: int = None,
        top_k: int = 2,
        curvature: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.expert_dim = expert_dim or d_model * 4
        self.top_k = top_k
        self.curvature = curvature
        
        # Expert centroids in Poincaré ball (learnable)
        # Initialize on a hypersphere with small radius
        centroids = torch.randn(num_experts, d_model)
        centroids = centroids / centroids.norm(dim=-1, keepdim=True) * 0.5
        self.centroids = nn.Parameter(centroids)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, self.expert_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.expert_dim, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        ])
        
        # For projecting to Poincaré ball
        self.input_proj = nn.Linear(d_model, d_model)
        
        # Statistics
        self.expert_counts = torch.zeros(num_experts)
    
    def _poincare_distance(
        self,
        x: torch.Tensor,  # [B, N, D]
        c: torch.Tensor,  # [E, D]
    ) -> torch.Tensor:
        """
        Compute Poincaré distance from each token to each centroid.
        
        Returns: distances [B, N, E]
        """
        # Expand for broadcasting
        x = x.unsqueeze(-2)  # [B, N, 1, D]
        c = c.unsqueeze(0).unsqueeze(0)  # [1, 1, E, D]
        
        # Compute norms (clamp to stay in ball interior)
        x_norm_sq = (x * x).sum(-1).clamp(max=0.99)  # [B, N, 1]
        c_norm_sq = (c * c).sum(-1).clamp(max=0.99)  # [1, 1, E]
        
        # ||x - c||^2
        diff = x - c  # [B, N, E, D]
        diff_sq = (diff * diff).sum(-1)  # [B, N, E]
        
        # Poincaré distance
        denom = (1 - x_norm_sq) * (1 - c_norm_sq) + 1e-7
        arg = 1 + 2 * diff_sq / denom
        arg = arg.clamp(min=1.0 + 1e-7)
        
        dist = torch.acosh(arg)
        return dist
    
    def _project_to_ball(self, x: torch.Tensor, max_norm: float = 0.95) -> torch.Tensor:
        """Project input to Poincaré ball interior."""
        norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-7)
        scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
        return x * scale
    
    def forward(
        self,
        x: torch.Tensor,
        return_routing: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, any]]:
        """
        Forward pass with hyperbolic routing.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            output: Expert-processed output [batch, seq_len, d_model]
            diagnostics: Routing statistics
        """
        batch, seq_len, d_model = x.shape
        
        # Project to Poincaré ball
        x_hyper = self.input_proj(x)
        x_hyper = self._project_to_ball(x_hyper)
        
        # Compute distances to all expert centroids
        centroids_proj = self._project_to_ball(self.centroids)
        distances = self._poincare_distance(x_hyper, centroids_proj)  # [B, N, E]
        
        # Convert distances to routing weights (closer = higher weight)
        # Use negative distance with softmax
        routing_logits = -distances / math.sqrt(self.curvature)
        
        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(routing_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # [B, N, k]
        
        # Compute expert outputs
        output = torch.zeros_like(x)
        
        for k in range(self.top_k):
            expert_idx = top_k_indices[..., k]  # [B, N]
            weight = top_k_weights[..., k].unsqueeze(-1)  # [B, N, 1]
            
            # Gather tokens for each expert (simplified - could be batched better)
            for e in range(self.num_experts):
                mask = (expert_idx == e)  # [B, N]
                if mask.any():
                    expert_input = x[mask]  # [num_tokens, D]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += weight[mask].squeeze(-1).unsqueeze(-1) * expert_output
                    
                    # Update statistics
                    self.expert_counts[e] += mask.sum().item()
        
        diagnostics = {
            'routing_entropy': -(top_k_weights * top_k_weights.log().clamp(min=-100)).sum(-1).mean().item(),
            'expert_utilization': (self.expert_counts > 0).float().mean().item(),
            'max_distance': distances.max().item(),
            'min_distance': distances.min().item(),
        }
        
        if return_routing:
            diagnostics['routing_weights'] = top_k_weights
            diagnostics['routing_indices'] = top_k_indices
        
        return output, diagnostics
    
    def reset_stats(self):
        """Reset expert utilization statistics."""
        self.expert_counts.zero_()
    
    def get_expert_utilization(self) -> torch.Tensor:
        """Get normalized expert utilization."""
        total = self.expert_counts.sum()
        if total > 0:
            return self.expert_counts / total
        return self.expert_counts


class HyperbolicVoronoiRouter(nn.Module):
    """
    Fast Voronoi-based routing in hyperbolic space.
    
    Uses the Klein model for faster linear separability of Voronoi regions.
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        
        # Expert hyperplanes in Klein model (for linear Voronoi boundaries)
        self.hyperplanes = nn.Parameter(torch.randn(num_experts, d_model + 1))
    
    def _poincare_to_klein(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from Poincaré ball to Klein model."""
        norm_sq = (x * x).sum(-1, keepdim=True).clamp(max=0.99)
        return 2 * x / (1 + norm_sq)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fast expert assignment using Klein model hyperplanes.
        
        Returns expert indices for each token.
        """
        # Convert to Klein model
        x_klein = self._poincare_to_klein(x)
        
        # Add homogeneous coordinate
        ones = torch.ones(*x_klein.shape[:-1], 1, device=x.device)
        x_homo = torch.cat([x_klein, ones], dim=-1)  # [B, N, D+1]
        
        # Compute signed distance to each hyperplane
        # In Klein model, Voronoi boundaries are hyperplanes
        distances = torch.einsum('...d,ed->...e', x_homo, self.hyperplanes)
        
        # Assign to expert with maximum "inner" score
        return distances.argmax(dim=-1)


def create_hyperbolic_moe(
    d_model: int = 256,
    num_experts: int = 8,
    top_k: int = 2,
) -> HyperbolicMoE:
    """Factory function for Hyperbolic MoE."""
    return HyperbolicMoE(
        d_model=d_model,
        num_experts=num_experts,
        top_k=top_k,
    )
