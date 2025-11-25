# src/models/phase7/hyperbolic_attention.py
# Implementation of Hyperbolic Attention Mechanisms

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ##############################################################################
# # HYPERBOLIC UTILITY FUNCTIONS
# ##############################################################################

# Epsilon for numerical stability
EPS = 1e-8

def poincare_distance(x, y, dim=-1):
    """
    Computes pairwise Poincaré distance between two tensors x and y.
    Optimized for multi-head attention using batch matrix multiplication (bmm).

    Args:
        x (torch.Tensor): shape (batch, heads, seq_len, dim)
        y (torch.Tensor): shape (batch, heads, seq_len, dim)
        dim (int): The feature dimension.

    Returns:
        torch.Tensor: Pairwise distance matrix, shape (batch, heads, seq_len, seq_len)
    """
    # Using bmm for efficiency: (B, H, N, D) @ (B, H, D, M) -> (B, H, N, M)
    # where N is query_len and M is key_len
    x_norm_sq = (x * x).sum(dim=dim, keepdim=True)
    y_norm_sq = (y * y).sum(dim=dim, keepdim=True)

    # For pairwise distances, we need to compute ||x_i - y_j||^2
    # This can be expanded as ||x_i||^2 - 2<x_i, y_j> + ||y_j||^2

    # Let x be (B, H, N, D) and y be (B, H, M, D)
    # We want to compute the distance matrix of size (B, H, N, M)

    y_t = y.transpose(-2, -1) # (B, H, D, M)
    xy_dot = torch.matmul(x, y_t) # (B, H, N, M)

    # x_norm_sq is (B, H, N, 1), y_norm_sq is (B, H, M, 1)
    # We need to broadcast them to (B, H, N, M)
    x_norm_sq_broadcast = x_norm_sq.expand(-1, -1, -1, y.shape[-2])
    y_norm_sq_broadcast = y_norm_sq.transpose(-2, -1).expand(-1, -1, x.shape[-2], -1)

    diff_norm_sq = x_norm_sq_broadcast - 2 * xy_dot + y_norm_sq_broadcast

    # Denominator term: (1 - ||x_i||^2)(1 - ||y_j||^2)
    denom = (1 - x_norm_sq_broadcast) * (1 - y_norm_sq_broadcast)

    # Argument of acosh
    arg = 1 + 2 * diff_norm_sq / denom.clamp_min(EPS)

    # The distance can sometimes be slightly < 1 due to floating point errors,
    # which results in NaN for acosh. Clamp to ensure stability.
    return torch.acosh(arg.clamp_min(1.0 + EPS))


def exp_map_at_origin(v, dim=-1):
    """
    Exponential map at the origin of the Poincaré ball.
    Maps a vector `v` from the tangent space at the origin to the manifold.
    Formula: tanh(||v||/2) * (v / ||v||)
    """
    v_norm = v.norm(dim=dim, keepdim=True).clamp_min(EPS)

    # The formula can be simplified for c=1 curvature.
    # We assume standard curvature for now.
    mapped_v = F.tanh(v_norm) * (v / v_norm)
    return mapped_v

def log_map_at_origin(y, dim=-1):
    """
    Logarithmic map at the origin of the Poincaré ball.
    Maps a point `y` from the manifold to the tangent space at the origin.
    Formula: atanh(||y||) * (y / ||y||)
    """
    y_norm = y.norm(dim=dim, keepdim=True).clamp_min(EPS)

    # Clamp y_norm to be slightly less than 1 to avoid infinity in atanh
    y_norm_clamped = y_norm.clamp_max(1.0 - EPS)

    return torch.atanh(y_norm_clamped) * (y / y_norm)


# ##############################################################################
# # SINGLE HEAD HYPERBOLIC ATTENTION
# ##############################################################################

class SingleHeadHyperbolicAttention(nn.Module):
    """
    Implements a single head of hyperbolic attention.
    Assumes q, k, and v are already in the Poincaré ball.
    """
    def __init__(self, d_model: int):
        super().__init__()
        # Learnable scale parameter, initialized near 1.0
        # Using softplus to ensure beta is always positive
        self.beta = nn.Parameter(torch.tensor(0.0)) # softplus(0) is approx 0.69

        # Learnable bias parameter
        self.c = nn.Parameter(torch.tensor(0.0))

    def forward(self, q, k, v):
        """
        Args:
            q (torch.Tensor): Queries in the Poincaré ball. Shape: (batch, seq_len, dim)
            k (torch.Tensor): Keys in the Poincaré ball. Shape: (batch, seq_len, dim)
            v (torch.Tensor): Values in the Poincaré ball. Shape: (batch, seq_len, dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The aggregated value vector in the Poincaré ball and the attention weights.
        """
        # 1. Compute pairwise hyperbolic distances
        # Unsqueeze for the 'heads' dimension to use the optimized distance function
        dist_qk = poincare_distance(q.unsqueeze(1), k.unsqueeze(1)) # (batch, 1, seq_len, seq_len)
        dist_qk = dist_qk.squeeze(1) # (batch, seq_len, seq_len)

        # 2. Calculate attention scores
        # The negative sign is crucial: smaller distance means higher attention
        beta_positive = F.softplus(self.beta)
        attention_scores = -beta_positive * dist_qk - self.c
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 3. Aggregate values in the tangent space
        # 3.1 Map v from manifold to tangent space at origin
        v_tangent = log_map_at_origin(v) # (batch, seq_len, dim)

        # 3.2 Perform weighted sum in tangent space
        # attention_weights: (batch, seq_len, seq_len), v_tangent: (batch, seq_len, dim)
        # output_tangent: (batch, seq_len, dim)
        output_tangent = torch.bmm(attention_weights, v_tangent)

        # 3.3 Map the result back to the manifold
        output_hyperbolic = exp_map_at_origin(output_tangent)

        return output_hyperbolic, attention_weights


# ##############################################################################
# # MULTI-HEAD HYPERBOLIC ATTENTION
# ##############################################################################

class HyperbolicMultiHeadAttention(nn.Module):
    """
    Implements multi-head hyperbolic attention.
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Linear projections for Q, K, V. These project from Euclidean to the tangent space.
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # The core hyperbolic attention mechanism
        # We can reuse the single-head implementation across all heads.
        self.attention = SingleHeadHyperbolicAttention(self.d_head)

        # Final output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor in Euclidean space. Shape: (batch, seq_len, d_model)

        Returns:
            torch.Tensor: Final output tensor in Euclidean space.
        """
        batch_size, seq_len, _ = x.shape

        # 1. Project to tangent space and then map to hyperbolic space
        q_tangent = self.W_q(x)
        k_tangent = self.W_k(x)
        v_tangent = self.W_v(x)

        q_hyp = exp_map_at_origin(q_tangent)
        k_hyp = exp_map_at_origin(k_tangent)
        v_hyp = exp_map_at_origin(v_tangent)

        # 2. Reshape for multi-head processing
        # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_head) -> (batch * num_heads, seq_len, d_head)
        q_hyp = q_hyp.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k_hyp = k_hyp.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v_hyp = v_hyp.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        # Reshape for batch processing by SingleHeadHyperbolicAttention
        q_hyp = q_hyp.reshape(batch_size * self.num_heads, seq_len, self.d_head)
        k_hyp = k_hyp.reshape(batch_size * self.num_heads, seq_len, self.d_head)
        v_hyp = v_hyp.reshape(batch_size * self.num_heads, seq_len, self.d_head)

        # 3. Apply attention mechanism
        # Note: The 'poincare_distance' is already expecting a 'heads' dimension,
        # so a more optimized version could avoid this reshape.
        # However, for clean modularity, we use the single-head module as is.
        output_hyperbolic, _ = self.attention(q_hyp, k_hyp, v_hyp)

        # 4. Concatenate heads and map back to tangent space
        # (batch * num_heads, seq_len, d_head) -> (batch, num_heads, seq_len, d_head)
        output_hyperbolic = output_hyperbolic.view(batch_size, self.num_heads, seq_len, self.d_head)
        # (batch, num_heads, seq_len, d_head) -> (batch, seq_len, num_heads, d_head) -> (batch, seq_len, d_model)
        output_hyperbolic_concat = output_hyperbolic.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        output_tangent = log_map_at_origin(output_hyperbolic_concat)

        # 5. Final linear projection
        final_output = self.W_o(output_tangent)

        return final_output
