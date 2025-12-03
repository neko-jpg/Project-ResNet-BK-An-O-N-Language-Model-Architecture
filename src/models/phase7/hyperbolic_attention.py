# src/models/phase7/hyperbolic_attention.py
# Implementation of Hyperbolic Attention Mechanisms

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.models.bitnet import BitNetLinear, LowRankLinear

# ##############################################################################
# # HYPERBOLIC UTILITY FUNCTIONS
# ##############################################################################

# Epsilon for numerical stability, widened for mixed precision
EPS = 1e-5

def poincare_distance(x, y, c, dim=-1):
    """
    Computes pairwise Poincaré distance with curvature c.
    d(x,y) = (1/sqrt(c)) * acosh(1 + 2*c*||x-y||^2 / ((1-c*||x||^2)(1-c*||y||^2)))
    """
    with torch.cuda.amp.autocast(enabled=False):
        x = x.float()
        y = y.float()
        c = c.float()

        sqrt_c = torch.sqrt(c)
        x_norm_sq = (x * x).sum(dim=dim, keepdim=True)
        y_norm_sq = (y * y).sum(dim=dim, keepdim=True)

        y_t = y.transpose(-2, -1)
        xy_dot = torch.matmul(x, y_t)

        x_norm_sq_broadcast = x_norm_sq.expand(-1, -1, -1, y.shape[-2])
        y_norm_sq_broadcast = y_norm_sq.transpose(-2, -1).expand(-1, -1, x.shape[-2], -1)
        diff_norm_sq = x_norm_sq_broadcast - 2 * xy_dot + y_norm_sq_broadcast

        # Denominator term: (1 - c*||x_i||^2)(1 - c*||y_j||^2)
        denom = (1 - c * x_norm_sq_broadcast) * (1 - c * y_norm_sq_broadcast)

        # Argument of acosh
        numerator = 2 * c * diff_norm_sq
        arg = 1 + numerator / denom.clamp_min(EPS)

        dist = (1. / sqrt_c) * torch.acosh(arg.clamp_min(1.0 + EPS))

    return dist.to(x.dtype)


def exp_map_at_origin(v, c, dim=-1):
    """
    Exponential map with curvature c.
    Formula: (1/sqrt(c)) * tanh(sqrt(c) * ||v||) * (v / ||v||)
    """
    with torch.cuda.amp.autocast(enabled=False):
        v = v.float()
        c = c.float()
        sqrt_c = torch.sqrt(c)
        v_norm = v.norm(dim=dim, keepdim=True).clamp_min(EPS)

        mapped_v = (1. / sqrt_c) * F.tanh(sqrt_c * v_norm) * (v / v_norm)

    return mapped_v.to(v.dtype)

def log_map_at_origin(y, c, dim=-1):
    """
    Logarithmic map with curvature c.
    Formula: (1/sqrt(c)) * atanh(sqrt(c) * ||y||) * (y / ||y||)
    """
    with torch.cuda.amp.autocast(enabled=False):
        y = y.float()
        c = c.float()
        sqrt_c = torch.sqrt(c)
        y_norm = y.norm(dim=dim, keepdim=True).clamp_min(EPS)

        # Clamp arg of atanh to be < 1.0
        # sqrt(c) * ||y|| < 1  => ||y|| < 1/sqrt(c)
        max_norm = (1. / sqrt_c) - EPS
        y_norm_clamped = y_norm.clamp_max(max_norm)

        arg = sqrt_c * y_norm_clamped

        result = (1. / sqrt_c) * torch.atanh(arg) * (y / y_norm)

    return result.to(y.dtype)


# ##############################################################################
# # SINGLE HEAD HYPERBOLIC ATTENTION
# ##############################################################################

class SingleHeadHyperbolicAttention(nn.Module):
    """
    Implements a single head of hyperbolic attention.
    Assumes q, k are in the Poincaré ball, and v_tangent is in the tangent space.
    """
    def __init__(self, d_model: int):
        super().__init__()
        # Learnable scale parameter, initialized near 1.0
        # Using softplus to ensure beta is always positive
        self.beta = nn.Parameter(torch.tensor(0.0)) # softplus(0) is approx 0.69

        # Learnable bias parameter
        self.attention_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, q, k, v_tangent, c, mask=None):
        """
        Args:
            q (torch.Tensor): Queries in the Poincaré ball. Shape: (batch, seq_len, dim)
            k (torch.Tensor): Keys in the Poincaré ball. Shape: (batch, seq_len, dim)
            v_tangent (torch.Tensor): Values in the tangent space. Shape: (batch, seq_len, dim)
            c (torch.Tensor): Curvature tensor.
            mask (torch.Tensor, optional): Attention mask. Shape: (batch, seq_len, seq_len).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, dict]: Aggregated value, attention weights, and diagnostics.
        """
        # 1. Compute pairwise hyperbolic distances
        dist_qk_unsqueezed = poincare_distance(q.unsqueeze(1), k.unsqueeze(1), c=c)
        dist_qk = dist_qk_unsqueezed.squeeze(1)

        # 2. Calculate attention scores
        beta_positive = F.softplus(self.beta)
        attention_scores = -beta_positive * dist_qk - self.attention_bias

        # Apply causal mask
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -torch.finfo(attention_scores.dtype).max)

        attention_weights = F.softmax(attention_scores, dim=-1)

        # 3. Aggregate values, which are already in the tangent space
        output_tangent = torch.bmm(attention_weights, v_tangent)
        output_hyperbolic = exp_map_at_origin(output_tangent, c=c)

        # 4. Create diagnostics
        with torch.no_grad():
            diagnostics = {
                'hyperbolic_dist_mean': dist_qk.mean(),
                'hyperbolic_dist_max': dist_qk.max(),
                'hyperbolic_dist_min': dist_qk.min(),
            }

        return output_hyperbolic, attention_weights, diagnostics


# ##############################################################################
# # MULTI-HEAD HYPERBOLIC ATTENTION
# ##############################################################################

class HyperbolicMultiHeadAttention(nn.Module):
    """
    Implements multi-head hyperbolic attention.
    
    カーネル選択:
    - 'fast': 最速版（近似双曲距離、通常Attentionに近い速度）
    - 'v2': 最適化版（正確な双曲距離、事前計算で高速化）
    - 'v1': 従来版（完全な双曲距離計算）
    """
    def __init__(self, d_model: int, num_heads: int, use_triton_kernel: bool = True, kernel_version: str = 'fast',
                 use_bitnet: bool = False, low_rank_attention: bool = False, low_rank_rank: int = 64):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.use_triton_kernel = use_triton_kernel
        self.kernel_version = kernel_version
        self.use_bitnet = use_bitnet
        self.low_rank_attention = low_rank_attention
        self.low_rank_rank = low_rank_rank

        self.triton_kernel_function = None
        if self.use_triton_kernel:
            try:
                if kernel_version == 'fast':
                    # 最速版: 近似双曲距離
                    from src.kernels.hyperbolic_attention_fast import fast_hyperbolic_attention
                    self.triton_kernel_function = fast_hyperbolic_attention
                elif kernel_version == 'v2':
                    # 最適化版: 事前計算 + autotune
                    from src.kernels.hyperbolic_attention_triton_v2 import hyperbolic_attention_triton_v2
                    self.triton_kernel_function = hyperbolic_attention_triton_v2
                else:
                    # 従来版
                    from src.kernels.hyperbolic_attention_triton import hyperbolic_attention_triton
                    self.triton_kernel_function = hyperbolic_attention_triton
            except (ImportError, ModuleNotFoundError) as e:
                import warnings
                warnings.warn(
                    f"Triton kernel '{kernel_version}' could not be imported: {e}. "
                    "Falling back to PyTorch implementation."
                )
                self.triton_kernel_function = None

        # Linear projections for Q, K, V. These project from Euclidean to the tangent space.
        if self.low_rank_attention:
            self.W_q = LowRankLinear(d_model, d_model, self.low_rank_rank, bias=False, use_bitnet=self.use_bitnet)
            self.W_k = LowRankLinear(d_model, d_model, self.low_rank_rank, bias=False, use_bitnet=self.use_bitnet)
            self.W_v = LowRankLinear(d_model, d_model, self.low_rank_rank, bias=False, use_bitnet=self.use_bitnet)
            self.W_o = LowRankLinear(d_model, d_model, self.low_rank_rank, bias=False, use_bitnet=self.use_bitnet)
        elif self.use_bitnet:
            self.W_q = BitNetLinear(d_model, d_model, bias=False)
            self.W_k = BitNetLinear(d_model, d_model, bias=False)
            self.W_v = BitNetLinear(d_model, d_model, bias=False)
            self.W_o = BitNetLinear(d_model, d_model, bias=False)
        else:
            self.W_q = nn.Linear(d_model, d_model, bias=False)
            self.W_k = nn.Linear(d_model, d_model, bias=False)
            self.W_v = nn.Linear(d_model, d_model, bias=False)
            self.W_o = nn.Linear(d_model, d_model, bias=False)

        # The core hyperbolic attention mechanism
        # We can reuse the single-head implementation across all heads.
        self.attention = SingleHeadHyperbolicAttention(self.d_head)

        # Learnable curvature parameter, must be positive.
        self.log_c = nn.Parameter(torch.tensor(0.0))

        # Final output projection
        # self.W_o is initialized above

        self._init_weights()

    def _init_weights(self):
        from src.utils.prime_init import prime_bump_init_
        
        def init_layer(layer):
            if isinstance(layer, LowRankLinear):
                prime_bump_init_(layer.U.weight)
                prime_bump_init_(layer.V.weight)
            elif hasattr(layer, 'weight'):
                prime_bump_init_(layer.weight)
                
        init_layer(self.W_q)
        init_layer(self.W_k)
        init_layer(self.W_v)
        init_layer(self.W_o)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, return_diagnostics: bool = True):
        """
        Args:
            x (torch.Tensor): Input tensor in Euclidean space. Shape: (batch, seq_len, d_model)
            mask (torch.Tensor, optional): Attention mask. Shape: (batch, 1, 1, seq_len).
            return_diagnostics (bool): If True, returns a dictionary of monitoring metrics.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, dict]:
            - Final output tensor in Euclidean space.
            - (Optional) Dictionary with diagnostic metrics.
        """
        original_dtype = x.dtype
        batch_size, seq_len, _ = x.shape

        # --- Start of Numerical Stability Section ---
        # All hyperbolic computations are wrapped in autocast(enabled=False) to enforce float32
        # and prevent numerical instability issues common in mixed-precision training.
        with torch.cuda.amp.autocast(enabled=False):
            x_f32 = x.float()

            c = F.softplus(self.log_c.float())

            q_tangent = self.W_q(x_f32)
            k_tangent = self.W_k(x_f32)
            v_tangent = self.W_v(x_f32)

            q_tangent = q_tangent.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
            k_tangent = k_tangent.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
            v_tangent = v_tangent.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

            # マスクは Triton カーネルにも対応させるため bool に正規化しておく
            mask_bool = None
            if mask is not None:
                mask_bool = mask.to(device=x.device, dtype=torch.bool)
                if mask_bool.dim() == 4:
                    if mask_bool.shape[0] == 1 and batch_size != 1:
                        mask_bool = mask_bool.expand(batch_size, -1, seq_len, seq_len)
                else:
                    # 非対応形状は明示的に失敗させる
                    raise ValueError(f"Unsupported mask shape {mask.shape}; expected (B, H|1, N, N)")

            # Determine if we can use the Triton kernel
            can_use_triton = self.use_triton_kernel and hasattr(self, 'triton_kernel_function') and self.triton_kernel_function is not None

            if can_use_triton:
                # New optimized Triton kernel interface
                # Input: q, k, v in tangent space [B, H, N, D]
                # The kernel handles exp_map internally
                beta_pos = F.softplus(self.attention.beta.float())
                causal = mask_bool is not None  # Use causal masking if mask provided
                
                output_hyperbolic = self.triton_kernel_function(
                    q_tangent,
                    k_tangent,
                    v_tangent,
                    c,
                    beta_pos,
                    causal,
                )

                output_tangent_heads = log_map_at_origin(output_hyperbolic, c=c)
                output_tangent_concat = output_tangent_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
                final_output = self.W_o(output_tangent_concat)

                # Only compute diagnostics if requested (expensive operation)
                head_diagnostics = {}
                if return_diagnostics:
                    with torch.no_grad():
                        # Sample only first 8 positions for efficiency
                        q_sample = q_tangent[:, :, :8, :]
                        k_sample = k_tangent[:, :, :8, :]
                        q_hyp_sample = exp_map_at_origin(q_sample, c=c)
                        k_hyp_sample = exp_map_at_origin(k_sample, c=c)
                        dist_sample = poincare_distance(q_hyp_sample, k_hyp_sample, c=c)
                        head_diagnostics = {
                            'hyperbolic_dist_mean': dist_sample.mean(),
                            'hyperbolic_dist_max': dist_sample.max(),
                            'hyperbolic_dist_min': dist_sample.min(),
                        }
                        self.triton_q_norms = torch.norm(q_hyp_sample, dim=-1)
                        self.triton_k_norms = torch.norm(k_hyp_sample, dim=-1)

                q_hyp = None  # Signal that diagnostics are handled separately
            else:
                # PyTorch implementation
                q_hyp = exp_map_at_origin(q_tangent, c=c)
                k_hyp = exp_map_at_origin(k_tangent, c=c)
                # v_tangent remains in the tangent space, no exp_map needed

                q_hyp_flat = q_hyp.reshape(batch_size * self.num_heads, seq_len, self.d_head)
                k_hyp_flat = k_hyp.reshape(batch_size * self.num_heads, seq_len, self.d_head)
                v_tangent_flat = v_tangent.reshape(batch_size * self.num_heads, seq_len, self.d_head)

                # Reshape mask for multi-head attention
                if mask_bool is not None:
                    # Original mask shape: (B, 1, N, N)
                    # Expand for heads and reshape for flattened batch
                    mask_flat = mask_bool.expand(batch_size, self.num_heads, seq_len, seq_len).reshape(batch_size * self.num_heads, seq_len, seq_len)
                else:
                    mask_flat = None

                output_hyperbolic_flat, _, head_diagnostics = self.attention(q_hyp_flat, k_hyp_flat, v_tangent_flat, c=c, mask=mask_flat)

                output_hyperbolic = output_hyperbolic_flat.view(batch_size, self.num_heads, seq_len, self.d_head)
                output_tangent_heads = log_map_at_origin(output_hyperbolic, c=c)
                output_tangent_concat = output_tangent_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
                final_output = self.W_o(output_tangent_concat)

            final_output = final_output.to(original_dtype)
        # --- End of Numerical Stability Section ---

        if return_diagnostics:
            with torch.no_grad():
                diagnostics = { 'curvature': c.item() }
                if q_hyp is not None:
                    # PyTorch path: compute norms from q_hyp
                    q_norms = torch.norm(q_hyp, dim=-1)
                    k_norms = torch.norm(k_hyp, dim=-1)
                    diagnostics.update({
                        'boundary_collapse_q_mean_norm': q_norms.mean(),
                        'boundary_collapse_k_mean_norm': k_norms.mean(),
                        'boundary_collapse_q_max_norm': q_norms.max(),
                        'boundary_collapse_k_max_norm': k_norms.max(),
                    })
                elif hasattr(self, 'triton_q_norms'):
                    # Triton path: use pre-computed norms
                    diagnostics.update({
                        'boundary_collapse_q_mean_norm': self.triton_q_norms.mean(),
                        'boundary_collapse_k_mean_norm': self.triton_k_norms.mean(),
                        'boundary_collapse_q_max_norm': self.triton_q_norms.max(),
                        'boundary_collapse_k_max_norm': self.triton_k_norms.max(),
                    })

                diagnostics.update(head_diagnostics)
            return final_output, diagnostics
        else:
            return final_output
