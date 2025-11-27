"""
Hyperbolic Attention Triton Kernel - Optimized Implementation

Flash Attentionスタイルのタイルベース実装で、双曲幾何学的アテンションを高速化。
Poincaré球モデルでの距離計算とexp/log写像を効率的に実装。

物理的直観:
- Poincaré球は負の曲率を持つ双曲空間のモデル
- 距離は測地線（最短経路）に沿って測定
- exp_map: 接空間から多様体への写像
- log_map: 多様体から接空間への写像
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple, Optional

# 数値安定性のための定数
EPS = 1e-6
MAX_TANH_ARG = 15.0  # tanh(15) ≈ 1.0


@triton.jit
def _hyperbolic_fwd_kernel(
    # Input pointers
    Q, K, V,  # [B, H, N, D]
    C,  # curvature scalar
    BETA,  # temperature scalar
    # Output pointers
    Out,  # [B, H, N, D]
    L,  # [B, H, N] - logsumexp for backward
    # Dimensions
    N, D, H,
    # Strides for Q
    stride_qb, stride_qh, stride_qn, stride_qd,
    # Strides for K
    stride_kb, stride_kh, stride_kn, stride_kd,
    # Strides for V
    stride_vb, stride_vh, stride_vn, stride_vd,
    # Strides for Out
    stride_ob, stride_oh, stride_on, stride_od,
    # Strides for L
    stride_lb, stride_lh, stride_ln,
    # Block sizes and constants
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EPS: tl.constexpr,
    MAX_TANH: tl.constexpr,
):
    """
    Forward kernel for Hyperbolic Attention.
    
    Uses online softmax (Flash Attention style) for memory efficiency.
    Computes Poincaré distance between query and key points.
    """
    # Program IDs
    pid_m = tl.program_id(0)  # Which block of queries
    pid_bh = tl.program_id(1)  # Which batch-head
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Masks
    mask_m = offs_m < N
    mask_d = offs_d < D
    
    # Load curvature and beta
    c = tl.load(C).to(tl.float32)
    c = tl.maximum(c, EPS)
    sqrt_c = tl.sqrt(c)
    beta = tl.load(BETA).to(tl.float32)
    
    # Pointers for Q block [BLOCK_M, D]
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + \
             offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    
    # Load Q block
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    
    # Compute q_hyp = exp_map(q) with numerical stability
    q_norm_sq = tl.sum(q * q, axis=1)  # [BLOCK_M]
    q_norm = tl.sqrt(q_norm_sq + EPS)
    tanh_arg = tl.minimum(sqrt_c * q_norm, MAX_TANH)
    q_scale = tl.math.tanh(tanh_arg) / (sqrt_c * q_norm + EPS)
    q_hyp = q * q_scale[:, None]  # [BLOCK_M, D]
    q_hyp_norm_sq = tl.sum(q_hyp * q_hyp, axis=1)  # [BLOCK_M]
    
    # Initialize accumulators for online softmax
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)  # max scores
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # sum of exp
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)  # output accumulator
    
    # Iterate over K, V blocks
    for start_n in range(0, N, BLOCK_N):
        offs_n_curr = start_n + offs_n
        mask_n = offs_n_curr < N
        
        # Load K block [D, BLOCK_N] (transposed for efficient matmul)
        k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + \
                 offs_n_curr[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
        
        # Compute k_hyp = exp_map(k)
        k_norm_sq = tl.sum(k * k, axis=0)  # [BLOCK_N]
        k_norm = tl.sqrt(k_norm_sq + EPS)
        tanh_arg_k = tl.minimum(sqrt_c * k_norm, MAX_TANH)
        k_scale = tl.math.tanh(tanh_arg_k) / (sqrt_c * k_norm + EPS)
        k_hyp = k * k_scale[None, :]  # [D, BLOCK_N]
        k_hyp_norm_sq = tl.sum(k_hyp * k_hyp, axis=0)  # [BLOCK_N]
        
        # Compute Poincaré distance: d(q, k)
        # Formula: d = (1/sqrt(c)) * acosh(1 + 2c * ||q-k||^2 / ((1-c||q||^2)(1-c||k||^2)))
        qk_dot = tl.dot(q_hyp, k_hyp)  # [BLOCK_M, BLOCK_N]
        diff_norm_sq = q_hyp_norm_sq[:, None] - 2.0 * qk_dot + k_hyp_norm_sq[None, :]
        diff_norm_sq = tl.maximum(diff_norm_sq, 0.0)
        
        denom = (1.0 - c * q_hyp_norm_sq[:, None]) * (1.0 - c * k_hyp_norm_sq[None, :])
        denom = tl.maximum(denom, EPS)
        
        arg = 1.0 + 2.0 * c * diff_norm_sq / denom
        arg = tl.maximum(arg, 1.0 + EPS)
        
        # acosh(x) = log(x + sqrt(x^2 - 1))
        dist = (1.0 / sqrt_c) * tl.log(arg + tl.sqrt(arg * arg - 1.0 + EPS))
        
        # Compute attention scores: -beta * dist
        scores = -beta * dist  # [BLOCK_M, BLOCK_N]
        scores = tl.where(mask_n[None, :], scores, float('-inf'))
        
        # Causal mask (optional - can be controlled by parameter)
        causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
        scores = tl.where(causal_mask, scores, float('-inf'))
        
        # Online softmax update
        m_i_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(scores - m_i_new[:, None])
        
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        # Load V block [BLOCK_N, D]
        v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + \
                 offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        
        # Compute log_map(v) for tangent space projection
        v_norm_sq = tl.sum(v * v, axis=1)  # [BLOCK_N]
        v_norm = tl.sqrt(v_norm_sq + EPS)
        v_norm_clamped = tl.minimum(v_norm, (1.0 / sqrt_c) - EPS)
        v_arg = sqrt_c * v_norm_clamped
        v_arg = tl.minimum(v_arg, 0.99)  # Prevent log(0)
        # atanh(x) = 0.5 * log((1+x)/(1-x))
        atanh_v = 0.5 * tl.log((1.0 + v_arg + EPS) / (1.0 - v_arg + EPS))
        v_tangent = v * (atanh_v / (sqrt_c * v_norm + EPS))[:, None]  # [BLOCK_N, D]
        
        # Accumulate: acc += p @ v_tangent
        acc += tl.dot(p.to(v_tangent.dtype), v_tangent)
        m_i = m_i_new
    
    # Normalize by sum of exp
    l_i = tl.maximum(l_i, EPS)
    acc = acc / l_i[:, None]
    
    # Apply exp_map to output (back to hyperbolic space)
    acc_norm_sq = tl.sum(acc * acc, axis=1)
    acc_norm = tl.sqrt(acc_norm_sq + EPS)
    tanh_arg_out = tl.minimum(sqrt_c * acc_norm, MAX_TANH)
    out_scale = tl.math.tanh(tanh_arg_out) / (sqrt_c * acc_norm + EPS)
    out = acc * out_scale[:, None]
    
    # Store outputs
    out_ptrs = Out + pid_b * stride_ob + pid_h * stride_oh + \
               offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, out.to(Out.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])
    
    # Store L for backward pass
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ln
    tl.store(l_ptrs, l_i, mask=mask_m)


class HyperbolicAttentionTriton(torch.autograd.Function):
    """
    Triton-accelerated Hyperbolic Attention with autograd support.
    
    Forward: Triton kernel
    Backward: PyTorch reference (for correctness, can be optimized later)
    """
    
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,  # [B, H, N, D]
        k: torch.Tensor,
        v: torch.Tensor,
        c: torch.Tensor,  # curvature scalar
        beta: torch.Tensor,  # temperature scalar
        causal: bool = True,
    ) -> torch.Tensor:
        """Forward pass using Triton kernel."""
        B, H, N, D = q.shape
        
        # Ensure contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        # Allocate outputs
        out = torch.empty_like(q)
        L = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
        
        # Block sizes (tuned for common GPU architectures)
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_D = min(64, triton.next_power_of_2(D))
        
        # Grid
        grid = (triton.cdiv(N, BLOCK_M), B * H)
        
        # Launch kernel
        _hyperbolic_fwd_kernel[grid](
            q, k, v, c, beta,
            out, L,
            N, D, H,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            EPS=1e-6,
            MAX_TANH=15.0,
        )
        
        # Save for backward
        ctx.save_for_backward(q, k, v, c, beta, out, L)
        ctx.causal = causal
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass using PyTorch for correctness."""
        q, k, v, c, beta, out, L = ctx.saved_tensors
        
        # Use PyTorch autograd for backward (reliable gradients)
        # Re-compute forward with autograd enabled
        with torch.enable_grad():
            q_grad = q.detach().requires_grad_(True)
            k_grad = k.detach().requires_grad_(True)
            v_grad = v.detach().requires_grad_(True)
            c_grad = c.detach().requires_grad_(True)
            beta_grad = beta.detach().requires_grad_(True)
            
            # PyTorch reference forward
            out_ref = _hyperbolic_attention_pytorch(
                q_grad, k_grad, v_grad, c_grad, beta_grad, ctx.causal
            )
            
            # Backward
            out_ref.backward(grad_output)
        
        return (
            q_grad.grad,
            k_grad.grad,
            v_grad.grad,
            c_grad.grad,
            beta_grad.grad,
            None,  # causal
        )


def _hyperbolic_attention_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    c: torch.Tensor,
    beta: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """
    PyTorch reference implementation for backward pass.
    
    物理的直観:
    - Poincaré球での距離に基づくアテンション
    - 近い点ほど高いアテンション重み
    """
    B, H, N, D = q.shape
    device = q.device
    dtype = q.dtype
    
    # Work in float32 for numerical stability
    q = q.float()
    k = k.float()
    v = v.float()
    c = c.float()
    beta = beta.float()
    
    sqrt_c = torch.sqrt(c.clamp(min=EPS))
    
    # exp_map for Q and K
    def exp_map(x):
        norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
        tanh_arg = (sqrt_c * norm).clamp(max=MAX_TANH_ARG)
        return x * (torch.tanh(tanh_arg) / (sqrt_c * norm))
    
    # log_map for V
    def log_map(x):
        norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
        norm_clamped = norm.clamp(max=(1.0 / sqrt_c) - EPS)
        arg = (sqrt_c * norm_clamped).clamp(max=0.99)
        atanh = 0.5 * torch.log((1.0 + arg + EPS) / (1.0 - arg + EPS))
        return x * (atanh / (sqrt_c * norm))
    
    q_hyp = exp_map(q)
    k_hyp = exp_map(k)
    v_tangent = log_map(v)
    
    # Poincaré distance
    q_norm_sq = (q_hyp * q_hyp).sum(dim=-1, keepdim=True)
    k_norm_sq = (k_hyp * k_hyp).sum(dim=-1, keepdim=True)
    
    qk_dot = torch.matmul(q_hyp, k_hyp.transpose(-2, -1))
    diff_norm_sq = q_norm_sq - 2.0 * qk_dot + k_norm_sq.transpose(-2, -1)
    diff_norm_sq = diff_norm_sq.clamp(min=0.0)
    
    denom = (1.0 - c * q_norm_sq) * (1.0 - c * k_norm_sq.transpose(-2, -1))
    denom = denom.clamp(min=EPS)
    
    arg = 1.0 + 2.0 * c * diff_norm_sq / denom
    arg = arg.clamp(min=1.0 + EPS)
    
    dist = (1.0 / sqrt_c) * torch.acosh(arg)
    
    # Attention scores
    scores = -beta * dist
    
    # Causal mask
    if causal:
        mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
    
    # Softmax
    attn = F.softmax(scores, dim=-1)
    
    # Weighted sum in tangent space
    acc = torch.matmul(attn, v_tangent)
    
    # exp_map back to hyperbolic space
    out = exp_map(acc)
    
    return out.to(dtype)


def hyperbolic_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    c: torch.Tensor,
    beta: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """
    Hyperbolic Attention with Triton acceleration.
    
    Args:
        q: Query tensor [B, H, N, D]
        k: Key tensor [B, H, N, D]
        v: Value tensor [B, H, N, D]
        c: Curvature (positive scalar)
        beta: Temperature (positive scalar)
        causal: Whether to apply causal mask
    
    Returns:
        Output tensor [B, H, N, D]
    """
    return HyperbolicAttentionTriton.apply(q, k, v, c, beta, causal)
