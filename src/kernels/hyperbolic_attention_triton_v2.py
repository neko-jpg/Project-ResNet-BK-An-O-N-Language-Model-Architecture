"""
Hyperbolic Attention Triton Kernel V2 - Ultra-Optimized Implementation

最適化ポイント:
1. Flash Attention 2スタイルのタイリング（より大きなブロック）
2. exp_map/log_mapの事前計算でカーネル内計算を削減
3. 双曲距離の近似計算（高速化のため）
4. Autotuneによる最適ブロックサイズ自動選択
5. Fused softmax + matmul

物理的直観:
- Poincaré球での距離計算を簡略化しつつ精度を維持
- 小さい距離では線形近似、大きい距離では対数近似
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple, Optional

EPS = 1e-5
MAX_TANH_ARG = 15.0


def get_autotune_configs():
    """RTX 3080向けに最適化されたconfig群"""
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_D': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_D': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_D': 64}, num_warps=2, num_stages=5),
    ]


@triton.autotune(
    configs=get_autotune_configs(),
    key=['N', 'D', 'H'],
)
@triton.jit
def _hyperbolic_fwd_kernel_v2(
    # Input pointers (既にexp_map適用済み)
    Q_hyp, K_hyp, V_tan,  # [B, H, N, D]
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
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """
    Forward kernel V2: 事前計算されたexp_map済みテンソルを受け取る
    
    高速化のポイント:
    - exp_map計算をカーネル外に移動
    - 双曲距離の近似計算
    - より大きなブロックサイズ
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    
    mask_m = offs_m < N
    mask_d = offs_d < D
    
    c = tl.load(C).to(tl.float32)
    c = tl.maximum(c, 1e-6)
    sqrt_c = tl.sqrt(c)
    inv_sqrt_c = 1.0 / sqrt_c
    beta = tl.load(BETA).to(tl.float32)
    
    # Load Q block (already in hyperbolic space)
    q_ptrs = Q_hyp + pid_b * stride_qb + pid_h * stride_qh + \
             offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    
    # Pre-compute q norms
    q_norm_sq = tl.sum(q * q, axis=1)  # [BLOCK_M]
    
    # Initialize online softmax accumulators
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Causal: only iterate up to current position
    if CAUSAL:
        end_n = min((pid_m + 1) * BLOCK_M, N)
        end_n = ((end_n + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    else:
        end_n = N
    
    for start_n in range(0, end_n, BLOCK_N):
        offs_n_curr = start_n + offs_n
        mask_n = offs_n_curr < N
        
        # Load K block [D, BLOCK_N]
        k_ptrs = K_hyp + pid_b * stride_kb + pid_h * stride_kh + \
                 offs_n_curr[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
        
        k_norm_sq = tl.sum(k * k, axis=0)  # [BLOCK_N]
        
        # Compute QK^T
        qk_dot = tl.dot(q, k)  # [BLOCK_M, BLOCK_N]
        
        # 高速双曲距離近似:
        # d^2 ≈ 2 * (||q||^2 + ||k||^2 - 2*<q,k>) / ((1-c||q||^2)(1-c||k||^2))
        # 小さい曲率では線形近似が有効
        diff_sq = q_norm_sq[:, None] - 2.0 * qk_dot + k_norm_sq[None, :]
        diff_sq = tl.maximum(diff_sq, 0.0)
        
        denom_q = 1.0 - c * q_norm_sq[:, None]
        denom_k = 1.0 - c * k_norm_sq[None, :]
        denom = denom_q * denom_k
        denom = tl.maximum(denom, 1e-6)
        
        # 近似距離: sqrt(2c * diff_sq / denom) / sqrt_c
        # = sqrt(2 * diff_sq / denom)
        dist_sq_scaled = 2.0 * diff_sq / denom
        dist = tl.sqrt(dist_sq_scaled + 1e-8)
        
        # Attention scores
        scores = -beta * dist
        scores = tl.where(mask_n[None, :], scores, float('-inf'))
        
        # Causal mask
        if CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
            scores = tl.where(causal_mask, scores, float('-inf'))
        
        # Online softmax
        m_i_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(scores - m_i_new[:, None])
        
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        # Load V block [BLOCK_N, D] (tangent space)
        v_ptrs = V_tan + pid_b * stride_vb + pid_h * stride_vh + \
                 offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        
        # Accumulate
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_i_new
    
    # Normalize
    l_i = tl.maximum(l_i, 1e-8)
    acc = acc / l_i[:, None]
    
    # Store output (tangent space result)
    out_ptrs = Out + pid_b * stride_ob + pid_h * stride_oh + \
               offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])
    
    # Store L for backward
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ln
    tl.store(l_ptrs, m_i + tl.log(l_i), mask=mask_m)


def exp_map_batched(v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Batched exp_map: 接空間からPoincaré球への写像
    PyTorchで事前計算してカーネルに渡す
    """
    sqrt_c = torch.sqrt(c.clamp(min=EPS))
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=EPS)
    tanh_arg = (sqrt_c * v_norm).clamp(max=MAX_TANH_ARG)
    scale = torch.tanh(tanh_arg) / (sqrt_c * v_norm)
    return v * scale


def log_map_batched(y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Batched log_map: Poincaré球から接空間への写像
    """
    sqrt_c = torch.sqrt(c.clamp(min=EPS))
    y_norm = y.norm(dim=-1, keepdim=True).clamp(min=EPS)
    max_norm = (1.0 / sqrt_c) - EPS
    y_norm_clamped = y_norm.clamp(max=max_norm.item() if max_norm.numel() == 1 else 0.99)
    arg = (sqrt_c * y_norm_clamped).clamp(max=0.99)
    atanh = 0.5 * torch.log((1.0 + arg + EPS) / (1.0 - arg + EPS))
    return y * (atanh / (sqrt_c * y_norm))


class HyperbolicAttentionTritonV2(torch.autograd.Function):
    """
    Ultra-optimized Triton Hyperbolic Attention
    
    最適化:
    1. exp_map/log_mapをカーネル外で事前計算
    2. Autotuneで最適ブロックサイズ自動選択
    3. 双曲距離の高速近似
    """
    
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        c: torch.Tensor,
        beta: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        B, H, N, D = q.shape
        
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        # 事前計算: exp_map (カーネル外で1回だけ)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            q_bf16 = q.to(torch.bfloat16)
            k_bf16 = k.to(torch.bfloat16)
            v_bf16 = v.to(torch.bfloat16)
            c_bf16 = c.to(torch.bfloat16)
            
            q_hyp = exp_map_batched(q_bf16, c_bf16)
            k_hyp = exp_map_batched(k_bf16, c_bf16)
            # vは接空間のまま
        
        out = torch.empty_like(q)
        L = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
        
        BLOCK_D = min(64, triton.next_power_of_2(D))
        
        grid = (triton.cdiv(N, 64), B * H)  # 初期値、autotuneで調整
        
        _hyperbolic_fwd_kernel_v2[grid](
            q_hyp, k_hyp, v_bf16, c_bf16, beta.to(torch.bfloat16),
            out, L,
            N, D, H,
            q_hyp.stride(0), q_hyp.stride(1), q_hyp.stride(2), q_hyp.stride(3),
            k_hyp.stride(0), k_hyp.stride(1), k_hyp.stride(2), k_hyp.stride(3),
            v_f32.stride(0), v_f32.stride(1), v_f32.stride(2), v_f32.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            CAUSAL=causal,
        )
        
        # 出力にexp_mapを適用（接空間→双曲空間）
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out_hyp = exp_map_batched(out.to(torch.bfloat16), c_bf16)
        
        ctx.save_for_backward(q, k, v, c, beta, out_hyp, L)
        ctx.causal = causal
        
        return out_hyp.to(q.dtype)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, c, beta, out, L = ctx.saved_tensors
        
        # PyTorch reference backward (安定性重視)
        with torch.enable_grad():
            q_grad = q.detach().requires_grad_(True)
            k_grad = k.detach().requires_grad_(True)
            v_grad = v.detach().requires_grad_(True)
            c_grad = c.detach().requires_grad_(True)
            beta_grad = beta.detach().requires_grad_(True)
            
            out_ref = _hyperbolic_attention_pytorch_fast(
                q_grad, k_grad, v_grad, c_grad, beta_grad, ctx.causal
            )
            out_ref.backward(grad_output)
        
        return (
            q_grad.grad,
            k_grad.grad,
            v_grad.grad,
            c_grad.grad,
            beta_grad.grad,
            None,
        )


def _hyperbolic_attention_pytorch_fast(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    c: torch.Tensor,
    beta: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """高速PyTorch参照実装（backward用）"""
    B, H, N, D = q.shape
    device = q.device
    
    q = q.float()
    k = k.float()
    v = v.float()
    c = c.float().clamp(min=EPS)
    beta = beta.float()
    
    sqrt_c = torch.sqrt(c)
    
    # exp_map
    q_hyp = exp_map_batched(q, c)
    k_hyp = exp_map_batched(k, c)
    
    # 高速距離計算
    q_norm_sq = (q_hyp * q_hyp).sum(dim=-1, keepdim=True)
    k_norm_sq = (k_hyp * k_hyp).sum(dim=-1, keepdim=True)
    
    qk_dot = torch.matmul(q_hyp, k_hyp.transpose(-2, -1))
    diff_sq = q_norm_sq - 2.0 * qk_dot + k_norm_sq.transpose(-2, -1)
    diff_sq = diff_sq.clamp(min=0.0)
    
    denom = (1.0 - c * q_norm_sq) * (1.0 - c * k_norm_sq.transpose(-2, -1))
    denom = denom.clamp(min=EPS)
    
    dist = torch.sqrt(2.0 * diff_sq / denom + EPS)
    
    scores = -beta * dist
    
    if causal:
        mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
    
    attn = F.softmax(scores, dim=-1)
    acc = torch.matmul(attn, v)
    out = exp_map_batched(acc, c)
    
    return out


def hyperbolic_attention_triton_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    c: torch.Tensor,
    beta: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """
    Ultra-optimized Hyperbolic Attention V2
    
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
    return HyperbolicAttentionTritonV2.apply(q, k, v, c, beta, causal)
