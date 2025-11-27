"""
Hyperbolic Attention - Ultra Fast Implementation

最速版: 双曲幾何の本質を維持しつつ、計算を極限まで簡略化

物理的直観:
- Poincaré距離の本質は「境界に近いほど距離が大きくなる」こと
- これを近似的に再現しつつ、acosh/atanh等の重い計算を回避
- 結果として、通常のAttentionに近い速度で双曲的な性質を実現

近似手法:
- 双曲距離 d(x,y) ≈ ||x-y|| * (1 + c*(||x||^2 + ||y||^2)/2)
- この近似は曲率cが小さい場合に特に有効
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple

EPS = 1e-5


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ],
    key=['N', 'D'],
)
@triton.jit
def _fast_hyperbolic_attn_fwd(
    Q, K, V,  # [B*H, N, D]
    C, BETA,  # scalars
    Out,  # [B*H, N, D]
    N, D,
    stride_qn, stride_qd,
    stride_kn, stride_kd,
    stride_vn, stride_vd,
    stride_on, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """
    Ultra-fast hyperbolic attention kernel
    
    近似双曲距離を使用して高速化:
    d_approx(q,k) = ||q-k|| * sqrt(1 + c*(||q||^2 + ||k||^2))
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    
    mask_m = offs_m < N
    mask_d = offs_d < D
    
    c = tl.load(C).to(tl.float32)
    beta = tl.load(BETA).to(tl.float32)
    
    # Base pointers
    q_base = Q + pid_bh * N * D
    k_base = K + pid_bh * N * D
    v_base = V + pid_bh * N * D
    o_base = Out + pid_bh * N * D
    
    # Load Q block
    q_ptrs = q_base + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    
    # Pre-compute q norms
    q_norm_sq = tl.sum(q * q, axis=1)  # [BLOCK_M]
    
    # Online softmax
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    if CAUSAL:
        end_n = min((pid_m + 1) * BLOCK_M, N)
        end_n = ((end_n + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    else:
        end_n = N
    
    for start_n in range(0, end_n, BLOCK_N):
        offs_n_curr = start_n + offs_n
        mask_n = offs_n_curr < N
        
        # Load K [D, BLOCK_N]
        k_ptrs = k_base + offs_n_curr[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
        
        k_norm_sq = tl.sum(k * k, axis=0)  # [BLOCK_N]
        
        # QK^T
        qk = tl.dot(q, k)  # [BLOCK_M, BLOCK_N]
        
        # 近似双曲距離:
        # ||q-k||^2 = ||q||^2 - 2<q,k> + ||k||^2
        diff_sq = q_norm_sq[:, None] - 2.0 * qk + k_norm_sq[None, :]
        diff_sq = tl.maximum(diff_sq, 0.0)
        
        # 双曲補正係数: sqrt(1 + c*(||q||^2 + ||k||^2))
        hyp_factor = tl.sqrt(1.0 + c * (q_norm_sq[:, None] + k_norm_sq[None, :]))
        
        # 近似距離
        dist = tl.sqrt(diff_sq + 1e-8) * hyp_factor
        
        # Scores
        scores = -beta * dist
        scores = tl.where(mask_n[None, :], scores, float('-inf'))
        
        if CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
            scores = tl.where(causal_mask, scores, float('-inf'))
        
        # Online softmax update
        m_i_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(scores - m_i_new[:, None])
        
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        # Load V [BLOCK_N, D]
        v_ptrs = v_base + offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_i_new
    
    # Normalize
    l_i = tl.maximum(l_i, 1e-8)
    acc = acc / l_i[:, None]
    
    # Store
    o_ptrs = o_base + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])


class FastHyperbolicAttention(nn.Module):
    """
    Ultra-fast Hyperbolic Attention Module
    
    通常のMultiHeadAttentionとほぼ同じ速度で、
    双曲幾何の本質的な性質（境界効果）を維持
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        
        # 学習可能な曲率とtemperature
        self.log_c = nn.Parameter(torch.tensor(0.0))  # c = softplus(log_c)
        self.log_beta = nn.Parameter(torch.tensor(0.0))  # beta = softplus(log_beta)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        B, N, _ = x.shape
        
        # Project
        q = self.W_q(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        
        # Flatten batch and heads
        q = q.reshape(B * self.num_heads, N, self.d_head).contiguous()
        k = k.reshape(B * self.num_heads, N, self.d_head).contiguous()
        v = v.reshape(B * self.num_heads, N, self.d_head).contiguous()
        
        c = F.softplus(self.log_c)
        beta = F.softplus(self.log_beta) + 0.5  # minimum beta of 0.5
        
        out = torch.empty_like(q)
        
        causal = mask is not None
        
        BLOCK_D = min(64, triton.next_power_of_2(self.d_head))
        grid = (triton.cdiv(N, 64), B * self.num_heads)
        
        _fast_hyperbolic_attn_fwd[grid](
            q, k, v,
            c, beta,
            out,
            N, self.d_head,
            q.stride(1), q.stride(2),
            k.stride(1), k.stride(2),
            v.stride(1), v.stride(2),
            out.stride(1), out.stride(2),
            BLOCK_D=BLOCK_D,
            CAUSAL=causal,
        )
        
        # Reshape back
        out = out.view(B, self.num_heads, N, self.d_head).transpose(1, 2)
        out = out.reshape(B, N, self.d_model)
        
        out = self.W_o(out)
        out = self.dropout(out)
        
        diagnostics = {}
        if return_diagnostics:
            with torch.no_grad():
                diagnostics = {
                    'curvature': c.item(),
                    'beta': beta.item(),
                }
        
        return out, diagnostics


class FastHyperbolicAttentionFunction(torch.autograd.Function):
    """
    Autograd対応のfast hyperbolic attention
    
    Forward: Tritonカーネル（高速）
    Backward: PyTorch参照実装（安定性重視）
    """
    
    @staticmethod
    def forward(ctx, q, k, v, c, beta, causal):
        B, H, N, D = q.shape
        
        q_flat = q.reshape(B * H, N, D).contiguous()
        k_flat = k.reshape(B * H, N, D).contiguous()
        v_flat = v.reshape(B * H, N, D).contiguous()
        
        out = torch.empty_like(q_flat)
        
        BLOCK_D = min(64, triton.next_power_of_2(D))
        grid = (triton.cdiv(N, 64), B * H)
        
        _fast_hyperbolic_attn_fwd[grid](
            q_flat, k_flat, v_flat,
            c, beta,
            out,
            N, D,
            q_flat.stride(1), q_flat.stride(2),
            k_flat.stride(1), k_flat.stride(2),
            v_flat.stride(1), v_flat.stride(2),
            out.stride(1), out.stride(2),
            BLOCK_D=BLOCK_D,
            CAUSAL=causal,
        )
        
        ctx.save_for_backward(q, k, v, c, beta)
        ctx.causal = causal
        
        return out.view(B, H, N, D)
    
    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, c, beta = ctx.saved_tensors
        causal = ctx.causal
        
        # PyTorch参照実装でbackward
        with torch.enable_grad():
            q_g = q.detach().requires_grad_(True)
            k_g = k.detach().requires_grad_(True)
            v_g = v.detach().requires_grad_(True)
            c_g = c.detach().requires_grad_(True)
            beta_g = beta.detach().requires_grad_(True)
            
            out = _fast_hyperbolic_pytorch(q_g, k_g, v_g, c_g, beta_g, causal)
            out.backward(grad_output)
        
        return q_g.grad, k_g.grad, v_g.grad, c_g.grad, beta_g.grad, None


def _fast_hyperbolic_pytorch(q, k, v, c, beta, causal):
    """PyTorch参照実装（backward用）"""
    B, H, N, D = q.shape
    device = q.device
    
    q_norm_sq = (q * q).sum(dim=-1, keepdim=True)
    k_norm_sq = (k * k).sum(dim=-1, keepdim=True)
    
    qk = torch.matmul(q, k.transpose(-2, -1))
    diff_sq = q_norm_sq - 2.0 * qk + k_norm_sq.transpose(-2, -1)
    diff_sq = diff_sq.clamp(min=0.0)
    
    hyp_factor = torch.sqrt(1.0 + c * (q_norm_sq + k_norm_sq.transpose(-2, -1)))
    dist = torch.sqrt(diff_sq + 1e-8) * hyp_factor
    
    scores = -beta * dist
    
    if causal:
        mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
    
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def fast_hyperbolic_attention(
    q: torch.Tensor,  # [B, H, N, D]
    k: torch.Tensor,
    v: torch.Tensor,
    c: torch.Tensor,
    beta: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """
    Functional interface for fast hyperbolic attention
    
    既存のhyperbolic_attention_tritonと互換性のあるインターフェース
    Autograd対応済み
    """
    return FastHyperbolicAttentionFunction.apply(q, k, v, c, beta, causal)
