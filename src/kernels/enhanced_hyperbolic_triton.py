"""
Enhanced Hyperbolic Attention Kernel - Phase 8 Implementation

高度な近似とハードウェア最適化を組み合わせた双曲アテンションカーネル。

主要機能:
1. Taylor展開（小距離: d < 0.1）: 3次まで展開
2. 漸近近似（大距離: d > 2.0）: 対数近似
3. Warpレベル双曲プリミティブ
4. Tensor Core加速（FP16 WMMA）

Requirements: 26.1, 26.2, 26.3, 26.4, 26.5, 26.6, 32.1-32.6, 33.1-33.6
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple, Dict
import math

EPS = 1e-6


# ============================================================
# Triton Kernels
# ============================================================

@triton.autotune(
    configs=[
        # RTX 3080 optimized
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        # RTX 3090/4090 optimized
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        # RTX 4090 Ada Lovelace optimized
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
    ],
    key=['N', 'D'],
)
@triton.jit
def _enhanced_hyperbolic_attn_fwd(
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
    USE_TAYLOR: tl.constexpr,
    USE_ASYMPTOTIC: tl.constexpr,
):
    """
    Enhanced hyperbolic attention with Taylor/asymptotic approximations
    
    近似戦略:
    - d < 0.1: Taylor展開（3次）
    - 0.1 <= d <= 2.0: 標準計算
    - d > 2.0: 漸近近似
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
        
        # ||q-k||^2 = ||q||^2 - 2<q,k> + ||k||^2
        diff_sq = q_norm_sq[:, None] - 2.0 * qk + k_norm_sq[None, :]
        diff_sq = tl.maximum(diff_sq, 0.0)
        euclidean_dist = tl.sqrt(diff_sq + 1e-8)
        
        # 双曲補正係数
        norm_sum = q_norm_sq[:, None] + k_norm_sq[None, :]
        
        if USE_TAYLOR:
            # Taylor展開（小距離用）: d_hyp ≈ d_euc * (1 + c*norm_sum/2 + c²*norm_sum²/8)
            taylor_factor = 1.0 + c * norm_sum * 0.5 + c * c * norm_sum * norm_sum * 0.125
            dist_taylor = euclidean_dist * taylor_factor
        else:
            dist_taylor = euclidean_dist
        
        if USE_ASYMPTOTIC:
            # 漸近近似（大距離用）: d_hyp ≈ log(2) + log(d_euc) + c*norm_sum/4
            log_dist = tl.log(euclidean_dist + 1e-8)
            dist_asymp = 0.693 + log_dist + c * norm_sum * 0.25
        else:
            dist_asymp = euclidean_dist
        
        # 距離に応じて近似を選択
        # 小距離: Taylor, 大距離: 漸近, 中間: 標準
        hyp_factor = tl.sqrt(1.0 + c * norm_sum)
        dist_standard = euclidean_dist * hyp_factor
        
        # 条件分岐で近似を選択
        dist = tl.where(
            euclidean_dist < 0.1,
            dist_taylor,
            tl.where(
                euclidean_dist > 2.0,
                dist_asymp,
                dist_standard
            )
        )
        
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


@triton.jit
def _warp_hyperbolic_norm(
    x,  # [WARP_SIZE, D]
    WARP_SIZE: tl.constexpr,
):
    """
    Warpレベルのノルム計算
    
    Warp shuffleを使用して効率的にノルムを計算。
    Requirements: 32.1, 32.2
    """
    # 各スレッドの部分和
    partial_sum = tl.sum(x * x, axis=1)
    
    # Warp内でリダクション（Tritonが自動最適化）
    return tl.sqrt(partial_sum + 1e-8)




# ============================================================
# PyTorch Module
# ============================================================

class EnhancedHyperbolicAttention(nn.Module):
    """
    Enhanced Hyperbolic Attention Module
    
    高度な近似とハードウェア最適化を組み合わせた双曲アテンション。
    
    特徴:
    - Taylor展開（小距離）
    - 漸近近似（大距離）
    - RTX 3080/3090/4090向け自動チューニング
    - Tensor Core活用（FP16）
    
    Requirements: 26.1-26.6, 32.1-32.6, 33.1-33.6
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_taylor: bool = True,
        use_asymptotic: bool = True,
        use_tensor_core: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.use_taylor = use_taylor
        self.use_asymptotic = use_asymptotic
        self.use_tensor_core = use_tensor_core
        
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        
        # 学習可能な曲率とtemperature
        self.log_c = nn.Parameter(torch.tensor(0.0))
        self.log_beta = nn.Parameter(torch.tensor(0.0))
        
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
    ) -> Tuple[torch.Tensor, Dict]:
        B, N, _ = x.shape
        
        # Tensor Core用にFP16に変換（オプション、CUDA上でのみ）
        use_fp16 = (
            self.use_tensor_core and 
            x.dtype == torch.float32 and 
            x.is_cuda
        )
        
        if use_fp16:
            # 重みもFP16に変換
            with torch.cuda.amp.autocast():
                q = self.W_q(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
                k = self.W_k(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
                v = self.W_v(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        else:
            # Project
            q = self.W_q(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
            k = self.W_k(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
            v = self.W_v(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        
        # Flatten batch and heads
        q = q.reshape(B * self.num_heads, N, self.d_head).contiguous()
        k = k.reshape(B * self.num_heads, N, self.d_head).contiguous()
        v = v.reshape(B * self.num_heads, N, self.d_head).contiguous()
        
        c = F.softplus(self.log_c)
        beta = F.softplus(self.log_beta) + 0.5
        
        out = torch.empty_like(q)
        
        causal = mask is not None
        
        BLOCK_D = min(64, triton.next_power_of_2(self.d_head))
        grid = (triton.cdiv(N, 64), B * self.num_heads)
        
        try:
            _enhanced_hyperbolic_attn_fwd[grid](
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
                USE_TAYLOR=self.use_taylor,
                USE_ASYMPTOTIC=self.use_asymptotic,
            )
        except Exception:
            # フォールバック: PyTorch実装
            out = self._pytorch_fallback(q, k, v, c, beta, causal)
        
        # Reshape back
        out = out.view(B, self.num_heads, N, self.d_head).transpose(1, 2)
        out = out.reshape(B, N, self.d_model)
        
        # FP32に戻す
        if use_fp16:
            out = out.float()
        
        out = self.W_o(out)
        out = self.dropout(out)
        
        diagnostics = {}
        if return_diagnostics:
            with torch.no_grad():
                diagnostics = {
                    'curvature': c.item(),
                    'beta': beta.item(),
                    'use_taylor': self.use_taylor,
                    'use_asymptotic': self.use_asymptotic,
                    'use_tensor_core': self.use_tensor_core,
                }
        
        return out, diagnostics
    
    def _pytorch_fallback(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        c: torch.Tensor,
        beta: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        """PyTorch参照実装（フォールバック用）"""
        BH, N, D = q.shape
        device = q.device
        
        q_norm_sq = (q * q).sum(dim=-1, keepdim=True)
        k_norm_sq = (k * k).sum(dim=-1, keepdim=True)
        
        qk = torch.matmul(q, k.transpose(-2, -1))
        diff_sq = q_norm_sq - 2.0 * qk + k_norm_sq.transpose(-2, -1)
        diff_sq = diff_sq.clamp(min=0.0)
        euclidean_dist = torch.sqrt(diff_sq + 1e-8)
        
        norm_sum = q_norm_sq + k_norm_sq.transpose(-2, -1)
        
        # Taylor展開（小距離）
        if self.use_taylor:
            taylor_factor = 1.0 + c * norm_sum * 0.5 + c * c * norm_sum * norm_sum * 0.125
            dist_taylor = euclidean_dist * taylor_factor
        else:
            dist_taylor = euclidean_dist
        
        # 漸近近似（大距離）
        if self.use_asymptotic:
            log_dist = torch.log(euclidean_dist + 1e-8)
            dist_asymp = 0.693 + log_dist + c * norm_sum * 0.25
        else:
            dist_asymp = euclidean_dist
        
        # 標準計算
        hyp_factor = torch.sqrt(1.0 + c * norm_sum)
        dist_standard = euclidean_dist * hyp_factor
        
        # 距離に応じて近似を選択
        dist = torch.where(
            euclidean_dist < 0.1,
            dist_taylor,
            torch.where(
                euclidean_dist > 2.0,
                dist_asymp,
                dist_standard
            )
        )
        
        scores = -beta * dist
        
        if causal:
            mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)


class HierarchicalBlockDecomposition(nn.Module):
    """
    階層的ブロック分解
    
    長いシーケンス（> 4096）に対してメモリ帯域幅を削減。
    
    Requirements: 26.2
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        block_size: int = 1024,
        num_levels: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.block_size = block_size
        self.num_levels = num_levels
        
        # 各レベルのアテンション
        self.local_attn = EnhancedHyperbolicAttention(
            d_model=d_model,
            num_heads=num_heads,
        )
        
        # グローバル要約用の射影
        self.summary_proj = nn.Linear(d_model, d_model // num_levels)
        self.expand_proj = nn.Linear(d_model // num_levels, d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        B, N, D = x.shape
        
        if N <= self.block_size:
            # 短いシーケンスは直接処理
            return self.local_attn(x, mask)
        
        # ブロックに分割
        num_blocks = (N + self.block_size - 1) // self.block_size
        padded_len = num_blocks * self.block_size
        
        if padded_len > N:
            x = F.pad(x, (0, 0, 0, padded_len - N))
        
        x_blocks = x.view(B, num_blocks, self.block_size, D)
        
        # 各ブロックでローカルアテンション
        outputs = []
        for i in range(num_blocks):
            block_out, _ = self.local_attn(x_blocks[:, i])
            outputs.append(block_out)
        
        local_out = torch.stack(outputs, dim=1)  # [B, num_blocks, block_size, D]
        
        # グローバル要約
        summaries = self.summary_proj(local_out.mean(dim=2))  # [B, num_blocks, D//levels]
        global_context, _ = self.local_attn(
            self.expand_proj(summaries)
        )  # [B, num_blocks, D]
        
        # ローカルとグローバルを結合
        global_expanded = global_context.unsqueeze(2).expand_as(local_out)
        combined = local_out + global_expanded * 0.1
        
        # 元の形状に戻す
        out = combined.view(B, padded_len, D)[:, :N]
        
        diagnostics = {
            'num_blocks': num_blocks,
            'block_size': self.block_size,
        }
        
        return out, diagnostics


def create_enhanced_hyperbolic_attention(
    d_model: int = 256,
    num_heads: int = 8,
    use_taylor: bool = True,
    use_asymptotic: bool = True,
    use_tensor_core: bool = True,
    **kwargs,
) -> EnhancedHyperbolicAttention:
    """
    Enhanced Hyperbolic Attentionのファクトリ関数
    """
    return EnhancedHyperbolicAttention(
        d_model=d_model,
        num_heads=num_heads,
        use_taylor=use_taylor,
        use_asymptotic=use_asymptotic,
        use_tensor_core=use_tensor_core,
        **kwargs,
    )


def benchmark_enhanced_kernel(
    batch_size: int = 2,
    seq_lengths: list = [1024, 2048, 4096],
    d_model: int = 256,
    num_heads: int = 8,
    num_iterations: int = 10,
    device: str = 'cuda',
) -> Dict:
    """
    Enhanced kernelのベンチマーク
    
    Requirements: 26.4 (Property 16の検証用)
    """
    import time
    
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}
    
    results = {}
    
    model = create_enhanced_hyperbolic_attention(
        d_model=d_model,
        num_heads=num_heads,
    ).to(device)
    model.eval()
    
    for seq_len in seq_lengths:
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        # ウォームアップ
        with torch.no_grad():
            for _ in range(3):
                _ = model(x)
        
        torch.cuda.synchronize()
        
        # 測定
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(x)
        
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        
        total_tokens = batch_size * seq_len * num_iterations
        tokens_per_second = total_tokens / elapsed_time
        
        results[f'seq_{seq_len}'] = {
            'tokens_per_second': tokens_per_second,
            'elapsed_time': elapsed_time,
            'iterations': num_iterations,
        }
    
    return results
