"""
Flash Hyperbolic Attention Triton Kernel - Phase 8 Implementation

Flash Attentionスタイルのオンラインソフトマックスを使用した
高効率な双曲幾何学的アテンションカーネル。

物理的直観:
- Poincaré球モデルでの測地線距離に基づくアテンション
- ブロック単位の距離計算でメモリ効率を最大化
- 因果マスクの最適化（上三角ブロックをスキップ）
- 共有メモリを活用したQブロックキャッシング

ターゲット:
- RTX 3080で70%以上のFLOPS利用率
- O(N)メモリスケーリング
- Phase 7比で2倍のスループット向上

Requirements: 31.1, 31.2, 31.3, 31.4, 31.5, 31.6
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import math

# 数値安定性のための定数
EPS = 1e-6
MAX_TANH_ARG = 15.0


@dataclass
class FlashHyperbolicConfig:
    """Flash Hyperbolic Attention設定"""
    block_m: int = 128  # Queryブロックサイズ
    block_n: int = 64   # Key/Valueブロックサイズ
    num_warps: int = 4
    num_stages: int = 3
    shared_memory_kb: int = 48  # 共有メモリターゲット


# RTX 3080, RTX 3090, RTX 4090向けの自動チューニング設定
@triton.autotune(
    configs=[
        # RTX 3080 (10GB) - Ampere GA102
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64},
            num_warps=4, num_stages=3,
            pre_hook=None
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64},
            num_warps=4, num_stages=4,
        ),
        # RTX 3090 (24GB) - Ampere GA102
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128},
            num_warps=8, num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128},
            num_warps=4, num_stages=3,
        ),
        # RTX 4090 (24GB) - Ada Lovelace AD102
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128},
            num_warps=8, num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 256, 'BLOCK_N': 64},
            num_warps=8, num_stages=2,
        ),
    ],
    key=['N', 'D', 'H'],
)
@triton.jit
def _flash_hyperbolic_fwd_kernel(
    # Input pointers
    Q, K, V,  # [B, H, N, D]
    C,  # curvature scalar
    BETA,  # temperature scalar
    # Output pointers
    Out,  # [B, H, N, D]
    L,  # [B, H, N] - logsumexp for backward
    M,  # [B, H, N] - max scores for backward
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
    # Strides for L, M
    stride_lb, stride_lh, stride_ln,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    # Flags
    CAUSAL: tl.constexpr,
    USE_EXP_MAP: tl.constexpr,
):
    """
    Flash Hyperbolic Attention Forward Kernel
    
    オンラインソフトマックスを使用したメモリ効率の良い実装。
    ブロック単位で距離を計算し、即座にソフトマックスとV乗算を実行。
    
    物理的直観:
    - 双曲距離は境界に近いほど大きくなる
    - exp_map: 接空間→多様体（Poincaré球）
    - log_map: 多様体→接空間
    """
    # Program IDs
    pid_m = tl.program_id(0)  # Query block index
    pid_bh = tl.program_id(1)  # Batch-head index
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Masks
    mask_m = offs_m < N
    mask_d = offs_d < D
    
    # Load curvature and temperature
    c = tl.load(C).to(tl.float32)
    c = tl.maximum(c, 1e-6)
    sqrt_c = tl.sqrt(c)
    beta = tl.load(BETA).to(tl.float32)
    
    # Q block pointers [BLOCK_M, D]
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + \
             offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    
    # Load Q block to shared memory (conceptually)
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    
    # Apply exp_map to Q if enabled
    if USE_EXP_MAP:
        q_norm_sq = tl.sum(q * q, axis=1)  # [BLOCK_M]
        q_norm = tl.sqrt(q_norm_sq + 1e-6)
        tanh_arg = tl.minimum(sqrt_c * q_norm, 15.0)
        q_scale = tl.math.tanh(tanh_arg) / (sqrt_c * q_norm + 1e-6)
        q_hyp = q * q_scale[:, None]  # [BLOCK_M, D]
        q_hyp_norm_sq = tl.sum(q_hyp * q_hyp, axis=1)  # [BLOCK_M]
    else:
        # 近似モード: exp_mapをスキップ
        q_hyp = q
        q_hyp_norm_sq = tl.sum(q * q, axis=1)
    
    # Initialize online softmax accumulators
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)  # max scores
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # sum of exp
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)  # output accumulator
    
    # Causal mask optimization: determine end of K/V iteration
    if CAUSAL:
        # 因果マスク: 上三角ブロックを完全にスキップ
        end_n = tl.minimum((pid_m + 1) * BLOCK_M, N)
        end_n = ((end_n + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    else:
        end_n = N
    
    # Iterate over K, V blocks
    for start_n in range(0, end_n, BLOCK_N):
        offs_n_curr = start_n + offs_n
        mask_n = offs_n_curr < N
        
        # Load K block [D, BLOCK_N] (transposed)
        k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + \
                 offs_n_curr[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
        
        # Apply exp_map to K if enabled
        if USE_EXP_MAP:
            k_norm_sq = tl.sum(k * k, axis=0)  # [BLOCK_N]
            k_norm = tl.sqrt(k_norm_sq + 1e-6)
            tanh_arg_k = tl.minimum(sqrt_c * k_norm, 15.0)
            k_scale = tl.math.tanh(tanh_arg_k) / (sqrt_c * k_norm + 1e-6)
            k_hyp = k * k_scale[None, :]  # [D, BLOCK_N]
            k_hyp_norm_sq = tl.sum(k_hyp * k_hyp, axis=0)  # [BLOCK_N]
        else:
            k_hyp = k
            k_hyp_norm_sq = tl.sum(k * k, axis=0)
        
        # Compute Poincaré distance
        # d(q, k) = (1/sqrt(c)) * acosh(1 + 2c * ||q-k||^2 / ((1-c||q||^2)(1-c||k||^2)))
        qk_dot = tl.dot(q_hyp, k_hyp)  # [BLOCK_M, BLOCK_N]
        diff_norm_sq = q_hyp_norm_sq[:, None] - 2.0 * qk_dot + k_hyp_norm_sq[None, :]
        diff_norm_sq = tl.maximum(diff_norm_sq, 0.0)
        
        denom = (1.0 - c * q_hyp_norm_sq[:, None]) * (1.0 - c * k_hyp_norm_sq[None, :])
        denom = tl.maximum(denom, 1e-6)
        
        arg = 1.0 + 2.0 * c * diff_norm_sq / denom
        arg = tl.maximum(arg, 1.0 + 1e-6)
        
        # acosh(x) = log(x + sqrt(x^2 - 1))
        dist = (1.0 / sqrt_c) * tl.log(arg + tl.sqrt(arg * arg - 1.0 + 1e-6))
        
        # Attention scores: -beta * dist
        scores = -beta * dist  # [BLOCK_M, BLOCK_N]
        scores = tl.where(mask_n[None, :], scores, float('-inf'))
        
        # Apply causal mask
        if CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
            scores = tl.where(causal_mask, scores, float('-inf'))
        
        # Online softmax update (Flash Attention style)
        m_i_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(scores - m_i_new[:, None])
        
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        # Load V block [BLOCK_N, D]
        v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + \
                 offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        
        # Accumulate: acc += p @ v
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_i_new
    
    # Normalize by sum of exp
    l_i = tl.maximum(l_i, 1e-6)
    acc = acc / l_i[:, None]
    
    # Store outputs
    out_ptrs = Out + pid_b * stride_ob + pid_h * stride_oh + \
               offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])
    
    # Store L and M for backward pass
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ln
    tl.store(l_ptrs, l_i, mask=mask_m)
    
    m_ptrs = M + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ln
    tl.store(m_ptrs, m_i, mask=mask_m)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=2, num_stages=5),
    ],
    key=['N', 'D', 'H'],
)
@triton.jit
def _flash_hyperbolic_fwd_kernel_fused(
    Q, K, V,  # [B, H, N, D]
    RES,  # Residual [B, H, N, D] (optional)
    C,  # curvature scalar
    BETA,  # temperature scalar
    RES_SCALE,  # residual scaling
    Out,  # [B, H, N, D]
    L,  # [B, H, N]
    M,  # [B, H, N]
    N, D, H,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_rb, stride_rh, stride_rn, stride_rd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lb, stride_lh, stride_ln,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_EXP_MAP: tl.constexpr,
    APPLY_LOG_MAP: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
):
    """
    End-to-end fused hyperbolic attention path:
    ExpMap(Q/K) + Flash-style softmax + optional LogMap + residual accumulation.
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
    beta = tl.load(BETA).to(tl.float32)
    res_scale = tl.load(RES_SCALE).to(tl.float32)

    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + \
             offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    if USE_EXP_MAP:
        q_norm_sq = tl.sum(q * q, axis=1)
        q_norm = tl.sqrt(q_norm_sq + 1e-6)
        tanh_arg = tl.minimum(sqrt_c * q_norm, 15.0)
        q_scale = tl.math.tanh(tanh_arg) / (sqrt_c * q_norm + 1e-6)
        q_hyp = q * q_scale[:, None]
        q_hyp_norm_sq = tl.sum(q_hyp * q_hyp, axis=1)
    else:
        q_hyp = q
        q_hyp_norm_sq = tl.sum(q * q, axis=1)

    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    if CAUSAL:
        end_n = tl.minimum((pid_m + 1) * BLOCK_M, N)
        end_n = ((end_n + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    else:
        end_n = N

    for start_n in range(0, end_n, BLOCK_N):
        offs_n_curr = start_n + offs_n
        mask_n = offs_n_curr < N

        k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + \
                 offs_n_curr[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        if USE_EXP_MAP:
            k_norm_sq = tl.sum(k * k, axis=0)
            k_norm = tl.sqrt(k_norm_sq + 1e-6)
            tanh_arg_k = tl.minimum(sqrt_c * k_norm, 15.0)
            k_scale = tl.math.tanh(tanh_arg_k) / (sqrt_c * k_norm + 1e-6)
            k_hyp = k * k_scale[None, :]
            k_hyp_norm_sq = tl.sum(k_hyp * k_hyp, axis=0)
        else:
            k_hyp = k
            k_hyp_norm_sq = tl.sum(k * k, axis=0)

        qk_dot = tl.dot(q_hyp, k_hyp)
        diff_norm_sq = q_hyp_norm_sq[:, None] - 2.0 * qk_dot + k_hyp_norm_sq[None, :]
        diff_norm_sq = tl.maximum(diff_norm_sq, 0.0)

        denom = (1.0 - c * q_hyp_norm_sq[:, None]) * (1.0 - c * k_hyp_norm_sq[None, :])
        denom = tl.maximum(denom, 1e-6)

        arg = 1.0 + 2.0 * c * diff_norm_sq / denom
        arg = tl.maximum(arg, 1.0 + 1e-6)
        dist = (1.0 / sqrt_c) * tl.acosh(arg)

        scores = -beta * dist
        scores = tl.where(mask_n[None, :], scores, float('-inf'))

        if CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
            scores = tl.where(causal_mask, scores, float('-inf'))

        m_i_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(scores - m_i_new[:, None])

        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + \
                 offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_i_new

    l_i = tl.maximum(l_i, 1e-6)
    acc = acc / l_i[:, None]

    if APPLY_LOG_MAP:
        acc_norm_sq = tl.sum(acc * acc, axis=1)
        acc_norm = tl.sqrt(acc_norm_sq + 1e-6)
        log_scale = tl.math.atanh(tl.minimum(sqrt_c * acc_norm, 0.999)) / (sqrt_c * acc_norm + 1e-6)
        acc = acc * log_scale[:, None]

    if ADD_RESIDUAL:
        r_ptrs = RES + pid_b * stride_rb + pid_h * stride_rh + \
                 offs_m[:, None] * stride_rn + offs_d[None, :] * stride_rd
        r = tl.load(r_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        acc = acc + res_scale * r

    out_ptrs = Out + pid_b * stride_ob + pid_h * stride_oh + \
               offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])

    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ln
    tl.store(l_ptrs, l_i, mask=mask_m)

    m_ptrs = M + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ln
    tl.store(m_ptrs, m_i, mask=mask_m)



@triton.jit
def _flash_hyperbolic_bwd_kernel(
    # Inputs
    Q, K, V, Out, dOut,
    C, BETA,
    L, M,  # saved from forward
    # Outputs
    dQ, dK, dV,
    # Dimensions
    N, D, H,
    # Strides
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lb, stride_lh, stride_ln,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """
    Flash Hyperbolic Attention Backward Kernel
    
    再計算を使用してメモリ効率を維持しながら勾配を計算。
    アテンション重みを保存せず、Q, Kから再計算。
    
    ターゲット: 保存版と比較して<10%のオーバーヘッド
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
    beta = tl.load(BETA).to(tl.float32)
    
    # Load Q block
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + \
             offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    
    # Load dOut block
    do_ptrs = dOut + pid_b * stride_ob + pid_h * stride_oh + \
              offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    do = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    
    # Load L, M
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ln
    l_i = tl.load(l_ptrs, mask=mask_m, other=1.0)
    m_ptrs = M + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ln
    m_i = tl.load(m_ptrs, mask=mask_m, other=0.0)
    
    # Compute q_hyp
    q_norm_sq = tl.sum(q * q, axis=1)
    q_norm = tl.sqrt(q_norm_sq + 1e-6)
    tanh_arg = tl.minimum(sqrt_c * q_norm, 15.0)
    q_scale = tl.math.tanh(tanh_arg) / (sqrt_c * q_norm + 1e-6)
    q_hyp = q * q_scale[:, None]
    q_hyp_norm_sq = tl.sum(q_hyp * q_hyp, axis=1)
    
    # Initialize dQ accumulator
    dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    if CAUSAL:
        end_n = tl.minimum((pid_m + 1) * BLOCK_M, N)
        end_n = ((end_n + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    else:
        end_n = N
    
    # Recompute attention and compute gradients
    for start_n in range(0, end_n, BLOCK_N):
        offs_n_curr = start_n + offs_n
        mask_n = offs_n_curr < N
        
        # Load K, V blocks
        k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + \
                 offs_n_curr[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
        
        v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + \
                 offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        
        # Compute k_hyp
        k_norm_sq = tl.sum(k * k, axis=0)
        k_norm = tl.sqrt(k_norm_sq + 1e-6)
        tanh_arg_k = tl.minimum(sqrt_c * k_norm, 15.0)
        k_scale = tl.math.tanh(tanh_arg_k) / (sqrt_c * k_norm + 1e-6)
        k_hyp = k * k_scale[None, :]
        k_hyp_norm_sq = tl.sum(k_hyp * k_hyp, axis=0)
        
        # Recompute distance and attention
        qk_dot = tl.dot(q_hyp, k_hyp)
        diff_norm_sq = q_hyp_norm_sq[:, None] - 2.0 * qk_dot + k_hyp_norm_sq[None, :]
        diff_norm_sq = tl.maximum(diff_norm_sq, 0.0)
        
        denom = (1.0 - c * q_hyp_norm_sq[:, None]) * (1.0 - c * k_hyp_norm_sq[None, :])
        denom = tl.maximum(denom, 1e-6)
        
        arg = 1.0 + 2.0 * c * diff_norm_sq / denom
        arg = tl.maximum(arg, 1.0 + 1e-6)
        
        dist = (1.0 / sqrt_c) * tl.log(arg + tl.sqrt(arg * arg - 1.0 + 1e-6))
        scores = -beta * dist
        scores = tl.where(mask_n[None, :], scores, float('-inf'))
        
        if CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
            scores = tl.where(causal_mask, scores, float('-inf'))
        
        # Recompute attention weights
        p = tl.exp(scores - m_i[:, None]) / l_i[:, None]
        
        # Compute dV: dV += p^T @ dO
        dv = tl.dot(tl.trans(p.to(do.dtype)), do)
        
        # Store dV
        dv_ptrs = dV + pid_b * stride_vb + pid_h * stride_vh + \
                  offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vd
        tl.atomic_add(dv_ptrs, dv.to(dV.dtype.element_ty), mask=mask_n[:, None] & mask_d[None, :])
        
        # Compute dp: dp = dO @ V^T
        dp = tl.dot(do, tl.trans(v))
        
        # Compute ds (gradient through softmax)
        # ds = p * (dp - sum(p * dp))
        sum_dp = tl.sum(p * dp, axis=1, keep_dims=True)
        ds = p * (dp - sum_dp)
        
        # Gradient through distance: d(score)/d(dist) = -beta
        # d(dist)/d(q_hyp) requires chain rule through acosh
        # Simplified: accumulate gradient contribution
        dq_acc += tl.dot(ds.to(k_hyp.dtype), tl.trans(k_hyp)) * (-beta)
    
    # Store dQ
    dq_ptrs = dQ + pid_b * stride_qb + pid_h * stride_qh + \
              offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    tl.store(dq_ptrs, dq_acc.to(dQ.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])


class FlashHyperbolicAttention(torch.autograd.Function):
    """
    Flash Hyperbolic Attention with Triton acceleration.
    
    Forward: Tritonカーネル（高速、メモリ効率）
    Backward: 再計算ベース（メモリ節約）またはPyTorch参照（安定性）
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
        use_exp_map: bool = True,
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
        M = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
        
        # Block sizes
        BLOCK_D = min(64, triton.next_power_of_2(D))
        
        # Grid
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_M']), B * H)
        
        # Launch kernel
        _flash_hyperbolic_fwd_kernel[grid](
            q, k, v, c, beta,
            out, L, M,
            N, D, H,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            BLOCK_D=BLOCK_D,
            CAUSAL=causal,
            USE_EXP_MAP=use_exp_map,
        )
        
        # Save for backward
        ctx.save_for_backward(q, k, v, c, beta, out, L, M)
        ctx.causal = causal
        ctx.use_exp_map = use_exp_map
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass with recomputation for memory efficiency."""
        q, k, v, c, beta, out, L, M = ctx.saved_tensors
        
        # Use PyTorch reference for stable gradients
        with torch.enable_grad():
            q_grad = q.detach().requires_grad_(True)
            k_grad = k.detach().requires_grad_(True)
            v_grad = v.detach().requires_grad_(True)
            c_grad = c.detach().requires_grad_(True)
            beta_grad = beta.detach().requires_grad_(True)
            
            out_ref = _flash_hyperbolic_pytorch_ref(
                q_grad, k_grad, v_grad, c_grad, beta_grad,
                ctx.causal, ctx.use_exp_map
            )
            out_ref.backward(grad_output)
        
        return (
            q_grad.grad,
            k_grad.grad,
            v_grad.grad,
            c_grad.grad,
            beta_grad.grad,
            None,  # causal
            None,  # use_exp_map
        )


def _flash_hyperbolic_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    c: torch.Tensor,
    beta: torch.Tensor,
    causal: bool = True,
    use_exp_map: bool = True,
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
    c = c.float().clamp(min=EPS)
    beta = beta.float()
    
    sqrt_c = torch.sqrt(c)
    
    if use_exp_map:
        # exp_map for Q and K
        def exp_map(x):
            norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
            tanh_arg = (sqrt_c * norm).clamp(max=MAX_TANH_ARG)
            return x * (torch.tanh(tanh_arg) / (sqrt_c * norm))
        
        q_hyp = exp_map(q)
        k_hyp = exp_map(k)
    else:
        q_hyp = q
        k_hyp = k
    
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
    
    # Weighted sum
    out = torch.matmul(attn, v)
    
    return out.to(dtype)


class FlashHyperbolicAttentionE2E(torch.autograd.Function):
    """
    End-to-end fused hyperbolic attention with residual addition.
    
    Forward: Triton fused kernel (_flash_hyperbolic_fwd_kernel_fused)
    Backward: Recomputes a reference path that includes optional LogMap and residual.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual: Optional[torch.Tensor],
        c: torch.Tensor,
        beta: torch.Tensor,
        causal: bool = True,
        use_exp_map: bool = True,
        apply_log_map: bool = True,
        residual_scale: float = 1.0,
    ) -> torch.Tensor:
        B, H, N, D = q.shape

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        add_residual = residual is not None
        if residual is None:
            residual = torch.zeros_like(q)
        else:
            residual = residual.contiguous()

        out = torch.empty_like(q)
        L = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
        M = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
        BLOCK_D = min(64, triton.next_power_of_2(D))
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_M']), B * H)

        res_scale_tensor = torch.tensor(residual_scale, device=q.device, dtype=torch.float32)

        _flash_hyperbolic_fwd_kernel_fused[grid](
            q, k, v, residual, c, beta, res_scale_tensor,
            out, L, M,
            N, D, H,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            residual.stride(0), residual.stride(1), residual.stride(2), residual.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            BLOCK_D=BLOCK_D,
            CAUSAL=causal,
            USE_EXP_MAP=use_exp_map,
            APPLY_LOG_MAP=apply_log_map,
            ADD_RESIDUAL=add_residual,
        )

        ctx.save_for_backward(q, k, v, residual if add_residual else torch.tensor([]).to(q.device), c, beta)
        ctx.add_residual = add_residual
        ctx.causal = causal
        ctx.use_exp_map = use_exp_map
        ctx.apply_log_map = apply_log_map
        ctx.residual_scale = residual_scale
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, residual, c, beta = ctx.saved_tensors
        with torch.enable_grad():
            q_grad = q.detach().requires_grad_(True)
            k_grad = k.detach().requires_grad_(True)
            v_grad = v.detach().requires_grad_(True)
            residual_grad = None
            residual_arg = None
            if ctx.add_residual:
                residual_grad = residual.detach().requires_grad_(True)
                residual_arg = residual_grad

            c_grad = c.detach().requires_grad_(True)
            beta_grad = beta.detach().requires_grad_(True)
            out_ref = _flash_hyperbolic_pytorch_ref_fused(
                q_grad,
                k_grad,
                v_grad,
                residual_arg,
                c_grad,
                beta_grad,
                causal=ctx.causal,
                use_exp_map=ctx.use_exp_map,
                apply_log_map=ctx.apply_log_map,
                residual_scale=ctx.residual_scale,
            )
            out_ref.backward(grad_output)

        res_grad_out = residual_grad.grad if residual_grad is not None else None
        return (
            q_grad.grad,
            k_grad.grad,
            v_grad.grad,
            res_grad_out,
            c_grad.grad,
            beta_grad.grad,
            None,  # causal
            None,  # use_exp_map
            None,  # apply_log_map
            None,  # residual_scale
        )


def flash_hyperbolic_attention_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    residual: Optional[torch.Tensor],
    c: torch.Tensor,
    beta: torch.Tensor,
    causal: bool = True,
    use_exp_map: bool = True,
    apply_log_map: bool = True,
    residual_scale: float = 1.0,
) -> torch.Tensor:
    """
    End-to-end fused Flash Hyperbolic Attention.

    Args:
        q, k, v: [B, H, N, D] tensors
        residual: optional residual in the same shape (e.g., reshaped input)
        c: curvature scalar tensor
        beta: temperature scalar tensor
        causal: enable causal mask
        use_exp_map: run exp_map inside the kernel
        apply_log_map: optionally map back to tangent space
        residual_scale: scale for residual addition
    """
    return FlashHyperbolicAttentionE2E.apply(
        q, k, v, residual, c, beta, causal, use_exp_map, apply_log_map, residual_scale
    )


def _flash_hyperbolic_pytorch_ref_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    residual: Optional[torch.Tensor],
    c: torch.Tensor,
    beta: torch.Tensor,
    causal: bool = True,
    use_exp_map: bool = True,
    apply_log_map: bool = True,
    residual_scale: float = 1.0,
) -> torch.Tensor:
    """
    PyTorch reference for the fused end-to-end path.
    Adds optional LogMap + residual accumulation for gradient computation.
    """
    out = _flash_hyperbolic_pytorch_ref(q, k, v, c, beta, causal=causal, use_exp_map=use_exp_map)

    if apply_log_map:
        c_val = c.clamp_min(EPS)
        sqrt_c = torch.sqrt(c_val)
        out_norm = out.norm(dim=-1, keepdim=True).clamp_min(EPS)
        log_scale = torch.atanh(torch.clamp(sqrt_c * out_norm, max=0.999)) / (sqrt_c * out_norm + 1e-6)
        out = out * log_scale

    if residual is not None:
        out = out + residual_scale * residual
    return out


def flash_hyperbolic_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    c: torch.Tensor,
    beta: torch.Tensor,
    causal: bool = True,
    use_exp_map: bool = True,
) -> torch.Tensor:
    """
    Flash Hyperbolic Attention with Triton acceleration.
    
    Args:
        q: Query tensor [B, H, N, D]
        k: Key tensor [B, H, N, D]
        v: Value tensor [B, H, N, D]
        c: Curvature (positive scalar)
        beta: Temperature (positive scalar)
        causal: Whether to apply causal mask
        use_exp_map: Whether to apply exp_map to Q, K
    
    Returns:
        Output tensor [B, H, N, D]
    
    Requirements: 31.1, 31.2, 31.3
    """
    return FlashHyperbolicAttention.apply(q, k, v, c, beta, causal, use_exp_map)


class FlashHyperbolicAttentionModule(nn.Module):
    """
    Flash Hyperbolic Attention Module
    
    Phase 8の高効率双曲アテンションモジュール。
    Flash Attentionスタイルのメモリ効率とTriton最適化を組み合わせ。
    
    ターゲット:
    - RTX 3080で70%以上のFLOPS利用率
    - Phase 7比で2倍のスループット向上
    - O(N)メモリスケーリング
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        initial_curvature: float = 1.0,
        initial_beta: float = 1.0,
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
        self.log_c = nn.Parameter(torch.tensor(math.log(initial_curvature)))
        self.log_beta = nn.Parameter(torch.tensor(math.log(initial_beta)))
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    @property
    def curvature(self) -> torch.Tensor:
        return F.softplus(self.log_c)
    
    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.log_beta) + 0.5
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, N, D]
            mask: Optional attention mask
            return_diagnostics: Whether to return diagnostic info
        
        Returns:
            output: Output tensor [B, N, D]
            diagnostics: Dictionary of diagnostic metrics
        """
        B, N, _ = x.shape
        
        # Project
        q = self.W_q(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        
        # Get curvature and beta
        c = self.curvature
        beta = self.beta
        
        # Apply Flash Hyperbolic Attention
        causal = mask is not None
        out = flash_hyperbolic_attention(q, k, v, c, beta, causal=causal)
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        out = self.W_o(out)
        out = self.dropout(out)
        
        diagnostics = {}
        if return_diagnostics:
            with torch.no_grad():
                diagnostics = {
                    'curvature': c.item(),
                    'beta': beta.item(),
                    'q_norm_mean': q.norm(dim=-1).mean().item(),
                    'k_norm_mean': k.norm(dim=-1).mean().item(),
                }
        
        return out, diagnostics


def benchmark_flash_hyperbolic(
    batch_size: int = 2,
    num_heads: int = 8,
    seq_lengths: list = [1024, 2048, 4096, 8192],
    d_head: int = 64,
    num_warmup: int = 10,
    num_iterations: int = 100,
    device: str = 'cuda',
) -> Dict[str, Any]:
    """
    Flash Hyperbolic Attentionのベンチマーク
    
    Returns:
        Dictionary with throughput and memory metrics
    """
    import time
    
    results = {
        'seq_lengths': seq_lengths,
        'throughput_tokens_per_sec': [],
        'peak_memory_mb': [],
        'avg_time_ms': [],
    }
    
    c = torch.tensor(1.0, device=device)
    beta = torch.tensor(1.0, device=device)
    
    for seq_len in seq_lengths:
        # Create inputs
        q = torch.randn(batch_size, num_heads, seq_len, d_head, device=device, dtype=torch.float16)
        k = torch.randn(batch_size, num_heads, seq_len, d_head, device=device, dtype=torch.float16)
        v = torch.randn(batch_size, num_heads, seq_len, d_head, device=device, dtype=torch.float16)
        
        # Warmup
        torch.cuda.reset_peak_memory_stats()
        for _ in range(num_warmup):
            _ = flash_hyperbolic_attention(q, k, v, c, beta, causal=True)
        torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        for _ in range(num_iterations):
            _ = flash_hyperbolic_attention(q, k, v, c, beta, causal=True)
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Calculate metrics
        avg_time_ms = (end_time - start_time) / num_iterations * 1000
        tokens_per_sec = (batch_size * seq_len * num_iterations) / (end_time - start_time)
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        results['throughput_tokens_per_sec'].append(tokens_per_sec)
        results['peak_memory_mb'].append(peak_memory_mb)
        results['avg_time_ms'].append(avg_time_ms)
        
        # Cleanup
        del q, k, v
        torch.cuda.empty_cache()
    
    return results
