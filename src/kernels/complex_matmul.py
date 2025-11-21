"""
Complex Matrix Multiplication Kernel (Triton) for Phase 3

このモジュールは、複素行列積 (A + iB)(C + iD) = (AC - BD) + i(AD + BC) を
Tritonを用いて高速化します。

Requirements:
    - Requirement 8.1: Triton Kernel Implementation
    - Requirement 8.2: Python Wrapper & Fallback
    - Requirement 8.3: Benchmark
"""

import torch
import warnings
from typing import Tuple

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    # ダミーのデコレータ定義
    class MockTriton:
        def jit(self, func):
            return func
    triton = MockTriton()
    tl = None

# ========================================
# Triton Kernel
# ========================================

@triton.jit
def complex_matmul_kernel(
    # Pointers to matrices
    a_real_ptr, a_imag_ptr,
    b_real_ptr, b_imag_ptr,
    c_real_ptr, c_imag_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    複素行列積カーネル: C = A * B

    A = Ar + iAi
    B = Br + iBi
    C = (ArBr - AiBi) + i(ArBi + AiBr)
    """
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Pointers
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # A pointers
    a_real_ptrs = a_real_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    a_imag_ptrs = a_imag_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)

    # B pointers
    b_real_ptrs = b_real_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    b_imag_ptrs = b_imag_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Accumulators
    acc_real = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_imag = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load A
        a_real = tl.load(a_real_ptrs)
        a_imag = tl.load(a_imag_ptrs)

        # Load B
        b_real = tl.load(b_real_ptrs)
        b_imag = tl.load(b_imag_ptrs)

        # Complex Multiplication
        # Real part: ArBr - AiBi
        acc_real += tl.dot(a_real, b_real)
        acc_real -= tl.dot(a_imag, b_imag)

        # Imag part: ArBi + AiBr
        acc_imag += tl.dot(a_real, b_imag)
        acc_imag += tl.dot(a_imag, b_real)

        # Advance pointers
        a_real_ptrs += BLOCK_SIZE_K * stride_ak
        a_imag_ptrs += BLOCK_SIZE_K * stride_ak
        b_real_ptrs += BLOCK_SIZE_K * stride_bk
        b_imag_ptrs += BLOCK_SIZE_K * stride_bk

    # Store result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    c_real_ptrs = c_real_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_imag_ptrs = c_imag_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # Write back (cast to half if needed, but usually input/output is same dtype)
    tl.store(c_real_ptrs, acc_real, mask=c_mask)
    tl.store(c_imag_ptrs, acc_imag, mask=c_mask)


# ========================================
# Python Wrapper
# ========================================

def complex_matmul(
    a_real: torch.Tensor, a_imag: torch.Tensor,
    b_real: torch.Tensor, b_imag: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    複素行列積の計算（Triton最適化またはフォールバック）

    Args:
        a_real, a_imag: 行列Aの実部と虚部 (M, K)
        b_real, b_imag: 行列Bの実部と虚部 (K, N)

    Returns:
        c_real, c_imag: 結果行列Cの実部と虚部 (M, N)
    """
    # 入力検証
    assert a_real.shape == a_imag.shape
    assert b_real.shape == b_imag.shape
    assert a_real.shape[1] == b_real.shape[0], f"Shape mismatch: {a_real.shape} vs {b_real.shape}"

    # デバイスチェック & Triton可用性チェック
    if TRITON_AVAILABLE and a_real.is_cuda:
        return _complex_matmul_triton(a_real, a_imag, b_real, b_imag)
    else:
        return _complex_matmul_fallback(a_real, a_imag, b_real, b_imag)


def _complex_matmul_fallback(
    a_real: torch.Tensor, a_imag: torch.Tensor,
    b_real: torch.Tensor, b_imag: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch Fallback Implementation"""
    # (Ar + iAi)(Br + iBi) = (ArBr - AiBi) + i(ArBi + AiBr)

    # ArBr
    ar_br = torch.matmul(a_real, b_real)
    # AiBi
    ai_bi = torch.matmul(a_imag, b_imag)
    # ArBi
    ar_bi = torch.matmul(a_real, b_imag)
    # AiBr
    ai_br = torch.matmul(a_imag, b_real)

    c_real = ar_br - ai_bi
    c_imag = ar_bi + ai_br

    return c_real, c_imag


def _complex_matmul_triton(
    a_real: torch.Tensor, a_imag: torch.Tensor,
    b_real: torch.Tensor, b_imag: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton Implementation Wrapper"""
    # Check constraints
    assert a_real.is_contiguous(), "A must be contiguous"
    assert b_real.is_contiguous(), "B must be contiguous"

    M, K = a_real.shape
    K2, N = b_real.shape
    assert K == K2

    # Allocate output
    c_real = torch.empty((M, N), device=a_real.device, dtype=a_real.dtype)
    c_imag = torch.empty((M, N), device=a_real.device, dtype=a_real.dtype)

    # 1D launch kernel configuration
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    complex_matmul_kernel[grid](
        a_real, a_imag,
        b_real, b_imag,
        c_real, c_imag,
        M, N, K,
        a_real.stride(0), a_real.stride(1),
        b_real.stride(0), b_real.stride(1),
        c_real.stride(0), c_real.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8
    )

    return c_real, c_imag
