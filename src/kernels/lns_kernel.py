"""
Logarithmic Number System (LNS) Triton Kernel - Phase 1.3

物理的直観 (Physical Intuition):
乗算器(FMA)を加算器(ADD)に変換することで、計算コストと消費電力を削減します。
FMAはADDの約3-5倍の電力を消費し、スループットも2倍低速です。

Mathematical Foundation:
LNS transforms multiplication into addition in logarithmic domain:
  Standard: c = a × b  (expensive FMA operation)
  LNS: log(c) = log(a) + log(b)  (cheap addition)

Matrix Multiplication:
  C = A @ B = Σₖ A[i,k] × B[k,j]
  
LNS Approximation (Max-Log):
  log(C[i,j]) ≈ maxₖ(log(A[i,k]) + log(B[k,j]))
  
Physical Intuition:
  - 乗算器(FMA) → 加算器(ADD)への変換
  - 消費電力: FMA は ADD の約3-5倍
  - スループット: ADD は FMA より2倍高速

Numerical Considerations:
The max-log approximation introduces bias but maintains:
- Monotonicity: arg max preserved
- Sparsity: Dominant terms emphasized
- Stability: No exponential overflow

References:
- Requirements: 3.1, 3.2
- Design: Section "Logarithmic Number System (LNS) Kernel"
- Algorithm: Max-log approximation for accumulation
"""

import torch
import math
from typing import Optional

# Try to import Triton, but provide graceful fallback
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


if TRITON_AVAILABLE:
    @triton.jit
    def lns_matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # Strides
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        """
        LNS Matrix Multiplication Triton Kernel.
        
        物理的直観:
        対数領域での行列積を計算。乗算をすべて加算に変換し、
        FMA命令を使わずにADD命令のみで計算を実行します。
        
        Algorithm:
        1. Load blocks of A and B
        2. Compute log_prod = log_a + log_b (element-wise addition)
        3. Accumulate using max-log: acc = max(acc, log_prod)
        4. Store result
        
        Complexity: O(MNK) operations, but each operation is ADD not FMA
        Memory: O(1) per thread (register-only accumulation)
        
        Args:
            a_ptr: Pointer to matrix A (assumed in log domain or will be converted)
            b_ptr: Pointer to matrix B (assumed in log domain or will be converted)
            c_ptr: Pointer to output matrix C (in log domain)
            M, N, K: Matrix dimensions (C = A @ B where A is MxK, B is KxN)
            stride_am, stride_ak: Strides for matrix A
            stride_bk, stride_bn: Strides for matrix B
            stride_cm, stride_cn: Strides for matrix C
            BLOCK_SIZE_M, N, K: Block sizes for tiling
        
        Requirements: 3.1, 3.2
        """
        # Get program ID (which block this thread belongs to)
        pid = tl.program_id(axis=0)
        
        # Calculate 2D block indices from 1D program ID
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m
        
        # Create offsets for this block
        # Requirement 3.1: Implement configurable block sizes
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        
        # Initialize pointers for A and B blocks
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        
        # Initialize accumulator
        # Requirement 3.2: Implement max-log accumulation for summation
        # 物理的直観: Max-Log近似を使用
        # log(x + y) ≈ max(log(x), log(y)) when x >> y or y >> x
        # This approximation emphasizes dominant terms and maintains sparsity
        accumulator = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), float('-inf'), dtype=tl.float32)
        
        # Loop over K dimension in blocks
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load blocks of A and B
            # Mask to handle non-multiple-of-BLOCK_SIZE dimensions
            k_remaining = K - k * BLOCK_SIZE_K
            a_mask = offs_k[None, :] < k_remaining
            b_mask = offs_k[:, None] < k_remaining
            
            a = tl.load(a_ptrs, mask=a_mask, other=float('-inf'))
            b = tl.load(b_ptrs, mask=b_mask, other=float('-inf'))
            
            # --- LNS COMPUTATION ---
            # Requirement 3.1: Implement log-domain addition
            # In log domain: log(a * b) = log(a) + log(b)
            # 物理的直観: 乗算を加算に変換
            log_prod = a + b
            
            # Requirement 3.2: Max-log accumulation
            # Instead of: acc += exp(log_prod) (expensive)
            # We use: acc = max(acc, log_prod) (cheap approximation)
            # 
            # 物理的直観: 支配的な項のみを保持
            # This approximation works well for sparse activations
            # where one term dominates the sum
            accumulator = tl.maximum(accumulator, log_prod)
            
            # Advance pointers to next K block
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        
        # Store result
        c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
        c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)


class LNSMatmulFunction(torch.autograd.Function):
    """
    Custom autograd function for LNS matrix multiplication with backward pass.
    
    Task 8.3: Implement backward pass for max-log approximation
    
    Forward: Uses max-log approximation for efficient computation
    Backward: Uses straight-through estimator (STE) for gradient flow
    
    物理的直観:
    Max演算は微分不可能だが、Straight-Through Estimatorにより
    勾配を「そのまま通す」ことで学習を可能にします。
    これは、Gumbel-Softmaxなどでも使われる手法です。
    """
    
    @staticmethod
    def forward(ctx, a, b, block_size_m, block_size_n, block_size_k, gradient_clip_value):
        """
        Forward pass using LNS kernel.
        
        Args:
            ctx: Context for saving tensors for backward
            a: (M, K) input matrix in log domain
            b: (K, N) input matrix in log domain
            block_size_m, block_size_n, block_size_k: Block sizes
            gradient_clip_value: Gradient clipping threshold
        
        Returns:
            c: (M, N) output matrix in log domain
        """
        # Validate inputs
        assert a.ndim == 2 and b.ndim == 2, "Inputs must be 2D matrices"
        assert a.shape[1] == b.shape[0], f"Incompatible dimensions: {a.shape} @ {b.shape}"
        assert a.is_contiguous(), "Matrix A must be contiguous"
        assert b.is_contiguous(), "Matrix B must be contiguous"
        
        # Check CUDA availability
        if not TRITON_AVAILABLE or not torch.cuda.is_available() or not a.is_cuda:
            raise RuntimeError(
                "LNS kernel requires CUDA and Triton. "
                "Use CPU fallback or standard torch.matmul instead."
            )
        
        # Get dimensions
        M, K = a.shape
        K2, N = b.shape
        assert K == K2, f"Inner dimensions must match: {K} != {K2}"
        
        # Allocate output
        c = torch.empty((M, N), device=a.device, dtype=torch.float32)
        
        # Launch kernel
        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
        )
        
        lns_matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n,
            BLOCK_SIZE_K=block_size_k,
        )
        
        # Save for backward
        ctx.save_for_backward(a, b, c)
        ctx.gradient_clip_value = gradient_clip_value
        
        return c
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using straight-through estimator.
        
        Task 8.3: Use straight-through estimator for non-differentiable max
        
        物理的直観:
        Max-log近似 log(Σ exp(x_i)) ≈ max(x_i) は微分不可能。
        STEでは、forward時はmax、backward時は勾配をそのまま通します。
        
        Approximation:
        ∂max(x_i)/∂x_j ≈ 1 for all j (straight-through)
        
        This is a biased estimator but works well in practice for:
        - Sparse activations (where max is a good approximation)
        - Inference-focused models (where exact gradients less critical)
        
        Args:
            ctx: Context with saved tensors
            grad_output: (M, N) gradient from upstream
        
        Returns:
            Tuple of gradients: (grad_a, grad_b, None, None, None, None)
        """
        a, b, c = ctx.saved_tensors
        gradient_clip_value = ctx.gradient_clip_value
        
        # Task 8.3: Add gradient clipping at kernel level
        # Clip gradient to prevent explosion
        if gradient_clip_value is not None and gradient_clip_value > 0:
            grad_output = torch.clamp(
                grad_output,
                -gradient_clip_value,
                gradient_clip_value
            )
        
        # Straight-through estimator:
        # Treat max-log as if it were standard log-sum-exp
        # ∂(log Σ exp(x_i))/∂x_j = exp(x_j) / Σ exp(x_i)
        # 
        # But we approximate: ∂max(x_i)/∂x_j ≈ uniform distribution
        # This is equivalent to treating the operation as linear
        
        # Gradient w.r.t. a: grad_a = grad_output @ b^T
        # In log domain: log(grad_a) = log(grad_output) + log(b^T)
        # But grad_output is not in log domain, so we use standard matmul
        grad_a = torch.matmul(grad_output, b.t())
        
        # Gradient w.r.t. b: grad_b = a^T @ grad_output
        grad_b = torch.matmul(a.t(), grad_output)
        
        # Apply gradient clipping again after matmul
        if gradient_clip_value is not None and gradient_clip_value > 0:
            grad_a = torch.clamp(grad_a, -gradient_clip_value, gradient_clip_value)
            grad_b = torch.clamp(grad_b, -gradient_clip_value, gradient_clip_value)
        
        # Return gradients for (a, b, block_size_m, block_size_n, block_size_k, gradient_clip_value)
        # Only a and b need gradients
        return grad_a, grad_b, None, None, None, None


def lns_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size_m: int = 128,
    block_size_n: int = 128,
    block_size_k: int = 32,
    gradient_clip_value: Optional[float] = 10.0,
) -> torch.Tensor:
    """
    LNS matrix multiplication using Triton kernel with gradient support.
    
    物理的直観:
    対数領域での行列積。乗算器を使わず加算器のみで計算。
    推論時の消費電力を大幅に削減します。
    
    This function performs matrix multiplication in logarithmic domain,
    replacing expensive FMA operations with cheap additions.
    
    Task 8.3: Backward pass implemented using straight-through estimator
    
    Args:
        a: Input matrix A (M, K) - assumed to be in log domain
        b: Input matrix B (K, N) - assumed to be in log domain
        block_size_m: Block size for M dimension (default: 128)
        block_size_n: Block size for N dimension (default: 128)
        block_size_k: Block size for K dimension (default: 32)
        gradient_clip_value: Gradient clipping threshold (default: 10.0)
                            Set to None to disable clipping
    
    Returns:
        c: Output matrix C (M, N) in log domain
    
    Note:
        For neural networks, inputs/outputs are typically NOT in log domain.
        This kernel is most useful for:
        1. Inference-only deployment (convert weights to log once)
        2. Specific layers where log-domain makes sense (e.g., attention scores)
        3. Research/experimentation with log-domain training
    
    Requirements: 3.1, 3.2, Task 8.3
    """
    return LNSMatmulFunction.apply(
        a, b, block_size_m, block_size_n, block_size_k, gradient_clip_value
    )


__all__ = [
    'lns_matmul',
    'lns_matmul_kernel',
]
