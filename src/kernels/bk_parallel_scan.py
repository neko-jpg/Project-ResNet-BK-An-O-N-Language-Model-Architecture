"""
BK-Core Parallel Scan: O(log N) Associative Scan for Tridiagonal Inverse

Replaces sequential O(N) theta/phi recursions with parallel associative scan.

Algorithm:
    The recurrence θ_{i+1} = a_i * θ_i - k_i * θ_{i-1} is a 2nd-order linear recurrence.
    We convert it to 2x2 matrix form where each step is a matrix multiplication:
    
    [θ_{i+1}]   [a_i  -k_i] [θ_i    ]
    [θ_i    ] = [1    0   ] [θ_{i-1}]
    
    This allows parallel prefix computation using associative scan:
    M_final = M_N @ M_{N-1} @ ... @ M_1 in O(log N) parallel steps

Expected Performance:
    - Sequential O(N): ~10ms for N=4096
    - Parallel O(log N): ~1ms for N=4096 (10x speedup)

Author: Project MUSE Team
"""

import torch
import warnings
from typing import Tuple, Optional

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


# =============================================================================
# 2x2 Complex Matrix Operations (Triton JIT)
# =============================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def complex_2x2_matmul(
        # Matrix A: [[a11, a12], [a21, a22]]
        a11_r, a11_i, a12_r, a12_i,
        a21_r, a21_i, a22_r, a22_i,
        # Matrix B: [[b11, b12], [b21, b22]]
        b11_r, b11_i, b12_r, b12_i,
        b21_r, b21_i, b22_r, b22_i,
    ):
        """Compute C = A @ B for 2x2 complex matrices."""
        # c11 = a11*b11 + a12*b21
        c11_r = (a11_r * b11_r - a11_i * b11_i) + (a12_r * b21_r - a12_i * b21_i)
        c11_i = (a11_r * b11_i + a11_i * b11_r) + (a12_r * b21_i + a12_i * b21_r)
        
        # c12 = a11*b12 + a12*b22
        c12_r = (a11_r * b12_r - a11_i * b12_i) + (a12_r * b22_r - a12_i * b22_i)
        c12_i = (a11_r * b12_i + a11_i * b12_r) + (a12_r * b22_i + a12_i * b22_r)
        
        # c21 = a21*b11 + a22*b21
        c21_r = (a21_r * b11_r - a21_i * b11_i) + (a22_r * b21_r - a22_i * b21_i)
        c21_i = (a21_r * b11_i + a21_i * b11_r) + (a22_r * b21_i + a22_i * b21_r)
        
        # c22 = a21*b12 + a22*b22
        c22_r = (a21_r * b12_r - a21_i * b12_i) + (a22_r * b22_r - a22_i * b22_i)
        c22_i = (a21_r * b12_i + a21_i * b12_r) + (a22_r * b22_i + a22_i * b22_r)
        
        return c11_r, c11_i, c12_r, c12_i, c21_r, c21_i, c22_r, c22_i


    @triton.jit
    def scan_combine_2x2_matrices(
        left_packed,  # (..., 8) packed matrix: [m11r, m11i, m12r, m12i, m21r, m21i, m22r, m22i]
        right_packed,  # (..., 8) packed matrix
    ):
        """
        Associative combine function: computes right @ left (for prefix scan).
        Returns the packed result matrix.
        """
        # Unpack left matrix
        l11_r = left_packed[0]
        l11_i = left_packed[1]
        l12_r = left_packed[2]
        l12_i = left_packed[3]
        l21_r = left_packed[4]
        l21_i = left_packed[5]
        l22_r = left_packed[6]
        l22_i = left_packed[7]
        
        # Unpack right matrix
        r11_r = right_packed[0]
        r11_i = right_packed[1]
        r12_r = right_packed[2]
        r12_i = right_packed[3]
        r21_r = right_packed[4]
        r21_i = right_packed[5]
        r22_r = right_packed[6]
        r22_i = right_packed[7]
        
        # Compute right @ left
        c11_r, c11_i, c12_r, c12_i, c21_r, c21_i, c22_r, c22_i = complex_2x2_matmul(
            r11_r, r11_i, r12_r, r12_i, r21_r, r21_i, r22_r, r22_i,
            l11_r, l11_i, l12_r, l12_i, l21_r, l21_i, l22_r, l22_i,
        )
        
        # Pack result
        return tl.join(c11_r, c11_i, c12_r, c12_i, c21_r, c21_i, c22_r, c22_i)


    @triton.jit
    def bk_parallel_scan_kernel(
        # Input: Recurrence coefficients
        a_ptr,  # (B, N) main diagonal - z
        k_ptr,  # (B, N-1) product c[i]*b[i]
        # Output: theta values
        theta_r_ptr,  # (B, N+1) real part of theta
        theta_i_ptr,  # (B, N+1) imag part of theta
        # Dimensions
        B: tl.constexpr,
        N: tl.constexpr,
        # Strides
        stride_b_a, stride_n_a,
        stride_b_k, stride_n_k,
        stride_b_theta, stride_n_theta,
        # Block size
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Parallel associative scan for theta recursion.
        
        The recurrence: θ_{i+1} = a_i * θ_i - k_i * θ_{i-1}
        Matrix form: [θ_{i+1}, θ_i]^T = M_i @ [θ_i, θ_{i-1}]^T
        where M_i = [[a_i, -k_i], [1, 0]]
        
        We compute M_N @ M_{N-1} @ ... @ M_1 using parallel scan.
        """
        batch_id = tl.program_id(0)
        
        # Each thread handles a block of sequence positions
        block_start = tl.program_id(1) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        # Load coefficients for this block
        a_offset = batch_id * stride_b_a + offsets * stride_n_a
        a_r = tl.load(a_ptr + a_offset, mask=mask, other=1.0)
        
        k_offset = batch_id * stride_b_k + offsets * stride_n_k
        k_mask = offsets < (N - 1)
        k_r = tl.load(k_ptr + k_offset, mask=k_mask, other=0.0)
        
        # Build 2x2 transfer matrices for each position
        # M_i = [[a_i, -k_i], [1, 0]] (real parts only for now)
        m11_r = a_r
        m11_i = tl.zeros_like(a_r)
        m12_r = -k_r
        m12_i = tl.zeros_like(k_r)
        m21_r = tl.full_like(a_r, 1.0)
        m21_i = tl.zeros_like(a_r)
        m22_r = tl.zeros_like(a_r)
        m22_i = tl.zeros_like(a_r)
        
        # Pack matrices: (BLOCK_SIZE, 8)
        packed = tl.join(m11_r, m11_i, m12_r, m12_i, m21_r, m21_i, m22_r, m22_i)
        
        # Perform parallel associative scan
        scanned = tl.associative_scan(packed, axis=0, combine_fn=scan_combine_2x2_matrices)
        
        # Unpack scanned results
        # The scanned matrix at position i is M_i @ M_{i-1} @ ... @ M_1
        # Multiply by initial state [1, 0]^T to get [θ_{i+1}, θ_i]^T
        
        # θ_{i+1} = M[0,0] * 1 + M[0,1] * 0 = M[0,0] = scanned[..., 0]
        theta_r = scanned[..., 0]  # Real part of M[0,0]
        theta_i = scanned[..., 1]  # Imag part of M[0,0]
        
        # Store results
        theta_offset = batch_id * stride_b_theta + (offsets + 1) * stride_n_theta
        tl.store(theta_r_ptr + theta_offset, theta_r, mask=mask)
        tl.store(theta_i_ptr + theta_offset, theta_i, mask=mask)


# =============================================================================
# PyTorch Fallback Implementation
# =============================================================================

def _build_transfer_matrices_torch(a_shifted: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Build 2x2 transfer matrices for the recurrence.
    
    Args:
        a_shifted: (B, N) diagonal values a - z (complex)
        k: (B, N-1) product c[i]*b[i] (complex)
    
    Returns:
        matrices: (B, N, 2, 2) transfer matrices (complex)
    """
    B, N = a_shifted.shape
    device = a_shifted.device
    dtype = a_shifted.dtype
    
    # Initialize matrices
    matrices = torch.zeros(B, N, 2, 2, dtype=dtype, device=device)
    
    # M_i = [[a_i, -k_i], [1, 0]]
    matrices[:, :, 0, 0] = a_shifted  # a_i
    matrices[:, :-1, 0, 1] = -k  # -k_i (only for i < N-1)
    matrices[:, :, 1, 0] = 1.0  # Always 1
    # matrices[:, :, 1, 1] = 0.0  # Already zero
    
    return matrices


def _parallel_scan_torch(matrices: torch.Tensor) -> torch.Tensor:
    """
    Parallel prefix scan for 2x2 matrix multiplication.
    Uses iterative doubling (Hillis-Steele algorithm).
    
    Args:
        matrices: (B, N, 2, 2) transfer matrices
    
    Returns:
        cumulative: (B, N, 2, 2) cumulative products M_i @ ... @ M_1
    """
    B, N, _, _ = matrices.shape
    
    # Clone to avoid modifying input
    result = matrices.clone()
    
    # Hillis-Steele parallel scan
    # O(N log N) work but O(log N) depth for parallelism
    stride = 1
    while stride < N:
        # For positions >= stride, multiply with result[pos - stride]
        # result[i] = result[i] @ result[i - stride]
        shifted = torch.roll(result, shifts=stride, dims=1)
        
        # Create mask for valid positions
        mask = torch.arange(N, device=matrices.device) >= stride
        mask = mask.view(1, N, 1, 1).expand(B, N, 2, 2)
        
        # Matrix multiplication: result = result @ shifted
        new_result = torch.where(
            mask,
            torch.einsum('bnij,bnjk->bnik', result, shifted),
            result
        )
        result = new_result
        stride *= 2
    
    return result


def bk_parallel_theta_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    z: complex,
) -> torch.Tensor:
    """
    Compute theta recursion using parallel scan (PyTorch implementation).
    
    θ_0 = 1, θ_1 = a_0 - z
    θ_{i+1} = (a_i - z) * θ_i - c_{i-1} * b_{i-1} * θ_{i-1}
    
    Args:
        a: (B, N) main diagonal
        b: (B, N-1) super-diagonal
        c: (B, N-1) sub-diagonal
        z: complex shift
    
    Returns:
        theta: (B, N+1) theta values (complex)
    """
    B, N = a.shape
    device = a.device
    
    # Convert to complex128 for precision
    a_c = a.to(torch.complex128)
    b_c = b.to(torch.complex128)
    c_c = c.to(torch.complex128)
    z_c = torch.tensor(z, dtype=torch.complex128, device=device)
    
    # a_shifted = a - z
    a_shifted = a_c - z_c
    
    # k_i = c_i * b_i
    k = c_c * b_c  # (B, N-1)
    
    # Build transfer matrices
    matrices = _build_transfer_matrices_torch(a_shifted, k)  # (B, N, 2, 2)
    
    # Parallel scan
    cumulative = _parallel_scan_torch(matrices)  # (B, N, 2, 2)
    
    # Extract theta values
    # Initial state: [θ_1, θ_0] = [a_0 - z, 1]
    # After scan at position i: cumulative[i] @ [θ_1, θ_0] = [θ_{i+1}, θ_i]
    
    theta = torch.zeros(B, N + 1, dtype=torch.complex128, device=device)
    theta[:, 0] = 1.0  # θ_0 = 1
    theta[:, 1] = a_shifted[:, 0]  # θ_1 = a_0 - z
    
    # For i >= 1: θ_{i+1} = cumulative[i, 0, 0] * θ_1 + cumulative[i, 0, 1] * θ_0
    if N > 1:
        theta[:, 2:] = (
            cumulative[:, 1:, 0, 0] * theta[:, 1:2] + 
            cumulative[:, 1:, 0, 1] * theta[:, 0:1]
        )
    
    return theta


def bk_parallel_phi_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    z: complex,
) -> torch.Tensor:
    """
    Compute phi recursion using parallel scan (backward sweep).
    
    φ_{N-1} = 1, φ_{N-2} = a_{N-1} - z
    φ_i = (a_{i+1} - z) * φ_{i+1} - c_i * b_i * φ_{i+2}
    
    This is the same recurrence as theta but reversed.
    """
    B, N = a.shape
    device = a.device
    
    # Reverse the inputs and apply theta scan
    a_reversed = a.flip(dims=[1])
    b_reversed = b.flip(dims=[1])
    c_reversed = c.flip(dims=[1])
    
    # Compute theta on reversed sequence
    phi_reversed = bk_parallel_theta_torch(a_reversed, b_reversed, c_reversed, z)
    
    # Reverse back to get phi
    phi = phi_reversed.flip(dims=[1])
    
    return phi


def bk_parallel_inverse_diagonal(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    z: complex,
    use_triton: bool = True,
) -> torch.Tensor:
    """
    Compute diagonal of (T - zI)^{-1} using parallel associative scan.
    
    This achieves O(log N) parallel depth compared to O(N) sequential.
    
    Args:
        a: (B, N) main diagonal
        b: (B, N-1) super-diagonal
        c: (B, N-1) sub-diagonal
        z: complex shift
        use_triton: whether to use Triton kernel (if available)
    
    Returns:
        diag_inv: (B, N) diagonal of inverse (complex64)
    
    KPI Target: ≥10x speedup, ≥99.9% numerical correlation
    """
    B, N = a.shape
    device = a.device
    
    # Use PyTorch implementation (Triton kernel coming in next iteration)
    # Compute theta (forward) and phi (backward) using parallel scan
    theta = bk_parallel_theta_torch(a, b, c, z)  # (B, N+1)
    phi = bk_parallel_phi_torch(a, b, c, z)  # (B, N+1)
    
    # Compute diagonal inverse
    # G_ii = θ_i * φ_i / θ_N
    theta_n = theta[:, -1:]  # (B, 1)
    
    # Avoid division by zero
    theta_n_safe = torch.where(
        theta_n.abs() < 1e-10,
        torch.ones_like(theta_n) * 1e-10,
        theta_n
    )
    
    # G_ii = θ_i * φ_i / θ_N for i = 0, ..., N-1
    diag_inv = (theta[:, :-1] * phi[:, :-1]) / theta_n_safe
    
    # Numerical stability: clamp magnitude
    max_mag = 50.0
    mag = diag_inv.abs()
    factor = torch.where(mag > max_mag, max_mag / (mag + 1e-9), torch.ones_like(mag))
    diag_inv = diag_inv * factor
    
    return diag_inv.to(torch.complex64)


def is_parallel_scan_available() -> bool:
    """Check if parallel scan is available (always True, fallback to PyTorch)."""
    return True


__all__ = [
    'bk_parallel_inverse_diagonal',
    'bk_parallel_theta_torch',
    'bk_parallel_phi_torch',
    'is_parallel_scan_available',
]
