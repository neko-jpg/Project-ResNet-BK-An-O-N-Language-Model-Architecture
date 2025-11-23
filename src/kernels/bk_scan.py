"""
BK-Core Triton Kernel: O(N) Parallel Associative Scan

Implements parallel associative scan for computing diag((H - zI)^-1) using Triton.
Replaces serial loop with Hillis-Steele parallel scan for GPU efficiency.

Algorithm:
- Theta/Phi recursions are mapped to 2x2 matrix multiplications.
- Parallel associative scan (prefix product) computes the state transition matrices.
- Supports chunked processing for N > BLOCK_SIZE.

Update:
- Uses Packed Tensor Scan to support Triton versions that require single-tensor input for associative_scan.
- Input matrices are pre-packed in PyTorch to shape (B, N, 8).
"""

import torch
import warnings

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    # Dummy triton module for type hinting / decorator simulation if needed
    class tl:
        constexpr = int
        @staticmethod
        def program_id(x): return 0
        @staticmethod
        def arange(x, y): return torch.arange(x, y)
        @staticmethod
        def load(x, mask=None, other=0): return x
        @staticmethod
        def store(x, y, mask=None): pass
        @staticmethod
        def where(c, x, y): return x if c else y
        @staticmethod
        def associative_scan(x, axis, combine_fn): return x
        @staticmethod
        def debug_barrier(): pass
        @staticmethod
        def cat(tensors, dim=0): return tensors[0]
        @staticmethod
        def make_block_ptr(*args, **kwargs): return None
        @staticmethod
        def advance(ptr, offsets): return ptr

    # Dummy JIT decorator
    def jit(fn):
        return fn

    class triton:
        jit = jit

# ============================================================================
# Pure Python Implementations (for logic verification & JIT source)
# ============================================================================

def complex_mul_impl(r1, i1, r2, i2):
    """(r1 + i1*j) * (r2 + i2*j)"""
    return r1 * r2 - i1 * i2, r1 * i2 + i1 * r2

def complex_mat_mul_2x2_impl(
    a11r, a11i, a12r, a12i, a21r, a21i, a22r, a22i,
    b11r, b11i, b12r, b12i, b21r, b21i, b22r, b22i,
):
    """C = A * B (2x2 complex matrices)"""
    # c11 = a11*b11 + a12*b21
    t1r, t1i = complex_mul_impl(a11r, a11i, b11r, b11i)
    t2r, t2i = complex_mul_impl(a12r, a12i, b21r, b21i)
    c11r, c11i = t1r + t2r, t1i + t2i
    
    # c12 = a11*b12 + a12*b22
    t1r, t1i = complex_mul_impl(a11r, a11i, b12r, b12i)
    t2r, t2i = complex_mul_impl(a12r, a12i, b22r, b22i)
    c12r, c12i = t1r + t2r, t1i + t2i
    
    # c21 = a21*b11 + a22*b21
    t1r, t1i = complex_mul_impl(a21r, a21i, b11r, b11i)
    t2r, t2i = complex_mul_impl(a22r, a22i, b21r, b21i)
    c21r, c21i = t1r + t2r, t1i + t2i
    
    # c22 = a21*b12 + a22*b22
    t1r, t1i = complex_mul_impl(a21r, a21i, b12r, b12i)
    t2r, t2i = complex_mul_impl(a22r, a22i, b22r, b22i)
    c22r, c22i = t1r + t2r, t1i + t2i
    
    return c11r, c11i, c12r, c12i, c21r, c21i, c22r, c22i

# ============================================================================
# Triton JIT Functions
# ============================================================================

if TRITON_AVAILABLE:
    complex_mul = triton.jit(complex_mul_impl)
    complex_mat_mul_2x2 = triton.jit(complex_mat_mul_2x2_impl)
else:
    complex_mul = complex_mul_impl
    complex_mat_mul_2x2 = complex_mat_mul_2x2_impl

@triton.jit
def scan_op_packed(a, b):
    """
    Associative combine function for scan on PACKED tensors.
    Input/Output: (..., 8) tensor where last dim is [m11r, m11i, m12r, m12i, m21r, m21i, m22r, m22i]

    We compute B @ A (since we want M_k @ ... @ M_1)
    """
    # Unpack a
    a11r = a[:, 0]
    a11i = a[:, 1]
    a12r = a[:, 2]
    a12i = a[:, 3]
    a21r = a[:, 4]
    a21i = a[:, 5]
    a22r = a[:, 6]
    a22i = a[:, 7]

    # Unpack b
    b11r = b[:, 0]
    b11i = b[:, 1]
    b12r = b[:, 2]
    b12i = b[:, 3]
    b21r = b[:, 4]
    b21i = b[:, 5]
    b22r = b[:, 6]
    b22i = b[:, 7]

    # Compute C = B @ A
    c11r, c11i, c12r, c12i, c21r, c21i, c22r, c22i = complex_mat_mul_2x2(
        b11r, b11i, b12r, b12i, b21r, b21i, b22r, b22i,
        a11r, a11i, a12r, a12i, a21r, a21i, a22r, a22i
    )

    # Pack C using tl.cat
    # Requires constructing (BLOCK, 1) tensors then cat along dim 1
    return tl.cat(
        (
            c11r[:, None], c11i[:, None],
            c12r[:, None], c12i[:, None],
            c21r[:, None], c21i[:, None],
            c22r[:, None], c22i[:, None]
        ),
        dim=1
    )


# ============================================================================
# Forward Scan Kernel (Packed)
# ============================================================================

@triton.jit
def bk_scan_fwd_kernel_packed(
    # Input: Packed Matrices (B, N, 8)
    matrices_ptr,
    # Output pointers
    theta_r_ptr, theta_i_ptr,
    # Dimensions
    B, N,
    # Strides
    stride_b_mat, stride_n_mat,
    stride_b_theta, stride_n_theta,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Parallel Forward Scan using Packed Tensors.
    """
    pid = tl.program_id(0)

    # Pointers
    m_ptr_base = matrices_ptr + pid * stride_b_mat
    theta_r = theta_r_ptr + pid * stride_b_theta
    theta_i = theta_i_ptr + pid * stride_b_theta
    
    # Initialize Theta[0]=1, Theta[1]=alpha[0] (which is M[0]_11 if packed correctly)
    tl.store(theta_r, 1.0)
    tl.store(theta_i, 0.0)
    
    cur_r, cur_i = 1.0, 0.0 # theta_0
    prev_r, prev_i = 0.0, 0.0 # theta_-1 (dummy)

    # Loop over chunks
    for off in range(0, N, BLOCK_SIZE):
        ks = off + tl.arange(0, BLOCK_SIZE)
        mask = ks < N

        # Load M chunk: (BLOCK, 8)
        # Using simple pointer arithmetic + broadcasting for indices
        cols = tl.arange(0, 8)
        # ptrs: (BLOCK, 8)
        ptrs = m_ptr_base + ks[:, None] * stride_n_mat + cols[None, :]

        # Load with mask. Note: Inputs are padded with Identity, so OOB masking logic is implicit.
        # But we still use mask for safety if padding wasn't enough (though we force padding).
        packed_m = tl.load(ptrs, mask=mask[:, None], other=0.0)

        # Run Scan
        accumulated_m = tl.associative_scan(packed_m, 0, scan_op_packed)

        # Unpack accumulated M
        p11r, p11i, p12r, p12i, p21r, p21i, p22r, p22i = (
            accumulated_m[:, 0], accumulated_m[:, 1],
            accumulated_m[:, 2], accumulated_m[:, 3],
            accumulated_m[:, 4], accumulated_m[:, 5],
            accumulated_m[:, 6], accumulated_m[:, 7]
        )

        # Matrix-Vector Mul: P @ v_prev
        t1r, t1i = complex_mul(p11r, p11i, cur_r, cur_i)
        t2r, t2i = complex_mul(p12r, p12i, prev_r, prev_i)
        new_cur_r = t1r + t2r
        new_cur_i = t1i + t2i

        # Store theta
        store_idx = ks + 1
        mask_store = store_idx <= N

        tl.store(theta_r + store_idx * stride_n_theta, new_cur_r, mask=mask_store)
        tl.store(theta_i + store_idx * stride_n_theta, new_cur_i, mask=mask_store)
        
        tl.debug_barrier()

        # Load state for next iteration
        # Read from the LAST computed element in the chunk.
        # Since we padded, we can read at off + BLOCK_SIZE even if it's padding region.
        last_idx = off + BLOCK_SIZE

        cur_r = tl.load(theta_r + last_idx * stride_n_theta)
        cur_i = tl.load(theta_i + last_idx * stride_n_theta)
        prev_r = tl.load(theta_r + (last_idx - 1) * stride_n_theta)
        prev_i = tl.load(theta_i + (last_idx - 1) * stride_n_theta)

        tl.debug_barrier()


# ============================================================================
# Python Interface Functions
# ============================================================================

def pack_scan_input(alpha, beta):
    """
    Pack alpha, beta into (B, N, 8) transition matrices for Forward Scan.
    M_k = [[alpha[k], beta[k-1]], [1, 0]]
    """
    B, N = alpha.shape
    device = alpha.device
    
    # Prepare beta with shift (beta[-1] -> 0, beta[0] -> beta[0])
    # alpha: 0..N-1. beta: 0..N-2 (size N-1).
    # M_k uses beta[k-1].
    # M_0 uses beta[-1] (0).
    if beta.shape[1] == N - 1:
        zero_col = torch.zeros(B, 1, dtype=beta.dtype, device=device)
        beta_shifted = torch.cat([zero_col, beta], dim=1) # (B, N)
    else:
        beta_shifted = beta

    # Construct (B, N, 8)
    M = torch.zeros(B, N, 8, dtype=torch.float32, device=device)
    
    M[..., 0] = alpha.real
    M[..., 1] = alpha.imag
    M[..., 2] = beta_shifted.real
    M[..., 3] = beta_shifted.imag
    M[..., 4] = 1.0
    M[..., 5] = 0.0
    M[..., 6] = 0.0
    M[..., 7] = 0.0
    
    return M

def pack_scan_input_backward(alpha, beta):
    """
    Pack for Backward Scan by reversing inputs.
    """
    N = alpha.shape[1]
    # Reverse Alpha
    alpha_rev = alpha.flip(1)

    # Reverse Beta
    if beta.shape[1] == N - 1:
        beta_rev = beta.flip(1)
    else:
        beta_rev = beta.flip(1)
        
    return pack_scan_input(alpha_rev, beta_rev)


def bk_scan_triton_forward(alpha, beta):
    if not TRITON_AVAILABLE:
        raise ImportError("Triton is not available.")

    B, N = alpha.shape
    device = alpha.device
    
    # 1. Pack
    M = pack_scan_input(alpha, beta) # (B, N, 8)
    
    # 2. Pad to BLOCK_SIZE
    BLOCK_SIZE = 128
    if N >= 1024: BLOCK_SIZE = 1024
    elif N >= 512: BLOCK_SIZE = 512
    elif N >= 256: BLOCK_SIZE = 256
    
    remainder = N % BLOCK_SIZE
    if remainder != 0:
        pad_len = BLOCK_SIZE - remainder
        pad = torch.zeros(B, pad_len, 8, dtype=M.dtype, device=device)
        pad[..., 0] = 1.0 # m11r
        pad[..., 6] = 1.0 # m22r
        M_padded = torch.cat([M, pad], dim=1)
        N_padded = N + pad_len
    else:
        M_padded = M
        N_padded = N

    # 3. Output buffers
    theta_r = torch.empty(B, N_padded + 1, dtype=torch.float32, device=device)
    theta_i = torch.empty(B, N_padded + 1, dtype=torch.float32, device=device)

    # 4. Run Kernel
    grid = (B,)
    bk_scan_fwd_kernel_packed[grid](
        M_padded,
        theta_r, theta_i,
        B, N_padded,
        M_padded.stride(0), M_padded.stride(1),
        theta_r.stride(0), theta_r.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    res = torch.complex(theta_r[:, :N+1], theta_i[:, :N+1])
    return res


def bk_scan_triton_backward(alpha, beta, N):
    if not TRITON_AVAILABLE:
        raise ImportError("Triton is not available.")

    B = alpha.shape[0]
    device = alpha.device

    # 1. Pack Reverse
    M = pack_scan_input_backward(alpha, beta)
    
    # 2. Pad
    BLOCK_SIZE = 128
    if N >= 1024: BLOCK_SIZE = 1024
    elif N >= 512: BLOCK_SIZE = 512
    elif N >= 256: BLOCK_SIZE = 256
    
    remainder = N % BLOCK_SIZE
    if remainder != 0:
        pad_len = BLOCK_SIZE - remainder
        pad = torch.zeros(B, pad_len, 8, dtype=M.dtype, device=device)
        pad[..., 0] = 1.0
        pad[..., 6] = 1.0
        M_padded = torch.cat([M, pad], dim=1)
        N_padded = N + pad_len
    else:
        M_padded = M
        N_padded = N

    # 3. Run Forward Kernel
    theta_r = torch.empty(B, N_padded + 1, dtype=torch.float32, device=device)
    theta_i = torch.empty(B, N_padded + 1, dtype=torch.float32, device=device)

    grid = (B,)
    bk_scan_fwd_kernel_packed[grid](
        M_padded,
        theta_r, theta_i,
        B, N_padded,
        M_padded.stride(0), M_padded.stride(1),
        theta_r.stride(0), theta_r.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # 4. Extract and Reverse Back
    res_psi = torch.complex(theta_r[:, :N], theta_i[:, :N])
    phi = res_psi.flip(1)

    return phi


def bk_scan_triton(a, b, c, z):
    """
    Complete Triton-accelerated BK-Core computation.
    """
    if not TRITON_AVAILABLE:
        raise ImportError("Triton is not available.")

    B, N = a.shape
    device = a.device
    
    a_c = a.to(torch.complex64)
    b_c = b.to(torch.complex64)
    c_c = c.to(torch.complex64)

    # Fix: use detach to avoid warnings
    if isinstance(z, torch.Tensor):
        z_c = z.clone().detach().to(dtype=torch.complex64, device=device)
    else:
        z_c = torch.tensor(z, dtype=torch.complex64, device=device)
    
    alpha = a_c - z_c
    beta = -c_c * b_c
    
    theta = bk_scan_triton_forward(alpha, beta) # Size N+1
    phi = bk_scan_triton_backward(alpha, beta, N) # Size N

    # Combine
    det_T = theta[:, -1:] # (B, 1)
    theta_trunc = theta[:, :-1] # 0..N-1
    
    eps = 1e-18
    diag_inv = theta_trunc * phi / (det_T + eps)
    
    return diag_inv

def is_triton_available():
    return TRITON_AVAILABLE
