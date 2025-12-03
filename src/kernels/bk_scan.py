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
def bk_scan_fused_kernel(
    # Input pointers (Real, Imag)
    a_r_ptr, a_i_ptr,
    b_r_ptr, b_i_ptr,
    c_r_ptr, c_i_ptr,
    # Output pointers
    theta_r_ptr, theta_i_ptr,
    # Scalar z
    z_r, z_i,
    # Dimensions
    B, N,
    # Strides
    stride_b_a, stride_n_a,
    stride_b_b, stride_n_b,
    stride_b_c, stride_n_c,
    stride_b_theta, stride_n_theta,
    # Block size
    BLOCK_SIZE: tl.constexpr,
    # Direction (0: Forward, 1: Backward)
    IS_BACKWARD: tl.constexpr,
):
    """
    Fused Parallel Scan Kernel.
    Computes alpha, beta on the fly and performs scan.
    """
    pid = tl.program_id(0)

    # Pointers
    theta_r = theta_r_ptr + pid * stride_b_theta
    theta_i = theta_i_ptr + pid * stride_b_theta
    
    # Initialize Theta[0]=1, Theta[1]=alpha[0]
    tl.store(theta_r, 1.0)
    tl.store(theta_i, 0.0)
    
    cur_r, cur_i = 1.0, 0.0 # theta_0
    prev_r, prev_i = 0.0, 0.0 # theta_-1 (dummy)

    # Loop over chunks
    for off in range(0, N, BLOCK_SIZE):
        ks = off + tl.arange(0, BLOCK_SIZE)
        mask = ks < N

        # Compute M on the fly
        # M_k = [[alpha[k], beta[k-1]], [1, 0]]
        
        # Load alpha[k] = a[k] - z
        # Handle strides for backward (if IS_BACKWARD, pointers are adjusted outside or strides are negative)
        # We assume strides are passed correctly.
        
        # Load a[k]
        a_off = ks * stride_n_a
        ar = tl.load(a_r_ptr + pid * stride_b_a + a_off, mask=mask, other=0.0)
        ai = tl.load(a_i_ptr + pid * stride_b_a + a_off, mask=mask, other=0.0)
        
        alpha_r = ar - z_r
        alpha_i = ai - z_i
        
        # Load beta[k-1] = -c[k-1] * b[k-1]
        # For k=0 (global), beta is 0.
        # If IS_BACKWARD, logic might differ?
        # In backward: beta_rev[k-1]. beta_rev is flipped beta.
        # beta[k] = -c[k]*b[k].
        # beta_rev[0] = beta[N-2].
        # beta_rev[k] = beta[N-2-k].
        # So we need to load c, b at index N-2-(k-1) = N-1-k?
        # If we use negative strides for a, b, c, we get a[N-1-k].
        # But beta needs shift.
        
        # Let's handle beta logic:
        # We need beta_val for each k.
        # If k=0, beta_val=0.
        # Else, load b, c.
        
        # We can use a trick: Load b, c at k-1 (or appropriate index).
        # If stride is negative, k-1 moves in the other direction.
        
        # Let's compute indices for b, c.
        # Forward: k -> k-1.
        # Backward: k -> k-1 (in reversed sequence).
        # If we set pointer to start at end, and stride -1.
        # a_ptr points to a[N-1]. stride -1.
        # a[k] (kernel) -> a[N-1-k] (memory). Correct.
        
        # For beta:
        # Forward: beta[k-1] -> -c[k-1]*b[k-1].
        # Backward: beta_rev[k-1] -> beta[N-2-(k-1)] = beta[N-1-k].
        # beta[j] = -c[j]*b[j].
        # So we need -c[N-1-k]*b[N-1-k].
        # This is exactly accessing c, b at the SAME index as a!
        # Wait, beta[k-1] in forward uses index k-1.
        # beta_rev[k-1] in backward uses index N-1-k.
        # a_rev[k] uses index N-1-k.
        # So in backward, beta uses the SAME index as alpha?
        # Let's check:
        # M_k (backward) uses beta_rev[k-1].
        # beta_rev[0] = beta[N-2]. (index N-2).
        # a_rev[1] = a[N-2]. (index N-2).
        # So M_1 uses beta corresponding to index N-2.
        # M_1 uses alpha corresponding to index N-2.
        # So yes, in backward, beta and alpha are aligned!
        # But in Forward, M_k uses alpha[k] and beta[k-1]. Misaligned by 1.
        
        # So:
        # Forward: Load b, c at k-1.
        # Backward: Load b, c at k (aligned with a).
        
        # We can use `tl.where` to handle k=0.
        
        beta_r = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        beta_i = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        # Indices for b, c
        if IS_BACKWARD:
            # Aligned with k
            bc_idx = ks
            valid_bc = (ks < N) & (ks < N - 1) # beta is size N-1?
            # Actually beta is defined for 0..N-2.
            # In backward, we access N-2..0.
            # At k=0 (kernel), we need beta[N-2].
            # a[0] (kernel) is a[N-1].
            # a[1] (kernel) is a[N-2].
            # So at k=1, we need beta[N-2].
            # So beta is aligned with a[k]!
            # But wait, M_0 uses 0.
            # M_1 uses beta_rev[0] = beta[N-2].
            # a_rev[1] = a[N-2].
            # So yes, aligned.
            # But at k=0, beta is 0.
            
            # Load b, c at bc_idx
            # We need to handle the fact that b, c might be size N-1 or N.
            # Usually b, c are size N-1.
            # If size N-1, they correspond to 0..N-2.
            # In backward, we access N-2 down to 0.
            # This corresponds to a indices N-2 down to 0.
            # So valid when k >= 1.
            
            # Adjust pointers for backward:
            # a_ptr starts at N-1.
            # b_ptr starts at N-2.
            # stride -1.
            # At k=1: b_ptr + 1*(-1) = N-3? No.
            # We need b[N-2] at k=1.
            # If b_ptr starts at N-2. b_ptr + (1-1)*(-1) = N-2.
            # So we need offset k-1.
            pass
        else:
            # Forward: k-1.
            pass

        # Unified logic:
        # We need to load from (k-1) relative to the direction.
        # If Forward: ptr + (k-1)*stride.
        # If Backward: ptr + (k-1)*stride.
        # And handle k=0 case (beta=0).
        
        # Load b, c
        # We use a shift of -1.
        shift = -1
        bc_ks = ks + shift
        
        # Mask for valid b, c read
        # k=0 -> bc_ks=-1 -> Invalid.
        mask_bc = (bc_ks >= 0) & (bc_ks < N - 1) & mask
        
        # Calculate offsets
        b_off = bc_ks * stride_n_b
        c_off = bc_ks * stride_n_c
        
        br = tl.load(b_r_ptr + pid * stride_b_b + b_off, mask=mask_bc, other=0.0)
        bi = tl.load(b_i_ptr + pid * stride_b_b + b_off, mask=mask_bc, other=0.0)
        cr = tl.load(c_r_ptr + pid * stride_b_c + c_off, mask=mask_bc, other=0.0)
        ci = tl.load(c_i_ptr + pid * stride_b_c + c_off, mask=mask_bc, other=0.0)
        
        # beta = -c * b
        # -(cr + ici)*(br + ibi) = -((cr*br - ci*bi) + i(cr*bi + ci*br))
        beta_real = -(cr * br - ci * bi)
        beta_imag = -(cr * bi + ci * br)
        
        # If k=0, beta is 0.
        is_first = ks == 0
        beta_real = tl.where(is_first, 0.0, beta_real)
        beta_imag = tl.where(is_first, 0.0, beta_imag)
        
        # Pack M
        # M = [[alpha, beta], [1, 0]]
        # We construct a (BLOCK, 8) tensor for scan_op_packed
        # Layout: m11r, m11i, m12r, m12i, m21r, m21i, m22r, m22i
        
        packed_m = tl.zeros((BLOCK_SIZE, 8), dtype=tl.float32)
        # m11 = alpha
        packed_m = packed_m + tl.where(mask[:, None], alpha_r[:, None], 0.0) * (tl.arange(0, 8) == 0)
        packed_m = packed_m + tl.where(mask[:, None], alpha_i[:, None], 0.0) * (tl.arange(0, 8) == 1)
        # m12 = beta
        packed_m = packed_m + tl.where(mask[:, None], beta_real[:, None], 0.0) * (tl.arange(0, 8) == 2)
        packed_m = packed_m + tl.where(mask[:, None], beta_imag[:, None], 0.0) * (tl.arange(0, 8) == 3)
        # m21 = 1
        packed_m = packed_m + tl.where(mask[:, None], 1.0, 0.0) * (tl.arange(0, 8) == 4)
        # m22 = 0 (already zero)
        
        # Run Scan
        accumulated_m = tl.associative_scan(packed_m, 0, scan_op_packed)

        # Unpack and update state (same as before)
        p11r, p11i, p12r, p12i, p21r, p21i, p22r, p22i = (
            accumulated_m[:, 0], accumulated_m[:, 1],
            accumulated_m[:, 2], accumulated_m[:, 3],
            accumulated_m[:, 4], accumulated_m[:, 5],
            accumulated_m[:, 6], accumulated_m[:, 7]
        )

        t1r, t1i = complex_mul(p11r, p11i, cur_r, cur_i)
        t2r, t2i = complex_mul(p12r, p12i, prev_r, prev_i)
        new_cur_r = t1r + t2r
        new_cur_i = t1i + t2i

        store_idx = ks + 1
        mask_store = store_idx <= N

        tl.store(theta_r + store_idx * stride_n_theta, new_cur_r, mask=mask_store)
        tl.store(theta_i + store_idx * stride_n_theta, new_cur_i, mask=mask_store)
        
        tl.debug_barrier()

        last_idx = off + BLOCK_SIZE
        cur_r = tl.load(theta_r + last_idx * stride_n_theta)
        cur_i = tl.load(theta_i + last_idx * stride_n_theta)
        prev_r = tl.load(theta_r + (last_idx - 1) * stride_n_theta)
        prev_i = tl.load(theta_i + (last_idx - 1) * stride_n_theta)

        tl.debug_barrier()

def bk_scan_triton(a, b, c, z):
    """
    Complete Triton-accelerated BK-Core computation (Fused).
    """
    if not TRITON_AVAILABLE:
        raise ImportError("Triton is not available.")

    B, N = a.shape
    device = a.device
    
    # Ensure inputs are contiguous or handle strides
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    # Prepare z
    if isinstance(z, torch.Tensor):
        z_val = z.detach().cpu().numpy() if z.numel() == 1 else z.item()
    else:
        z_val = z
    
    # Handle complex z
    if isinstance(z_val, complex):
        z_r, z_i = z_val.real, z_val.imag
    else:
        z_r, z_i = float(z_val), 0.0

    # Output buffers
    theta_r = torch.empty(B, N + 1, dtype=torch.float32, device=device)
    theta_i = torch.empty(B, N + 1, dtype=torch.float32, device=device)
    
    # Forward Scan
    grid = (B,)
    BLOCK_SIZE = 128 # Tune this
    
    # Forward Pointers
    # a, b, c are real/imag separated?
    # Input a, b, c are complex tensors.
    # We need to pass pointers to real and imag parts.
    # Since they are complex64 (or 128), real and imag are interleaved.
    # We can cast to float and use stride 2.
    
    a_float = torch.view_as_real(a.to(torch.complex64)) # (B, N, 2)
    b_float = torch.view_as_real(b.to(torch.complex64))
    c_float = torch.view_as_real(c.to(torch.complex64))
    
    # Strides for float view
    # stride_n_a for real part is stride(1) * 2? No, stride(1) of complex is 1 (element).
    # stride(1) of float view is 2.
    # stride(2) is 1.
    # Real part: ptr + 0. Imag part: ptr + 1.
    
    # We pass base pointers and strides.
    # a_r_ptr = a_float.data_ptr()
    # a_i_ptr = a_float.data_ptr() + 4 (sizeof float)
    
    # Or we can just load as float2? Triton supports pointer arithmetic.
    # Let's pass separate pointers for simplicity in python wrapper.
    
    bk_scan_fused_kernel[grid](
        a_float.data_ptr(), a_float.data_ptr() + 4,
        b_float.data_ptr(), b_float.data_ptr() + 4,
        c_float.data_ptr(), c_float.data_ptr() + 4,
        theta_r, theta_i,
        z_r, z_i,
        B, N,
        a_float.stride(0), a_float.stride(1),
        b_float.stride(0), b_float.stride(1),
        c_float.stride(0), c_float.stride(1),
        theta_r.stride(0), theta_r.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        IS_BACKWARD=0
    )
    
    theta = torch.complex(theta_r, theta_i)
    
    # Backward Scan
    # We need to reverse inputs.
    # We can use negative strides or adjust pointers.
    # a_rev[k] = a[N-1-k].
    # Ptr should point to a[N-1]. Stride should be -stride.
    
    # Adjust pointers to end
    # a_float is (B, N, 2).
    # End of row i: a_float[i, N-1, 0].
    # Offset = i * stride_b + (N-1) * stride_n.
    # But we pass base ptr + pid * stride_b.
    # So we need to adjust base ptr to point to N-1 column.
    # And set stride_n to negative.
    
    offset_last = (N - 1) * a_float.stride(1) * 4 # bytes
    
    phi_r = torch.empty(B, N + 1, dtype=torch.float32, device=device)
    phi_i = torch.empty(B, N + 1, dtype=torch.float32, device=device)
    
    bk_scan_fused_kernel[grid](
        a_float.data_ptr() + offset_last, a_float.data_ptr() + offset_last + 4,
        b_float.data_ptr() + offset_last, b_float.data_ptr() + offset_last + 4,
        c_float.data_ptr() + offset_last, c_float.data_ptr() + offset_last + 4,
        phi_r, phi_i,
        z_r, z_i,
        B, N,
        a_float.stride(0), -a_float.stride(1),
        b_float.stride(0), -b_float.stride(1),
        c_float.stride(0), -c_float.stride(1),
        phi_r.stride(0), phi_r.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        IS_BACKWARD=1
    )
    
    phi_raw = torch.complex(phi_r, phi_i)
    # phi_raw is theta of reversed input.
    # We need to reverse it back to get phi.
    # And phi is size N (0..N-1).
    # phi_raw is size N+1.
    # phi[k] corresponds to suffix k..N-1.
    # phi_raw[k] corresponds to prefix 0..k of reversed.
    # prefix 0..k of reversed is suffix N-1-k..N-1.
    # So phi_raw[k] is phi[N-1-k]?
    # Let's check sizes.
    # phi_raw has N+1 elements.
    # phi_raw[0] = 1.
    # phi_raw[1] = M_rev[0].
    # ...
    # We want phi[k].
    # phi = phi_raw.flip(1)
    # phi[0] should be phi_raw[N]?
    # Yes.
    
    phi = phi_raw[:, :N].flip(1) # Take first N and flip?
    # phi_raw is 0..N.
    # phi_raw[0] is identity.
    # phi_raw[N] is total product.
    # We want phi[0]..phi[N-1].
    # phi[k] is product from k to N-1.
    # phi[N-1] is M_{N-1}.
    # phi_raw[1] is M_rev[0] = M_{N-1}.
    # So phi[N-1] = phi_raw[1].
    # phi[0] = M_0...M_{N-1} = phi_raw[N].
    # So phi = phi_raw[1:N+1].flip(1).
    
    phi = phi_raw[:, 1:].flip(1)

    # Combine
    det_T = theta[:, -1:] # (B, 1)
    theta_trunc = theta[:, :-1] # 0..N-1
    
    eps = 1e-18
    diag_inv = theta_trunc * phi / (det_T + eps)
    
    return diag_inv

def is_triton_available():
    return TRITON_AVAILABLE

def is_triton_available():
    return TRITON_AVAILABLE
