"""
BK-Core Triton Kernel: O(N) Parallel Associative Scan

Implements parallel associative scan for computing diag((H - zI)^-1) using Triton.
Replaces serial loop with Hillis-Steele parallel scan for GPU efficiency.

Algorithm:
- Theta/Phi recursions are mapped to 2x2 matrix multiplications.
- Parallel associative scan (prefix product) computes the state transition matrices.
- Supports chunked processing for N > BLOCK_SIZE.
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Complex Number & Matrix Utilities
# ============================================================================

@triton.jit
def complex_mul(r1, i1, r2, i2):
    """(r1 + i1*j) * (r2 + i2*j)"""
    return r1 * r2 - i1 * i2, r1 * i2 + i1 * r2

@triton.jit
def complex_mat_mul_2x2(
    a11r, a11i, a12r, a12i, a21r, a21i, a22r, a22i,
    b11r, b11i, b12r, b12i, b21r, b21i, b22r, b22i,
):
    """C = A * B (2x2 complex matrices)"""
    # c11 = a11*b11 + a12*b21
    t1r, t1i = complex_mul(a11r, a11i, b11r, b11i)
    t2r, t2i = complex_mul(a12r, a12i, b21r, b21i)
    c11r, c11i = t1r + t2r, t1i + t2i
    
    # c12 = a11*b12 + a12*b22
    t1r, t1i = complex_mul(a11r, a11i, b12r, b12i)
    t2r, t2i = complex_mul(a12r, a12i, b22r, b22i)
    c12r, c12i = t1r + t2r, t1i + t2i
    
    # c21 = a21*b11 + a22*b21
    t1r, t1i = complex_mul(a21r, a21i, b11r, b11i)
    t2r, t2i = complex_mul(a22r, a22i, b21r, b21i)
    c21r, c21i = t1r + t2r, t1i + t2i
    
    # c22 = a21*b12 + a22*b22
    t1r, t1i = complex_mul(a21r, a21i, b12r, b12i)
    t2r, t2i = complex_mul(a22r, a22i, b22r, b22i)
    c22r, c22i = t1r + t2r, t1i + t2i
    
    return c11r, c11i, c12r, c12i, c21r, c21i, c22r, c22i

@triton.jit
def scan_op(
    # Accumulator (Left) - 8 components
    a11r, a11i, a12r, a12i, a21r, a21i, a22r, a22i,
    # Value (Right) - 8 components
    b11r, b11i, b12r, b12i, b21r, b21i, b22r, b22i
):
    """
    Associative combine function for scan.
    We want the sequence M1, M2, M3 to result in M3 @ M2 @ M1.
    Standard scan computes Acc @ Val? Or Val @ Acc?
    Triton associative_scan typically sums: x0, x0+x1.
    If we want M1, M2@M1, M3@M2@M1.
    Then new accumulator = NewVal @ OldAcc.
    """
    return complex_mat_mul_2x2(
        b11r, b11i, b12r, b12i, b21r, b21i, b22r, b22i, # Matrix B (New Value)
        a11r, a11i, a12r, a12i, a21r, a21i, a22r, a22i  # Matrix A (Accumulator)
    )


# ============================================================================
# Forward Scan Kernel (Parallelized)
# ============================================================================

@triton.jit
def bk_scan_fwd_kernel(
    # Input pointers
    alpha_r_ptr, alpha_i_ptr,
    beta_r_ptr, beta_i_ptr,
    # Output pointers
    theta_r_ptr, theta_i_ptr,
    # Dimensions
    B, N,
    # Strides
    stride_b_alpha, stride_n_alpha,
    stride_b_beta, stride_n_beta,
    stride_b_theta, stride_n_theta,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Parallel Forward Scan for Theta recursion.
    Theta[i] = Alpha[i-1]*Theta[i-1] + Beta[i-2]*Theta[i-2]
    """
    pid = tl.program_id(0)

    # Pointers for this batch
    alpha_r = alpha_r_ptr + pid * stride_b_alpha
    alpha_i = alpha_i_ptr + pid * stride_b_alpha
    beta_r = beta_r_ptr + pid * stride_b_beta
    beta_i = beta_i_ptr + pid * stride_b_beta
    theta_r = theta_r_ptr + pid * stride_b_theta
    theta_i = theta_i_ptr + pid * stride_b_theta
    
    # Initial State: Theta[-1]=0, Theta[0]=1
    # Matrix form propagates (Theta[k], Theta[k-1]).
    # Initial vector v0 = [1, 0] (for k=0).
    # Wait, recurrence is for k >= 2 (indices 2..N).
    # Theta[0] = 1, Theta[1] = Alpha[0].
    # So initial state at k=1 is v1 = [Theta[1], Theta[0]] = [Alpha[0], 1].
    
    # Initialize Theta[0] and Theta[1] directly
    tl.store(theta_r, 1.0)
    tl.store(theta_i, 0.0)
    
    if N > 0:
        a0r = tl.load(alpha_r)
        a0i = tl.load(alpha_i)
        tl.store(theta_r + stride_n_theta, a0r)
        tl.store(theta_i + stride_n_theta, a0i)

        # Current state vector [Theta_cur, Theta_prev]
        # At start of loop (k=2), cur=Theta[1], prev=Theta[0]
        cur_r, cur_i = a0r, a0i
        prev_r, prev_i = 1.0, 0.0

    # Loop over chunks for k = 2 to N
    # k goes from 2 to N.
    # Logic: We compute Theta[k].
    # M_k uses alpha[k-1], beta[k-2].
    
    for off in range(2, N + 1, BLOCK_SIZE):
        ks = off + tl.arange(0, BLOCK_SIZE)
        mask = ks <= N # k goes up to N (inclusive for theta, but alpha is length N)
        # Note: Theta has length N+1. Indices 0..N.
        # We compute indices 2..N.

        # Load params for M_k
        # alpha index: k-1
        # beta index: k-2

        # Mask check for loads
        load_mask = ks <= N

        # Load Alpha[k-1]
        a_idx = ks - 1
        ar = tl.load(alpha_r + a_idx * stride_n_alpha, mask=load_mask, other=0.0)
        ai = tl.load(alpha_i + a_idx * stride_n_alpha, mask=load_mask, other=0.0)

        # Load Beta[k-2]
        b_idx = ks - 2
        br = tl.load(beta_r + b_idx * stride_n_beta, mask=load_mask, other=0.0)
        bi = tl.load(beta_i + b_idx * stride_n_beta, mask=load_mask, other=0.0)

        # Construct Matrices M_k
        # [ alpha  beta ]
        # [   1      0  ]
        m11r, m11i = ar, ai
        m12r, m12i = br, bi
        m21r, m21i = 1.0, 0.0
        m22r, m22i = 0.0, 0.0

        # Apply mask to identity for out-of-bounds (so they don't affect scan)
        # Identity matrix: [1 0; 0 1]
        is_oob = ks > N
        m11r = tl.where(is_oob, 1.0, m11r)
        m11i = tl.where(is_oob, 0.0, m11i)
        m12r = tl.where(is_oob, 0.0, m12r)
        m12i = tl.where(is_oob, 0.0, m12i)
        m21r = tl.where(is_oob, 0.0, m21r) # Was 1.0
        m21i = tl.where(is_oob, 0.0, m21i)
        m22r = tl.where(is_oob, 1.0, m22r) # Was 0.0

        # Run Parallel Scan
        p11r, p11i, p12r, p12i, p21r, p21i, p22r, p22i = tl.associative_scan(
            (m11r, m11i, m12r, m12i, m21r, m21i, m22r, m22i),
            0, # Axis
            scan_op
        )

        # Multiply prefix matrices P_k by current state vector v_{start-1}
        # v_new = P_k @ v_prev
        # v_prev = [cur, prev]

        # theta_new = p11*cur + p12*prev
        t1r, t1i = complex_mul(p11r, p11i, cur_r, cur_i)
        t2r, t2i = complex_mul(p12r, p12i, prev_r, prev_i)
        theta_new_r = t1r + t2r
        theta_new_i = t1i + t2i

        # Store result
        tl.store(theta_r + ks * stride_n_theta, theta_new_r, mask=load_mask)
        tl.store(theta_i + ks * stride_n_theta, theta_new_i, mask=load_mask)
        
        # Update running state for next chunk
        # Last element of this chunk becomes the new [cur, prev]
        # But wait, P_k gives [Theta_k, Theta_{k-1}].
        # So we can just extract the last valid state.
        
        # We need the state at index `off + BLOCK_SIZE - 1` (last computed)
        # But if `N` < end of chunk, we need state at `N`.
        # However, loop continues? No, masking handles valid range.
        # But for next chunk, we need the state from the last VALID index.
        
        # Optimization: Just read the last thread's result?
        # Since we padded with Identity, the last thread in block (if oob) will just propagate.
        # So the last thread (index BLOCK_SIZE-1) holds the state after all valid updates.
        # Yes!
        
        # Get last thread's P matrix components
        last_idx = BLOCK_SIZE - 1
        
        # Last P matrix
        L11r = tl.load(p11r + last_idx * 0) # broadcasting workaround if scalar? No, p11r is tensor.
        # Actually p11r is a tensor of size BLOCK_SIZE.
        # We can't index it like a pointer easily in Triton unless we shuffle?
        # Triton `associative_scan` returns tensor.
        # We want the last element.
        # We can simply re-compute the multiplication for the last element?
        # Or use a reduction?
        # Or `tl.sum` is not appropriate.
        # Actually, since all threads have `p...`, we can just take the value from the last thread.
        # But threads cannot read other threads' registers directly without shuffles.
        
        # Trick: Store to shared memory or re-calculate?
        # Or: output is stored to global memory. Load it back?
        # Loading from global is safe (barrier needed?).
        # Triton doesn't expose barrier easily within `range` loop unless implied.
        
        # Better: Update state using the last stored value in global memory.
        # This acts as a barrier across chunks (implicitly, as we loop).

        # Wait, if we use `tl.load` from global memory we just wrote, we need consistency.
        # But within the same kernel, global memory visibility?
        # Usually fine if same thread reads, but we need the result of the LAST thread.
        # This requires communication.

        # Alternative: The running state is scalar variables `cur_r`, `cur_i`.
        # We need to update them.
        # Since we can't extract from `p` tensor easily (it's distributed?),
        # actually `associative_scan` result is distributed.
        # We need to broadcast the last element to all threads?
        # No, only the `main` control flow needs it for the next loop iteration.
        # But Triton is SPMD. ALL threads execute the loop.
        # So ALL threads need the new `cur, prev`.

        # How to get the last element of a scan to all threads?
        # In Triton, we can assume the block fits in SM?
        # We can't use `permute`?

        # Safe fallback: Since we stored to global memory `theta`, we can LOAD it back.
        # To get state for next chunk (start index `next_off`), we need `theta[next_off-1]` and `theta[next_off-2]`.
        # `next_off` is `off + BLOCK`.
        # So we load `theta[off + BLOCK - 1]` and `theta[off + BLOCK - 2]`.
        # But check bounds.

        last_k = off + BLOCK_SIZE - 1
        if last_k > N:
            last_k = N

        # Determine the last valid k processed in this chunk
        # Actually, simpler:
        # We just need `theta` at `last_k` and `last_k-1` to start next chunk?
        # No, next chunk starts at `last_k + 1`.
        # So we need `theta[last_k]` and `theta[last_k-1]`.

        # Barrier to ensure stores are visible?
        tl.debug_barrier()

        # Load state for next iteration
        cur_r = tl.load(theta_r + last_k * stride_n_theta)
        cur_i = tl.load(theta_i + last_k * stride_n_theta)
        prev_r = tl.load(theta_r + (last_k - 1) * stride_n_theta)
        prev_i = tl.load(theta_i + (last_k - 1) * stride_n_theta)

        tl.debug_barrier()


# ============================================================================
# Backward Scan Kernel (Parallelized)
# ============================================================================

@triton.jit
def bk_scan_bwd_kernel(
    # Input pointers
    alpha_r_ptr, alpha_i_ptr,
    beta_r_ptr, beta_i_ptr,
    # Output pointers
    phi_r_ptr, phi_i_ptr,
    # Dimensions
    B, N,
    # Strides
    stride_b_alpha, stride_n_alpha,
    stride_b_beta, stride_n_beta,
    stride_b_phi, stride_n_phi,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Parallel Backward Scan for Phi recursion.
    Phi[i] = Alpha[i+1]*Phi[i+1] + Beta[i]*Phi[i+2]
    """
    pid = tl.program_id(0)
    
    # Pointers
    alpha_r = alpha_r_ptr + pid * stride_b_alpha
    alpha_i = alpha_i_ptr + pid * stride_b_alpha
    beta_r = beta_r_ptr + pid * stride_b_beta
    beta_i = beta_i_ptr + pid * stride_b_beta
    phi_r = phi_r_ptr + pid * stride_b_phi
    phi_i = phi_i_ptr + pid * stride_b_phi
    
    # Initialize Phi[N-1]=1, Phi[N-2]=Alpha[N-1]
    if N > 0:
        tl.store(phi_r + (N - 1) * stride_n_phi, 1.0)
        tl.store(phi_i + (N - 1) * stride_n_phi, 0.0)
    
    if N > 1:
        # Alpha[N-1]
        a_last_r = tl.load(alpha_r + (N - 1) * stride_n_alpha)
        a_last_i = tl.load(alpha_i + (N - 1) * stride_n_alpha)
        tl.store(phi_r + (N - 2) * stride_n_phi, a_last_r)
        tl.store(phi_i + (N - 2) * stride_n_phi, a_last_i)

        cur_r, cur_i = a_last_r, a_last_i
        prev_r, prev_i = 1.0, 0.0

    # Loop over chunks backwards from N-3 down to 0
    # Logic: We compute Phi[i] for i = N-3 ... 0
    # Matrix M_i transitions from [Phi[i+1], Phi[i+2]] to [Phi[i], Phi[i+1]].
    # M_i = [ Alpha[i+1]  Beta[i] ]
    #       [     1         0     ]
    
    # We process chunks of `i`.
    # To use parallel scan, we need to linearize the chunk.
    # Chunk indices: start_i ... start_i - BLOCK + 1
    # We map thread k to index `start_i - k`.

    for start_i in range(N - 3, -1, -BLOCK_SIZE):
        # Threads 0..BLOCK-1 handle indices `start_i - k`
        ks = tl.arange(0, BLOCK_SIZE)
        indices = start_i - ks

        mask = indices >= 0

        # Load M_i params
        # i = indices
        # Alpha[i+1], Beta[i]

        a_idx = indices + 1
        b_idx = indices

        ar = tl.load(alpha_r + a_idx * stride_n_alpha, mask=mask, other=0.0)
        ai = tl.load(alpha_i + a_idx * stride_n_alpha, mask=mask, other=0.0)
        br = tl.load(beta_r + b_idx * stride_n_beta, mask=mask, other=0.0)
        bi = tl.load(beta_i + b_idx * stride_n_beta, mask=mask, other=0.0)

        # Construct Matrices (Identity if masked)
        is_oob = indices < 0
        m11r = tl.where(is_oob, 1.0, ar)
        m11i = tl.where(is_oob, 0.0, ai)
        m12r = tl.where(is_oob, 0.0, br)
        m12i = tl.where(is_oob, 0.0, bi)
        m21r = tl.where(is_oob, 0.0, 1.0)
        m21i = tl.where(is_oob, 0.0, 0.0)
        m22r = tl.where(is_oob, 1.0, 0.0)

        # Run Scan
        # Scan gives product M_{i} ... M_{start_i}
        # Since we scan on `ks`, k=0 is start_i. k=1 is start_i-1.
        # Order: M_{start_i}, M_{start_i-1} ...
        # This matches the recursion direction (decreasing i).

        p11r, p11i, p12r, p12i, p21r, p21i, p22r, p22i = tl.associative_scan(
            (m11r, m11i, m12r, m12i, m21r, m21i, m22r, m22i),
            0,
            scan_op
        )

        # Apply to state
        t1r, t1i = complex_mul(p11r, p11i, cur_r, cur_i)
        t2r, t2i = complex_mul(p12r, p12i, prev_r, prev_i)
        phi_new_r = t1r + t2r
        phi_new_i = t1i + t2i
        
        # Store
        tl.store(phi_r + indices * stride_n_phi, phi_new_r, mask=mask)
        tl.store(phi_i + indices * stride_n_phi, phi_new_i, mask=mask)
        
        # Update State
        last_k = BLOCK_SIZE - 1
        # Find the last valid index processed
        # If chunk is full, it's `start_i - (BLOCK-1)`.
        # If partial (at the end of loop, indices -> 0), last valid is 0.
        
        tl.debug_barrier()
        
        # Determine next state index
        # We finished `indices[-1]`. Next state comes from `indices[-1]` and `indices[-1]+1`.
        # Which corresponds to stored `phi`.
        
        last_valid_idx = start_i - BLOCK_SIZE + 1
        if last_valid_idx < 0:
            last_valid_idx = 0

        cur_r = tl.load(phi_r + last_valid_idx * stride_n_phi)
        cur_i = tl.load(phi_i + last_valid_idx * stride_n_phi)
        # Prev is actually the one "before" current in recursion, which is i+1 (higher index)
        # In recursion: Phi[i] depends on Phi[i+1], Phi[i+2].
        # State was [Phi[i+1], Phi[i+2]].
        # After update, state is [Phi[i], Phi[i+1]].
        # So we load Phi[last] and Phi[last+1].
        
        prev_r = tl.load(phi_r + (last_valid_idx + 1) * stride_n_phi)
        prev_i = tl.load(phi_i + (last_valid_idx + 1) * stride_n_phi)
        
        tl.debug_barrier()


# ============================================================================
# Python Interface Functions
# ============================================================================

def bk_scan_triton_forward(alpha, beta):
    """
    Triton-accelerated forward scan for BK-Core.
    """
    B, N = alpha.shape
    device = alpha.device
    
    alpha_r = alpha.real.contiguous()
    alpha_i = alpha.imag.contiguous()
    beta_r = beta.real.contiguous()
    beta_i = beta.imag.contiguous()
    
    theta_r = torch.empty(B, N + 1, dtype=torch.float32, device=device)
    theta_i = torch.empty(B, N + 1, dtype=torch.float32, device=device)
    
    grid = (B,)
    # Choose optimal block size
    BLOCK_SIZE = 128
    if N >= 1024: BLOCK_SIZE = 1024
    elif N >= 512: BLOCK_SIZE = 512
    elif N >= 256: BLOCK_SIZE = 256
    
    bk_scan_fwd_kernel[grid](
        alpha_r, alpha_i,
        beta_r, beta_i,
        theta_r, theta_i,
        B, N,
        alpha_r.stride(0), alpha_r.stride(1),
        beta_r.stride(0), beta_r.stride(1),
        theta_r.stride(0), theta_r.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return torch.complex(theta_r, theta_i)


def bk_scan_triton_backward(alpha, beta, N):
    """
    Triton-accelerated backward scan for BK-Core.
    """
    B = alpha.shape[0]
    device = alpha.device
    
    alpha_r = alpha.real.contiguous()
    alpha_i = alpha.imag.contiguous()
    beta_r = beta.real.contiguous()
    beta_i = beta.imag.contiguous()
    
    phi_r = torch.empty(B, N, dtype=torch.float32, device=device)
    phi_i = torch.empty(B, N, dtype=torch.float32, device=device)
    
    grid = (B,)
    BLOCK_SIZE = 128
    if N >= 1024: BLOCK_SIZE = 1024
    elif N >= 512: BLOCK_SIZE = 512
    elif N >= 256: BLOCK_SIZE = 256
    
    bk_scan_bwd_kernel[grid](
        alpha_r, alpha_i,
        beta_r, beta_i,
        phi_r, phi_i,
        B, N,
        alpha_r.stride(0), alpha_r.stride(1),
        beta_r.stride(0), beta_r.stride(1),
        phi_r.stride(0), phi_r.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return torch.complex(phi_r, phi_i)


def bk_scan_triton(a, b, c, z):
    """
    Complete Triton-accelerated BK-Core computation.
    """
    B, N = a.shape
    device = a.device
    
    a_c = a.to(torch.complex64)
    b_c = b.to(torch.complex64)
    c_c = c.to(torch.complex64)
    z_c = torch.tensor(z, dtype=torch.complex64, device=device)
    
    alpha = a_c - z_c
    beta = -c_c * b_c
    
    theta = bk_scan_triton_forward(alpha, beta)
    phi = bk_scan_triton_backward(alpha, beta, N)
    
    det_T = theta[:, -1:]
    eps = 1e-18
    diag_inv = theta[:, :-1] * phi / (det_T + eps)
    
    return diag_inv

def is_triton_available():
    try:
        import triton
        return True
    except:
        return False
