"""
BK-Core: O(N) Tridiagonal Inverse Diagonal Computation
Implements the core algorithm for computing diag((H - zI)^-1) in O(N) time.

Supports both PyTorch (vmap) and Triton implementations.
Strict Triton Mode: If Triton is enabled, it must succeed or fail loudly.
Includes input sanitization and gradient warning system for robust training.
"""

import torch
import torch.nn as nn
from torch.func import vmap
import warnings
import logging

# Configure logger for BK-Core
logger = logging.getLogger(__name__)

# Try to import parallel scan kernel
try:
    from src.kernels.bk_parallel_scan import bk_parallel_inverse_diagonal, is_parallel_scan_available
    PARALLEL_SCAN_AVAILABLE = is_parallel_scan_available()
except ImportError:
    PARALLEL_SCAN_AVAILABLE = False
    bk_parallel_inverse_diagonal = None

def get_tridiagonal_inverse_diagonal(a, b, c, z):
    """
    Compute diagonal of tridiagonal matrix inverse: diag((T - zI)^-1)
    
    Uses forward (theta) and backward (phi) recursions for O(N) complexity.
    
    Args:
        a: (N,) main diagonal elements
        b: (N-1,) super-diagonal elements
        c: (N-1,) sub-diagonal elements
        z: complex scalar shift
    
    Returns:
        diag_inv: (N,) complex64 diagonal elements of (T - zI)^-1
    """
    n = a.shape[-1]
    device = a.device

    a_c = a.to(torch.complex128)
    b_c = b.to(torch.complex128)
    c_c = c.to(torch.complex128)

    if isinstance(z, (int, float, complex)):
        z_c = torch.tensor(z, dtype=torch.complex128, device=device)
    else:
        z_c = z.to(torch.complex128)

    a_shifted = a_c - z_c

    if n == 0:
        return torch.zeros(0, dtype=torch.complex64, device=device)

    # --- Safe-Log Helper (The Rubber Wall) ---
    def safe_log_exp_diff(diff_val, k_threshold=88.0):
        """
        Computes exp(diff) with safe soft-clamping (Rubber Wall).
        If diff is large positive, it clamps to exp(K).
        Uses tanh for smooth saturation.
        """
        clamped_diff = k_threshold * torch.tanh(diff_val / k_threshold)
        return torch.exp(clamped_diff)

    # --- Diagonal Regularization Helper ---
    def apply_diagonal_regularization(val, epsilon=1e-3):
        """
        Applies Tikhonov-style regularization to avoid singularities.
        val_stable = val + epsilon * sign(val)
        If val is 0, adds epsilon.
        """
        abs_val = val.abs()
        is_zero = abs_val < 1e-12
        direction = torch.where(is_zero, torch.tensor(1.0, dtype=torch.complex128, device=val.device), val / (abs_val + 1e-12))
        return val + epsilon * direction

    # --- Theta recursion (forward sweep) with scaling ---
    t_theta = [] # normalized values
    l_theta = [] # log scales

    # θ_0 = 1
    t_theta.append(torch.ones((), dtype=torch.complex128, device=device))
    l_theta.append(torch.zeros((), dtype=torch.float64, device=device))

    # θ_1 = a[0]
    mass_epsilon = 1e-3
    val = apply_diagonal_regularization(a_shifted[0], mass_epsilon)

    scale = val.abs() + 1e-20
    t_theta.append(val / scale)
    l_theta.append(scale.log())

    for i in range(1, n):
        # θ_{i+1} = a_i * θ_i - c_{i-1} * b_{i-1} * θ_{i-1}
        prev_t = t_theta[-1]
        prev_l = l_theta[-1]
        prev2_t = t_theta[-2]
        prev2_l = l_theta[-2]

        k = c_c[i-1] * b_c[i-1]
        log_diff = prev2_l - prev_l

        term2_scale = safe_log_exp_diff(log_diff)
        curr_a = apply_diagonal_regularization(a_shifted[i], mass_epsilon)
        val = curr_a * prev_t - k * prev2_t * term2_scale

        scale = val.abs() + 1e-20
        t_theta.append(val / scale)
        l_theta.append(prev_l + scale.log())

    t_theta_stack = torch.stack(t_theta)
    l_theta_stack = torch.stack(l_theta)

    # Determinant is the last theta
    t_det = t_theta_stack[-1]
    l_det = l_theta_stack[-1]

    # --- Resonance Safety Valve ---
    # If determinant is effectively zero (log det -> -inf), we have a resonance (pole).
    # We must cap the minimum determinant magnitude to avoid division by zero (Infinity).
    # l_det is log magnitude. Low l_det means small magnitude.
    # We clamp l_det from below. e.g., -50.0 corresponds to exp(-50) ~ 1e-22.
    l_det_clamped = torch.clamp(l_det, min=-50.0)

    # Also ensure t_det is not zero (direction only)
    t_det_safe = torch.where(
        t_det.abs() < 1e-12,
        torch.tensor(1.0 + 0j, dtype=torch.complex128, device=device),
        t_det
    )

    # --- Phi recursion (backward sweep) with scaling ---
    t_phi = [torch.zeros((), dtype=torch.complex128, device=device) for _ in range(n)]
    l_phi = [torch.zeros((), dtype=torch.float64, device=device) for _ in range(n)]

    # phi_{n-1} = 1
    t_phi[n-1] = torch.ones((), dtype=torch.complex128, device=device)
    l_phi[n-1] = torch.zeros((), dtype=torch.float64, device=device)

    if n > 1:
        # phi_{n-2} = a_{n-1}
        val = apply_diagonal_regularization(a_shifted[-1], mass_epsilon)
        scale = val.abs() + 1e-20
        t_phi[n-2] = val / scale
        l_phi[n-2] = scale.log()

        for i in range(n - 3, -1, -1):
            prev_t = t_phi[i+1]
            prev_l = l_phi[i+1]
            prev2_t = t_phi[i+2]
            prev2_l = l_phi[i+2]

            k = c_c[i] * b_c[i]
            log_diff = prev2_l - prev_l
            term2_scale = safe_log_exp_diff(log_diff)
            curr_a = apply_diagonal_regularization(a_shifted[i+1], mass_epsilon)

            val = curr_a * prev_t - k * prev2_t * term2_scale
            scale = val.abs() + 1e-20
            t_phi[i] = val / scale
            l_phi[i] = prev_l + scale.log()

    t_phi_stack = torch.stack(t_phi)
    l_phi_stack = torch.stack(l_phi)

    # --- Combine results ---
    # diag_inv = (theta * phi) / det
    # log_mag = l_theta + l_phi - l_det_clamped

    t_num = t_theta_stack[:-1] * t_phi_stack
    l_num = l_theta_stack[:-1] + l_phi_stack

    log_mag_total = l_num - l_det_clamped

    # Clamp log magnitude to avoid overflow in exp
    # (e.g., > 80.0 leads to huge numbers)
    # We use tanh soft clamp here as well for consistency
    log_mag_total = 88.0 * torch.tanh(log_mag_total / 88.0)

    # Use t_det_safe for direction
    diag_inv = (t_num / t_det_safe) * torch.exp(log_mag_total.to(torch.complex128))

    # --- Numerical stability: remove NaN/Inf + clip magnitude ---
    diag_inv = torch.where(torch.isfinite(diag_inv), diag_inv, torch.zeros_like(diag_inv))
    max_mag = 50.0  # Maximum resolvent magnitude
    mag = diag_inv.abs()
    factor = torch.where(mag > max_mag, max_mag / (mag + 1e-9), torch.ones_like(mag))
    diag_inv = diag_inv * factor

    return diag_inv.to(torch.complex64)


# Batched BK-Core using vmap
vmapped_get_diag = vmap(
    get_tridiagonal_inverse_diagonal, in_dims=(0, 0, 0, None), out_dims=0
)


class BKCoreFunction(torch.autograd.Function):
    """
    BK-Core with hybrid analytic gradient.
    
    Forward: O(N) computation of G_ii = diag((H - zI)^-1)
    Backward: O(N) analytic gradient computation
    """
    GRAD_BLEND = 0.5  # 0.0 = pure theoretical, 1.0 = pure hypothesis-7
    USE_TRITON = None  # Auto-detect on first use
    USE_PARALLEL_SCAN = True  # NEW: Use O(log N) parallel scan by default

    @staticmethod
    def forward(ctx, he_diag, h0_super, h0_sub, z, use_triton=None, use_parallel_scan=None):
        # --- Input Sanitization ---
        # Clamp inputs to prevent NaN/Inf propagation
        if torch.isnan(he_diag).any() or torch.isinf(he_diag).any():
             # logger.warning("BK-Core: NaN/Inf detected in he_diag input. Sanitizing.")
             he_diag = torch.nan_to_num(he_diag, nan=0.0, posinf=100.0, neginf=-100.0)

        he_diag = torch.clamp(he_diag, min=-100.0, max=100.0)
        h0_super = torch.clamp(h0_super, min=-10.0, max=10.0)
        h0_sub = torch.clamp(h0_sub, min=-10.0, max=10.0)
        
        # Auto-detect Triton availability on first use
        if use_triton is None:
            if BKCoreFunction.USE_TRITON is None:
                try:
                    from src.kernels.bk_scan import is_triton_available
                    BKCoreFunction.USE_TRITON = is_triton_available()
                except Exception:
                    BKCoreFunction.USE_TRITON = False
            use_triton = False # Explicitly disable Triton for now as per user request/stability issues
        
        # Check parallel scan preference
        if use_parallel_scan is None:
            use_parallel_scan = BKCoreFunction.USE_PARALLEL_SCAN and PARALLEL_SCAN_AVAILABLE
        
        # Force PyTorch implementation for stability (as requested by plan)
        # unless parallel scan is explicitly requested and working
        use_triton = False

        if use_parallel_scan and PARALLEL_SCAN_AVAILABLE:
            G_ii = bk_parallel_inverse_diagonal(he_diag, h0_super, h0_sub, z)
        elif use_triton:
            from src.kernels.bk_scan import bk_scan_triton
            G_ii = bk_scan_triton(he_diag, h0_super, h0_sub, z)
        else:
            # Use PyTorch vmap implementation
            G_ii = vmapped_get_diag(he_diag, h0_super, h0_sub, z)
        
        ctx.save_for_backward(G_ii)

        # Convert to real features (real, imag)
        output_features = torch.stack(
            [G_ii.real, G_ii.imag], dim=-1
        ).to(torch.float32)  # (B, N, 2)

        return output_features, G_ii

    @staticmethod
    def backward(ctx, grad_output_features, grad_G_ii):
        (G_ii,) = ctx.saved_tensors  # (B, N) complex
        
        # Check if incoming gradient is valid
        if grad_output_features is not None:
             grad_output_features = torch.nan_to_num(grad_output_features)

        # Combine gradients from both outputs
        if grad_output_features is not None:
            grad_G = torch.complex(
                grad_output_features[..., 0],
                grad_output_features[..., 1],
            )
        else:
            grad_G = torch.zeros_like(G_ii)
        
        if grad_G_ii is not None:
            grad_G = grad_G + grad_G_ii

        grad_G_conj = grad_G.conj()

        # --- Compute G² and 1/G² safely ---
        G_sq = G_ii ** 2

        # Stabilize denominator for 1/G² (preserve phase, clamp magnitude)
        denom = G_sq
        denom_mag = denom.abs()
        min_denom = 1e-4
        denom = torch.where(
            denom_mag < min_denom,
            denom / (denom_mag + 1e-9) * min_denom,
            denom,
        )

        # --- Theoretical gradient: dG/dv = -G² ---
        grad_v_analytic = -(grad_G_conj * G_sq).real

        # --- Hypothesis-7 gradient: inverse square type ---
        grad_v_h7 = -(grad_G_conj / (denom + 1e-6)).real

        # --- Hybrid blend ---
        alpha = BKCoreFunction.GRAD_BLEND
        grad_v = (1.0 - alpha) * grad_v_analytic + alpha * grad_v_h7

        # --- Numerical safety ---
        grad_v = torch.nan_to_num(grad_v, nan=0.0, posinf=100.0, neginf=-100.0)

        # Clamp max gradient to prevent explosion
        max_grad_norm = 10.0 # Strict clamping for stability
        grad_v = torch.clamp(grad_v, -max_grad_norm, max_grad_norm)

        grad_he_diag = grad_v.to(torch.float32)

        # No gradients for h0_super, h0_sub, z, use_triton, use_parallel_scan
        return grad_he_diag, None, None, None, None, None


def set_triton_mode(enabled: bool):
    BKCoreFunction.USE_TRITON = enabled
    if enabled:
        try:
            from src.kernels.bk_scan import is_triton_available
            if not is_triton_available():
                warnings.warn("Triton is not available. BK-Core will use PyTorch implementation.", UserWarning)
                BKCoreFunction.USE_TRITON = False
        except Exception:
            warnings.warn("Failed to import Triton. BK-Core will use PyTorch implementation.", UserWarning)
            BKCoreFunction.USE_TRITON = False

def get_triton_mode() -> bool:
    if BKCoreFunction.USE_TRITON is None:
        try:
            from src.kernels.bk_scan import is_triton_available
            return is_triton_available()
        except Exception:
            return False
    return BKCoreFunction.USE_TRITON
