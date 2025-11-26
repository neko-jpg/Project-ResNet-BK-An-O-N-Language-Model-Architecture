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

    # --- Theta recursion (forward sweep) with scaling ---
    # We use log-scale to prevent overflow: theta_i = t_i * exp(l_i)
    # where |t_i| is close to 1.

    t_theta = [] # normalized values
    l_theta = [] # log scales

    # θ_0 = 1
    t_theta.append(torch.ones((), dtype=torch.complex128, device=device))
    l_theta.append(torch.zeros((), dtype=torch.float64, device=device))

    # θ_1 = a[0]
    val = a_shifted[0]
    scale = val.abs() + 1e-20
    t_theta.append(val / scale)
    l_theta.append(scale.log())

    for i in range(1, n):
        # θ_{i+1} = a_i * θ_i - c_{i-1} * b_{i-1} * θ_{i-1}
        # θ_{i+1} = e^{l_i} ( a_i * t_i - k * t_{i-1} * e^{l_{i-1} - l_i} )

        prev_t = t_theta[-1]
        prev_l = l_theta[-1]
        prev2_t = t_theta[-2]
        prev2_l = l_theta[-2]

        k = c_c[i-1] * b_c[i-1]

        log_diff = prev2_l - prev_l
        term2_scale = torch.exp(log_diff)

        val = a_shifted[i] * prev_t - k * prev2_t * term2_scale

        scale = val.abs() + 1e-20
        t_theta.append(val / scale)
        l_theta.append(prev_l + scale.log())

    t_theta_stack = torch.stack(t_theta)
    l_theta_stack = torch.stack(l_theta)

    # Determinant is the last theta
    t_det = t_theta_stack[-1]
    l_det = l_theta_stack[-1]

    # --- Phi recursion (backward sweep) with scaling ---
    t_phi = [torch.zeros((), dtype=torch.complex128, device=device) for _ in range(n)]
    l_phi = [torch.zeros((), dtype=torch.float64, device=device) for _ in range(n)]

    # phi_{n-1} = 1
    t_phi[n-1] = torch.ones((), dtype=torch.complex128, device=device)
    l_phi[n-1] = torch.zeros((), dtype=torch.float64, device=device)

    if n > 1:
        # phi_{n-2} = a_{n-1}
        val = a_shifted[-1]
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
            term2_scale = torch.exp(log_diff)

            val = a_shifted[i+1] * prev_t - k * prev2_t * term2_scale

            scale = val.abs() + 1e-20
            t_phi[i] = val / scale
            l_phi[i] = prev_l + scale.log()

    t_phi_stack = torch.stack(t_phi)
    l_phi_stack = torch.stack(l_phi)

    # --- Combine results ---
    t_num = t_theta_stack[:-1] * t_phi_stack
    l_num = l_theta_stack[:-1] + l_phi_stack

    log_mag_total = l_num - l_det

    # Clamp log magnitude to avoid overflow in exp
    # (e.g., > 50.0 leads to > 1e21 which is huge but finite)
    log_mag_total = torch.clamp(log_mag_total, max=50.0)

    eps = torch.tensor(1e-18, dtype=torch.complex128, device=device)
    diag_inv = (t_num / (t_det + eps)) * torch.exp(log_mag_total.to(torch.complex128))

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
    
    Gradient blending:
      - Theoretical: dG/dv = -G²
      - Hypothesis-7: dL/dv ~ -(dL/dG) / G²
      - Hybrid: dL/dv = (1-α)*theoretical + α*hypothesis7
    
    Supports both PyTorch (vmap) and Triton implementations.
    """
    GRAD_BLEND = 0.5  # 0.0 = pure theoretical, 1.0 = pure hypothesis-7
    USE_TRITON = None  # Auto-detect on first use

    @staticmethod
    def forward(ctx, he_diag, h0_super, h0_sub, z, use_triton=None):
        """
        Forward pass: compute G_ii features.
        
        Args:
            he_diag: (B, N) effective Hamiltonian diagonal
            h0_super: (B, N-1) super-diagonal
            h0_sub: (B, N-1) sub-diagonal
            z: complex scalar shift
            use_triton: bool or None - force Triton on/off, None=auto-detect
        
        Returns:
            features: (B, N, 2) [real(G_ii), imag(G_ii)]
        """
        # --- Input Sanitization ---
        if not torch.isfinite(he_diag).all():
            msg = "BKCoreFunction forward: 'he_diag' contains NaN or Inf."
            warnings.warn(msg, RuntimeWarning)
            logger.warning(msg)
            # Attempt to sanitize
            he_diag = torch.nan_to_num(he_diag, nan=0.0, posinf=100.0, neginf=-100.0)

        # Auto-detect Triton availability on first use
        if use_triton is None:
            if BKCoreFunction.USE_TRITON is None:
                try:
                    from src.kernels.bk_scan import is_triton_available
                    BKCoreFunction.USE_TRITON = is_triton_available()
                    if BKCoreFunction.USE_TRITON:
                        pass # logger.info("BK-Core: Triton acceleration enabled")
                except Exception:
                    BKCoreFunction.USE_TRITON = False
            use_triton = False
        
        # Strict Triton implementation (No Fallback)
        if use_triton:
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
    def backward(ctx, grad_output_features):
        """
        Backward pass: hybrid analytic gradient.
        
        Args:
            grad_output_features: (B, N, 2) gradient w.r.t. output features
        
        Returns:
            grad_he_diag: (B, N) gradient w.r.t. effective Hamiltonian diagonal
        """
        (G_ii,) = ctx.saved_tensors  # (B, N) complex
        
        # Check if incoming gradient is valid
        if not torch.isfinite(grad_output_features).all():
             msg = "BKCoreFunction backward: Incoming 'grad_output_features' contains NaN or Inf."
             warnings.warn(msg, RuntimeWarning)
             logger.warning(msg)
             grad_output_features = torch.nan_to_num(grad_output_features)

        # dL/dG = dL/dRe(G) + i*dL/dIm(G)
        grad_G = torch.complex(
            grad_output_features[..., 0],
            grad_output_features[..., 1],
        )
        # We need dL/dv = Re( grad_G.conj() * dG/dv )
        # dG/dv = -G^2
        # So dL/dv = - Re( grad_G.conj() * G^2 )
        grad_G_conj = grad_G.conj()

        # --- Compute G² and 1/G² safely ---
        G_sq = G_ii ** 2

        # Stabilize denominator for 1/G² (preserve phase, clamp magnitude)
        denom = G_sq
        denom_mag = denom.abs()
        min_denom = 1e-3  # Below this, 1/G² explodes
        denom = torch.where(
            denom_mag < min_denom,
            denom / (denom_mag + 1e-9) * min_denom,
            denom,
        )

        # --- Theoretical gradient: dG/dv = -G² ---
        grad_v_analytic = -(grad_G_conj * G_sq).real

        # --- Hypothesis-7 gradient: inverse square type ---
        # Heuristic: Scale by inverse of G^2 (consistency with conjugate logic)
        grad_v_h7 = -(grad_G_conj / (denom + 1e-6)).real

        # --- Hybrid blend ---
        alpha = BKCoreFunction.GRAD_BLEND
        grad_v = (1.0 - alpha) * grad_v_analytic + alpha * grad_v_h7

        # --- Numerical safety ---
        # Check for anomalies in calculated gradient
        if not torch.isfinite(grad_v).all():
            msg = "BKCoreFunction backward: Calculated 'grad_v' contains NaN/Inf."
            warnings.warn(msg, RuntimeWarning)
            logger.warning(msg)
            grad_v = torch.nan_to_num(grad_v)

        # Clamp and warn if clamping is active (significant clipping)
        max_grad_norm = 1000.0
        if grad_v.abs().max() > max_grad_norm:
             # Warn only if it's significantly over (e.g. > 10000) or just debug log
             # We clip at 1000.0
             # logger.debug(f"BKCoreFunction backward: Clipping gradient magnitude from {grad_v.abs().max()} to {max_grad_norm}")
             pass

        grad_v = torch.clamp(grad_v, -max_grad_norm, max_grad_norm)

        grad_he_diag = grad_v.to(torch.float32)

        # No gradients for h0_super, h0_sub, z, use_triton
        return grad_he_diag, None, None, None, None



def set_triton_mode(enabled: bool):
    """
    Globally enable or disable Triton acceleration for BK-Core.
    
    Args:
        enabled: True to use Triton (if available), False to use PyTorch vmap
    
    Example:
        >>> from src.models.bk_core import set_triton_mode
        >>> set_triton_mode(True)  # Enable Triton
        >>> set_triton_mode(False)  # Disable Triton (use PyTorch)
    """
    BKCoreFunction.USE_TRITON = enabled
    if enabled:
        try:
            from src.kernels.bk_scan import is_triton_available
            if not is_triton_available():
                warnings.warn(
                    "Triton is not available. BK-Core will use PyTorch implementation.",
                    UserWarning
                )
                BKCoreFunction.USE_TRITON = False
        except Exception:
            warnings.warn(
                "Failed to import Triton. BK-Core will use PyTorch implementation.",
                UserWarning
            )
            BKCoreFunction.USE_TRITON = False


def get_triton_mode() -> bool:
    """
    Check if Triton acceleration is currently enabled.
    
    Returns:
        True if Triton is enabled, False otherwise
    """
    if BKCoreFunction.USE_TRITON is None:
        # Auto-detect
        try:
            from src.kernels.bk_scan import is_triton_available
            return is_triton_available()
        except Exception:
            return False
    return BKCoreFunction.USE_TRITON
