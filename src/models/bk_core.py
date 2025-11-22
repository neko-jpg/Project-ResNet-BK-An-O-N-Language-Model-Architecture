"""
BK-Core: O(N) Tridiagonal Inverse Diagonal Computation
Implements the core algorithm for computing diag((H - zI)^-1) in O(N) time.

Supports both PyTorch (vmap) and Triton implementations with automatic fallback.
"""

import torch
import torch.nn as nn
from torch.func import vmap
import warnings


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
    z_c = z.to(torch.complex128)
    a_shifted = a_c - z_c

    if n == 0:
        return torch.zeros(0, dtype=torch.complex64, device=device)

    # --- Theta recursion (forward sweep) ---
    theta = []
    theta.append(torch.ones((), dtype=torch.complex128, device=device))  # θ_0
    theta.append(a_shifted[0])                                          # θ_1

    for i in range(1, n):
        theta.append(a_shifted[i] * theta[i] - c_c[i-1] * b_c[i-1] * theta[i-1])

    theta_stack = torch.stack(theta)
    det_T = theta_stack[-1]

    # --- Phi recursion (backward sweep) ---
    phi = [torch.zeros((), dtype=torch.complex128, device=device) for _ in range(n)]
    phi[n-1] = torch.ones((), dtype=torch.complex128, device=device)

    if n > 1:
        phi[n-2] = a_shifted[-1]
        for i in range(n - 3, -1, -1):
            phi[i] = a_shifted[i+1] * phi[i+1] - c_c[i] * b_c[i] * phi[i+2]

    phi_stack = torch.stack(phi)

    eps = torch.tensor(1e-18, dtype=torch.complex128, device=device)
    diag_inv = theta_stack[:-1] * phi_stack / (det_T + eps)

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
        # Auto-detect Triton availability on first use
        if use_triton is None:
            if BKCoreFunction.USE_TRITON is None:
                try:
                    from src.kernels.bk_scan import is_triton_available
                    BKCoreFunction.USE_TRITON = is_triton_available()
                    if BKCoreFunction.USE_TRITON:
                        print("BK-Core: Triton acceleration enabled")
                except Exception:
                    BKCoreFunction.USE_TRITON = False
            use_triton = BKCoreFunction.USE_TRITON
        
        # Try Triton implementation with fallback
        if use_triton:
            try:
                from src.kernels.bk_scan import bk_scan_triton
                G_ii = bk_scan_triton(he_diag, h0_super, h0_sub, z)

                # Double-check numerical stability (Task 3)
                if not torch.isfinite(G_ii).all():
                    raise ValueError("Triton kernel produced non-finite values (NaN/Inf)")

            except Exception as e:
                warnings.warn(
                    f"Triton kernel failed or unstable: {e}. Falling back to PyTorch implementation.",
                    UserWarning
                )
                G_ii = vmapped_get_diag(he_diag, h0_super, h0_sub, z)
        else:
            # Use PyTorch vmap implementation
            G_ii = vmapped_get_diag(he_diag, h0_super, h0_sub, z)
        
        ctx.save_for_backward(G_ii)

        # Convert to real features (real, imag)
        output_features = torch.stack(
            [G_ii.real, G_ii.imag], dim=-1
        ).to(torch.float32)  # (B, N, 2)

        return output_features

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
        
        # dL/dG = dL/dRe(G) + i*dL/dIm(G)
        grad_G = torch.complex(
            grad_output_features[..., 0],
            grad_output_features[..., 1],
        )

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
        grad_v_analytic = -(grad_G * G_sq).real

        # --- Hypothesis-7 gradient: inverse square type ---
        grad_v_h7 = -(grad_G / (denom + 1e-6)).real

        # --- Hybrid blend ---
        alpha = BKCoreFunction.GRAD_BLEND
        grad_v = (1.0 - alpha) * grad_v_analytic + alpha * grad_v_h7

        # --- Numerical safety ---
        grad_v = torch.where(torch.isfinite(grad_v), grad_v, torch.zeros_like(grad_v))
        grad_v = torch.clamp(grad_v, -1000.0, 1000.0)

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
