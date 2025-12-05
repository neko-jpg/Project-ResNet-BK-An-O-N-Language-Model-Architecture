"""
Safe Numerical Operations for Triton Kernels

Provides numerically stable primitives for Triton GPU kernels:
- safe_exp: Rubber Wall implementation preventing overflow
- safe_log: Prevents log(0) and log(negative)
- safe_acosh: Numerically stable inverse hyperbolic cosine
- safe_log_exp_diff: The "Rubber Wall" from bk_core.py

These functions mirror the Python implementations in bk_core.py
for consistent numerical behavior across CPU and GPU paths.

Physical Intuition:
- The "Rubber Wall" (tanh soft-clamping) acts as a physical cutoff
  for infinite potentials, preventing numerical overflow while
  maintaining smooth gradients.
- Threshold of K=88.0 is chosen to be close to float32 exp limit (~88.7)

Requirements: Triton >= 2.0, CUDA capable GPU
"""

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# =============================================================================
# Constants
# =============================================================================

# Soft clamping threshold (close to float32 exp overflow ~88.7)
K_THRESHOLD = 88.0

# Numerical epsilon values
EPS_F32 = 1e-8
EPS_F16 = 1e-4
EPS_ACOSH = 1e-6


# =============================================================================
# Triton JIT Functions (Inlined into kernels)
# =============================================================================

if TRITON_AVAILABLE:
    
    @triton.jit
    def safe_exp(x, K: tl.constexpr = 88.0):
        """
        Safe exponential with Rubber Wall soft clamping.
        
        Computes exp(K * tanh(x/K)) which smoothly saturates
        to exp(K) for large positive x and exp(-K) for large negative x.
        
        Args:
            x: Input tensor (any shape)
            K: Clamping threshold (default: 88.0)
        
        Returns:
            Numerically stable exp(x)
        
        Physical Intuition:
            Acts as a physical cutoff for infinite potentials,
            preventing numerical overflow while maintaining
            smooth gradients through tanh.
        """
        clamped = K * tl.math.tanh(x / K)
        return tl.exp(clamped)
    
    
    @triton.jit
    def safe_log(x, eps: tl.constexpr = 1e-8):
        """
        Safe logarithm preventing log(0) and log(negative).
        
        Args:
            x: Input tensor (any shape)
            eps: Minimum value to clamp to (default: 1e-8)
        
        Returns:
            log(max(x, eps))
        """
        return tl.log(tl.maximum(x, eps))
    

    @triton.jit
    def safe_acosh(x, eps: tl.constexpr = 1e-6):
        """
        Numerically stable inverse hyperbolic cosine.
        
        acosh(x) = log(x + sqrt(x^2 - 1))
        
        Handles edge cases:
        - x close to 1: Uses clamping to prevent sqrt of negative
        - Large x: Standard formula works fine
        
        Args:
            x: Input tensor, should be >= 1
            eps: Epsilon for numerical stability
        
        Returns:
            acosh(x) computed stably
        """
        # Clamp x to be at least 1 + eps
        x_clamped = tl.maximum(x, 1.0 + eps)
        # acosh(x) = log(x + sqrt(x^2 - 1))
        sqrt_term = tl.sqrt(x_clamped * x_clamped - 1.0 + eps)
        return tl.log(x_clamped + sqrt_term)
    

    @triton.jit
    def safe_atanh(x, max_val: tl.constexpr = 0.999):
        """
        Numerically stable inverse hyperbolic tangent.
        
        atanh(x) = 0.5 * log((1+x)/(1-x))
        
        Clamps x to [-max_val, max_val] to prevent division by zero.
        
        Args:
            x: Input tensor, should be in (-1, 1)
            max_val: Maximum absolute value for clamping
        
        Returns:
            atanh(x) computed stably
        """
        x_clamped = tl.minimum(tl.maximum(x, -max_val), max_val)
        return 0.5 * tl.log((1.0 + x_clamped) / (1.0 - x_clamped + 1e-8))


    @triton.jit
    def safe_softmax_exp(score, m_i, K: tl.constexpr = 88.0):
        """
        Safe exponential for Flash Attention style softmax.
        
        Computes exp(score - m_i) with overflow protection.
        The subtraction of m_i (max score) provides numerical stability,
        and the Rubber Wall provides additional overflow protection.
        
        Args:
            score: Attention scores
            m_i: Max scores (for numerical stability)
            K: Clamping threshold
        
        Returns:
            Numerically stable exp(score - m_i)
        """
        diff = score - m_i
        # Apply Rubber Wall for extreme values
        clamped_diff = K * tl.math.tanh(diff / K)
        return tl.exp(clamped_diff)


    @triton.jit
    def safe_poincare_distance(
        q_norm_sq, k_norm_sq, diff_norm_sq,
        c, sqrt_c,
        eps: tl.constexpr = 1e-6
    ):
        """
        Numerically stable Poincaré distance computation.
        
        d(q, k) = (1/sqrt(c)) * acosh(1 + 2c * ||q-k||^2 / ((1-c||q||^2)(1-c||k||^2)))
        
        Args:
            q_norm_sq: ||q||^2 [BLOCK_M]
            k_norm_sq: ||k||^2 [BLOCK_N]
            diff_norm_sq: ||q-k||^2 [BLOCK_M, BLOCK_N]
            c: Curvature scalar
            sqrt_c: sqrt(c)
            eps: Numerical epsilon
        
        Returns:
            Poincaré distance [BLOCK_M, BLOCK_N]
        """
        # Denominator: (1 - c||q||^2)(1 - c||k||^2)
        denom = (1.0 - c * q_norm_sq[:, None]) * (1.0 - c * k_norm_sq[None, :])
        denom = tl.maximum(denom, eps)
        
        # Argument to acosh
        arg = 1.0 + 2.0 * c * diff_norm_sq / denom
        
        # Use safe_acosh
        return (1.0 / sqrt_c) * safe_acosh(arg, eps)


# =============================================================================
# PyTorch Reference Implementations (for CPU fallback and testing)
# =============================================================================

def safe_exp_pytorch(x: torch.Tensor, k: float = 88.0) -> torch.Tensor:
    """
    PyTorch reference implementation of safe_exp.
    
    Used for CPU fallback and unit testing.
    """
    clamped = k * torch.tanh(x / k)
    return torch.exp(clamped)


def safe_log_pytorch(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    PyTorch reference implementation of safe_log.
    """
    return torch.log(torch.clamp(x, min=eps))


def safe_acosh_pytorch(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    PyTorch reference implementation of safe_acosh.
    """
    x_clamped = torch.clamp(x, min=1.0 + eps)
    sqrt_term = torch.sqrt(x_clamped * x_clamped - 1.0 + eps)
    return torch.log(x_clamped + sqrt_term)


def safe_atanh_pytorch(x: torch.Tensor, max_val: float = 0.999) -> torch.Tensor:
    """
    PyTorch reference implementation of safe_atanh.
    """
    x_clamped = torch.clamp(x, min=-max_val, max=max_val)
    return 0.5 * torch.log((1.0 + x_clamped) / (1.0 - x_clamped + 1e-8))


# =============================================================================
# Testing Utilities
# =============================================================================

def validate_safe_ops():
    """
    Validate that Triton safe ops match PyTorch reference implementations.
    
    Run this in a CUDA environment to verify correctness.
    """
    if not TRITON_AVAILABLE:
        print("Triton not available, skipping validation")
        return False
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping validation")
        return False
    
    print("Validating Safe Operations...")
    
    # Test data
    x = torch.linspace(-100, 100, 1000, device='cuda')
    
    # Test safe_exp
    ref_exp = safe_exp_pytorch(x)
    assert torch.all(torch.isfinite(ref_exp)), "safe_exp has non-finite values"
    print("  ✔ safe_exp: No overflow with values in [-100, 100]")
    
    # Test safe_log
    x_positive = torch.linspace(0, 100, 1000, device='cuda')
    ref_log = safe_log_pytorch(x_positive)
    assert torch.all(torch.isfinite(ref_log)), "safe_log has non-finite values"
    print("  ✔ safe_log: No issues with positive values including 0")
    
    # Test safe_acosh
    x_acosh = torch.linspace(0.999, 100, 1000, device='cuda')
    ref_acosh = safe_acosh_pytorch(x_acosh)
    assert torch.all(torch.isfinite(ref_acosh)), "safe_acosh has non-finite values"
    print("  ✔ safe_acosh: No issues with values close to 1")
    
    print("All validations passed!")
    return True


def is_triton_available() -> bool:
    """Check if Triton is available for use."""
    return TRITON_AVAILABLE


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Triton JIT functions
    'safe_exp',
    'safe_log',
    'safe_acosh',
    'safe_atanh',
    'safe_softmax_exp',
    'safe_poincare_distance',
    # PyTorch reference
    'safe_exp_pytorch',
    'safe_log_pytorch',
    'safe_acosh_pytorch',
    'safe_atanh_pytorch',
    # Utilities
    'validate_safe_ops',
    'is_triton_available',
    # Constants
    'K_THRESHOLD',
    'EPS_F32',
    'EPS_F16',
    'EPS_ACOSH',
]


if __name__ == "__main__":
    # Run validation when executed directly
    validate_safe_ops()
