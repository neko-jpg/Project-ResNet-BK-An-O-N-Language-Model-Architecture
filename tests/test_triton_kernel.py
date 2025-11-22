
import torch
import pytest

# Import scan_op is hard because it uses triton.jit decorators which might fail if triton is not installed
# or if we are not in a GPU environment.
# However, for this test file meant for the user, we assume they might have Triton.
# If not, we skip.

try:
    import triton
    import triton.language as tl
    from src.kernels.bk_scan import bk_scan_fwd_kernel, bk_scan_bwd_kernel, scan_op
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
def test_bk_scan_fwd_small_sequence():
    """
    End-to-End test for Triton Kernel on a small sequence (N=4).
    Verifies M3 @ M2 @ M1 logic against PyTorch CPU reference.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    B = 1
    N = 4
    device = "cuda"

    # Inputs
    # Alpha, Beta are complex scalars defining the matrices
    # M_k = [alpha[k-1] beta[k-2]; 1 0]

    alpha = torch.randn(B, N, dtype=torch.complex64, device=device)
    beta = torch.randn(B, N, dtype=torch.complex64, device=device)

    # Run Triton Kernel wrapper (which we would need to import or simulate)
    # For this test to be self-contained or use the codebase, let's use the python wrapper if available
    from src.kernels.bk_scan import bk_scan_triton_forward

    # Run Triton
    theta_triton = bk_scan_triton_forward(alpha, beta) # (B, N+1)

    # Run PyTorch CPU Reference
    alpha_cpu = alpha.cpu()
    beta_cpu = beta.cpu()

    # Theta[0] = 1, Theta[1] = Alpha[0]
    # Theta[k] = Alpha[k-1]*Theta[k-1] + Beta[k-2]*Theta[k-2]

    theta_ref = torch.zeros(B, N+1, dtype=torch.complex64)
    theta_ref[:, 0] = 1.0
    theta_ref[:, 1] = alpha_cpu[:, 0]

    for k in range(2, N+1):
        # M_k applies to [Theta[k-1], Theta[k-2]]^T
        # Theta[k] is the top component of M_k @ ...
        # But simple recursion check:
        t_prev = theta_ref[:, k-1]
        t_prev2 = theta_ref[:, k-2]

        a_val = alpha_cpu[:, k-1]
        b_val = beta_cpu[:, k-2]

        theta_ref[:, k] = a_val * t_prev + b_val * t_prev2

    # Compare
    theta_triton_cpu = theta_triton.cpu()
    diff = (theta_triton_cpu - theta_ref).abs().max().item()

    print(f"Max difference Triton vs CPU Loop: {diff}")
    assert diff < 1e-4, f"Mismatch too high: {diff}"

if __name__ == "__main__":
    # Helper to run manually
    try:
        test_bk_scan_fwd_small_sequence()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed or skipped: {e}")
