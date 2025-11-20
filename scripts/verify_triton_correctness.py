"""
BK-Core Triton Numerical Correctness Verification

Verifies that Triton implementation produces numerically equivalent results
to the reference PyTorch implementation.

Test conditions:
- Sequence lengths: [512, 1024, 2048, 4096]
- Batch sizes: [1, 4, 8, 16]
- Random inputs: 100 trials
- Complex number verification: separate real/imag error checks

Success criteria:
- MSE error < 1e-6 for all test cases
- NaN occurrence rate: 0% (100 random trials)
"""

import torch
import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.bk_core import BKCoreFunction, set_triton_mode


def compute_mse(output1, output2):
    """
    Compute Mean Squared Error between two outputs.
    
    Args:
        output1: (B, N, 2) first output
        output2: (B, N, 2) second output
    
    Returns:
        mse_real: MSE for real part
        mse_imag: MSE for imaginary part
        mse_total: Total MSE
    """
    diff = output1 - output2
    mse_real = (diff[..., 0] ** 2).mean().item()
    mse_imag = (diff[..., 1] ** 2).mean().item()
    mse_total = (diff ** 2).mean().item()
    return mse_real, mse_imag, mse_total


def check_nan_rate(output):
    """
    Check NaN occurrence rate in output.
    
    Args:
        output: (B, N, 2) output tensor
    
    Returns:
        nan_rate: Fraction of NaN values
    """
    total_elements = output.numel()
    nan_count = torch.isnan(output).sum().item()
    return nan_count / total_elements


def verify_single_case(batch_size, seq_len, device="cuda", seed=None):
    """
    Verify correctness for a single test case.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        device: Device to run on
        seed: Random seed (optional)
    
    Returns:
        results: Dictionary with verification results
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate test data
    he_diag = torch.randn(batch_size, seq_len, device=device)
    h0_super = torch.randn(batch_size, seq_len - 1, device=device)
    h0_sub = torch.randn(batch_size, seq_len - 1, device=device)
    z = torch.tensor(0.1 + 0.1j, dtype=torch.complex64, device=device)
    
    # PyTorch implementation
    set_triton_mode(False)
    output_pytorch = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, False)
    
    # Triton implementation
    set_triton_mode(True)
    output_triton = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, True)
    
    # Compute errors
    mse_real, mse_imag, mse_total = compute_mse(output_pytorch, output_triton)
    
    # Check NaN rates
    nan_rate_pytorch = check_nan_rate(output_pytorch)
    nan_rate_triton = check_nan_rate(output_triton)
    
    results = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "mse_real": mse_real,
        "mse_imag": mse_imag,
        "mse_total": mse_total,
        "nan_rate_pytorch": nan_rate_pytorch,
        "nan_rate_triton": nan_rate_triton,
        "passed": mse_total < 1e-6 and nan_rate_triton == 0.0,
    }
    
    return results


def verify_all_configurations(device="cuda"):
    """
    Verify correctness across all test configurations.
    
    Args:
        device: Device to run on
    
    Returns:
        all_results: List of results for all configurations
        summary: Summary statistics
    """
    print("BK-Core Triton Numerical Correctness Verification")
    print("=" * 70)
    print()
    
    # Check device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Check Triton availability
    try:
        from src.kernels.bk_scan import is_triton_available
        if not is_triton_available():
            print("ERROR: Triton not available")
            return [], {"triton_available": False}
    except Exception as e:
        print(f"ERROR: Failed to import Triton: {e}")
        return [], {"triton_available": False}
    
    # Test configurations
    seq_lengths = [512, 1024, 2048, 4096]
    batch_sizes = [1, 4, 8, 16]
    
    all_results = []
    passed_count = 0
    total_count = 0
    
    print("Testing different configurations...")
    print()
    
    for seq_len in seq_lengths:
        for batch_size in batch_sizes:
            print(f"Testing: Batch={batch_size}, SeqLen={seq_len}...", end=" ")
            
            try:
                results = verify_single_case(batch_size, seq_len, device, seed=42)
                all_results.append(results)
                
                if results["passed"]:
                    print(f"✓ PASS (MSE: {results['mse_total']:.2e})")
                    passed_count += 1
                else:
                    print(f"✗ FAIL (MSE: {results['mse_total']:.2e})")
                
                total_count += 1
                
            except Exception as e:
                print(f"✗ ERROR: {e}")
                all_results.append({
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "error": str(e),
                    "passed": False,
                })
                total_count += 1
    
    print()
    print("-" * 70)
    print()
    
    # ========================================================================
    # Random Input NaN Test (100 trials)
    # ========================================================================
    print("Testing NaN occurrence rate (100 random trials)...")
    print()
    
    nan_trials = 100
    nan_count_pytorch = 0
    nan_count_triton = 0
    
    for trial in range(nan_trials):
        # Random configuration
        batch_size = np.random.choice([1, 4, 8, 16])
        seq_len = np.random.choice([512, 1024, 2048])
        
        results = verify_single_case(batch_size, seq_len, device, seed=trial)
        
        if results["nan_rate_pytorch"] > 0:
            nan_count_pytorch += 1
        if results["nan_rate_triton"] > 0:
            nan_count_triton += 1
        
        if (trial + 1) % 20 == 0:
            print(f"  Progress: {trial + 1}/{nan_trials}")
    
    nan_rate_pytorch_pct = (nan_count_pytorch / nan_trials) * 100
    nan_rate_triton_pct = (nan_count_triton / nan_trials) * 100
    
    print()
    print(f"NaN occurrence rate:")
    print(f"  PyTorch: {nan_rate_pytorch_pct:.1f}% ({nan_count_pytorch}/{nan_trials} trials)")
    print(f"  Triton:  {nan_rate_triton_pct:.1f}% ({nan_count_triton}/{nan_trials} trials)")
    print()
    
    # ========================================================================
    # Summary
    # ========================================================================
    summary = {
        "triton_available": True,
        "total_tests": total_count,
        "passed_tests": passed_count,
        "failed_tests": total_count - passed_count,
        "pass_rate": passed_count / total_count if total_count > 0 else 0.0,
        "nan_rate_pytorch": nan_rate_pytorch_pct,
        "nan_rate_triton": nan_rate_triton_pct,
        "success": (
            passed_count == total_count and
            nan_rate_triton_pct == 0.0
        ),
    }
    
    # Compute error statistics
    if all_results:
        valid_results = [r for r in all_results if "mse_total" in r]
        if valid_results:
            mse_values = [r["mse_total"] for r in valid_results]
            summary["max_mse"] = max(mse_values)
            summary["mean_mse"] = sum(mse_values) / len(mse_values)
    
    return all_results, summary


def main():
    """Run verification and report results."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    all_results, summary = verify_all_configurations(device)
    
    if not summary.get("triton_available", False):
        print("Triton not available, cannot verify")
        sys.exit(1)
    
    # Print summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Configuration tests: {summary['passed_tests']}/{summary['total_tests']} passed")
    print(f"Pass rate: {summary['pass_rate'] * 100:.1f}%")
    
    if "max_mse" in summary:
        print(f"Maximum MSE: {summary['max_mse']:.2e}")
        print(f"Mean MSE: {summary['mean_mse']:.2e}")
    
    print()
    print(f"NaN occurrence rate:")
    print(f"  PyTorch: {summary['nan_rate_pytorch']:.1f}%")
    print(f"  Triton:  {summary['nan_rate_triton']:.1f}%")
    print()
    
    # Success criteria
    success_criteria = [
        ("All configuration tests pass", summary['passed_tests'] == summary['total_tests']),
        ("MSE < 1e-6", summary.get('max_mse', float('inf')) < 1e-6),
        ("NaN rate = 0%", summary['nan_rate_triton'] == 0.0),
    ]
    
    print("Success Criteria:")
    for criterion, passed in success_criteria:
        status = "✓" if passed else "✗"
        print(f"  {status} {criterion}")
    
    print()
    
    if summary["success"]:
        print("✓ VERIFICATION PASSED")
    else:
        print("✗ VERIFICATION FAILED")
    
    # Save results to JSON
    output_dir = Path(__file__).parent.parent / "results" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "test_name": "BK-Core Triton Numerical Correctness Verification",
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "configuration_tests": all_results,
        "summary": summary,
        "success_criteria": [
            {"criterion": criterion, "passed": passed}
            for criterion, passed in success_criteria
        ],
    }
    
    output_file = output_dir / "bk_triton_correctness.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print()
    print(f"Results saved to: {output_file}")
    
    sys.exit(0 if summary["success"] else 1)


if __name__ == "__main__":
    main()
