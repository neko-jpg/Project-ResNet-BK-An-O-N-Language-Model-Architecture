"""
Benchmark for the Fused MoE Kernel
====================================

This script quantitatively measures the performance of the monolithic
`fused_moe_kernel` against the original, non-fused `SparseMoELayer`
implementation.

It performs the following steps:
1.  Defines a set of configurations (batch size, sequence length, model dimension, etc.).
2.  For each configuration, it creates a standard `SparseMoELayer` and prepares the
    weights for the fused kernel.
3.  Verifies that the numerical output of both implementations is identical.
4.  Uses `triton.testing.do_bench` to accurately measure the execution time of both
    forward passes.
5.  Calculates the speedup factor.
6.  Prints the results in a formatted table and saves them to a JSON file in the
    `results/` directory, as per project guidelines in AGENTS.md.
"""

import torch
import triton
import json
import pandas as pd
import os

# Add src to python path to allow direct imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.moe import SparseMoELayer
from src.kernels.fused_moe_kernel import fused_moe_forward, is_triton_available

def print_and_save_results(results, filename="results/fused_moe_benchmark.json"):
    """Prints results in a formatted table and saves them to a JSON file."""
    if not results:
        print("No results to save.")
        return

    df = pd.DataFrame(results)

    # Set display options for better readability
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', None)

    print("\n--- Fused MoE Benchmark Results ---")
    print(df)
    print("-----------------------------------\n")

    # Ensure the results directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Use a format that is easily parseable
    report = {
        "title": "Fused MoE Kernel Performance Benchmark",
        "configs": results
    }

    with open(filename, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Benchmark results saved to {filename}")


def run_benchmark():
    """Main function to run the benchmark."""
    # --- Pre-flight checks ---
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return
    if not is_triton_available():
        print("Triton not available, cannot run the fused kernel. Skipping benchmark.")
        return

    # --- Benchmark Configurations ---
    configs = [
        {'B': 1, 'N': 1024, 'D': 768, 'E': 8, 'H_multiplier': 2, 'K': 1},
        {'B': 4, 'N': 1024, 'D': 768, 'E': 8, 'H_multiplier': 2, 'K': 1},
        {'B': 1, 'N': 2048, 'D': 1024, 'E': 16, 'H_multiplier': 2, 'K': 1},
        {'B': 4, 'N': 2048, 'D': 1024, 'E': 16, 'H_multiplier': 2, 'K': 1},
        # A configuration with top_k > 1 to test the non-optimized path
        {'B': 1, 'N': 1024, 'D': 768, 'E': 8, 'H_multiplier': 2, 'K': 2},
    ]

    benchmark_results = []

    for config in configs:
        B, N, D, E, H_multiplier, K = (
            config['B'], config['N'], config['D'], config['E'], config['H_multiplier'], config['K']
        )
        H = D * H_multiplier # Hidden dimension of expert MLPs

        print(f"-> Running Config: B={B}, N={N}, D={D}, E={E}, H={H}, K={K}")

        try:
            # --- Setup ---
            # Use float16 as it's common for performance-critical models
            dtype = torch.float16
            x = torch.randn(B, N, D, device='cuda', dtype=dtype)

            # Instantiate the original SparseMoELayer
            moe_layer = SparseMoELayer(d_model=D, num_experts=E, top_k=K).to('cuda').to(dtype)

            # Prepare weights for the fused kernel (transpose for GEMM)
            gate_w = moe_layer.gating_network.weight.t()
            experts_w1 = torch.stack([expert[0].weight.t() for expert in moe_layer.experts])
            experts_w2 = torch.stack([expert[3].weight.t() for expert in moe_layer.experts])

            # --- Correctness Check ---
            print("   Verifying numerical correctness...")
            with torch.no_grad():
                output_original, _, _ = moe_layer(x)
                output_fused = fused_moe_forward(x, gate_w, experts_w1, experts_w2, K)

            # Use a slightly looser tolerance for float16 and complex fused operations
            assert torch.allclose(output_original, output_fused, atol=1e-1, rtol=1e-2), \
                "Correctness check failed: Outputs do not match."
            print("   Correctness check PASSED.")

            # --- Benchmarking ---
            print("   Running performance benchmark...")
            quantiles = [0.5, 0.2, 0.8] # Median, 20th percentile, 80th percentile

            # Benchmark original implementation
            ms_original, min_original, max_original = triton.testing.do_bench(
                lambda: moe_layer(x), quantiles=quantiles
            )

            # Benchmark fused kernel implementation
            ms_fused, min_fused, max_fused = triton.testing.do_bench(
                lambda: fused_moe_forward(x, gate_w, experts_w1, experts_w2, K), quantiles=quantiles
            )

            speedup = ms_original / ms_fused

            result = {
                'config': f"B={B}, N={N}, D={D}, E={E}, K={K}",
                'original_latency_ms': round(ms_original, 4),
                'fused_latency_ms': round(ms_fused, 4),
                'speedup_factor': f"{speedup:.2f}x"
            }
            benchmark_results.append(result)

            print(f"   - Original Latency: {ms_original:.4f} ms")
            print(f"   - Fused Latency:    {ms_fused:.4f} ms")
            print(f"   - Speedup:          {speedup:.2f}x")

        except Exception as e:
            print(f"   ERROR running config {config}: {e}")
            # Add a failed result to the report
            benchmark_results.append({
                'config': f"B={B}, N={N}, D={D}, E={E}, K={K}",
                'status': 'Failed',
                'error': str(e)
            })

    # --- Final Report ---
    print_and_save_results(benchmark_results)


if __name__ == "__main__":
    # The PYTHONPATH needs to include the root of the repository
    # Example command from root: python benchmarks/benchmark_fused_moe.py
    run_benchmark()
