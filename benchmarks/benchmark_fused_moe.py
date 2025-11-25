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

from src.models.resnet_bk import MoEResNetBKLayer
from src.kernels.fused_moe_kernel import is_triton_available, fused_moe_forward

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
        # d_model must be divisible by BLOCK_SIZE_D (64)
        # 2*d_model must be divisible by BLOCK_SIZE_H (128)
        {'B': 4, 'N': 1024, 'D': 768, 'E': 8, 'K': 1},
        {'B': 4, 'N': 2048, 'D': 1024, 'E': 16, 'K': 1},
        # Config that is not compatible with block layout to test fallback
        {'B': 4, 'N': 1024, 'D': 767, 'E': 8, 'K': 1},
        # Config to test top_k > 1 fallback
        {'B': 4, 'N': 1024, 'D': 768, 'E': 8, 'K': 2},
    ]

    benchmark_results = []

    for config in configs:
        B, N, D, E, K = config['B'], config['N'], config['D'], config['E'], config['K']

        print(f"-> Running Config: B={B}, N={N}, D={D}, E={E}, K={K}")

        try:
            dtype = torch.float16
            x = torch.randn(B, N, D, device='cuda', dtype=dtype)

            # --- Setup Models ---
            # Instantiate MoE layer with fused kernel DISABLED (original)
            model_original = MoEResNetBKLayer(
                d_model=D, n_seq=N, num_experts=E, top_k=K, use_fused_moe_kernel=False
            ).to('cuda').to(dtype)

            # Instantiate MoE layer with fused kernel ENABLED
            model_fused = MoEResNetBKLayer(
                d_model=D, n_seq=N, num_experts=E, top_k=K, use_fused_moe_kernel=True
            ).to('cuda').to(dtype)

            # Copy weights to ensure they are identical
            model_fused.load_state_dict(model_original.state_dict())

            # --- Correctness Check ---
            print("   Verifying numerical correctness...")
            with torch.no_grad():
                # We only need to compare the MoE FFN part of the output
                output_original, _, _ = model_original.moe_ffn(x)

                # Manually call the fused forward path for comparison
                gate_w_t = model_fused.gate_w_t
                w1_bl = model_fused.w1_bl
                w2_bl = model_fused.w2_bl
                output_fused = fused_moe_forward(x, gate_w_t, w1_bl, w2_bl, K, use_block_layout=True)

            # Check only if the fused kernel was supposed to be used
            if model_fused.use_fused_moe_kernel:
                 assert torch.allclose(output_original, output_fused, atol=1e-1, rtol=1e-2), \
                    "Correctness check failed: Outputs do not match."
                 print("   Correctness check PASSED.")
            else:
                print("   Skipping correctness check (fused kernel was disabled due to incompatibility).")


            # --- Benchmarking ---
            print("   Running performance benchmark...")
            quantiles = [0.5, 0.2, 0.8]

            ms_original, _, _ = triton.testing.do_bench(lambda: model_original.moe_ffn(x), quantiles=quantiles)

            ms_fused, _, _ = triton.testing.do_bench(lambda: fused_moe_forward(x, gate_w_t, w1_bl, w2_bl, K, use_block_layout=True), quantiles=quantiles)

            speedup = ms_original / ms_fused if ms_fused > 0 else float('inf')

            result = {
                'config': f"B={B}, N={N}, D={D}, E={E}, K={K}",
                'original_ms': round(ms_original, 4),
                'fused_ms': round(ms_fused, 4) if model_fused.use_fused_moe_kernel else "N/A (fallback)",
                'speedup': f"{speedup:.2f}x" if model_fused.use_fused_moe_kernel else "N/A"
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
