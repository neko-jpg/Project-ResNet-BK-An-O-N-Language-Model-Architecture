# benchmarks/benchmark_hyperbolic_attention.py

import torch
import time
import json
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.phase7.hyperbolic_attention import HyperbolicMultiHeadAttention

def benchmark(seq_lengths, d_model, num_heads, num_runs=10, warmup_runs=3, device='cuda'):
    """
    Benchmarks the HyperbolicMultiHeadAttention module.
    """
    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA not available, switching to CPU.")
        device = 'cpu'

    results = []

    for seq_len in seq_lengths:
        print(f"--- Benchmarking Sequence Length: {seq_len} ---")

        model = HyperbolicMultiHeadAttention(d_model=d_model, num_heads=num_heads).to(device)
        model.train()

        # Dummy input and loss function
        x = torch.randn(1, seq_len, d_model, device=device)

        # Warmup runs
        for _ in range(warmup_runs):
            output = model(x)
            loss = output.sum()
            loss.backward()
            model.zero_grad()

        # Synchronize for accurate timing
        if device == 'cuda':
            torch.cuda.synchronize()

        # Forward pass benchmark
        start_time_fwd = time.time()
        for _ in range(num_runs):
            _ = model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        end_time_fwd = time.time()
        avg_fwd_time = (end_time_fwd - start_time_fwd) / num_runs * 1000  # ms

        # Backward pass benchmark
        output = model(x)
        loss = output.sum()
        start_time_bwd = time.time()
        for _ in range(num_runs):
            loss.backward(retain_graph=True)
            model.zero_grad()
        if device == 'cuda':
            torch.cuda.synchronize()
        end_time_bwd = time.time()
        avg_bwd_time = (end_time_bwd - start_time_bwd) / num_runs * 1000 # ms

        total_time = avg_fwd_time + avg_bwd_time

        print(f"  Forward Pass : {avg_fwd_time:.3f} ms")
        print(f"  Backward Pass: {avg_bwd_time:.3f} ms")
        print(f"  Total Time   : {total_time:.3f} ms")

        results.append({
            "seq_len": seq_len,
            "d_model": d_model,
            "num_heads": num_heads,
            "forward_ms": avg_fwd_time,
            "backward_ms": avg_bwd_time,
            "total_ms": total_time
        })

    return results

def print_results_table(results):
    """Prints results in a formatted table."""
    print("\n--- Benchmark Summary ---")
    print(f"{'Seq Len':<10} | {'Forward (ms)':<15} | {'Backward (ms)':<15} | {'Total (ms)':<15}")
    print("-" * 60)
    for res in results:
        print(f"{res['seq_len']:<10} | {res['forward_ms']:.3f}{'':<10} | {res['backward_ms']:.3f}{'':<10} | {res['total_ms']:.3f}")

if __name__ == '__main__':
    # Configuration
    SEQ_LENGTHS = [64, 128, 256, 512]
    D_MODEL = 512
    NUM_HEADS = 8
    OUTPUT_FILE = "results/hyperbolic_attention_benchmark.json"

    # Run benchmark
    benchmark_results = benchmark(SEQ_LENGTHS, D_MODEL, NUM_HEADS)

    # Print and save results
    print_results_table(benchmark_results)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(benchmark_results, f, indent=4)

    print(f"\nResults saved to {OUTPUT_FILE}")
