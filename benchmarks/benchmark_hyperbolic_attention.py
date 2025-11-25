import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import triton

from src.models.phase7.hyperbolic_attention import HyperbolicMultiHeadAttention

def benchmark(d_model, num_heads, seq_length, use_triton, device="cuda", dtype=torch.float16):
    """
    Measures the performance of the hyperbolic attention layer for forward and backward passes.
    """
    if not torch.cuda.is_available() or not triton.is_available():
        print(f"Skipping benchmark for Triton={use_triton} as CUDA/Triton is not available.")
        return float('nan'), float('nan'), float('nan')

    model = HyperbolicMultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
        use_triton_kernel=use_triton
    ).to(device).to(dtype)

    x = torch.randn(1, seq_length, d_model, device=device, dtype=dtype)

    # We need gradients for the backward pass benchmark
    x.requires_grad = True

    # Warmup iterations
    for _ in range(10):
        output, _ = model(x)
        # Backward pass requires a scalar loss
        output.sum().backward()
        # Gradients need to be cleared after each backward pass
        model.zero_grad()
        x.grad = None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    # --- Forward Pass Benchmark ---
    start_time = time.time()
    for _ in range(50):
        _ = model(x)
    torch.cuda.synchronize()
    forward_time = (time.time() - start_time) / 50 * 1000  # in ms

    # --- Backward Pass Benchmark ---
    # Need to re-run the forward pass to get the output for backward
    start_time = time.time()
    for _ in range(50):
        output, _ = model(x)
        output.sum().backward()
        model.zero_grad()
        x.grad = None
    torch.cuda.synchronize()
    backward_time = (time.time() - start_time) / 50 * 1000 # in ms

    # Memory usage in GB
    max_memory = torch.cuda.max_memory_allocated(device) / 1e9

    return forward_time, backward_time, max_memory


def run_benchmark_suite():
    """
    Runs the benchmark across different sequence lengths and saves the results.
    """
    results = []
    # Use a range of sequence lengths to show scaling properties
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]
    d_model = 512
    num_heads = 8

    print("--- Running Hyperbolic Attention Benchmark Suite ---")
    print(f"Config: d_model={d_model}, num_heads={num_heads}")

    for seq_len in seq_lengths:
        # Benchmark PyTorch implementation
        fwd_torch, bwd_torch, mem_torch = benchmark(d_model, num_heads, seq_len, use_triton=False)
        if not pd.isna(fwd_torch):
            results.append({
                "Sequence Length": seq_len,
                "Implementation": "PyTorch",
                "Forward (ms)": fwd_torch,
                "Backward (ms)": bwd_torch,
                "VRAM (GB)": mem_torch
            })
            print(f"SeqLen: {seq_len}, Impl: PyTorch, Fwd: {fwd_torch:.2f}ms, Bwd: {bwd_torch:.2f}ms, Mem: {mem_torch:.2f}GB")

        # Benchmark Triton implementation
        fwd_triton, bwd_triton, mem_triton = benchmark(d_model, num_heads, seq_len, use_triton=True)
        if not pd.isna(fwd_triton):
            results.append({
                "Sequence Length": seq_len,
                "Implementation": "Triton",
                "Forward (ms)": fwd_triton,
                "Backward (ms)": bwd_triton,
                "VRAM (GB)": mem_triton
            })
            print(f"SeqLen: {seq_len}, Impl: Triton, Fwd: {fwd_triton:.2f}ms, Bwd: {bwd_triton:.2f}ms, Mem: {mem_triton:.2f}GB")

    if not results:
        print("\nNo benchmarks were run, likely due to missing CUDA/Triton.")
        return

    df = pd.DataFrame(results)
    print("\n--- Benchmark Results ---")
    print(df)

    # Save results to a CSV file
    df.to_csv("results/hyperbolic_attention_benchmark.csv", index=False)
    print("\nResults saved to results/hyperbolic_attention_benchmark.csv")

    # Plotting the results
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Hyperbolic Attention Benchmark: PyTorch vs. Triton')

    sns.lineplot(data=df, x='Sequence Length', y='Forward (ms)', hue='Implementation', ax=axes[0], marker='o')
    axes[0].set_title('Forward Pass Duration')
    axes[0].set_ylabel('Time (ms)')
    axes[0].grid(True)

    sns.lineplot(data=df, x='Sequence Length', y='Backward (ms)', hue='Implementation', ax=axes[1], marker='o')
    axes[1].set_title('Backward Pass Duration')
    axes[1].set_ylabel('Time (ms)')
    axes[1].grid(True)

    sns.lineplot(data=df, x='Sequence Length', y='VRAM (GB)', hue='Implementation', ax=axes[2], marker='o')
    axes[2].set_title('Peak VRAM Usage')
    axes[2].set_ylabel('Memory (GB)')
    axes[2].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_filename = "results/hyperbolic_attention_benchmark.png"
    plt.savefig(plot_filename)
    print(f"Plots saved to {plot_filename}")


if __name__ == "__main__":
    # Create results directory if it doesn't exist
    import os
    os.makedirs("results", exist_ok=True)
    run_benchmark_suite()
