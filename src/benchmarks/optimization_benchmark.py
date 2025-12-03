
import torch
import time
import argparse
from src.models.semiseparable_matrix import SemiseparableMatrix
from src.models.phase1.htt_embedding import HolographicTTEmbedding

def benchmark_bitnet(batch_size=32, seq_len=1024, rank=16, device='cuda'):
    print(f"\nBenchmarking BitNet (B={batch_size}, N={seq_len}, r={rank})...")
    
    # Standard Model
    model_std = SemiseparableMatrix(seq_len, rank=rank, device=device, use_bitnet=False).to(device)
    
    # BitNet Model
    model_bit = SemiseparableMatrix(seq_len, rank=rank, device=device, use_bitnet=True).to(device)
    
    x = torch.randn(batch_size, seq_len, device=device)
    
    # Warmup
    for _ in range(10):
        model_std.matvec(x)
        model_bit.matvec(x)
    
    torch.cuda.synchronize()
    
    # Measure Standard
    start = time.time()
    for _ in range(100):
        model_std.matvec(x)
    torch.cuda.synchronize()
    std_time = (time.time() - start) / 100
    
    # Measure BitNet
    start = time.time()
    for _ in range(100):
        model_bit.matvec(x)
    torch.cuda.synchronize()
    bit_time = (time.time() - start) / 100
    
    print(f"Standard: {std_time*1000:.3f} ms")
    print(f"BitNet:   {bit_time*1000:.3f} ms")
    print(f"Speedup:  {std_time/bit_time:.2f}x")

def benchmark_htt_fused(batch_size=32, seq_len=1024, device='cuda'):
    print(f"\nBenchmarking HTT Fused (B={batch_size}, L={seq_len})...")
    
    vocab_size = 10000
    d_model = 512
    
    # Standard HTT (uses einsum or unfused triton)
    model_std = HolographicTTEmbedding(vocab_size, d_model, rank=16).to(device)
    model_std.use_triton_kernel = False # Force einsum
    
    # Fused HTT
    model_fused = HolographicTTEmbedding(vocab_size, d_model, rank=16).to(device)
    model_fused.use_triton_kernel = True
    model_fused.quantize() # Enable fused kernel
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Warmup
    for _ in range(10):
        model_std(input_ids)
        model_fused(input_ids)
    
    torch.cuda.synchronize()
    
    # Measure Standard
    start = time.time()
    for _ in range(100):
        model_std(input_ids)
    torch.cuda.synchronize()
    std_time = (time.time() - start) / 100
    
    # Measure Fused
    start = time.time()
    for _ in range(100):
        model_fused(input_ids)
    torch.cuda.synchronize()
    fused_time = (time.time() - start) / 100
    
    print(f"Standard: {std_time*1000:.3f} ms")
    print(f"Fused:    {fused_time*1000:.3f} ms")
    print(f"Speedup:  {std_time/fused_time:.2f}x")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
    else:
        benchmark_bitnet(device=args.device)
        benchmark_htt_fused(device=args.device)
