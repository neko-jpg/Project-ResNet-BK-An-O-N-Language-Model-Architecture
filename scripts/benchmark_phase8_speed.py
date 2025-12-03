#!/usr/bin/env python3
"""
Speed Benchmark Script for Phase 8

Benchmarks training speed before and after optimizations.
"""

import argparse
import time
import torch
import torch.optim as optim
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.phase8.integrated_model import Phase8IntegratedModel, Phase8Config


def benchmark_forward_pass(model, input_ids, num_iterations=50):
    """Benchmark forward pass speed."""
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_iterations):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits, _ = model(input_ids)
        torch.cuda.synchronize()
    
    end = time.time()
    elapsed = end - start
    tokens_processed = input_ids.numel() * num_iterations
    tokens_per_sec = tokens_processed / elapsed
    
    return tokens_per_sec, elapsed


def benchmark_training_step(model, optimizer, input_ids, targets, num_iterations=50):
    """Benchmark training step (forward + backward + optimizer)."""
    torch.cuda.synchronize()
    start = time.time()
    
    scaler = torch.cuda.amp.GradScaler()
    
    for _ in range(num_iterations):
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits, _ = model(input_ids)
            logits = logits.view(-1, model.config.vocab_size)
            loss = torch.nn.functional.cross_entropy(logits, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        torch.cuda.synchronize()
    
    end = time.time()
    elapsed = end - start
    tokens_processed = input_ids.numel() * num_iterations
    tokens_per_sec = tokens_processed / elapsed
    
    return tokens_per_sec, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d-model', type=int, default=512)
    parser.add_argument('--n-layers', type=int, default=8)
    parser.add_argument('--n-seq', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--low-rank-rank', type=int, default=16)
    parser.add_argument('--use-compile', action='store_true')
    parser.add_argument('--use-flash-attn2', action='store_true')
    parser.add_argument('--iterations', type=int, default=50)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("❌  CUDA not available, benchmark requires GPU")
        return
    
    print("=" * 60)
    print(f"Phase 8 Speed Benchmark")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  d_model: {args.d_model}")
    print(f"  n_layers: {args.n_layers}")
    print(f"  n_seq: {args.n_seq}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  low_rank_rank: {args.low_rank_rank}")
    print(f"  use_compile: {args.use_compile}")
    print(f"  use_flash_attn2: {args.use_flash_attn2}")
    print()
    
    # Create model
    config = Phase8Config(
        vocab_size=50257,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_seq=args.n_seq,
        num_heads=min(args.d_model // 64, 16),
        low_rank_ffn=True,
        low_rank_attention=True,
        low_rank_rank=args.low_rank_rank,
        use_bitnet=True,
        use_bk_hyperbolic=True,
        use_ar_ssm_fusion=True,
        use_gradient_checkpointing=False,  # Disable for fair comparison
        use_mixed_precision=True,
        use_torch_compile=args.use_compile,
        compile_mode="max-autotune" if args.use_compile else "default",
        use_flash_attention_2=args.use_flash_attn2,
    )
    
    model = Phase8IntegratedModel(config).to(device)
    
    # Compile if requested
    if args.use_compile:
        print("⚡ Compiling model with torch.compile...")
        model = torch.compile(model, mode=config.compile_mode)
        print("✅ Compilation complete")
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, eps=1e-6)
    
    # Create dummy data
    input_ids = torch.randint(0, config.vocab_size, (args.batch_size, args.n_seq)).to(device)
    targets = torch.randint(0, config.vocab_size, (args.batch_size * args.n_seq,)).to(device)
    
    # Warmup
    print(f"Warming up ({args.iterations // 5} iterations)...")
    model.train()
    for _ in range(args.iterations // 5):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits, _ = model(input_ids)
    print()
    
    # Benchmark forward only
    print(f"Benchmarking forward pass ({args.iterations} iterations)...")
    model.eval()
    with torch.no_grad():
        forward_tokens_per_sec, forward_time = benchmark_forward_pass(model, input_ids, args.iterations)
    print(f"  Forward-only: {forward_tokens_per_sec:.2f} tokens/sec")
    print(f"  Time: {forward_time:.2f}s")
    print()
    
    # Benchmark training step
    print(f"Benchmarking training step ({args.iterations} iterations)...")
    model.train()
    training_tokens_per_sec, training_time = benchmark_training_step(
        model, optimizer, input_ids, targets, args.iterations
    )
    print(f"  Training: {training_tokens_per_sec:.2f} tokens/sec")
    print(f"  Time: {training_time:.2f}s")
    print()
    
    # Memory stats
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU Memory: {peak_memory:.2f} GB")
    
    print("=" * 60)
    print(f"Summary:")
    print(f"  Forward: {forward_tokens_per_sec:.2f} tokens/sec")
    print(f"  Training: {training_tokens_per_sec:.2f} tokens/sec")
    if training_tokens_per_sec >= 1000:
        print(f"  ✅ TARGET MET: >1000 tokens/sec!")
    else:
        gap = 1000 - training_tokens_per_sec
        print(f"  ⚠️  Short of target by {gap:.0f} tokens/sec")
    print("=" * 60)


if __name__ == '__main__':
    main()
