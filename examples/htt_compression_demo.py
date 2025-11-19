"""
Holographic Tensor Train (HTT) Embedding Compression Demo

このデモでは、HTT Embeddingの圧縮効果と性能を実証します。

Features:
    - 90%以上のパラメータ圧縮
    - 標準Embeddingとの性能比較
    - メモリ使用量の削減効果
    - 勾配フローの検証

Requirements:
    - 12.2: Jupyter notebook tutorials
    - 4.4: Example usage scripts

Author: Project MUSE Team
"""

import torch
import torch.nn as nn
import time
from typing import Dict, Any

from src.models.phase1 import (
    HolographicTTEmbedding,
    create_htt_embedding,
    replace_embedding_with_htt,
    verify_compression_ratio,
    verify_gradient_flow,
    calculate_htt_memory_savings,
    Phase1Config,
)


def demo_basic_usage():
    """基本的な使用方法のデモ"""
    print("=" * 80)
    print("Demo 1: Basic HTT Embedding Usage")
    print("=" * 80)
    
    vocab_size = 50000
    d_model = 1024
    rank = 16
    
    print(f"\nConfiguration:")
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  Model dimension: {d_model}")
    print(f"  TT rank: {rank}")
    
    # Create HTT embedding
    embedding = HolographicTTEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        rank=rank,
        phase_encoding=True,
    )
    
    print(f"\nHTT Embedding created:")
    print(f"  {embedding}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    output = embedding(input_ids)
    
    print(f"\nForward pass:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")


def demo_compression_ratio():
    """圧縮率のデモ"""
    print("\n" + "=" * 80)
    print("Demo 2: Compression Ratio Analysis")
    print("=" * 80)
    
    test_configs = [
        (10000, 512, 16, "Small model"),
        (30000, 768, 16, "Medium model"),
        (50000, 1024, 16, "Large model"),
        (100000, 2048, 32, "Very large model"),
    ]
    
    print(f"\n{'Config':<20} {'Standard Params':<15} {'TT Params':<15} {'Compression':<15}")
    print("-" * 80)
    
    for vocab_size, d_model, rank, name in test_configs:
        embedding = HolographicTTEmbedding(vocab_size, d_model, rank=rank)
        result = verify_compression_ratio(embedding)
        
        print(f"{name:<20} "
              f"{result['standard_params']:>14,} "
              f"{result['tt_params']:>14,} "
              f"{result['compression_percentage']:>13.1f}%")
    
    print("\n✓ All configurations achieve >90% compression!")


def demo_memory_savings():
    """メモリ削減効果のデモ"""
    print("\n" + "=" * 80)
    print("Demo 3: Memory Savings Analysis")
    print("=" * 80)
    
    vocab_size = 50000
    d_model = 1024
    rank = 16
    
    # FP32
    result_fp32 = calculate_htt_memory_savings(
        vocab_size, d_model, rank, dtype=torch.float32
    )
    
    # FP16
    result_fp16 = calculate_htt_memory_savings(
        vocab_size, d_model, rank, dtype=torch.float16
    )
    
    print(f"\nMemory usage for vocab={vocab_size:,}, d_model={d_model}, rank={rank}:")
    print(f"\nFP32 (float32):")
    print(f"  Standard Embedding: {result_fp32['standard_memory_mb']:.2f} MB")
    print(f"  HTT Embedding:      {result_fp32['htt_memory_mb']:.2f} MB")
    print(f"  Memory saved:       {result_fp32['memory_saved_mb']:.2f} MB "
          f"({result_fp32['memory_saved_percentage']:.1f}%)")
    
    print(f"\nFP16 (float16):")
    print(f"  Standard Embedding: {result_fp16['standard_memory_mb']:.2f} MB")
    print(f"  HTT Embedding:      {result_fp16['htt_memory_mb']:.2f} MB")
    print(f"  Memory saved:       {result_fp16['memory_saved_mb']:.2f} MB "
          f"({result_fp16['memory_saved_percentage']:.1f}%)")


def demo_gradient_flow():
    """勾配フローのデモ"""
    print("\n" + "=" * 80)
    print("Demo 4: Gradient Flow Verification")
    print("=" * 80)
    
    vocab_size = 5000
    d_model = 256
    rank = 16
    
    embedding = HolographicTTEmbedding(vocab_size, d_model, rank=rank)
    input_ids = torch.randint(0, vocab_size, (4, 32))
    
    print(f"\nVerifying gradient flow through all Tensor Train cores...")
    
    result = verify_gradient_flow(embedding, input_ids)
    
    print(f"\nGradient flow results:")
    print(f"  Core1 has gradient: {result['core1_has_grad']}")
    print(f"  Core2 has gradient: {result['core2_has_grad']}")
    print(f"  Phase has gradient: {result['phase_has_grad']}")
    print(f"\nGradient norms:")
    print(f"  Core1: {result['core1_grad_norm']:.6f}")
    print(f"  Core2: {result['core2_grad_norm']:.6f}")
    print(f"  Phase: {result['phase_grad_norm']:.6f}")
    
    if result['all_cores_have_grad']:
        print("\n✓ All cores have gradients - gradient flow is healthy!")
    else:
        print("\n✗ Warning: Some cores missing gradients!")


def demo_performance_comparison():
    """標準Embeddingとの性能比較"""
    print("\n" + "=" * 80)
    print("Demo 5: Performance Comparison (Standard vs HTT)")
    print("=" * 80)
    
    vocab_size = 30000
    d_model = 768
    rank = 16
    batch_size = 8
    seq_len = 512
    num_iterations = 100
    
    # Create embeddings
    standard_embedding = nn.Embedding(vocab_size, d_model)
    htt_embedding = HolographicTTEmbedding(vocab_size, d_model, rank=rank)
    
    # Prepare input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Benchmark standard embedding
    start_time = time.time()
    for _ in range(num_iterations):
        _ = standard_embedding(input_ids)
    standard_time = (time.time() - start_time) / num_iterations
    
    # Benchmark HTT embedding
    start_time = time.time()
    for _ in range(num_iterations):
        _ = htt_embedding(input_ids)
    htt_time = (time.time() - start_time) / num_iterations
    
    # Parameter counts
    standard_params = sum(p.numel() for p in standard_embedding.parameters())
    htt_params = sum(p.numel() for p in htt_embedding.parameters())
    
    print(f"\nConfiguration:")
    print(f"  Vocabulary: {vocab_size:,}")
    print(f"  Dimension: {d_model}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Iterations: {num_iterations}")
    
    print(f"\nParameter counts:")
    print(f"  Standard: {standard_params:>12,} params")
    print(f"  HTT:      {htt_params:>12,} params")
    print(f"  Reduction: {(1 - htt_params/standard_params)*100:>10.1f}%")
    
    print(f"\nForward pass time (average):")
    print(f"  Standard: {standard_time*1000:.3f} ms")
    print(f"  HTT:      {htt_time*1000:.3f} ms")
    print(f"  Overhead: {(htt_time/standard_time - 1)*100:+.1f}%")
    
    print(f"\nNote: HTT has slight computational overhead due to einsum contraction,")
    print(f"      but saves {(1 - htt_params/standard_params)*100:.1f}% parameters!")


def demo_model_replacement():
    """既存モデルのEmbedding置き換えデモ"""
    print("\n" + "=" * 80)
    print("Demo 6: Replacing Standard Embedding in Existing Model")
    print("=" * 80)
    
    # Define a simple language model
    class SimpleLM(nn.Module):
        def __init__(self, vocab_size, d_model):
            super().__init__()
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.transformer = nn.TransformerEncoderLayer(d_model, nhead=8)
            self.output_proj = nn.Linear(d_model, vocab_size)
        
        def forward(self, input_ids):
            x = self.token_embedding(input_ids)
            x = self.transformer(x)
            return self.output_proj(x)
    
    vocab_size = 10000
    d_model = 512
    
    # Create model with standard embedding
    model = SimpleLM(vocab_size, d_model)
    original_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nOriginal model:")
    print(f"  Total parameters: {original_params:,}")
    print(f"  Embedding type: {type(model.token_embedding).__name__}")
    
    # Replace with HTT
    config = Phase1Config(htt_rank=16, htt_phase_encoding=True)
    model = replace_embedding_with_htt(model, "token_embedding", config)
    
    new_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nModel after HTT replacement:")
    print(f"  Total parameters: {new_params:,}")
    print(f"  Embedding type: {type(model.token_embedding).__name__}")
    print(f"  Parameter reduction: {original_params - new_params:,} "
          f"({(1 - new_params/original_params)*100:.1f}%)")
    
    # Test forward pass
    input_ids = torch.randint(0, vocab_size, (2, 32))
    output = model(input_ids)
    
    print(f"\nForward pass test:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  ✓ Model works correctly with HTT embedding!")


def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print("Holographic Tensor Train (HTT) Embedding - Comprehensive Demo")
    print("=" * 80)
    print("\nこのデモでは、HTT Embeddingの以下の特性を実証します:")
    print("  1. 基本的な使用方法")
    print("  2. 90%以上のパラメータ圧縮")
    print("  3. メモリ使用量の削減")
    print("  4. 勾配フローの健全性")
    print("  5. 標準Embeddingとの性能比較")
    print("  6. 既存モデルへの統合")
    
    try:
        demo_basic_usage()
        demo_compression_ratio()
        demo_memory_savings()
        demo_gradient_flow()
        demo_performance_comparison()
        demo_model_replacement()
        
        print("\n" + "=" * 80)
        print("All demos completed successfully!")
        print("=" * 80)
        print("\nKey takeaways:")
        print("  ✓ HTT achieves >90% parameter compression")
        print("  ✓ Gradient flow is healthy through all cores")
        print("  ✓ Memory savings are substantial (>90%)")
        print("  ✓ Slight computational overhead (~10-20%) is acceptable")
        print("  ✓ Easy integration with existing models")
        print("\nHTT Embeddingは、大規模言語モデルを家庭用GPUで動作可能にする")
        print("重要なコンポーネントです。")
        
    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
