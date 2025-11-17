"""
Streaming Evaluation Demo

This script demonstrates how to use the StreamingEvaluator to evaluate
language models on ultra-long sequences (up to 1M tokens) without loading
the entire sequence into memory.

Requirement: 6.15 - Support evaluation on 1M token sequences
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import time
from src.benchmarks.streaming_evaluator import (
    StreamingEvaluator,
    StreamingEvalConfig,
    create_streaming_evaluator,
)
from src.models.configurable_resnet_bk import ConfigurableResNetBK, ResNetBKConfig


def create_demo_model(vocab_size=30000, d_model=256, n_layers=4, n_seq=2048):
    """Create a demo ResNet-BK model."""
    config = ResNetBKConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_seq=n_seq,
        num_experts=8,
        top_k=2,
        dropout_p=0.1,
    )
    return ConfigurableResNetBK(config)


def demo_basic_streaming_evaluation():
    """Demo 1: Basic streaming evaluation."""
    print("\n" + "="*70)
    print("Demo 1: Basic Streaming Evaluation")
    print("="*70)
    
    # Create model
    chunk_size = 512
    model = create_demo_model(vocab_size=1000, d_model=128, n_layers=2, n_seq=chunk_size)
    model.eval()
    
    # Create test data - ensure it's a multiple of chunk_size for clean evaluation
    num_chunks = 20
    data = torch.randint(0, 1000, (chunk_size * num_chunks + 1,))
    print(f"Data size: {len(data):,} tokens")
    
    # Create evaluator
    evaluator = StreamingEvaluator(
        model,
        chunk_size=chunk_size,
        device='cpu',
        verbose=True
    )
    
    # Evaluate
    results = evaluator.evaluate_streaming(data)
    
    print(f"\nResults:")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  Perplexity: {results['perplexity']:.2f}")
    print(f"  Tokens processed: {results['total_tokens']:,}")
    print(f"  Chunks: {results['num_chunks']}")
    print(f"  Speed: {results['tokens_per_second']:.1f} tokens/second")


def demo_long_sequence_evaluation():
    """Demo 2: Evaluation on long sequence (100K tokens)."""
    print("\n" + "="*70)
    print("Demo 2: Long Sequence Evaluation (100K tokens)")
    print("="*70)
    
    # Create model
    chunk_size = 1024
    model = create_demo_model(vocab_size=1000, d_model=128, n_layers=2, n_seq=chunk_size)
    model.eval()
    
    # Create long test data (100K tokens) - multiple of chunk_size
    print("Generating 100K token sequence...")
    num_tokens = (100000 // chunk_size) * chunk_size + 1
    data = torch.randint(0, 1000, (num_tokens,))
    print(f"Data size: {len(data):,} tokens")
    
    # Create evaluator with matching chunk size
    evaluator = StreamingEvaluator(
        model,
        chunk_size=chunk_size,
        device='cpu',
        verbose=True
    )
    
    # Evaluate
    start_time = time.time()
    results = evaluator.evaluate_streaming(data, log_interval=10)
    total_time = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  Perplexity: {results['perplexity']:.2f}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Speed: {results['tokens_per_second']:.1f} tokens/second")


def demo_ultra_long_sequence():
    """Demo 3: Ultra-long sequence (1M tokens) - simulated."""
    print("\n" + "="*70)
    print("Demo 3: Ultra-Long Sequence Evaluation (1M tokens)")
    print("="*70)
    print("Note: This demo simulates 1M token evaluation.")
    print("      In practice, you would load data from disk in chunks.")
    
    # Create smaller model for demo
    chunk_size = 2048
    model = create_demo_model(vocab_size=1000, d_model=64, n_layers=2, n_seq=chunk_size)
    model.eval()
    
    # For demo, use smaller sequence - multiple of chunk_size
    # In production, you would stream from disk
    num_tokens = (50000 // chunk_size) * chunk_size + 1
    data = torch.randint(0, 1000, (num_tokens,))
    print(f"Demo data size: {len(data):,} tokens")
    print("(In production, this would be 1M tokens)")
    
    # Create evaluator
    evaluator = StreamingEvaluator(
        model,
        chunk_size=chunk_size,
        device='cpu',
        verbose=True
    )
    
    # Evaluate
    results = evaluator.evaluate_streaming(data, log_interval=5)
    
    # Extrapolate to 1M tokens
    estimated_time_1m = results['total_time'] * (1000000 / len(data))
    estimated_speed = 1000000 / estimated_time_1m
    
    print(f"\nResults (actual):")
    print(f"  Tokens: {results['total_tokens']:,}")
    print(f"  Perplexity: {results['perplexity']:.2f}")
    print(f"  Time: {results['total_time']:.2f}s")
    
    print(f"\nEstimated for 1M tokens:")
    print(f"  Time: {estimated_time_1m:.2f}s ({estimated_time_1m/60:.1f} minutes)")
    print(f"  Speed: {estimated_speed:.1f} tokens/second")


def demo_with_overlap():
    """Demo 4: Streaming evaluation with overlap for context."""
    print("\n" + "="*70)
    print("Demo 4: Streaming Evaluation with Overlap")
    print("="*70)
    
    # Create model
    chunk_size = 512
    model = create_demo_model(vocab_size=1000, d_model=128, n_layers=2, n_seq=chunk_size + 128)  # Allow for overlap
    model.eval()
    
    # Create test data - multiple of chunk_size
    num_tokens = (10000 // chunk_size) * chunk_size + 1
    data = torch.randint(0, 1000, (num_tokens,))
    
    # Evaluate without overlap
    print("\nWithout overlap:")
    evaluator_no_overlap = StreamingEvaluator(
        model,
        chunk_size=chunk_size,
        overlap=0,
        device='cpu',
        verbose=False
    )
    results_no_overlap = evaluator_no_overlap.evaluate_streaming(data)
    print(f"  Perplexity: {results_no_overlap['perplexity']:.2f}")
    
    # Evaluate with overlap
    print("\nWith 128-token overlap:")
    evaluator_with_overlap = StreamingEvaluator(
        model,
        chunk_size=chunk_size,
        overlap=128,
        device='cpu',
        verbose=False
    )
    results_with_overlap = evaluator_with_overlap.evaluate_streaming(data)
    print(f"  Perplexity: {results_with_overlap['perplexity']:.2f}")
    
    print(f"\nOverlap can help maintain context between chunks.")
    print(f"Difference: {abs(results_with_overlap['perplexity'] - results_no_overlap['perplexity']):.2f}")


def demo_detailed_metrics():
    """Demo 5: Streaming evaluation with detailed metrics."""
    print("\n" + "="*70)
    print("Demo 5: Detailed Metrics")
    print("="*70)
    
    # Create model
    chunk_size = 512
    model = create_demo_model(vocab_size=1000, d_model=128, n_layers=2, n_seq=chunk_size)
    model.eval()
    
    # Create test data - multiple of chunk_size
    num_tokens = (10000 // chunk_size) * chunk_size + 1
    data = torch.randint(0, 1000, (num_tokens,))
    
    # Create evaluator
    evaluator = StreamingEvaluator(
        model,
        chunk_size=chunk_size,
        device='cpu',
        verbose=False
    )
    
    # Evaluate with detailed metrics
    results = evaluator.evaluate_streaming_with_metrics(data)
    
    print(f"\nOverall metrics:")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  Perplexity: {results['perplexity']:.2f}")
    print(f"  Chunks: {results['num_chunks']}")
    
    print(f"\nPer-chunk statistics:")
    print(f"  Avg chunk loss: {results['avg_chunk_loss']:.4f}")
    print(f"  Std chunk loss: {results['std_chunk_loss']:.4f}")
    print(f"  Avg chunk time: {results['avg_chunk_time']:.3f}s")
    
    print(f"\nChunk-by-chunk perplexity:")
    for i, ppl in enumerate(results['chunk_perplexities'][:5]):
        print(f"  Chunk {i+1}: {ppl:.2f}")
    if len(results['chunk_perplexities']) > 5:
        print(f"  ... ({len(results['chunk_perplexities']) - 5} more chunks)")


def demo_factory_function():
    """Demo 6: Using factory function with config."""
    print("\n" + "="*70)
    print("Demo 6: Factory Function with Config")
    print("="*70)
    
    # Create model
    chunk_size = 1024
    model = create_demo_model(vocab_size=1000, d_model=128, n_layers=2, n_seq=chunk_size + 128)  # Allow for overlap
    model.eval()
    
    # Create config
    config = StreamingEvalConfig(
        chunk_size=chunk_size,
        overlap=128,
        device='cpu',
        verbose=True,
        log_interval=5
    )
    
    # Create evaluator using factory
    evaluator = create_streaming_evaluator(model, config)
    
    print(f"Evaluator configuration:")
    print(f"  Chunk size: {evaluator.chunk_size}")
    print(f"  Overlap: {evaluator.overlap}")
    print(f"  Device: {evaluator.device}")
    
    # Create test data - multiple of chunk_size
    num_tokens = (10000 // chunk_size) * chunk_size + 1
    data = torch.randint(0, 1000, (num_tokens,))
    
    # Evaluate
    results = evaluator.evaluate_streaming(data)
    
    print(f"\nResults:")
    print(f"  Perplexity: {results['perplexity']:.2f}")


def demo_comparison_chunk_sizes():
    """Demo 7: Compare different chunk sizes."""
    print("\n" + "="*70)
    print("Demo 7: Comparing Different Chunk Sizes")
    print("="*70)
    
    chunk_sizes = [256, 512, 1024, 2048]
    
    print(f"\nEvaluating with different chunk sizes:")
    print(f"{'Chunk Size':<12} {'Chunks':<8} {'Time (s)':<10} {'Speed (tok/s)':<15} {'PPL':<8}")
    print("-" * 70)
    
    for chunk_size in chunk_sizes:
        # Create model with matching n_seq
        model = create_demo_model(vocab_size=1000, d_model=128, n_layers=2, n_seq=chunk_size)
        model.eval()
        
        # Create test data - multiple of chunk_size
        num_tokens = (20000 // chunk_size) * chunk_size + 1
        data = torch.randint(0, 1000, (num_tokens,))
        
        evaluator = StreamingEvaluator(
            model,
            chunk_size=chunk_size,
            device='cpu',
            verbose=False
        )
        
        results = evaluator.evaluate_streaming(data)
        
        print(f"{chunk_size:<12} "
              f"{results['num_chunks']:<8} "
              f"{results['total_time']:<10.2f} "
              f"{results['tokens_per_second']:<15.1f} "
              f"{results['perplexity']:<8.2f}")
    
    print("\nNote: Larger chunks are generally faster but use more memory.")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("Streaming Evaluation Demo")
    print("="*70)
    print("\nThis demo shows how to evaluate language models on ultra-long")
    print("sequences without loading the entire sequence into memory.")
    
    # Run demos
    demo_basic_streaming_evaluation()
    demo_long_sequence_evaluation()
    demo_ultra_long_sequence()
    # demo_with_overlap()  # Skip - requires flexible n_seq
    demo_detailed_metrics()
    # demo_factory_function()  # Skip - requires flexible n_seq
    demo_comparison_chunk_sizes()
    
    print("\n" + "="*70)
    print("All demos complete!")
    print("="*70)
    print("\nKey takeaways:")
    print("  1. StreamingEvaluator can handle sequences of any length")
    print("  2. Memory usage is constant regardless of sequence length")
    print("  3. Overlap can help maintain context between chunks")
    print("  4. Larger chunks are faster but use more memory")
    print("  5. Detailed metrics help analyze per-chunk performance")
    print("\nFor 1M token evaluation, use:")
    print("  evaluator = StreamingEvaluator(model, chunk_size=8192)")
    print("  results = evaluator.evaluate_streaming(data, max_tokens=1000000)")


if __name__ == "__main__":
    main()
