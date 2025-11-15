"""
Demo: Learned Sequence Length for ResNet-BK

Demonstrates dynamic sequence length prediction and adaptation.
Shows how the model learns to use shorter sequences for simple inputs
and longer sequences for complex inputs.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append('src')

from models.resnet_bk import LanguageModel
from models.learned_sequence_length import (
    AdaptiveSequenceLengthWrapper,
    LearnedSequenceLengthTrainer
)


def create_synthetic_data(num_samples=1000, vocab_size=100, max_seq_len=128):
    """
    Create synthetic data with varying complexity.
    Simple sequences: short repeated patterns
    Complex sequences: random tokens
    """
    data = []
    
    for i in range(num_samples):
        # Alternate between simple and complex sequences
        if i % 2 == 0:
            # Simple: short repeated pattern (optimal length: 16-32)
            pattern_length = torch.randint(4, 8, (1,)).item()
            pattern = torch.randint(0, vocab_size, (pattern_length,))
            seq = pattern.repeat(max_seq_len // pattern_length + 1)[:max_seq_len]
        else:
            # Complex: random tokens (optimal length: full 128)
            seq = torch.randint(0, vocab_size, (max_seq_len,))
        
        data.append(seq)
    
    return torch.stack(data)


def main():
    print("=" * 80)
    print("Learned Sequence Length Demo")
    print("=" * 80)
    
    # Configuration
    vocab_size = 100
    d_model = 64
    n_layers = 4
    max_seq_len = 128
    num_length_bins = 8
    batch_size = 32
    num_epochs = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nConfiguration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of layers: {n_layers}")
    print(f"  Max sequence length: {max_seq_len}")
    print(f"  Number of length bins: {num_length_bins}")
    print(f"  Device: {device}")
    
    # Create synthetic data
    print(f"\nGenerating synthetic data...")
    train_data = create_synthetic_data(num_samples=1000, vocab_size=vocab_size, max_seq_len=max_seq_len)
    val_data = create_synthetic_data(num_samples=200, vocab_size=vocab_size, max_seq_len=max_seq_len)
    
    # Create targets (next token prediction)
    train_targets = torch.cat([train_data[:, 1:], train_data[:, :1]], dim=1)
    val_targets = torch.cat([val_data[:, 1:], val_data[:, :1]], dim=1)
    
    # Create dataloaders
    train_dataset = TensorDataset(train_data, train_targets.reshape(-1))
    val_dataset = TensorDataset(val_data, val_targets.reshape(-1))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Training samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    
    # Create base model
    print(f"\nCreating base ResNet-BK model...")
    base_model = LanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_seq=max_seq_len,
        num_experts=4,
        top_k=1,
        dropout_p=0.1
    ).to(device)
    
    num_params = sum(p.numel() for p in base_model.parameters())
    print(f"  Base model parameters: {num_params:,}")
    
    # Wrap with adaptive sequence length
    print(f"\nWrapping with adaptive sequence length...")
    model = AdaptiveSequenceLengthWrapper(
        base_model=base_model,
        max_seq_len=max_seq_len,
        num_length_bins=num_length_bins,
        length_penalty=0.01
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    predictor_params = sum(p.numel() for p in model.length_predictor.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Length predictor parameters: {predictor_params:,}")
    print(f"  Overhead: {predictor_params / total_params * 100:.2f}%")
    
    # Length bins
    length_bins = model.length_predictor.length_bins.tolist()
    print(f"\nLength bins: {length_bins}")
    
    # Create trainer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = LearnedSequenceLengthTrainer(model, optimizer, device=device)
    
    # Training loop
    print(f"\n{'='*80}")
    print("Training with Adaptive Sequence Length")
    print(f"{'='*80}\n")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        # Training
        model.train()
        trainer.reset_statistics()
        model.reset_length_statistics()
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            metrics = trainer.train_step(x_batch, y_batch, use_adaptive_length=True)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}: "
                      f"Loss={metrics['total_loss']:.4f}, "
                      f"CE={metrics['ce_loss']:.4f}, "
                      f"Length={metrics['avg_predicted_length']:.1f}, "
                      f"Speedup={metrics['speedup_estimate']:.2f}x")
        
        # Get training statistics
        train_stats = model.get_length_statistics()
        print(f"\n  Training Statistics:")
        print(f"    Avg predicted length: {train_stats['avg_predicted_length']:.1f}")
        print(f"    Avg speedup: {train_stats['avg_speedup']:.2f}x")
        print(f"    Length distribution:")
        for i, (length, pct) in enumerate(zip(train_stats['length_bins'], train_stats['length_distribution'])):
            print(f"      Length {length}: {pct:.1f}%")
        
        # Validation
        print(f"\n  Validation:")
        val_metrics = trainer.evaluate(val_loader, use_adaptive_length=True)
        print(f"    Loss: {val_metrics['loss']:.4f}")
        print(f"    Perplexity: {val_metrics['perplexity']:.2f}")
        print(f"    Avg predicted length: {val_metrics['avg_predicted_length']:.1f}")
        print(f"    Avg speedup: {val_metrics['avg_speedup']:.2f}x")
        
        print()
    
    # Final evaluation
    print(f"\n{'='*80}")
    print("Final Evaluation")
    print(f"{'='*80}\n")
    
    # Compare with and without adaptive length
    print("Without adaptive length:")
    val_metrics_no_adapt = trainer.evaluate(val_loader, use_adaptive_length=False)
    print(f"  Loss: {val_metrics_no_adapt['loss']:.4f}")
    print(f"  Perplexity: {val_metrics_no_adapt['perplexity']:.2f}")
    
    print("\nWith adaptive length:")
    val_metrics_adapt = trainer.evaluate(val_loader, use_adaptive_length=True)
    print(f"  Loss: {val_metrics_adapt['loss']:.4f}")
    print(f"  Perplexity: {val_metrics_adapt['perplexity']:.2f}")
    print(f"  Avg predicted length: {val_metrics_adapt['avg_predicted_length']:.1f}")
    print(f"  Avg speedup: {val_metrics_adapt['avg_speedup']:.2f}x")
    
    # Perplexity degradation
    ppl_degradation = (val_metrics_adapt['perplexity'] - val_metrics_no_adapt['perplexity']) / val_metrics_no_adapt['perplexity'] * 100
    print(f"\nPerplexity degradation: {ppl_degradation:+.2f}%")
    print(f"Speedup: {val_metrics_adapt['avg_speedup']:.2f}x")
    
    # Length distribution
    print(f"\nFinal length distribution:")
    for length, pct in zip(val_metrics_adapt['length_bins'], val_metrics_adapt['length_distribution']):
        bar = '█' * int(pct / 2)
        print(f"  Length {length:3d}: {bar} {pct:.1f}%")
    
    # Test on specific examples
    print(f"\n{'='*80}")
    print("Example Predictions")
    print(f"{'='*80}\n")
    
    model.eval()
    with torch.no_grad():
        # Simple sequence (repeated pattern)
        simple_seq = torch.tensor([1, 2, 3, 4] * 32).unsqueeze(0).to(device)
        _, simple_info = model(simple_seq, use_adaptive_length=True)
        print(f"Simple sequence (repeated pattern):")
        print(f"  Predicted length: {simple_info['predicted_lengths'][0].item()}")
        print(f"  Original length: {simple_info['original_length']}")
        print(f"  Speedup: {simple_info['speedup_estimate']:.2f}x")
        
        # Complex sequence (random)
        complex_seq = torch.randint(0, vocab_size, (1, max_seq_len)).to(device)
        _, complex_info = model(complex_seq, use_adaptive_length=True)
        print(f"\nComplex sequence (random):")
        print(f"  Predicted length: {complex_info['predicted_lengths'][0].item()}")
        print(f"  Original length: {complex_info['original_length']}")
        print(f"  Speedup: {complex_info['speedup_estimate']:.2f}x")
    
    print(f"\n{'='*80}")
    print("Demo completed successfully!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
