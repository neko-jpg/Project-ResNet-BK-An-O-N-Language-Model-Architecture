"""
Demonstration of Adaptive Computation Time (ACT) for ResNet-BK.

This script shows how ACT dynamically adjusts the number of layers
executed based on input difficulty.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from src.models.adaptive_computation import ACTLanguageModel, ACTTrainer


def demonstrate_act():
    """Demonstrate ACT with different thresholds."""
    print("=" * 60)
    print("Adaptive Computation Time (ACT) Demonstration")
    print("=" * 60)
    
    # Model configuration
    vocab_size = 10000
    d_model = 64
    n_layers = 4
    n_seq = 128
    batch_size = 4
    
    # Create sample data
    x_batch = torch.randint(0, vocab_size, (batch_size, n_seq))
    y_batch = torch.randint(0, vocab_size, (batch_size * n_seq,))
    
    print(f"\nModel Configuration:")
    print(f"  - Vocabulary size: {vocab_size}")
    print(f"  - Hidden dimension: {d_model}")
    print(f"  - Number of layers: {n_layers}")
    print(f"  - Sequence length: {n_seq}")
    print(f"  - Batch size: {batch_size}")
    
    # Test different ACT thresholds
    thresholds = [0.5, 0.8, 0.95, 0.99]
    lambdas = [0.001, 0.01, 0.1]
    
    print("\n" + "=" * 60)
    print("Testing Different ACT Thresholds")
    print("=" * 60)
    
    for threshold in thresholds:
        print(f"\n--- Threshold: {threshold} ---")
        
        model = ACTLanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_seq=n_seq,
            act_threshold=threshold,
            act_lambda=0.01
        )
        
        # Forward pass
        logits, ponder_cost = model(x_batch, return_ponder_cost=True)
        avg_layers = model.get_avg_layers_executed()
        
        # Compute loss
        total_loss, ce_loss, ponder_cost_val = model.compute_loss(
            logits, y_batch, ponder_cost
        )
        
        print(f"  Average layers executed: {avg_layers:.2f} / {n_layers}")
        print(f"  Ponder cost: {ponder_cost_val:.4f}")
        print(f"  CE loss: {ce_loss:.4f}")
        print(f"  Total loss: {total_loss:.4f}")
        print(f"  Speedup potential: {n_layers / avg_layers:.2f}x")
    
    print("\n" + "=" * 60)
    print("Testing Different Ponder Cost Weights (λ)")
    print("=" * 60)
    
    threshold = 0.95
    for lambda_val in lambdas:
        print(f"\n--- Lambda: {lambda_val} ---")
        
        model = ACTLanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_seq=n_seq,
            act_threshold=threshold,
            act_lambda=lambda_val
        )
        
        # Forward pass
        logits, ponder_cost = model(x_batch, return_ponder_cost=True)
        avg_layers = model.get_avg_layers_executed()
        
        # Compute loss
        total_loss, ce_loss, ponder_cost_val = model.compute_loss(
            logits, y_batch, ponder_cost
        )
        
        print(f"  Average layers executed: {avg_layers:.2f} / {n_layers}")
        print(f"  Ponder cost: {ponder_cost_val:.4f}")
        print(f"  CE loss: {ce_loss:.4f}")
        print(f"  Total loss: {total_loss:.4f}")
        print(f"  Ponder cost contribution: {(lambda_val * ponder_cost_val):.4f}")


def train_act_model():
    """Train ACT model for a few steps to show learning."""
    print("\n" + "=" * 60)
    print("Training ACT Model")
    print("=" * 60)
    
    # Model configuration
    vocab_size = 10000
    d_model = 64
    n_layers = 4
    n_seq = 128
    batch_size = 4
    num_steps = 10
    
    model = ACTLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_seq=n_seq,
        act_threshold=0.95,
        act_lambda=0.01
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = ACTTrainer(model, optimizer)
    
    print(f"\nTraining for {num_steps} steps...")
    print(f"{'Step':<6} {'Loss':<10} {'CE Loss':<10} {'Ponder':<10} {'Avg Layers':<12}")
    print("-" * 60)
    
    for step in range(num_steps):
        # Generate random batch
        x_batch = torch.randint(0, vocab_size, (batch_size, n_seq))
        y_batch = torch.randint(0, vocab_size, (batch_size * n_seq,))
        
        # Training step
        metrics = trainer.train_step(x_batch, y_batch)
        
        print(f"{step+1:<6} {metrics['total_loss']:<10.4f} "
              f"{metrics['ce_loss']:<10.4f} {metrics['ponder_cost']:<10.4f} "
              f"{metrics['avg_layers_executed']:<12.2f}")
    
    # Show average metrics
    avg_metrics = trainer.get_average_metrics()
    print("\n" + "-" * 60)
    print(f"Average CE Loss: {avg_metrics['avg_ce_loss']:.4f}")
    print(f"Average Ponder Cost: {avg_metrics['avg_ponder_cost']:.4f}")


def compare_act_vs_standard():
    """Compare ACT model with standard model."""
    print("\n" + "=" * 60)
    print("ACT vs Standard Model Comparison")
    print("=" * 60)
    
    vocab_size = 10000
    d_model = 64
    n_layers = 4
    n_seq = 128
    batch_size = 4
    
    x_batch = torch.randint(0, vocab_size, (batch_size, n_seq))
    
    # Standard model (all layers always executed)
    print("\n--- Standard Model ---")
    from src.models.resnet_bk import LanguageModel
    
    standard_model = LanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_seq=n_seq
    )
    
    with torch.no_grad():
        logits_standard = standard_model(x_batch)
    
    print(f"  Layers executed: {n_layers} (always)")
    print(f"  Output shape: {logits_standard.shape}")
    
    # ACT model (adaptive layers)
    print("\n--- ACT Model ---")
    act_model = ACTLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_seq=n_seq,
        act_threshold=0.95,
        act_lambda=0.01
    )
    
    with torch.no_grad():
        logits_act, ponder_cost = act_model(x_batch, return_ponder_cost=True)
    
    avg_layers = act_model.get_avg_layers_executed()
    
    print(f"  Layers executed: {avg_layers:.2f} (average)")
    print(f"  Output shape: {logits_act.shape}")
    print(f"  Ponder cost: {ponder_cost:.4f}")
    print(f"  Computational savings: {(1 - avg_layers/n_layers)*100:.1f}%")
    print(f"  Speedup: {n_layers/avg_layers:.2f}x")


if __name__ == '__main__':
    # Run demonstrations
    demonstrate_act()
    train_act_model()
    compare_act_vs_standard()
    
    print("\n" + "=" * 60)
    print("ACT Demonstration Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. Lower thresholds → fewer layers executed → faster inference")
    print("  2. Higher λ → stronger penalty on computation → encourages early halting")
    print("  3. ACT adapts computation to input difficulty")
    print("  4. Potential for 1.3-2x speedup with minimal accuracy loss")
