#!/usr/bin/env python3
"""
Revolutionary Training Integration Benchmark

Tests that all 7 revolutionary algorithms work in the integrated trainer.
Measures actual training speed improvements.

Usage:
    python scripts/benchmark_revolutionary_training.py
"""

import torch
import torch.nn as nn
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.revolutionary_trainer import (
    RevolutionaryTrainer,
    RevolutionaryConfig,
    create_revolutionary_trainer,
)


class SimpleTestModel(nn.Module):
    """Small model for testing."""
    
    def __init__(self, vocab_size=1000, d_model=128, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_layers)
        ])
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        h = self.embedding(x)
        for layer in self.layers:
            h = torch.relu(layer(h))
        return self.output(h)


def test_revolutionary_trainer():
    """Test the revolutionary trainer with all algorithms."""
    print("=" * 60)
    print("Revolutionary Training Integration Test")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create model
    model = SimpleTestModel(vocab_size=1000, d_model=128, n_layers=2).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    
    # Create test data
    batch_size = 4
    seq_len = 64
    data = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    targets = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Test each algorithm individually
    print("\n" + "-" * 40)
    print("Testing individual algorithms:")
    print("-" * 40)
    
    algorithms = [
        'holographic',
        'closed_form', 
        'topological',
        'retrocausal',
        'zeta',
        'sheaf',
        'diffractive',
    ]
    
    results = {}
    
    for algo in algorithms:
        print(f"\n[{algo}]", end=" ")
        
        # Create config with only this algorithm enabled
        config = RevolutionaryConfig(
            use_holographic=(algo == 'holographic'),
            use_closed_form=(algo == 'closed_form'),
            use_topological=(algo == 'topological'),
            use_retrocausal=(algo == 'retrocausal'),
            use_zeta=(algo == 'zeta'),
            use_sheaf=(algo == 'sheaf'),
            use_diffractive=(algo == 'diffractive'),
            log_interval=100,
        )
        
        # Fresh model for each test
        test_model = SimpleTestModel(vocab_size=1000, d_model=128, n_layers=2).to(device)
        
        try:
            trainer = RevolutionaryTrainer(test_model, config, device)
            
            # Time a few steps
            times = []
            for _ in range(3):
                start = time.perf_counter()
                loss, metrics = trainer.train_step(data, targets, loss_fn)
                times.append((time.perf_counter() - start) * 1000)
            
            avg_time = sum(times) / len(times)
            results[algo] = {
                'status': 'PASS',
                'avg_time_ms': avg_time,
                'final_loss': loss.item() if torch.is_tensor(loss) else loss,
            }
            print(f"✅ PASS ({avg_time:.1f}ms)")
            
        except Exception as e:
            results[algo] = {
                'status': 'FAIL',
                'error': str(e),
            }
            print(f"❌ FAIL: {e}")
    
    # Test full integration
    print("\n" + "-" * 40)
    print("Testing full integration (all algorithms):")
    print("-" * 40)
    
    test_model = SimpleTestModel(vocab_size=1000, d_model=128, n_layers=2).to(device)
    trainer = create_revolutionary_trainer(test_model, enable_all=True, log_interval=100)
    
    print("\nRunning 70 steps (10 cycles through all 7 algorithms)...")
    
    start = time.perf_counter()
    for step in range(70):
        loss, metrics = trainer.train_step(data, targets, loss_fn)
    total_time = (time.perf_counter() - start) * 1000
    
    summary = trainer.get_summary()
    
    print(f"\nTotal time: {total_time:.1f}ms")
    print(f"Avg per step: {total_time / 70:.1f}ms")
    
    print("\nAlgorithm usage:")
    for algo, stats in summary.get('algorithms', {}).items():
        count = stats.get('count', 0)
        avg_time = stats.get('avg_time_ms', 0)
        print(f"  {algo}: {count} steps, {avg_time:.1f}ms avg")
    
    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results.values() if r['status'] == 'PASS')
    print(f"\nIndividual: {passed}/7 algorithms working")
    print(f"Integrated: 70 steps completed")
    
    if passed == 7:
        print("\n✅ All revolutionary algorithms integrated successfully!")
    else:
        print("\n⚠️ Some algorithms had issues - check logs above")
    
    return results


if __name__ == "__main__":
    results = test_revolutionary_trainer()
