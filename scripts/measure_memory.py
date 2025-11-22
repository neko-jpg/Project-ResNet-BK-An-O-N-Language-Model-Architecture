import torch
import torch.nn as nn
import json
import os
import time
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.phase4.integrated_model import Phase4IntegratedModel
from src.models.phase4.memory_monitor import MemoryMonitor
from src.models.phase3.config import Phase3Config

class MockPhase3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        self.dialectic = nn.Linear(self.d_model, self.d_model) # Dummy layer for hook

    def forward(self, input_ids, labels=None, return_diagnostics=False):
        B, N = input_ids.shape
        # Simulate computation
        logits = torch.randn(B, N, self.vocab_size)
        loss = torch.tensor(0.0)

        # Manually trigger hook?
        # Phase4IntegratedModel registers hook on self.dialectic.
        # We must run self.dialectic to trigger it.
        dummy_hidden = torch.randn(B, N, self.d_model)
        _ = self.dialectic(dummy_hidden)

        return {'logits': logits, 'loss': loss, 'diagnostics': {}}

def run_benchmark():
    print("Starting Phase 4 Memory Benchmark (with Mock Phase 3)...")

    # Configuration
    d_model = 512
    seq_len = 4096
    vocab_size = 1000

    config = Phase3Config(
        d_model=d_model,
        vocab_size=vocab_size,
        n_layers=4,
        max_seq_len=seq_len
    )

    # Initialize Models
    print("Initializing Mock Phase 3 Model...")
    phase3_model = MockPhase3Model(config)

    print("Initializing Phase 4 Integrated Model...")
    model = Phase4IntegratedModel(
        phase3_model=phase3_model,
        enable_emotion=True,
        enable_dream=True,
        enable_holographic=True,
        enable_quantum=True,
        enable_topological=True,
        enable_ethics=True
    )
    model.eval()

    monitor = model.memory_monitor

    # Set mock usage to simulate high load
    if monitor.use_mock:
        print("Running in Mock Memory Mode (CPU)")
        # Simulate having only 1.5GB free (Total 8GB, Used 6.5GB)
        monitor.set_mock_usage(6.5)

    # Input
    input_ids = torch.randint(0, vocab_size, (1, seq_len))

    # Benchmark
    start_mem = monitor.get_memory_stats()
    print(f"Initial Memory: {start_mem}")

    try:
        with torch.no_grad():
            start_time = time.time()
            output = model(input_ids, return_diagnostics=True)
            end_time = time.time()

        end_mem = monitor.get_memory_stats()
        print(f"Final Memory: {end_mem}")

        diagnostics = output.get('diagnostics', {})
        bulk_info = diagnostics.get('bulk', {})

        # Verify Logic Triggered
        active_dim = bulk_info.get('active_bulk_dim', 'N/A')
        low_mem = bulk_info.get('low_memory_mode', 'N/A')
        print(f"Active Bulk Dim: {active_dim}")
        print(f"Low Memory Mode: {low_mem}")

        results = {
            'device': 'cpu' if monitor.use_mock else 'cuda',
            'd_model': d_model,
            'seq_len': seq_len,
            'initial_memory_mb': start_mem['used_mb'],
            'peak_memory_mb': end_mem['used_mb'],
            'latency_sec': end_time - start_time,
            'bulk_active_dim': active_dim,
            'low_memory_mode': low_mem,
            'success': True
        }

    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        results = {
            'success': False,
            'error': str(e)
        }

    # Save Results
    output_dir = Path('results/benchmarks')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'phase4_memory_benchmark.json'

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    run_benchmark()
