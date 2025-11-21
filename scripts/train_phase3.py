"""
Training Script for Phase 3 (Task 24)

Stage 1 -> Stage 2 -> Stage 3 Training Loop.
Includes diagnostics logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.phase3.factory import create_phase3_model, get_preset_config
from src.models.phase3.config import Phase3Config

def train_phase3():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, default=3, help='Target stage (1, 2, or 3)')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--preset', type=str, default='small')
    args = parser.parse_args()

    print(f"Starting Phase 3 Training (Target Stage: {args.stage})")

    # Config
    config = get_preset_config(args.preset)
    model = create_phase3_model(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Dummy Data Loader
    print("Generating dummy data...")
    vocab_size = config.vocab_size
    seq_len = 128 # Smaller for demo
    dataset_size = 100
    data = torch.randint(0, vocab_size, (dataset_size, seq_len))

    model.train()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        total_loss = 0

        for i in range(0, dataset_size, args.batch_size):
            batch = data[i:i+args.batch_size].to(device)

            optimizer.zero_grad()

            # Forward
            # Stage control could be implemented by freezing parts or different loss
            # Here we run full forward
            outputs = model(batch, labels=batch, return_diagnostics=True)

            loss = outputs['loss']
            loss.backward()

            # Clip grad
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

            # Diagnostics
            if i % 20 == 0:
                diag = outputs['diagnostics']
                dialectic_diag = diag['dialectic_diagnostics']
                print(f"  Step {i}: Loss={loss.item():.4f}, "
                      f"Contradiction={dialectic_diag['contradiction_score']:.4f}, "
                      f"EnergyDrift={dialectic_diag['energy_drift_max']:.2e}")

        avg_loss = total_loss / (dataset_size / args.batch_size)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # Anneal Temperature
        if hasattr(model, 'dialectic'):
            model.dialectic.anneal_temperature()
            print(f"  New Temperature: {model.dialectic.temperature:.4f}")

    # Save Checkpoint
    output_dir = project_root / "checkpoints" / "phase3"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / f"phase3_stage{args.stage}_final.pt")
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    train_phase3()
