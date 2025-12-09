#!/usr/bin/env python3
"""Debug script to trace the full training loop and understand why loss doesn't change."""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.phase8.integrated_model import Phase8IntegratedModel, Phase8Config
from src.optimizers.riemannian_muon_bit import RiemannianMuonBit

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create small model for quick test
    config = Phase8Config(
        vocab_size=1000,
        d_model=256,
        n_layers=2,
        n_seq=64,
        num_heads=4,
        use_bk_hyperbolic=False,
        use_ar_ssm_fusion=False,
        low_rank_ffn=True,
        low_rank_attention=True,
        low_rank_rank=16,
        use_bitnet=False,
    )
    
    model = Phase8IntegratedModel(config).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer (same settings as train_phase8.py)
    optimizer = RiemannianMuonBit(
        model.parameters(),
        lr=0.005,  # Same as config
        momentum=0.95,
        nesterov=True,
        use_orthogonalization=False,  # CRITICAL: disabled
        use_j_orthogonal=False,
        use_stochastic_rounding=False,
        warmup_steps=0,  # No internal warmup
    )
    
    # Track losses
    losses = []
    
    print("\n=== Training for 10 steps ===")
    model.train()
    
    for step in range(10):
        # Same dummy data as train_phase8.py but with CONSISTENT seed
        torch.manual_seed(42 + step)  # Vary per step but reproducible
        x = torch.randint(0, 1000, (1, 64)).to(device)
        y = torch.randint(0, 1000, (64,)).to(device)
        
        # Forward (no mixed precision for simplicity)
        logits, _ = model(x, return_diagnostics=False)
        logits = logits.view(-1, 1000)
        loss = F.cross_entropy(logits, y)
        
        losses.append(loss.item())
        
        # Backward
        loss.backward()
        
        # Check gradient stats
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"Step {step+1}: loss={loss.item():.4f}, avg_grad_norm={avg_grad_norm:.6f}")
    
    print(f"\n=== Summary ===")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Loss change: {losses[-1] - losses[0]:.6f}")
    
    if abs(losses[-1] - losses[0]) < 0.001:
        print("❌ LOSS DID NOT CHANGE - PROBLEM PERSISTS")
    else:
        print("✅ LOSS CHANGED - OPTIMIZER WORKING")

if __name__ == "__main__":
    main()
