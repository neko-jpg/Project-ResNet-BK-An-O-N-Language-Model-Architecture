#!/usr/bin/env python3
"""Debug script to verify if optimizer is actually updating weights."""

import torch
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
        use_bk_hyperbolic=False,  # Disable for simpler test
        use_ar_ssm_fusion=False,
        low_rank_ffn=True,
        low_rank_attention=True,
        low_rank_rank=16,
        use_bitnet=False,  # Disable BitNet for cleaner test
    )
    
    model = Phase8IntegratedModel(config).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = RiemannianMuonBit(
        model.parameters(),
        lr=0.01,  # Higher LR for visible change
        momentum=0.95,
        nesterov=True,
        use_orthogonalization=False,
        warmup_steps=0,
    )
    
    # Get initial weight snapshot
    initial_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_weights[name] = param.data.clone()
    
    print(f"Tracked {len(initial_weights)} parameters")
    
    # Training step
    model.train()
    x = torch.randint(0, 1000, (1, 64)).to(device)
    y = torch.randint(0, 1000, (64,)).to(device)
    
    # Forward
    logits, _ = model(x, return_diagnostics=False)
    logits = logits.view(-1, 1000)
    loss = torch.nn.functional.cross_entropy(logits, y)
    
    print(f"Initial loss: {loss.item():.4f}")
    
    # Backward
    loss.backward()
    
    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append((name, param.grad.norm().item()))
    
    grad_norms.sort(key=lambda x: -x[1])
    print(f"\nTop 5 gradient norms:")
    for name, norm in grad_norms[:5]:
        print(f"  {name}: {norm:.6f}")
    
    # Optimizer step
    print("\nCalling optimizer.step()...")
    optimizer.step()
    optimizer.zero_grad()
    
    # Check weight changes
    print("\nWeight changes after optimizer.step():")
    changes = []
    for name, param in model.named_parameters():
        if name in initial_weights:
            diff = (param.data - initial_weights[name]).abs().mean().item()
            max_diff = (param.data - initial_weights[name]).abs().max().item()
            changes.append((name, diff, max_diff))
    
    changes.sort(key=lambda x: -x[1])
    
    total_changed = sum(1 for _, diff, _ in changes if diff > 1e-8)
    print(f"Parameters with changes > 1e-8: {total_changed}/{len(changes)}")
    
    print("\nTop 10 changed parameters:")
    for name, mean_diff, max_diff in changes[:10]:
        print(f"  {name}: mean={mean_diff:.8f}, max={max_diff:.8f}")
    
    # Second forward to check loss change
    x2 = torch.randint(0, 1000, (1, 64)).to(device)
    y2 = torch.randint(0, 1000, (64,)).to(device)
    logits2, _ = model(x2, return_diagnostics=False)
    logits2 = logits2.view(-1, 1000)
    loss2 = torch.nn.functional.cross_entropy(logits2, y2)
    
    print(f"\nLoss after update: {loss2.item():.4f}")
    print(f"Loss diff: {loss2.item() - loss.item():.6f}")
    
    # Verdict
    if total_changed > 0:
        print("\n✅ WEIGHTS ARE BEING UPDATED")
    else:
        print("\n❌ WEIGHTS ARE NOT BEING UPDATED - BUG IN OPTIMIZER")

if __name__ == "__main__":
    main()
