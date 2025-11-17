"""
Birman-Schwinger Integration Demo

Demonstrates the integration of Birman-Schwinger core into ResNet-BK
with stability monitoring and Prime-Bump initialization.
"""

import torch
import torch.nn as nn
from src.models.resnet_bk import LanguageModel
from src.models.mourre_lap import StabilityDashboard


def demo_birman_schwinger_integration():
    """
    Demo: Birman-Schwinger core integration with stability monitoring.
    """
    print("=" * 80)
    print("Birman-Schwinger Integration Demo")
    print("=" * 80)
    
    # Configuration
    vocab_size = 1000
    d_model = 64
    n_layers = 2
    n_seq = 128
    batch_size = 4
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Create model with Birman-Schwinger core
    print("\n1. Creating model with Birman-Schwinger core...")
    model = LanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_seq=n_seq,
        num_experts=4,
        top_k=2,
        use_birman_schwinger=True,
        epsilon=1.0,
        use_mourre=True,
        use_lap=True,
        prime_bump_init=True,
        prime_bump_scale=0.02,
        k_max=3,
    ).to(device)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Using Birman-Schwinger: {model.use_birman_schwinger}")
    print(f"   Prime-Bump initialization: {model.prime_bump_init}")
    
    # Create stability dashboard
    print("\n2. Initializing stability dashboard...")
    dashboard = StabilityDashboard(n_seq=n_seq, device=device)
    
    # Generate sample input
    print("\n3. Running forward pass...")
    x = torch.randint(0, vocab_size, (batch_size, n_seq), device=device)
    
    # Forward pass
    with torch.no_grad():
        logits = model(x)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {logits.shape}")
    
    # Get stability diagnostics
    print("\n4. Collecting stability diagnostics...")
    diagnostics = model.get_stability_diagnostics()
    
    if diagnostics:
        print("\n   Schatten Norms:")
        print(f"      Mean S1: {diagnostics.get('mean_schatten_s1', 0.0):.4f}")
        print(f"      Max S1:  {diagnostics.get('max_schatten_s1', 0.0):.4f}")
        print(f"      Mean S2: {diagnostics.get('mean_schatten_s2', 0.0):.4f}")
        print(f"      Max S2:  {diagnostics.get('max_schatten_s2', 0.0):.4f}")
        
        print("\n   Condition Numbers:")
        print(f"      Mean: {diagnostics.get('mean_condition_number', 0.0):.2e}")
        print(f"      Max:  {diagnostics.get('max_condition_number', 0.0):.2e}")
        
        print("\n   Verification Rates:")
        print(f"      Mourre verified: {diagnostics.get('mourre_verified_rate', 0.0):.1%}")
        print(f"      S1 bound satisfied: {diagnostics.get('s1_bound_satisfied_rate', 0.0):.1%}")
        print(f"      S2 bound satisfied: {diagnostics.get('s2_bound_satisfied_rate', 0.0):.1%}")
        print(f"      All finite: {diagnostics.get('all_finite_rate', 1.0):.1%}")
        
        print(f"\n   Precision upgrades: {diagnostics.get('precision_upgrades', 0)}")
    else:
        print("   No diagnostics available (not using Birman-Schwinger)")
    
    # Test Prime-Bump potential
    if model.prime_bump_potential is not None:
        print("\n5. Prime-Bump Potential Statistics...")
        stats = model.prime_bump_potential.get_statistics()
        
        print(f"   Number of primes: {stats['num_primes']}")
        print(f"   Epsilon: {stats['epsilon']}")
        print(f"   k_max: {stats['k_max']}")
        print(f"   L1 norm: {stats['l1_norm']:.4f}")
        print(f"   L2 norm: {stats['l2_norm']:.4f}")
        print(f"   Mean potential: {stats['mean_potential']:.4f}")
        print(f"   Std potential: {stats['std_potential']:.4f}")
        print(f"   Overlap fraction: {stats['overlap_fraction']:.2%}")
        
        # Verify GUE statistics
        print("\n6. Verifying GUE statistics...")
        gue_results = model.prime_bump_potential.verify_gue_statistics()
        
        print(f"   Mean spacing: {gue_results['mean_spacing']:.4f} (expected: {gue_results['wigner_expected_mean']:.4f})")
        print(f"   Std spacing: {gue_results['std_spacing']:.4f} (expected: {gue_results['wigner_expected_std']:.4f})")
        print(f"   Wigner fit error: {gue_results['wigner_fit_error']:.4f}")
        print(f"   GUE verified: {gue_results['gue_verified']}")
    
    # Compare with original BK-Core
    print("\n7. Comparing with original BK-Core...")
    model_original = LanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_seq=n_seq,
        num_experts=4,
        top_k=2,
        use_birman_schwinger=False,  # Use original
        prime_bump_init=False,
    ).to(device)
    
    with torch.no_grad():
        logits_original = model_original(x)
    
    print(f"   Original BK-Core output shape: {logits_original.shape}")
    print(f"   Output difference (mean abs): {(logits - logits_original).abs().mean().item():.4f}")
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


def demo_stability_monitoring():
    """
    Demo: Real-time stability monitoring during training.
    """
    print("\n" + "=" * 80)
    print("Stability Monitoring Demo")
    print("=" * 80)
    
    # Configuration
    n_seq = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create stability dashboard
    dashboard = StabilityDashboard(n_seq=n_seq, device=device)
    
    print(f"\nDevice: {device}")
    print(f"Sequence length: {n_seq}")
    
    # Simulate training steps
    print("\nSimulating training steps...")
    
    for step in range(10):
        # Create dummy tensors
        H = torch.randn(n_seq, n_seq, device=device)
        H = (H + H.T) / 2  # Make symmetric
        
        K = torch.randn(2, n_seq, n_seq, device=device, dtype=torch.complex64)
        V = torch.randn(2, n_seq, device=device)
        
        # Update dashboard
        metrics = dashboard.update(
            step=step,
            H=H,
            K=K,
            V=V,
            z=1.0j,
            epsilon=1.0,
            tensors={'H': H, 'K': K, 'V': V}
        )
        
        if step % 3 == 0:
            print(f"\n   Step {step}:")
            print(f"      Condition number: {metrics.condition_number:.2e}")
            print(f"      Schatten S2: {metrics.schatten_s2:.4f}")
            print(f"      Mourre verified: {metrics.mourre_verified}")
            print(f"      LAP verified: {metrics.lap_verified}")
            print(f"      All finite: {metrics.all_finite}")
    
    # Get summary
    print("\n" + "-" * 80)
    print("Summary Statistics:")
    print("-" * 80)
    
    summary = dashboard.get_summary()
    
    print(f"\nTotal steps: {summary['total_steps']}")
    
    print("\nCondition Number:")
    print(f"   Mean: {summary['condition_number']['mean']:.2e}")
    print(f"   Max:  {summary['condition_number']['max']:.2e}")
    print(f"   Min:  {summary['condition_number']['min']:.2e}")
    
    print("\nSchatten Norms:")
    print(f"   S1 mean: {summary['schatten_s1']['mean']:.4f}")
    print(f"   S2 mean: {summary['schatten_s2']['mean']:.4f}")
    
    print("\nVerification Rates:")
    print(f"   Mourre verified: {summary['mourre_verified_rate']:.1%}")
    print(f"   LAP verified: {summary['lap_verified_rate']:.1%}")
    
    print(f"\nNumerical Health:")
    print(f"   NaN count: {summary['nan_count']}")
    print(f"   Inf count: {summary['inf_count']}")
    print(f"   Total alerts: {summary['total_alerts']}")
    
    # Show recent alerts
    alerts = dashboard.get_recent_alerts(5)
    if alerts:
        print("\nRecent Alerts:")
        for alert in alerts:
            print(f"   Step {alert['step']}: {alert['message']}")
    
    print("\n" + "=" * 80)
    print("Stability monitoring demo completed!")
    print("=" * 80)


if __name__ == '__main__':
    # Run demos
    demo_birman_schwinger_integration()
    demo_stability_monitoring()
