"""
Mourre Estimate and LAP Verification Demo

Demonstrates:
1. Mourre estimate verification
2. LAP uniform bounds verification
3. Real-time stability dashboard
4. Integration with Birman-Schwinger core
"""

import torch
import numpy as np
from src.models.mourre_lap import (
    MourreEstimateVerifier,
    LAPVerifier,
    StabilityDashboard,
    verify_birman_schwinger_stability,
)
from src.models.birman_schwinger_core import BirmanSchwingerCore


def demo_mourre_verification():
    """Demonstrate Mourre estimate verification."""
    print("=" * 60)
    print("Mourre Estimate Verification Demo")
    print("=" * 60)
    
    n_seq = 64
    verifier = MourreEstimateVerifier(n_seq)
    
    print(f"\nSequence length: {n_seq}")
    print(f"Free Hamiltonian H_0 shape: {verifier.H_0.shape}")
    print(f"Position operator A shape: {verifier.A.shape}")
    
    # Verify Mourre estimate
    results = verifier.verify()
    
    print(f"\nMourre Estimate Results:")
    print(f"  Verified: {results['verified']}")
    print(f"  Mourre constant: {results['mourre_constant']:.6f}")
    print(f"  Mean eigenvalue: {results['mean_eigenvalue']:.6f}")
    print(f"  Max eigenvalue: {results['max_eigenvalue']:.6f}")
    print(f"  Hermitian error: {results['hermitian_error']:.2e}")
    print(f"  Commutator norm: {results['commutator_norm']:.6f}")
    
    if results['verified']:
        print("\n✓ Mourre estimate verified successfully!")
    else:
        print("\n✗ Mourre estimate verification failed")


def demo_lap_verification():
    """Demonstrate LAP verification."""
    print("\n" + "=" * 60)
    print("Limiting Absorption Principle (LAP) Verification Demo")
    print("=" * 60)
    
    n_seq = 32
    s = 1.0
    verifier = LAPVerifier(n_seq, s=s)
    
    print(f"\nSequence length: {n_seq}")
    print(f"Weight exponent s: {s}")
    print(f"Weight function ⟨x⟩^{{-s}} shape: {verifier.weight.shape}")
    
    # Create free Hamiltonian
    mourre_verifier = MourreEstimateVerifier(n_seq)
    H_0 = mourre_verifier.H_0
    
    # Test uniform bounds
    eta_values = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    results = verifier.verify_uniform_bounds(
        H_0,
        lambda_=0.0,
        eta_values=eta_values,
        C_bound=100.0
    )
    
    print(f"\nLAP Uniform Bounds Results:")
    print(f"  Verified: {results['verified']}")
    print(f"  Max norm: {results['max_norm']:.4f}")
    print(f"  Min norm: {results['min_norm']:.4f}")
    print(f"  C bound: {results['C_bound']:.4f}")
    print(f"  Bounded growth: {results['bounded_growth']}")
    
    print(f"\n  Norms as η → 0:")
    for eta, norm in zip(eta_values, results['norms']):
        print(f"    η = {eta:.4f}: ||R|| = {norm:.4f}")
    
    # Test continuity
    continuity_results = verifier.verify_continuity_at_boundary(
        H_0,
        lambda_=0.0,
        eta_sequence=eta_values
    )
    
    print(f"\nLAP Continuity Results:")
    print(f"  Continuous: {continuity_results['continuous']}")
    print(f"  Max difference: {continuity_results['max_difference']:.4f}")
    
    if results['verified'] and continuity_results['continuous']:
        print("\n✓ LAP verified successfully!")
    else:
        print("\n✗ LAP verification failed")


def demo_stability_dashboard():
    """Demonstrate real-time stability dashboard."""
    print("\n" + "=" * 60)
    print("Real-time Stability Dashboard Demo")
    print("=" * 60)
    
    n_seq = 32
    dashboard = StabilityDashboard(n_seq, history_size=100)
    
    print(f"\nSequence length: {n_seq}")
    print(f"History size: {dashboard.history_size}")
    
    # Simulate training loop
    print("\nSimulating training loop...")
    num_steps = 20
    
    for step in range(num_steps):
        # Create mock data
        H = torch.eye(n_seq) * (2.0 + 0.1 * np.sin(step * 0.5))
        K = torch.randn(2, n_seq, n_seq, dtype=torch.complex64) * 0.1
        V = torch.randn(2, n_seq) * 0.5
        tensors = {
            'activations': torch.randn(2, n_seq, 64),
            'gradients': torch.randn(2, n_seq, 64) * 0.01,
        }
        
        # Update dashboard
        metrics = dashboard.update(
            step=step,
            H=H,
            K=K,
            V=V,
            tensors=tensors
        )
        
        if step % 5 == 0:
            print(f"\n  Step {step}:")
            print(f"    Mourre verified: {metrics.mourre_verified}")
            print(f"    LAP verified: {metrics.lap_verified}")
            print(f"    Condition number: {metrics.condition_number:.2f}")
            print(f"    Schatten S1: {metrics.schatten_s1:.4f}")
            print(f"    Schatten S2: {metrics.schatten_s2:.4f}")
            print(f"    All finite: {metrics.all_finite}")
    
    # Get summary
    summary = dashboard.get_summary()
    
    print(f"\n\nTraining Summary:")
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Condition number:")
    print(f"    Mean: {summary['condition_number']['mean']:.2f}")
    print(f"    Max: {summary['condition_number']['max']:.2f}")
    print(f"    Min: {summary['condition_number']['min']:.2f}")
    print(f"  Schatten S1:")
    print(f"    Mean: {summary['schatten_s1']['mean']:.4f}")
    print(f"    Max: {summary['schatten_s1']['max']:.4f}")
    print(f"  Schatten S2:")
    print(f"    Mean: {summary['schatten_s2']['mean']:.4f}")
    print(f"    Max: {summary['schatten_s2']['max']:.4f}")
    print(f"  Mourre verified rate: {summary['mourre_verified_rate']:.2%}")
    print(f"  LAP verified rate: {summary['lap_verified_rate']:.2%}")
    print(f"  NaN count: {summary['nan_count']}")
    print(f"  Inf count: {summary['inf_count']}")
    print(f"  Total alerts: {summary['total_alerts']}")
    
    # Show recent alerts
    recent_alerts = dashboard.get_recent_alerts(n=5)
    if recent_alerts:
        print(f"\n  Recent alerts:")
        for alert in recent_alerts:
            print(f"    Step {alert['step']}: {alert['message']}")
    else:
        print(f"\n  No alerts generated ✓")


def demo_comprehensive_verification():
    """Demonstrate comprehensive verification."""
    print("\n" + "=" * 60)
    print("Comprehensive Birman-Schwinger Stability Verification")
    print("=" * 60)
    
    n_seq = 64
    epsilon = 1.0
    
    print(f"\nSequence length: {n_seq}")
    print(f"Epsilon: {epsilon}")
    
    # Run comprehensive verification
    results = verify_birman_schwinger_stability(n_seq, epsilon)
    
    print(f"\nMourre Estimate:")
    print(f"  Verified: {results['mourre']['verified']}")
    print(f"  Mourre constant: {results['mourre']['mourre_constant']:.6f}")
    
    print(f"\nLAP Uniform Bounds:")
    print(f"  Verified: {results['lap_uniform_bounds']['verified']}")
    print(f"  Max norm: {results['lap_uniform_bounds']['max_norm']:.4f}")
    
    print(f"\nLAP Continuity:")
    print(f"  Continuous: {results['lap_continuity']['continuous']}")
    print(f"  Max difference: {results['lap_continuity']['max_difference']:.4f}")
    
    print(f"\nOverall:")
    print(f"  All verified: {results['all_verified']}")
    
    if results['all_verified']:
        print("\n✓ All stability checks passed!")
    else:
        print("\n✗ Some stability checks failed")


def demo_integration_with_bk_core():
    """Demonstrate integration with Birman-Schwinger core."""
    print("\n" + "=" * 60)
    print("Integration with Birman-Schwinger Core Demo")
    print("=" * 60)
    
    n_seq = 32
    batch_size = 2
    
    # Initialize components
    bk_core = BirmanSchwingerCore(
        n_seq=n_seq,
        epsilon=1.0,
        use_mourre=True,
        use_lap=True
    )
    dashboard = StabilityDashboard(n_seq)
    
    print(f"\nSequence length: {n_seq}")
    print(f"Batch size: {batch_size}")
    
    # Create mock potential
    v = torch.randn(batch_size, n_seq) * 0.5
    
    # Forward pass
    print("\nRunning forward pass...")
    features, diagnostics = bk_core(v, z=1.0j)
    
    print(f"\nBK-Core Diagnostics:")
    print(f"  Schatten S1: {diagnostics['schatten_s1']:.4f}")
    print(f"  Schatten S2: {diagnostics['schatten_s2']:.4f}")
    print(f"  Condition number: {diagnostics['condition_number']:.2f}")
    print(f"  Mourre verified: {diagnostics['mourre_verified']}")
    print(f"  All finite: {diagnostics['all_finite']}")
    print(f"  S1 bound satisfied: {diagnostics['s1_bound_satisfied']}")
    print(f"  S2 bound satisfied: {diagnostics['s2_bound_satisfied']}")
    
    # Update dashboard
    metrics = dashboard.update(
        step=0,
        V=v,
        tensors={'features': features}
    )
    
    print(f"\nDashboard Metrics:")
    print(f"  Mourre constant: {metrics.mourre_constant:.6f}")
    print(f"  Mourre verified: {metrics.mourre_verified}")
    print(f"  All finite: {metrics.all_finite}")
    
    # Get BK-Core statistics
    stats = bk_core.get_statistics()
    
    print(f"\nBK-Core Statistics:")
    print(f"  Mean Schatten S1: {stats['mean_schatten_s1']:.4f}")
    print(f"  Mean Schatten S2: {stats['mean_schatten_s2']:.4f}")
    print(f"  Mean condition number: {stats['mean_condition_number']:.2f}")
    print(f"  Max condition number: {stats['max_condition_number']:.2f}")
    print(f"  Precision upgrades: {stats['precision_upgrades']}")
    
    if diagnostics['all_finite'] and diagnostics['mourre_verified']:
        print("\n✓ Numerical stability verified!")
    else:
        print("\n✗ Numerical issues detected")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("MOURRE ESTIMATE AND LAP VERIFICATION DEMO")
    print("=" * 60)
    
    # Run demos
    demo_mourre_verification()
    demo_lap_verification()
    demo_stability_dashboard()
    demo_comprehensive_verification()
    demo_integration_with_bk_core()
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
