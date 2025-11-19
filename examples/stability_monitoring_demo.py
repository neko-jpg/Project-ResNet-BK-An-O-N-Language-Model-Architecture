"""
Birman-Schwinger Stability Monitor Demo

このデモは、Phase 1の安定性監視機能を実演します。

物理的直観 (Physical Intuition):
Birman-Schwinger演算子 K_ε の安定性を監視することで、
訓練中の発散を事前に検出し、自動的に回復アクションを提案します。

Requirements: 7.1, 7.2, 7.3, 7.4, 12.2
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.phase1 import (
    BKStabilityMonitor,
    StabilityThresholds,
    AdaptiveRankSemiseparableLayer,
)


def demo_basic_stability_monitoring():
    """
    基本的な安定性監視のデモ。
    
    安定なシステムと不安定なシステムの両方を監視し、
    警告と回復アクションの生成を実演します。
    """
    print("=" * 70)
    print("Demo 1: Basic Stability Monitoring")
    print("=" * 70)
    
    # Create stability monitor
    monitor = BKStabilityMonitor()
    
    # Scenario 1: Stable system
    print("\n--- Scenario 1: Stable System ---")
    B, N = 2, 10
    G_ii_stable = torch.randn(B, N, dtype=torch.complex64) * 0.1 + 1.0
    potential_stable = torch.randn(B, N) * 0.5
    epsilon = 0.1
    
    metrics_stable = monitor.check_stability(
        G_ii_stable, potential_stable, epsilon
    )
    
    print(f"Is Stable: {metrics_stable.is_stable}")
    print(f"Determinant Condition: {metrics_stable.det_condition:.2e}")
    print(f"Schatten S1 Norm: {metrics_stable.schatten_s1:.2f}")
    print(f"Schatten S2 Norm: {metrics_stable.schatten_s2:.2f}")
    print(f"Min Eigenvalue: {metrics_stable.min_eigenvalue:.2e}")
    print(f"Warnings: {len(metrics_stable.warnings)}")
    
    # Scenario 2: Unstable system
    print("\n--- Scenario 2: Unstable System ---")
    G_ii_unstable = torch.randn(B, N, dtype=torch.complex64) * 100.0
    potential_unstable = torch.randn(B, N) * 100.0
    epsilon_small = 0.0001
    
    metrics_unstable = monitor.check_stability(
        G_ii_unstable, potential_unstable, epsilon_small
    )
    
    print(f"Is Stable: {metrics_unstable.is_stable}")
    print(f"Determinant Condition: {metrics_unstable.det_condition:.2e}")
    print(f"Schatten S1 Norm: {metrics_unstable.schatten_s1:.2f}")
    print(f"Schatten S2 Norm: {metrics_unstable.schatten_s2:.2f}")
    print(f"Min Eigenvalue: {metrics_unstable.min_eigenvalue:.2e}")
    
    if not metrics_unstable.is_stable:
        print(f"\nWarnings ({len(metrics_unstable.warnings)}):")
        for i, warning in enumerate(metrics_unstable.warnings, 1):
            print(f"  {i}. {warning}")
        
        print(f"\nRecommended Actions ({len(metrics_unstable.recommended_actions)}):")
        for i, action in enumerate(metrics_unstable.recommended_actions, 1):
            print(f"  {i}. {action}")


def demo_stability_history_tracking():
    """
    安定性履歴追跡のデモ。
    
    複数のステップにわたって安定性メトリクスを追跡し、
    統計情報を可視化します。
    """
    print("\n" + "=" * 70)
    print("Demo 2: Stability History Tracking")
    print("=" * 70)
    
    # Create monitor with history
    monitor = BKStabilityMonitor(enable_history=True, history_size=100)
    
    # Simulate training steps with gradually degrading stability
    B, N = 2, 10
    num_steps = 50
    
    print(f"\nSimulating {num_steps} training steps...")
    
    for step in range(num_steps):
        # Gradually increase operator magnitude (simulating instability)
        magnitude = 0.1 + step * 0.05
        G_ii = torch.randn(B, N, dtype=torch.complex64) * magnitude + 1.0
        potential = torch.randn(B, N) * magnitude
        epsilon = 0.1
        
        metrics = monitor.check_stability(G_ii, potential, epsilon)
        
        if not metrics.is_stable and step % 10 == 0:
            print(f"  Step {step}: UNSTABLE - {len(metrics.warnings)} warnings")
    
    # Get summary
    summary = monitor.get_summary()
    print(f"\nSummary:")
    print(f"  Total Checks: {summary['total_checks']}")
    print(f"  Stability Violations: {summary['stability_violations']}")
    print(f"  Stability Rate: {summary['stability_rate']:.2%}")
    
    # Get history statistics
    if 'history_stats' in summary:
        print(f"\nHistory Statistics:")
        for metric_name, stats in summary['history_stats'].items():
            print(f"  {metric_name}:")
            print(f"    Mean: {stats['mean']:.2e}")
            print(f"    Std:  {stats['std']:.2e}")
            print(f"    Min:  {stats['min']:.2e}")
            print(f"    Max:  {stats['max']:.2e}")
    
    # Visualize history
    visualize_stability_history(monitor)


def demo_recovery_actions():
    """
    回復アクションのデモ。
    
    勾配クリッピング、スペクトルクリッピング、学習率削減などの
    自動回復アクションを実演します。
    """
    print("\n" + "=" * 70)
    print("Demo 3: Automatic Recovery Actions")
    print("=" * 70)
    
    monitor = BKStabilityMonitor()
    
    # Demo 3.1: Gradient Clipping
    print("\n--- Recovery Action 1: Gradient Clipping ---")
    param = torch.nn.Parameter(torch.randn(10, 10))
    param.grad = torch.randn_like(param) * 10.0  # Large gradient
    
    original_norm = param.grad.norm().item()
    print(f"Original Gradient Norm: {original_norm:.2f}")
    
    clipped_norm = monitor.apply_gradient_clipping([param], max_norm=1.0)
    print(f"Clipped Gradient Norm: {param.grad.norm().item():.2f}")
    print(f"Reduction: {(1 - param.grad.norm().item() / original_norm) * 100:.1f}%")
    
    # Demo 3.2: Spectral Clipping
    print("\n--- Recovery Action 2: Spectral Clipping ---")
    N = 10
    operator = torch.randn(N, N) * 10.0
    
    # Compute original norms
    U, S, Vh = torch.linalg.svd(operator, full_matrices=False)
    original_s1 = S.sum().item()
    original_s2 = torch.sqrt((S ** 2).sum()).item()
    
    print(f"Original Schatten S1 Norm: {original_s1:.2f}")
    print(f"Original Schatten S2 Norm: {original_s2:.2f}")
    
    # Apply spectral clipping
    clipped = monitor.apply_spectral_clipping(
        operator,
        max_s1_norm=50.0,
        max_s2_norm=25.0,
    )
    
    # Compute clipped norms
    U, S, Vh = torch.linalg.svd(clipped, full_matrices=False)
    clipped_s1 = S.sum().item()
    clipped_s2 = torch.sqrt((S ** 2).sum()).item()
    
    print(f"Clipped Schatten S1 Norm: {clipped_s1:.2f}")
    print(f"Clipped Schatten S2 Norm: {clipped_s2:.2f}")
    
    # Demo 3.3: Learning Rate Reduction
    print("\n--- Recovery Action 3: Learning Rate Reduction ---")
    current_lr = 1e-3
    print(f"Current Learning Rate: {current_lr:.2e}")
    
    new_lr = monitor.suggest_learning_rate_reduction(current_lr, reduction_factor=0.5)
    print(f"Suggested Learning Rate: {new_lr:.2e}")
    print(f"Reduction: 50%")


def demo_ar_ssm_integration():
    """
    AR-SSMレイヤーとの統合デモ。
    
    AR-SSMレイヤーに安定性監視を統合し、
    forward pass中の安定性チェックを実演します。
    """
    print("\n" + "=" * 70)
    print("Demo 4: AR-SSM Integration")
    print("=" * 70)
    
    # Create AR-SSM layer with stability monitoring
    monitor = BKStabilityMonitor()
    layer = AdaptiveRankSemiseparableLayer(
        d_model=64,
        max_rank=16,
        stability_monitor=monitor,
        enable_stability_checks=True,
    )
    
    print(f"Created AR-SSM Layer:")
    print(f"  d_model: {layer.d_model}")
    print(f"  max_rank: {layer.max_rank}")
    print(f"  Stability monitoring: {layer.enable_stability_checks}")
    
    # Forward pass
    B, L, D = 2, 10, 64
    x = torch.randn(B, L, D)
    
    print(f"\nForward pass with input shape: {x.shape}")
    y, diagnostics = layer(x)
    
    print(f"Output shape: {y.shape}")
    print(f"\nDiagnostics:")
    print(f"  Effective Rank: {diagnostics['effective_rank'].item():.2f}")
    print(f"  Gate L1 Loss: {diagnostics['gate_l1_loss'].item():.6f}")
    
    if 'condition_number' in diagnostics:
        print(f"  Condition Number: {diagnostics['condition_number']:.2e}")
    
    if 'is_singular' in diagnostics:
        print(f"  Is Singular: {diagnostics['is_singular']}")
    
    if 'stability_warning' in diagnostics:
        print(f"\n  ⚠️  Stability Warning:")
        print(f"    {diagnostics['stability_warning']}")


def visualize_stability_history(monitor: BKStabilityMonitor):
    """
    安定性履歴を可視化します。
    
    Args:
        monitor: 履歴データを持つBKStabilityMonitor
    """
    if not monitor.enable_history or len(monitor.det_history) == 0:
        print("\nNo history data to visualize.")
        return
    
    print("\nGenerating stability history plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Stability Monitoring History', fontsize=16)
    
    # Plot 1: Determinant Condition
    ax = axes[0, 0]
    ax.plot(monitor.det_history, 'b-', linewidth=2)
    ax.axhline(y=monitor.thresholds.det_threshold, color='r', linestyle='--', 
               label='Threshold')
    ax.set_xlabel('Step')
    ax.set_ylabel('det(I - K_ε)')
    ax.set_title('Determinant Condition')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Schatten Norms
    ax = axes[0, 1]
    ax.plot(monitor.schatten_s1_history, 'g-', linewidth=2, label='S1 Norm')
    ax.plot(monitor.schatten_s2_history, 'b-', linewidth=2, label='S2 Norm')
    ax.axhline(y=monitor.thresholds.schatten_s1_bound, color='r', linestyle='--', 
               alpha=0.5, label='S1 Bound')
    ax.axhline(y=monitor.thresholds.schatten_s2_bound, color='orange', linestyle='--', 
               alpha=0.5, label='S2 Bound')
    ax.set_xlabel('Step')
    ax.set_ylabel('Schatten Norm')
    ax.set_title('Schatten Norms')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Minimum Eigenvalue
    ax = axes[1, 0]
    ax.plot(monitor.min_eigenvalue_history, 'm-', linewidth=2)
    ax.axhline(y=monitor.thresholds.min_eigenvalue_threshold, color='r', 
               linestyle='--', label='Threshold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Min Eigenvalue')
    ax.set_title('Minimum Eigenvalue')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Gradient Norm
    ax = axes[1, 1]
    ax.plot(monitor.gradient_norm_history, 'c-', linewidth=2)
    ax.axhline(y=monitor.thresholds.gradient_norm_threshold, color='r', 
               linestyle='--', label='Threshold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = 'results/visualizations/stability_history.png'
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    except Exception as e:
        print(f"Could not save plot: {e}")
    
    plt.show()


def main():
    """メインデモ実行関数。"""
    print("\n" + "=" * 70)
    print("Birman-Schwinger Stability Monitor Demo")
    print("=" * 70)
    print("\nこのデモは、Phase 1の安定性監視機能を実演します。")
    print("Birman-Schwinger演算子の安定性を監視し、")
    print("訓練中の発散を事前に検出して回復アクションを提案します。")
    
    # Run demos
    demo_basic_stability_monitoring()
    demo_stability_history_tracking()
    demo_recovery_actions()
    demo_ar_ssm_integration()
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Stability monitor tracks det(I - K_ε), Schatten norms, eigenvalues")
    print("  2. Automatic warning generation when thresholds violated")
    print("  3. Recovery actions: gradient clipping, spectral clipping, LR reduction")
    print("  4. Seamless integration with AR-SSM layer")
    print("  5. History tracking for long-term stability analysis")


if __name__ == "__main__":
    main()
