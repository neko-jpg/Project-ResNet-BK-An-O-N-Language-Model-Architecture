"""
Demo: Clark Measure Computation for ε-Parametrized Model Family

This demo shows how to:
1. Compute Clark measures for different ε values
2. Verify probability measure properties
3. Measure total variation distance between measures
4. Visualize measure preservation during compression

Requirements: 4.5, 4.6, 4.7, 4.8
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.clark_measure import (
    ClarkMeasureComputer,
    EpsilonParametrizedFamily,
    visualize_clark_measures
)
from src.models.birman_schwinger_core import BirmanSchwingerCore


def demo_basic_clark_measure():
    """Demo 1: Basic Clark measure computation."""
    print("=" * 70)
    print("Demo 1: Basic Clark Measure Computation")
    print("=" * 70)
    
    # Create BirmanSchwingerCore
    n_seq = 64
    epsilon = 1.0
    core = BirmanSchwingerCore(n_seq=n_seq, epsilon=epsilon)
    
    # Generate sample potential
    batch_size = 4
    v = torch.randn(batch_size, n_seq) * 0.1
    
    # Forward pass to get G_ii
    features, diagnostics = core(v, z=1.0j)
    G_ii = torch.complex(features[..., 0], features[..., 1])
    
    print(f"G_ii shape: {G_ii.shape}")
    print(f"G_ii mean: {G_ii.mean().item():.6f}")
    
    # Compute Clark measure
    clark_computer = ClarkMeasureComputer(
        lambda_min=-5.0,
        lambda_max=5.0,
        num_points=500
    )
    
    measure = clark_computer.compute_measure(G_ii, epsilon)
    
    print(f"\nClark Measure Results:")
    print(f"  ε = {measure.epsilon}")
    print(f"  Total mass μ_ε(ℝ) = {measure.total_mass:.6f}")
    print(f"  Is valid probability measure: {measure.is_valid}")
    print(f"  Grid points: {len(measure.lambda_grid)}")
    print(f"  Measure range: [{measure.measure_values.min():.6f}, {measure.measure_values.max():.6f}]")
    
    # Verify probability measure properties
    is_valid = clark_computer.verify_probability_measure(measure, tolerance=0.1)
    print(f"\n✓ Probability measure verification: {'PASSED' if is_valid else 'FAILED'}")


def demo_epsilon_family():
    """Demo 2: ε-parametrized family of models."""
    print("\n" + "=" * 70)
    print("Demo 2: ε-Parametrized Model Family")
    print("=" * 70)
    
    # Create family manager
    epsilon_values = [1.0, 0.75, 0.5, 0.25, 0.1]
    family = EpsilonParametrizedFamily(
        epsilon_values=epsilon_values,
        lambda_min=-5.0,
        lambda_max=5.0,
        num_points=500
    )
    
    print(f"ε values: {epsilon_values}")
    
    # Simulate models at different ε values
    n_seq = 64
    batch_size = 4
    
    for epsilon in epsilon_values:
        print(f"\nComputing measure for ε={epsilon}...")
        
        # Create core with this epsilon
        core = BirmanSchwingerCore(n_seq=n_seq, epsilon=epsilon)
        
        # Generate sample potential (scaled by epsilon for variation)
        v = torch.randn(batch_size, n_seq) * 0.1 * epsilon
        
        # Forward pass
        features, _ = core(v, z=1.0j)
        G_ii = torch.complex(features[..., 0], features[..., 1])
        
        # Compute measure
        measure = family.clark_computer.compute_measure(G_ii, epsilon)
        family.measures[epsilon] = measure
        
        print(f"  Total mass: {measure.total_mass:.6f}")
        print(f"  Valid: {measure.is_valid}")


def demo_total_variation_distance():
    """Demo 3: Total variation distance between measures."""
    print("\n" + "=" * 70)
    print("Demo 3: Total Variation Distance")
    print("=" * 70)
    
    # Create two measures at different ε
    n_seq = 64
    batch_size = 4
    
    epsilon_values = [1.0, 0.5, 0.1]
    measures = {}
    
    clark_computer = ClarkMeasureComputer(
        lambda_min=-5.0,
        lambda_max=5.0,
        num_points=500
    )
    
    for epsilon in epsilon_values:
        core = BirmanSchwingerCore(n_seq=n_seq, epsilon=epsilon)
        v = torch.randn(batch_size, n_seq) * 0.1
        features, _ = core(v, z=1.0j)
        G_ii = torch.complex(features[..., 0], features[..., 1])
        
        measures[epsilon] = clark_computer.compute_measure(G_ii, epsilon)
    
    # Compute pairwise TV distances
    print("\nTotal Variation Distances:")
    print("-" * 50)
    
    for i, eps1 in enumerate(epsilon_values):
        for eps2 in epsilon_values[i+1:]:
            tv_dist = clark_computer.compute_total_variation_distance(
                measures[eps1],
                measures[eps2]
            )
            print(f"  ||μ_{eps1} - μ_{eps2}||_TV = {tv_dist:.6f}")
    
    # Check requirement 4.6: ||μ_1.0 - μ_0.1||_TV < 0.1
    if 1.0 in measures and 0.1 in measures:
        tv_1_to_01 = clark_computer.compute_total_variation_distance(
            measures[1.0],
            measures[0.1]
        )
        requirement_met = tv_1_to_01 < 0.1
        
        print(f"\n{'✓' if requirement_met else '✗'} Requirement 4.6: "
              f"||μ_1.0 - μ_0.1||_TV = {tv_1_to_01:.6f} "
              f"{'<' if requirement_met else '≥'} 0.1")


def demo_compression_verification():
    """Demo 4: Verify measure preservation during compression."""
    print("\n" + "=" * 70)
    print("Demo 4: Compression with Measure Preservation")
    print("=" * 70)
    
    # Create family
    family = EpsilonParametrizedFamily(
        epsilon_values=[1.0, 0.75, 0.5, 0.25, 0.1],
        lambda_min=-5.0,
        lambda_max=5.0,
        num_points=500
    )
    
    # Simulate progressive compression
    n_seq = 64
    batch_size = 4
    
    print("Progressive compression: ε = 1.0 → 0.75 → 0.5 → 0.25 → 0.1")
    print("-" * 70)
    
    for epsilon in family.epsilon_values:
        core = BirmanSchwingerCore(n_seq=n_seq, epsilon=epsilon)
        v = torch.randn(batch_size, n_seq) * 0.1
        features, _ = core(v, z=1.0j)
        G_ii = torch.complex(features[..., 0], features[..., 1])
        
        measure = family.clark_computer.compute_measure(G_ii, epsilon)
        family.measures[epsilon] = measure
        
        print(f"\nε={epsilon}:")
        print(f"  Total mass: {measure.total_mass:.6f}")
        print(f"  Valid: {measure.is_valid}")
    
    # Verify compression steps
    print("\n" + "=" * 70)
    print("Compression Verification:")
    print("=" * 70)
    
    compression_steps = [
        (1.0, 0.75),
        (0.75, 0.5),
        (0.5, 0.25),
        (0.25, 0.1),
        (1.0, 0.1)  # Full compression
    ]
    
    for eps_teacher, eps_student in compression_steps:
        preserved = family.verify_compression_preserves_measure(
            eps_teacher, eps_student, max_tv_distance=0.1
        )
        status = "✓ PRESERVED" if preserved else "✗ NOT PRESERVED"
        print(f"  {eps_teacher} → {eps_student}: {status}")
    
    # Generate report
    print("\n" + "=" * 70)
    print("Compression Report:")
    print("=" * 70)
    
    report = family.get_compression_report()
    print(f"  ε values: {report['epsilon_values']}")
    print(f"  Measures computed: {report['measures_computed']}")
    print(f"  All valid: {report['all_valid']}")
    print(f"\n  TV distances:")
    for key, value in report['tv_distances'].items():
        print(f"    {key}: {value:.6f}")


def demo_visualization():
    """Demo 5: Visualize Clark measures."""
    print("\n" + "=" * 70)
    print("Demo 5: Clark Measure Visualization")
    print("=" * 70)
    
    # Generate measures
    epsilon_values = [1.0, 0.75, 0.5, 0.25, 0.1]
    measures = {}
    
    n_seq = 64
    batch_size = 4
    
    clark_computer = ClarkMeasureComputer(
        lambda_min=-5.0,
        lambda_max=5.0,
        num_points=500
    )
    
    for epsilon in epsilon_values:
        core = BirmanSchwingerCore(n_seq=n_seq, epsilon=epsilon)
        v = torch.randn(batch_size, n_seq) * 0.1
        features, _ = core(v, z=1.0j)
        G_ii = torch.complex(features[..., 0], features[..., 1])
        
        measures[epsilon] = clark_computer.compute_measure(G_ii, epsilon)
    
    print("Generating visualization...")
    
    # Visualize
    try:
        visualize_clark_measures(
            measures,
            save_path='results/clark_measures.png'
        )
        print("✓ Visualization saved to results/clark_measures.png")
    except Exception as e:
        print(f"✗ Visualization failed: {e}")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("Clark Measure Computation Demo")
    print("Implementing Requirements 4.5, 4.6, 4.7, 4.8")
    print("=" * 70)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        demo_basic_clark_measure()
        demo_epsilon_family()
        demo_total_variation_distance()
        demo_compression_verification()
        demo_visualization()
        
        print("\n" + "=" * 70)
        print("All demos completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
