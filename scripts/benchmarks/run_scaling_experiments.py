"""
Run Model Size Scaling Experiments

This script runs scaling experiments for task 9.7:
- Train models with different d_model and n_layers configurations
- Measure scaling laws
- Generate analysis and plots

Usage:
    python run_scaling_experiments.py [--quick] [--device cuda]
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.benchmarks.scaling_experiments import ScalingExperiments


def main():
    parser = argparse.ArgumentParser(description='Run model size scaling experiments')
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test with fewer configurations'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for training'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmark_results/scaling',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MODEL SIZE SCALING EXPERIMENTS")
    print("=" * 80)
    print(f"Mode: {'Quick Test' if args.quick else 'Full Experiments'}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 80 + "\n")
    
    # Create experiments
    experiments = ScalingExperiments(output_dir=args.output_dir)
    
    # Configure experiments
    if args.quick:
        # Quick test: 2x2 = 4 experiments
        d_model_values = [64, 128]
        n_layers_values = [4, 8]
        print("Running quick test with 4 configurations:")
        print(f"  d_model: {d_model_values}")
        print(f"  n_layers: {n_layers_values}")
    else:
        # Full experiments: 4x4 = 16 experiments
        d_model_values = [64, 128, 256, 512]
        n_layers_values = [4, 8, 12, 16]
        print("Running full experiments with 16 configurations:")
        print(f"  d_model: {d_model_values}")
        print(f"  n_layers: {n_layers_values}")
    
    print()
    
    # Run experiments
    try:
        experiments.run_all_experiments(
            d_model_values=d_model_values,
            n_layers_values=n_layers_values,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device
        )
        
        print("\n" + "=" * 80)
        print("SUCCESS: All scaling experiments completed")
        print("=" * 80)
        print(f"\nResults saved to: {args.output_dir}")
        print("\nGenerated files:")
        print(f"  - all_scaling_results.json: All experiment results")
        print(f"  - scaling_law.json: Fitted scaling law parameters")
        print(f"  - scaling_laws.png: Visualization plots")
        print(f"  - d*_l*_results.json: Individual experiment results")
        
    except KeyboardInterrupt:
        print("\n\nExperiments interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError running experiments: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
