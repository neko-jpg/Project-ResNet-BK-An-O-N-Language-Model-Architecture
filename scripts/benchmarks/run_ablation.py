#!/usr/bin/env python3
"""Run ablation studies for paper."""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.resnet_bk import ResNetBK
from src.benchmarks.wikitext2_benchmark import WikiText2Benchmark


def run_ablation_experiment(components, seed, device='cuda'):
    """Run ablation experiment with specific components disabled."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Parse components to disable
    disable_prime_bump = 'prime_bump' not in components
    disable_scattering = 'scattering_router' not in components
    disable_lap = 'lap_stability' not in components
    disable_semiseparable = 'semiseparable' not in components
    
    # Initialize model with ablations
    model = ResNetBK(
        d_model=512,
        num_layers=12,
        use_prime_bump=not disable_prime_bump,
        use_scattering_router=not disable_scattering,
        use_lap_stability=not disable_lap,
        use_semiseparable=not disable_semiseparable
    )
    model = model.to(device)
    
    # Run benchmark
    benchmark = WikiText2Benchmark(model, device=device)
    results = benchmark.run()
    
    return {
        'components': components,
        'seed': seed,
        'perplexity': results['perplexity'],
        'loss': results['loss']
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--components', required=True, 
                       help='Comma-separated components to INCLUDE')
    parser.add_argument('--seeds', required=True, help='Comma-separated seeds')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    # Generate all ablation configurations
    all_components = args.components.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]
    
    # Test: full model, no components, and each component individually
    configs = [
        all_components,  # Full model
        [],  # No components
    ]
    for comp in all_components:
        configs.append([comp])  # Each component alone
    
    results = []
    total = len(configs) * len(seeds)
    count = 0
    
    for config in configs:
        for seed in seeds:
            count += 1
            config_str = ','.join(config) if config else 'none'
            print(f"[{count}/{total}] Running ablation: {config_str} (seed={seed})...")
            
            result = run_ablation_experiment(config, seed, args.device)
            results.append(result)
            print(f"  Perplexity: {result['perplexity']:.2f}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
