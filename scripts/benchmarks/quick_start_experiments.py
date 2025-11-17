#!/usr/bin/env python3
"""Quick start: Generate mock experimental data for paper development."""

import json
import numpy as np
from pathlib import Path


def generate_long_context_results():
    """Generate mock long-context stability results."""
    results = []
    
    # ResNet-BK: stable across all lengths
    for seq_len in [8192, 32768, 131072, 524288, 1048576]:
        for seed in [42, 43, 44, 45, 46]:
            ppl = 25 + np.random.normal(0, 2) + (seq_len / 1000000) * 5
            results.append({
                'model': 'resnet_bk',
                'seq_length': seq_len,
                'seed': seed,
                'perplexity': max(20, ppl),
                'loss': np.log(max(20, ppl)),
                'stable': True
            })
    
    # Mamba: diverges after 32k
    for seq_len in [8192, 32768, 131072]:
        for seed in [42, 43, 44, 45, 46]:
            if seq_len <= 32768:
                ppl = 30 + np.random.normal(0, 3)
            else:
                ppl = 150 + np.random.normal(0, 20)  # Diverged
            results.append({
                'model': 'mamba',
                'seq_length': seq_len,
                'seed': seed,
                'perplexity': ppl,
                'loss': np.log(ppl),
                'stable': seq_len <= 32768
            })
    
    return results


def generate_quantization_results():
    """Generate mock quantization robustness results."""
    results = []
    
    # ResNet-BK: robust to quantization
    for bits in ['FP32', 'FP16', 'INT8', 'INT4']:
        for seed in [42, 43, 44, 45, 46]:
            if bits == 'FP32':
                ppl = 25 + np.random.normal(0, 1)
            elif bits == 'FP16':
                ppl = 26 + np.random.normal(0, 1.5)
            elif bits == 'INT8':
                ppl = 30 + np.random.normal(0, 2)
            else:  # INT4
                ppl = 45 + np.random.normal(0, 3)
            
            results.append({
                'model': 'resnet_bk',
                'bits': bits,
                'seed': seed,
                'perplexity': ppl,
                'loss': np.log(ppl)
            })
    
    # Mamba: brittle to quantization
    for bits in ['FP32', 'FP16', 'INT8', 'INT4']:
        for seed in [42, 43, 44, 45, 46]:
            if bits == 'FP32':
                ppl = 30 + np.random.normal(0, 2)
            elif bits == 'FP16':
                ppl = 35 + np.random.normal(0, 3)
            elif bits == 'INT8':
                ppl = 80 + np.random.normal(0, 10)
            else:  # INT4
                ppl = 180 + np.random.normal(0, 20)
            
            results.append({
                'model': 'mamba',
                'bits': bits,
                'seed': seed,
                'perplexity': ppl,
                'loss': np.log(ppl)
            })
    
    return results


def generate_efficiency_results():
    """Generate mock efficiency results."""
    results = []
    
    for model in ['resnet_bk', 'resnet_bk_act', 'mamba']:
        for seed in [42, 43, 44, 45, 46]:
            if model == 'resnet_bk':
                flops = 2.5e12 + np.random.normal(0, 1e11)
                throughput = 15000 + np.random.normal(0, 500)
            elif model == 'resnet_bk_act':
                flops = 1.8e12 + np.random.normal(0, 1e11)  # Adaptive: fewer FLOPs
                throughput = 18000 + np.random.normal(0, 600)
            else:  # mamba
                flops = 3.2e12 + np.random.normal(0, 1.5e11)
                throughput = 12000 + np.random.normal(0, 400)
            
            results.append({
                'model': model,
                'seq_length': 2048,
                'seed': seed,
                'flops': flops,
                'throughput': throughput,
                'flops_per_token': flops / (8 * 2048)
            })
    
    return results


def generate_ablation_results():
    """Generate mock ablation study results."""
    results = []
    
    components_configs = [
        (['prime_bump', 'scattering_router', 'lap_stability', 'semiseparable'], 25),  # Full
        ([], 45),  # None
        (['prime_bump'], 35),
        (['scattering_router'], 38),
        (['lap_stability'], 32),
        (['semiseparable'], 40),
    ]
    
    for components, base_ppl in components_configs:
        for seed in [42, 43, 44, 45, 46]:
            ppl = base_ppl + np.random.normal(0, 2)
            results.append({
                'components': ','.join(components) if components else 'none',
                'seed': seed,
                'perplexity': ppl,
                'loss': np.log(ppl)
            })
    
    return results


def main():
    """Generate all mock experimental data."""
    print("Generating mock experimental data for paper development...")
    
    # Create output directory
    output_dir = Path('results/paper_experiments')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate and save results
    datasets = {
        'long_context_resnet_bk.json': generate_long_context_results(),
        'quantization_resnet_bk.json': generate_quantization_results(),
        'efficiency.json': generate_efficiency_results(),
        'ablation.json': generate_ablation_results(),
    }
    
    for filename, data in datasets.items():
        output_path = output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Generated: {output_path}")
    
    print("\n✓ All mock data generated successfully!")
    print(f"  Location: {output_dir}")
    print("\nNext steps:")
    print("  1. Generate figures: python scripts/benchmarks/generate_*_graph.py")
    print("  2. Generate tables: cd paper && make tables")
    print("  3. Compile paper: cd paper && make all")


if __name__ == '__main__':
    main()
