#!/usr/bin/env python3
"""Measure FLOPs and efficiency for paper."""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.resnet_bk import ResNetBK
from src.models.adaptive_computation import AdaptiveResNetBK
from src.models.mamba_baseline import MambaBaseline
from src.benchmarks.flops_counter import FLOPsCounter


def measure_model_flops(model_name, seq_length, seed, device='cuda'):
    """Measure FLOPs for a single model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize model
    if model_name == 'resnet_bk':
        model = ResNetBK(d_model=512, num_layers=12)
    elif model_name == 'resnet_bk_act':
        model = AdaptiveResNetBK(d_model=512, num_layers=12)
    elif model_name == 'mamba':
        model = MambaBaseline(d_model=512, n_layer=12)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    
    # Measure FLOPs
    counter = FLOPsCounter(model)
    batch_size = 8
    x = torch.randint(0, 50000, (batch_size, seq_length), device=device)
    
    flops = counter.count_flops(x)
    
    # Measure throughput
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    with torch.no_grad():
        _ = model(x)
    end.record()
    torch.cuda.synchronize()
    
    time_ms = start.elapsed_time(end)
    throughput = (batch_size * seq_length) / (time_ms / 1000)
    
    return {
        'model': model_name,
        'seq_length': seq_length,
        'seed': seed,
        'flops': flops,
        'throughput': throughput,
        'flops_per_token': flops / (batch_size * seq_length)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', required=True, help='Comma-separated model names')
    parser.add_argument('--seq_length', type=int, required=True)
    parser.add_argument('--seeds', required=True, help='Comma-separated seeds')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    models = args.models.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]
    
    results = []
    total = len(models) * len(seeds)
    count = 0
    
    for model_name in models:
        for seed in seeds:
            count += 1
            print(f"[{count}/{total}] Measuring {model_name} @ L={args.seq_length} (seed={seed})...")
            
            result = measure_model_flops(model_name, args.seq_length, seed, args.device)
            results.append(result)
            print(f"  FLOPs: {result['flops']:.2e}, Throughput: {result['throughput']:.0f} tok/s")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
