#!/usr/bin/env python3
"""Run quantization robustness experiments for paper."""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.quantized_bk_core import QuantizedBKCore
from src.models.mamba_baseline import MambaBaseline
from src.benchmarks.wikitext2_benchmark import WikiText2Benchmark


def run_quantization_experiment(model_name, bits, seed, device='cuda'):
    """Run single quantization experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize model
    if model_name == 'resnet_bk':
        if bits == 'FP32':
            model = QuantizedBKCore(d_model=512, num_layers=12, quantize=False)
        elif bits == 'FP16':
            model = QuantizedBKCore(d_model=512, num_layers=12, quantize=False).half()
        elif bits == 'INT8':
            model = QuantizedBKCore(d_model=512, num_layers=12, bits=8)
        elif bits == 'INT4':
            model = QuantizedBKCore(d_model=512, num_layers=12, bits=4)
    elif model_name == 'mamba':
        if bits == 'FP32':
            model = MambaBaseline(d_model=512, n_layer=12)
        elif bits == 'FP16':
            model = MambaBaseline(d_model=512, n_layer=12).half()
        else:
            # Mamba doesn't support INT quantization well
            return None
    
    model = model.to(device)
    
    # Run benchmark
    benchmark = WikiText2Benchmark(model, device=device)
    results = benchmark.run()
    
    return {
        'model': model_name,
        'bits': bits,
        'seed': seed,
        'perplexity': results['perplexity'],
        'loss': results['loss'],
        'throughput': results.get('throughput', 0)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['resnet_bk', 'mamba'])
    parser.add_argument('--bits', required=True, help='Comma-separated: FP32,FP16,INT8,INT4')
    parser.add_argument('--seeds', required=True, help='Comma-separated seeds')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    bits_list = args.bits.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]
    
    results = []
    total = len(bits_list) * len(seeds)
    count = 0
    
    for bits in bits_list:
        for seed in seeds:
            count += 1
            print(f"[{count}/{total}] Running {args.model} @ {bits} (seed={seed})...")
            
            result = run_quantization_experiment(args.model, bits, seed, args.device)
            if result:
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
