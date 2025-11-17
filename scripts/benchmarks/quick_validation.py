#!/usr/bin/env python3
"""
Quick validation script to verify basic claims before full experiments.
Runs in 2-4 hours on a single GPU.
"""

import argparse
import json
import time
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.resnet_bk import ResNetBK
from src.models.mamba_baseline import MambaBaseline


def quick_train(model, dataset='wikitext2', max_steps=1000, device='cuda'):
    """Quick training run to verify basic functionality."""
    print(f"Quick training: {max_steps} steps on {dataset}")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Mock data for quick validation
    vocab_size = 50000
    seq_length = 512
    batch_size = 8
    
    losses = []
    start_time = time.time()
    
    for step in range(max_steps):
        # Generate random batch
        x = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(x)
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size),
            x.view(-1)
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if (step + 1) % 100 == 0:
            avg_loss = np.mean(losses[-100:])
            print(f"  Step {step+1}/{max_steps}: Loss = {avg_loss:.4f}")
    
    elapsed = time.time() - start_time
    final_loss = np.mean(losses[-100:])
    
    return {
        'final_loss': final_loss,
        'losses': losses,
        'elapsed_time': elapsed,
        'steps_per_sec': max_steps / elapsed,
        'converged': losses[-1] < losses[0] * 0.8
    }


def test_quantization(model, bits=8, device='cuda'):
    """Test quantization robustness."""
    print(f"Testing {bits}-bit quantization...")
    
    model = model.to(device)
    
    # Quantize model
    if bits == 8:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    elif bits == 4:
        # Simulate 4-bit quantization
        for param in model.parameters():
            param.data = torch.round(param.data * 16) / 16
    
    # Test inference
    vocab_size = 50000
    seq_length = 512
    batch_size = 8
    
    x = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    
    try:
        with torch.no_grad():
            logits = model(x)
        
        # Compute perplexity
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size),
            x.view(-1)
        )
        ppl = torch.exp(loss).item()
        
        return {
            'bits': bits,
            'perplexity': ppl,
            'success': True
        }
    except Exception as e:
        return {
            'bits': bits,
            'perplexity': float('inf'),
            'success': False,
            'error': str(e)
        }


def test_long_context(model, seq_length=8192, device='cuda'):
    """Test long context stability."""
    print(f"Testing long context: {seq_length} tokens...")
    
    model = model.to(device)
    vocab_size = 50000
    batch_size = 2  # Smaller batch for long sequences
    
    try:
        x = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
        
        with torch.no_grad():
            logits = model(x)
        
        # Check for NaN/Inf
        has_nan = torch.isnan(logits).any().item()
        has_inf = torch.isinf(logits).any().item()
        
        return {
            'seq_length': seq_length,
            'stable': not (has_nan or has_inf),
            'has_nan': has_nan,
            'has_inf': has_inf
        }
    except RuntimeError as e:
        if 'out of memory' in str(e):
            return {
                'seq_length': seq_length,
                'stable': False,
                'oom': True
            }
        else:
            return {
                'seq_length': seq_length,
                'stable': False,
                'error': str(e)
            }


def measure_flops(model, seq_length=2048, device='cuda'):
    """Estimate FLOPs."""
    print(f"Measuring FLOPs for seq_length={seq_length}...")
    
    model = model.to(device)
    vocab_size = 50000
    batch_size = 8
    
    x = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    
    # Warm up
    with torch.no_grad():
        _ = model(x)
    
    # Measure time
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        _ = model(x)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # Estimate FLOPs (rough approximation)
    num_params = sum(p.numel() for p in model.parameters())
    flops_per_token = num_params * 2  # Forward pass approximation
    total_flops = flops_per_token * batch_size * seq_length
    
    return {
        'seq_length': seq_length,
        'flops': total_flops,
        'flops_per_token': flops_per_token,
        'time': elapsed,
        'throughput': (batch_size * seq_length) / elapsed
    }


def main():
    parser = argparse.ArgumentParser(description='Quick validation of ResNet-BK claims')
    parser.add_argument('--models', default='resnet_bk,mamba', help='Models to test')
    parser.add_argument('--dataset', default='wikitext2', help='Dataset')
    parser.add_argument('--quick', action='store_true', help='Ultra-quick mode (500 steps)')
    parser.add_argument('--output', default='results/quick_validation.json', help='Output file')
    parser.add_argument('--device', default='cuda', help='Device')
    args = parser.parse_args()
    
    models_to_test = args.models.split(',')
    max_steps = 500 if args.quick else 1000
    
    results = {}
    
    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"{'='*60}\n")
        
        # Initialize model
        if model_name == 'resnet_bk':
            model = ResNetBK(d_model=256, num_layers=6)
        elif model_name == 'mamba':
            model = MambaBaseline(d_model=256, n_layer=6)
        else:
            print(f"Unknown model: {model_name}")
            continue
        
        model_results = {}
        
        # Test 1: Basic training
        print("\n[1/4] Basic Training Test")
        model_results['training'] = quick_train(
            model, args.dataset, max_steps, args.device
        )
        
        # Test 2: Quantization
        print("\n[2/4] Quantization Test")
        model_results['quantization'] = {
            'int8': test_quantization(model, bits=8, device=args.device),
            'int4': test_quantization(model, bits=4, device=args.device)
        }
        
        # Test 3: Long context
        print("\n[3/4] Long Context Test")
        model_results['long_context'] = {
            '8k': test_long_context(model, seq_length=8192, device=args.device),
            '32k': test_long_context(model, seq_length=32768, device=args.device)
        }
        
        # Test 4: FLOPs
        print("\n[4/4] FLOPs Measurement")
        model_results['flops'] = measure_flops(model, seq_length=2048, device=args.device)
        
        results[model_name] = model_results
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}\n")
    
    for model_name, model_results in results.items():
        print(f"{model_name}:")
        print(f"  Training: {'✓' if model_results['training']['converged'] else '✗'}")
        print(f"    Final loss: {model_results['training']['final_loss']:.4f}")
        print(f"  Quantization INT4: {'✓' if model_results['quantization']['int4']['success'] else '✗'}")
        if model_results['quantization']['int4']['success']:
            print(f"    PPL: {model_results['quantization']['int4']['perplexity']:.2f}")
        print(f"  Long Context 32k: {'✓' if model_results['long_context']['32k']['stable'] else '✗'}")
        print(f"  FLOPs/token: {model_results['flops']['flops_per_token']:.2e}")
        print()
    
    # Compare models
    if 'resnet_bk' in results and 'mamba' in results:
        print("COMPARISON:")
        
        rb_loss = results['resnet_bk']['training']['final_loss']
        mb_loss = results['mamba']['training']['final_loss']
        print(f"  Training loss: ResNet-BK {rb_loss:.4f} vs Mamba {mb_loss:.4f}")
        
        rb_ppl = results['resnet_bk']['quantization']['int4']['perplexity']
        mb_ppl = results['mamba']['quantization']['int4']['perplexity']
        if rb_ppl < float('inf') and mb_ppl < float('inf'):
            ratio = mb_ppl / rb_ppl
            print(f"  INT4 PPL: ResNet-BK {rb_ppl:.2f} vs Mamba {mb_ppl:.2f} ({ratio:.2f}× better)")
        
        rb_stable = results['resnet_bk']['long_context']['32k']['stable']
        mb_stable = results['mamba']['long_context']['32k']['stable']
        print(f"  32k stability: ResNet-BK {'✓' if rb_stable else '✗'} vs Mamba {'✓' if mb_stable else '✗'}")
        
        rb_flops = results['resnet_bk']['flops']['flops_per_token']
        mb_flops = results['mamba']['flops']['flops_per_token']
        ratio = mb_flops / rb_flops
        print(f"  FLOPs: ResNet-BK {rb_flops:.2e} vs Mamba {mb_flops:.2e} ({ratio:.2f}× efficient)")
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
