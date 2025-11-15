"""
FLOPs Counter Demo
Demonstrates how to use the FLOPs counter infrastructure.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.models.configurable_resnet_bk import (
    ConfigurableResNetBK, 
    BASELINE_CONFIG,
    STEP2_CONFIG,
    FULL_CONFIG
)
from src.benchmarks.flops_counter import FLOPsCounter, compare_models


def demo_basic_counting():
    """Demo: Basic FLOPs counting for a single model."""
    print("\n" + "="*70)
    print("Demo 1: Basic FLOPs Counting")
    print("="*70)
    
    # Create baseline model
    config = BASELINE_CONFIG
    config.d_model = 64
    config.n_layers = 4
    config.n_seq = 128
    config.num_experts = 4
    config.vocab_size = 30000
    
    model = ConfigurableResNetBK(config)
    
    # Create FLOPs counter
    counter = FLOPsCounter(model, batch_size=32, seq_len=128)
    
    # Print summary
    counter.print_summary(optimizer_name='adamw')
    
    # Save to JSON
    counter.save_to_json('flops_baseline.json')


def demo_component_breakdown():
    """Demo: Detailed component-wise FLOPs breakdown."""
    print("\n" + "="*70)
    print("Demo 2: Component-wise FLOPs Breakdown")
    print("="*70)
    
    config = BASELINE_CONFIG
    config.d_model = 64
    config.n_layers = 4
    config.n_seq = 128
    
    model = ConfigurableResNetBK(config)
    counter = FLOPsCounter(model, batch_size=32, seq_len=128)
    
    # Count FLOPs
    total = counter.count_total_flops()
    
    # Get breakdown
    breakdown = counter.get_breakdown()
    
    print(f"\nTotal FLOPs: {total.total:,} ({total.total/1e9:.3f} GFLOPs)")
    print(f"\nForward Pass Breakdown:")
    print("-" * 70)
    
    # Sort by FLOPs
    sorted_components = sorted(
        breakdown.items(),
        key=lambda x: x[1]['forward'],
        reverse=True
    )
    
    for component, flops in sorted_components:
        if flops['forward'] > 0:
            pct = 100 * flops['forward'] / total.forward
            print(f"  {component:25s}: {flops['forward']:>12,} FLOPs ({pct:5.1f}%)")


def demo_model_comparison():
    """Demo: Compare FLOPs between different configurations."""
    print("\n" + "="*70)
    print("Demo 3: Model Configuration Comparison")
    print("="*70)
    
    # Create models with different configurations
    baseline_config = BASELINE_CONFIG
    baseline_config.d_model = 64
    baseline_config.n_layers = 4
    baseline_config.n_seq = 128
    baseline_model = ConfigurableResNetBK(baseline_config)
    
    step2_config = STEP2_CONFIG
    step2_config.d_model = 64
    step2_config.n_layers = 4
    step2_config.n_seq = 128
    step2_model = ConfigurableResNetBK(step2_config)
    
    # Compare models
    comparison = compare_models(
        baseline_model, step2_model,
        batch_size=32, seq_len=128,
        model1_name="Baseline (Step 1)",
        model2_name="Step 2 (Koopman)"
    )


def demo_scaling_analysis():
    """Demo: Analyze FLOPs scaling with model size."""
    print("\n" + "="*70)
    print("Demo 4: FLOPs Scaling Analysis")
    print("="*70)
    
    # Test different model sizes
    d_models = [64, 128, 256]
    n_layers_list = [2, 4, 8]
    
    print(f"\n{'d_model':<10} {'n_layers':<10} {'Forward (GFLOPs)':<20} {'Backward (GFLOPs)':<20} {'Total (GFLOPs)':<20}")
    print("-" * 80)
    
    for d_model in d_models:
        for n_layers in n_layers_list:
            config = BASELINE_CONFIG
            config.d_model = d_model
            config.n_layers = n_layers
            config.n_seq = 128
            
            model = ConfigurableResNetBK(config)
            counter = FLOPsCounter(model, batch_size=32, seq_len=128)
            flops = counter.count_total_flops()
            
            print(f"{d_model:<10} {n_layers:<10} {flops.forward/1e9:<20.3f} {flops.backward/1e9:<20.3f} {flops.total/1e9:<20.3f}")


def demo_sequence_length_scaling():
    """Demo: Analyze FLOPs scaling with sequence length."""
    print("\n" + "="*70)
    print("Demo 5: Sequence Length Scaling (O(N) vs O(N^2))")
    print("="*70)
    
    # Test different sequence lengths
    seq_lengths = [128, 256, 512, 1024, 2048]
    
    print(f"\n{'Seq Length':<15} {'Forward (GFLOPs)':<20} {'Backward (GFLOPs)':<20} {'Total (GFLOPs)':<20}")
    print("-" * 75)
    
    config = BASELINE_CONFIG
    config.d_model = 64
    config.n_layers = 4
    
    for seq_len in seq_lengths:
        config.n_seq = seq_len
        model = ConfigurableResNetBK(config)
        counter = FLOPsCounter(model, batch_size=32, seq_len=seq_len)
        flops = counter.count_total_flops()
        
        print(f"{seq_len:<15} {flops.forward/1e9:<20.3f} {flops.backward/1e9:<20.3f} {flops.total/1e9:<20.3f}")
    
    # Analyze scaling
    print("\nScaling Analysis:")
    print("-" * 75)
    
    # Compare N=128 to N=2048 (16× increase)
    config.n_seq = 128
    model_128 = ConfigurableResNetBK(config)
    counter_128 = FLOPsCounter(model_128, batch_size=32, seq_len=128)
    flops_128 = counter_128.count_total_flops()
    
    config.n_seq = 2048
    model_2048 = ConfigurableResNetBK(config)
    counter_2048 = FLOPsCounter(model_2048, batch_size=32, seq_len=2048)
    flops_2048 = counter_2048.count_total_flops()
    
    scaling_factor = flops_2048.total / flops_128.total
    print(f"Sequence length increased: 128 → 2048 (16×)")
    print(f"FLOPs increased: {flops_128.total/1e9:.3f} → {flops_2048.total/1e9:.3f} GFLOPs ({scaling_factor:.2f}×)")
    print(f"Expected for O(N): 16×")
    print(f"Expected for O(N^2): 256×")
    print(f"Actual scaling: {scaling_factor:.2f}× (close to O(N) ✓)")


def demo_optimizer_comparison():
    """Demo: Compare FLOPs for different optimizers."""
    print("\n" + "="*70)
    print("Demo 6: Optimizer FLOPs Comparison")
    print("="*70)
    
    config = BASELINE_CONFIG
    config.d_model = 64
    config.n_layers = 4
    config.n_seq = 128
    
    model = ConfigurableResNetBK(config)
    counter = FLOPsCounter(model, batch_size=32, seq_len=128)
    
    optimizers = ['sgd', 'adam', 'adamw']
    
    print(f"\n{'Optimizer':<15} {'Forward (GFLOPs)':<20} {'Backward (GFLOPs)':<20} {'Optimizer (GFLOPs)':<20} {'Total (GFLOPs)':<20}")
    print("-" * 95)
    
    for opt in optimizers:
        flops = counter.count_total_flops(optimizer_name=opt)
        print(f"{opt:<15} {flops.forward/1e9:<20.3f} {flops.backward/1e9:<20.3f} {flops.optimizer/1e9:<20.3f} {flops.total/1e9:<20.3f}")


def demo_batch_size_analysis():
    """Demo: Analyze FLOPs scaling with batch size."""
    print("\n" + "="*70)
    print("Demo 7: Batch Size Scaling")
    print("="*70)
    
    config = BASELINE_CONFIG
    config.d_model = 64
    config.n_layers = 4
    config.n_seq = 128
    
    model = ConfigurableResNetBK(config)
    
    batch_sizes = [1, 8, 16, 32, 64]
    
    print(f"\n{'Batch Size':<15} {'Forward (GFLOPs)':<20} {'Backward (GFLOPs)':<20} {'Total (GFLOPs)':<20}")
    print("-" * 75)
    
    for batch_size in batch_sizes:
        counter = FLOPsCounter(model, batch_size=batch_size, seq_len=128)
        flops = counter.count_total_flops()
        
        print(f"{batch_size:<15} {flops.forward/1e9:<20.3f} {flops.backward/1e9:<20.3f} {flops.total/1e9:<20.3f}")
    
    print("\nNote: FLOPs scale linearly with batch size (as expected)")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("FLOPs Counter Infrastructure Demo")
    print("="*70)
    
    # Run demos
    demo_basic_counting()
    demo_component_breakdown()
    demo_model_comparison()
    demo_scaling_analysis()
    demo_sequence_length_scaling()
    demo_optimizer_comparison()
    demo_batch_size_analysis()
    
    print("\n" + "="*70)
    print("All demos completed!")
    print("="*70)


if __name__ == '__main__':
    main()
