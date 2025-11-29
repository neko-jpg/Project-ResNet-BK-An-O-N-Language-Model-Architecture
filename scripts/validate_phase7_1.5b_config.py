#!/usr/bin/env python3
"""Validate Phase 7 1.5B configuration file."""

import yaml
from pathlib import Path

def validate_config():
    config_path = Path("configs/phase7_1.5b_triton.yaml")
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("="*60)
        print("✓ Phase 7 - 1.5B Configuration Validation")
        print("="*60)
        
        # Model architecture
        d_model = config.get('d_model', 0)
        n_layers = config.get('n_layers', 0)
        n_seq = config.get('n_seq', 0)
        num_heads = config.get('num_heads', 0)
        vocab_size = config.get('vocab_size', 50257)
        
        print(f"\n📐 Model Architecture:")
        print(f"  d_model: {d_model}")
        print(f"  n_layers: {n_layers}")
        print(f"  n_seq: {n_seq}")
        print(f"  num_heads: {num_heads}")
        print(f"  vocab_size: {vocab_size}")
        
        # Estimate parameters
        embedding_params = vocab_size * d_model
        transformer_params = n_layers * (
            4 * d_model * d_model +  # Attention (Q, K, V, O)
            8 * d_model * d_model    # FFN (2 layers, 4x expansion)
        )
        output_params = vocab_size * d_model
        total_params = embedding_params + transformer_params + output_params
        
        print(f"\n📊 Parameter Estimation:")
        print(f"  Embedding: {embedding_params / 1e6:.1f}M")
        print(f"  Transformer: {transformer_params / 1e6:.1f}M")
        print(f"  Output: {output_params / 1e6:.1f}M")
        print(f"  Total: {total_params / 1e9:.2f}B")
        
        # Triton settings
        use_triton = config.get('use_triton_kernel', False)
        triton_version = config.get('triton_kernel_version', 'unknown')
        
        print(f"\n⚡ Triton Configuration:")
        print(f"  use_triton_kernel: {use_triton}")
        print(f"  triton_kernel_version: {triton_version}")
        
        if not use_triton:
            print("  ⚠️  WARNING: Triton is not enabled!")
            return False
        
        # Optimizations
        print(f"\n🚀 Optimizations:")
        optimizations = [
            'use_mixed_precision',
            'use_gradient_checkpointing',
            'use_flash_attention',
            'use_fused_optimizer',
            'use_compile',
            'use_fused_kernels',
            'use_memory_efficient_attention',
        ]
        
        for opt in optimizations:
            value = config.get(opt, False)
            status = "✓" if value else "✗"
            print(f"  {status} {opt}: {value}")
        
        # Training settings
        batch_size = config.get('batch_size', 1)
        grad_accum = config.get('gradient_accumulation_steps', 1)
        effective_batch = batch_size * grad_accum
        
        print(f"\n🎯 Training Configuration:")
        print(f"  batch_size: {batch_size}")
        print(f"  gradient_accumulation_steps: {grad_accum}")
        print(f"  effective_batch_size: {effective_batch}")
        print(f"  learning_rate: {config.get('learning_rate', 0)}")
        print(f"  epochs: {config.get('epochs', 0)}")
        
        # Memory estimation
        bytes_per_param = 2  # FP16
        model_memory = total_params * bytes_per_param / (1024**3)
        activation_memory = batch_size * n_seq * d_model * n_layers * 4 / (1024**3)
        optimizer_memory = total_params * 8 / (1024**3)  # Adam: 2x params in FP32
        total_memory = model_memory + activation_memory + optimizer_memory
        
        print(f"\n💾 Memory Estimation:")
        print(f"  Model (FP16): {model_memory:.2f} GB")
        print(f"  Activations: {activation_memory:.2f} GB")
        print(f"  Optimizer: {optimizer_memory:.2f} GB")
        print(f"  Total: {total_memory:.2f} GB")
        
        if total_memory > 10:
            print(f"  ⚠️  WARNING: Estimated memory ({total_memory:.2f} GB) exceeds 10GB")
            print(f"     Consider reducing batch_size, d_model, or n_seq")
        
        print("\n" + "="*60)
        print("✅ Configuration is valid!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating config: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = validate_config()
    sys.exit(0 if success else 1)
