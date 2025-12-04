#!/usr/bin/env python3
"""
Phase 8 Training Script - 10B Parameter Scale Model
(ResNetBK + HTT + Hybrid Hyperbolic Attention + Low-Rank BitNet)

Features:
- 10B Parameter Scale Support via >99% Compression
- Low-Rank FFN & Attention with BitNet 1.58-bit Quantization
- BK-Core Hyperbolic Integration
- AR-SSM Hyperbolic Fusion

Usage:
    python scripts/train_phase8.py --d-model 4096 --n-layers 48 --low-rank-ffn --low-rank-attention --use-bitnet --dry-run
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
import torch
import torch.optim as optim
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.phase8.integrated_model import Phase8IntegratedModel, Phase8Config
from src.utils.data_utils import get_mixed_data_loader, get_data_loader
from src.training.curvature_scheduler import create_curvature_scheduler

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class Phase8TrainingConfig:
    """Phase 8 Training Configuration"""
    # Model Architecture (10B Scale Target: d_model=4096, n_layers=48)
    d_model: int = 4096
    n_layers: int = 48
    n_seq: int = 512
    num_heads: int = 32
    htt_rank: int = 16
    hyperbolic_window_size: int = 64
    
    # Phase 8 Specifics
    use_bk_hyperbolic: bool = True
    use_ar_ssm_fusion: bool = True
    
    # Compression (Critical for 10B)
    low_rank_ffn: bool = True
    low_rank_attention: bool = True
    low_rank_rank: int = 64
    use_bitnet: bool = True
    
    # Training
    batch_size: int = 1 # Small batch for 10B
    epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    max_steps: Optional[int] = None
    
    # Optimization
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_triton_kernel: bool = True
    triton_kernel_version: str = 'fast'
    
    # Data
    data_limit: int = 100_000_000
    vocab_size: int = 50257
    
    # Logging
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 200
    save_dir: str = "checkpoints/phase8"
    
    # Device
    device: str = "auto"
    seed: int = 42
    
    # Runtime
    dry_run: bool = False
    dataset_path: str = "configs/dataset_mixing.yaml"
    resume_from: Optional[str] = None

def parse_args() -> Phase8TrainingConfig:
    parser = argparse.ArgumentParser(description="Phase 8 Training Script")
    
    # Model
    parser.add_argument("--d-model", type=int, default=4096)
    parser.add_argument("--n-layers", type=int, default=48)
    parser.add_argument("--n-seq", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--low-rank-rank", type=int, default=64)
    
    # Flags
    parser.add_argument("--no-low-rank-ffn", action="store_false", dest="low_rank_ffn")
    parser.add_argument("--no-low-rank-attn", action="store_false", dest="low_rank_attention")
    parser.add_argument("--no-bitnet", action="store_false", dest="use_bitnet")
    parser.add_argument("--no-bk-hyperbolic", action="store_false", dest="use_bk_hyperbolic")
    parser.add_argument("--no-ar-ssm", action="store_false", dest="use_ar_ssm_fusion")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    
    # Optimization
    parser.add_argument("--extreme-compression", action="store_true", help="Enable Rank 16/32 for 8GB VRAM")
    parser.add_argument("--ultra-compression", action="store_true", help="Enable Rank 8 for <3GB VRAM")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    
    # Config file support
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset config")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint")

    parser.set_defaults(low_rank_ffn=True, low_rank_attention=True, use_bitnet=True, use_bk_hyperbolic=True, use_ar_ssm_fusion=True)
    
    args = parser.parse_args()

    # Load YAML config if provided
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
            # Override args with yaml config
            for k, v in yaml_config.items():
                # Convert yaml keys (e.g. use_bitnet) to args attributes
                k_norm = k.replace('-', '_')
                if hasattr(args, k_norm):
                    setattr(args, k_norm, v)
    
    # Extreme Compression Logic
    if args.extreme_compression:
        print("🚀 Extreme Compression Enabled (Target: 8GB VRAM)")
        args.low_rank_rank = 16
    
    # Ultra Compression Logic (<3GB)
    if args.ultra_compression:
        print("🌌 Ultra Compression Enabled (Target: <3GB VRAM)")
        args.low_rank_rank = 8  # Very aggressive
        # Force gradient checkpointing if not already set?
        # It's set in config creation if passed, but let's ensure it's on if memory is tight.
        # But Phase8Config defaults use_gradient_checkpointing=True usually?
        # Let's rely on config defaults for now, but Rank 8 is key.
        
        # Disable mixed precision for stability with ultra-low rank
        print("⚠️  Disabling Mixed Precision for Stability in Ultra Mode")
        # We need to ensure config uses this. Phase8TrainingConfig doesn't have use_mixed_precision arg directly?
        # It passes kwargs to Phase8Config.
        # Let's check Phase8TrainingConfig init.
    
    config = Phase8TrainingConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_seq=args.n_seq,
        num_heads=args.num_heads,
        low_rank_rank=args.low_rank_rank,
        low_rank_ffn=args.low_rank_ffn,
        low_rank_attention=args.low_rank_attention,
        use_bitnet=args.use_bitnet,
        use_bk_hyperbolic=args.use_bk_hyperbolic,
        use_ar_ssm_fusion=args.use_ar_ssm_fusion,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        dry_run=args.dry_run,
        use_mixed_precision=not args.ultra_compression, # Disable if ultra
    )
    config.compile = args.compile
    config.dataset_path = args.dataset if args.dataset else config.dataset_path
    config.resume_from = args.resume_from if args.resume_from else config.resume_from
    return config

def create_model(config: Phase8TrainingConfig, vocab_size: int, device: torch.device) -> Phase8IntegratedModel:
    model_config = Phase8Config(
        vocab_size=vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_seq=config.n_seq,
        num_heads=config.num_heads,
        htt_rank=config.htt_rank,
        hyperbolic_window_size=config.hyperbolic_window_size,
        
        # Phase 8
        use_bk_hyperbolic=config.use_bk_hyperbolic,
        use_ar_ssm_fusion=config.use_ar_ssm_fusion,
        
        # Compression
        low_rank_ffn=config.low_rank_ffn,
        low_rank_attention=config.low_rank_attention,
        low_rank_rank=config.low_rank_rank,
        use_bitnet=config.use_bitnet,
        
        # Optimization
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        use_mixed_precision=config.use_mixed_precision,
        use_triton_kernel=config.use_triton_kernel,
        triton_kernel_version=config.triton_kernel_version,
    )
    
    model = Phase8IntegratedModel(model_config)
    model = model.to(device)
    
    if hasattr(config, 'compile') and config.compile:
        print("⚡ Compiling model with torch.compile...")
        model = torch.compile(model, mode="max-autotune")
        
    return model

def train_phase8():
    config = parse_args()
    import torch
    import torch.optim as optim
    
    # Enable CuDNN Benchmark
    torch.backends.cudnn.benchmark = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 8 Training (10B Scale) on {device}")
    print(f"Config: d_model={config.d_model}, n_layers={config.n_layers}")
    print(f"Compression: LowRankFFN={config.low_rank_ffn}, LowRankAttn={config.low_rank_attention}, BitNet={config.use_bitnet}")
    print(f"Rank: {config.low_rank_rank}")
    
    # Create model
    model = create_model(config, config.vocab_size, device)
    
    # Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters (Dense Equivalent): {total_params:,}")
    
    # Estimate Memory
    mem_params = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Parameter Memory (Approx): {mem_params / 1024**2:.2f} MB")
    
    if config.dry_run:
        print("Dry Run: Running 10 steps...")
        
        # Fused AdamW if available
        use_fused = (device.type == 'cuda') and hasattr(optim.AdamW, 'fused')
        if use_fused:
            print("🚀 Using Fused AdamW")
            optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, fused=True)
    # Model Architecture (10B Scale Target: d_model=4096, n_layers=48)
    d_model: int = 4096
    n_layers: int = 48
    n_seq: int = 512
    num_heads: int = 32
    htt_rank: int = 16
    hyperbolic_window_size: int = 64
    
    # Phase 8 Specifics
    use_bk_hyperbolic: bool = True
    use_ar_ssm_fusion: bool = True
    
    # Compression (Critical for 10B)
    low_rank_ffn: bool = True
    low_rank_attention: bool = True
    low_rank_rank: int = 64
    use_bitnet: bool = True
    
    # Training
    batch_size: int = 1 # Small batch for 10B
    epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    max_steps: Optional[int] = None
    
    # Optimization
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_triton_kernel: bool = True
    triton_kernel_version: str = 'fast'
    
    # Data
    data_limit: int = 100_000_000
    vocab_size: int = 50257
    
    # Logging
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 200
    save_dir: str = "checkpoints/phase8"
    
    # Device
    device: str = "auto"
    seed: int = 42
    
    # Runtime
    dry_run: bool = False
    dataset_path: str = "configs/dataset_mixing.yaml"
    resume_from: Optional[str] = None

def parse_args() -> Phase8TrainingConfig:
    parser = argparse.ArgumentParser(description="Phase 8 Training Script")
    
    # Model
    parser.add_argument("--d-model", type=int, default=4096)
    parser.add_argument("--n-layers", type=int, default=48)
    parser.add_argument("--n-seq", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--low-rank-rank", type=int, default=64)
    
    # Flags
    parser.add_argument("--no-low-rank-ffn", action="store_false", dest="low_rank_ffn")
    parser.add_argument("--no-low-rank-attn", action="store_false", dest="low_rank_attention")
    parser.add_argument("--no-bitnet", action="store_false", dest="use_bitnet")
    parser.add_argument("--no-bk-hyperbolic", action="store_false", dest="use_bk_hyperbolic")
    parser.add_argument("--no-ar-ssm", action="store_false", dest="use_ar_ssm_fusion")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    
    # Optimization
    parser.add_argument("--extreme-compression", action="store_true", help="Enable Rank 16/32 for 8GB VRAM")
    parser.add_argument("--ultra-compression", action="store_true", help="Enable Rank 8 for <3GB VRAM")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    
    parser.set_defaults(low_rank_ffn=True, low_rank_attention=True, use_bitnet=True, use_bk_hyperbolic=True, use_ar_ssm_fusion=True)
    
    args = parser.parse_args()
    
    # Extreme Compression Logic
    if args.extreme_compression:
        print("🚀 Extreme Compression Enabled (Target: 8GB VRAM)")
        args.low_rank_rank = 16
    
    # Ultra Compression Logic (<3GB)
    if args.ultra_compression:
        print("🌌 Ultra Compression Enabled (Target: <3GB VRAM)")
        args.low_rank_rank = 8  # Very aggressive
        # Force gradient checkpointing if not already set?
        # It's set in config creation if passed, but let's ensure it's on if memory is tight.
        # But Phase8Config defaults use_gradient_checkpointing=True usually?
        # Let's rely on config defaults for now, but Rank 8 is key.
        
        # Disable mixed precision for stability with ultra-low rank
        print("⚠️  Disabling Mixed Precision for Stability in Ultra Mode")
        # We need to ensure config uses this. Phase8TrainingConfig doesn't have use_mixed_precision arg directly?
        # It passes kwargs to Phase8Config.
        # Let's check Phase8TrainingConfig init.
    
    config = Phase8TrainingConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_seq=args.n_seq,
        num_heads=args.num_heads,
        low_rank_rank=args.low_rank_rank,
        low_rank_ffn=args.low_rank_ffn,
        low_rank_attention=args.low_rank_attention,
        use_bitnet=args.use_bitnet,
        use_bk_hyperbolic=args.use_bk_hyperbolic,
        use_ar_ssm_fusion=args.use_ar_ssm_fusion,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        dry_run=args.dry_run,
        use_mixed_precision=not args.ultra_compression, # Disable if ultra
    )
    config.compile = args.compile
    return config

def create_model(config: Phase8TrainingConfig, vocab_size: int, device: torch.device) -> Phase8IntegratedModel:
    model_config = Phase8Config(
        vocab_size=vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_seq=config.n_seq,
        num_heads=config.num_heads,
        htt_rank=config.htt_rank,
        hyperbolic_window_size=config.hyperbolic_window_size,
        
        # Phase 8
        use_bk_hyperbolic=config.use_bk_hyperbolic,
        use_ar_ssm_fusion=config.use_ar_ssm_fusion,
        
        # Compression
        low_rank_ffn=config.low_rank_ffn,
        low_rank_attention=config.low_rank_attention,
        low_rank_rank=config.low_rank_rank,
        use_bitnet=config.use_bitnet,
        
        # Optimization
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        use_mixed_precision=config.use_mixed_precision,
        use_triton_kernel=config.use_triton_kernel,
        triton_kernel_version=config.triton_kernel_version,
    )
    
    model = Phase8IntegratedModel(model_config)
    model = model.to(device)
    
    if hasattr(config, 'compile') and config.compile:
        print("⚡ Compiling model with torch.compile...")
        model = torch.compile(model, mode="max-autotune")
        
    return model

def train_phase8():
    config = parse_args()
    import torch
    import torch.optim as optim
    
    # Enable CuDNN Benchmark
    torch.backends.cudnn.benchmark = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 8 Training (10B Scale) on {device}")
    print(f"Config: d_model={config.d_model}, n_layers={config.n_layers}")
    print(f"Compression: LowRankFFN={config.low_rank_ffn}, LowRankAttn={config.low_rank_attention}, BitNet={config.use_bitnet}")
    print(f"Rank: {config.low_rank_rank}")
    
    # Create model
    model = create_model(config, config.vocab_size, device)
    
    # Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters (Dense Equivalent): {total_params:,}")
    
    # Estimate Memory
    mem_params = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Parameter Memory (Approx): {mem_params / 1024**2:.2f} MB")
    
    if config.dry_run:
        print("Dry Run: Running 10 steps...")
        
        # Enable Anomaly Detection for Dry Run
        torch.autograd.set_detect_anomaly(True)
        
        # Fused AdamW if available
        use_fused = (device.type == 'cuda') and hasattr(optim.AdamW, 'fused')
        if use_fused:
            print("🚀 Using Fused AdamW")
            optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, fused=True)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
            
        model.train()
        
        # Simple Warmup Scheduler
        def get_lr(step, warmup_steps=5):
            if step < warmup_steps:
                return config.learning_rate * (step + 1) / warmup_steps
            return config.learning_rate
            
        # Use bfloat16 if available for better stability
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        amp_enabled = config.use_mixed_precision
        if amp_enabled:
            print(f"Using Mixed Precision: {dtype}")
            scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16)) # Scaler only needed for fp16
        
        for i in range(10):
            print(f"--- Step {i+1} ---")
            # Update LR for warmup
            lr = get_lr(i)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            optimizer.zero_grad()
            
            # Dummy input
            x = torch.randint(0, config.vocab_size, (config.batch_size, config.n_seq)).to(device)
            target = torch.randint(0, config.vocab_size, (config.batch_size * config.n_seq,)).to(device)
            
            try:
                with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=dtype):
                    logits, _ = model(x)
                    logits = logits.view(-1, config.vocab_size)
                    
                    # Check logits for NaN
                    if torch.isnan(logits).any():
                        print(f"🚨 NaN detected in logits at step {i+1}!")
                        print(f"Logits Max: {logits.max().item()}, Min: {logits.min().item()}")
                        break
                        
                    loss = torch.nn.functional.cross_entropy(logits, target)
                
                # Check for NaN loss immediately
                if torch.isnan(loss):
                    print(f"Step {i+1}: Loss is NaN! 🚨")
                    break
                
                if amp_enabled and dtype == torch.float16:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    
                    # Check gradients for NaN
                    has_nan_grad = False
                    for name, param in model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            print(f"🚨 NaN gradient detected in {name}!")
                            has_nan_grad = True
                            break
                    
                    if has_nan_grad:
                        print("Skipping optimizer step due to NaN gradients.")
                        break
                        
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                print(f"Step {i+1}: Loss {loss.item():.4f}")
                
            except RuntimeError as e:
                print(f"🚨 Runtime Error at step {i+1}: {e}")
                import traceback
                traceback.print_exc()
                break
            
        print("Dry Run Complete.")

if __name__ == "__main__":
    train_phase8()
