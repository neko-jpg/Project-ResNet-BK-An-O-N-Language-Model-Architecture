#!/usr/bin/env python3
"""
Phase 8 Training Script - 10B Parameter Scale Model
(ResNetBK + HTT + Hybrid Hyperbolic Attention + Low-Rank BitNet)

Features:
- 10B Parameter Scale Support via >99% Compression
- Low-Rank FFN & Attention with BitNet 1.58-bit Quantization
- BK-Core Hyperbolic Integration
- AR-SSM Hyperbolic Fusion
- Gradient Accumulation for Low-VRAM Training

Usage:
    python scripts/train_phase8.py --d-model 4096 --n-layers 48 --low-rank-ffn --low-rank-attention --use-bitnet --dry-run
"""

import argparse
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
from src.utils.data_utils import get_mixed_data_loader
from src.optimizers.muon import Muon

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
    grad_accum_steps: int = 16 # Simulate larger batch size
    epochs: int = 1
    learning_rate: float = 0.02 # Muon likes higher LR
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    max_steps: Optional[int] = None
    
    # Optimization
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_triton_kernel: bool = True
    triton_kernel_version: str = 'fast'
    optimizer_type: str = 'muon' # Default to Muon
    
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
    compile: bool = False

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
    parser.add_argument("--grad-accum-steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.02, help="Learning Rate (Muon default 0.02)")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    
    # Optimization
    parser.add_argument("--optimizer", type=str, default="muon", choices=["adamw", "muon"], help="Optimizer type")
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
        # Disable mixed precision for stability with ultra-low rank if needed, but keeping default for now
        
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
        grad_accum_steps=args.grad_accum_steps,
        epochs=args.epochs,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        dry_run=args.dry_run,
        use_mixed_precision=not args.ultra_compression,
        compile=args.compile,
        dataset_path=args.dataset if args.dataset else "configs/dataset_mixing.yaml",
        resume_from=args.resume_from,
        optimizer_type=args.optimizer
    )
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
    
    if config.resume_from and os.path.exists(config.resume_from):
        print(f"Loading checkpoint from {config.resume_from}...")
        try:
            state_dict = torch.load(config.resume_from, map_location=device)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            model.load_state_dict(state_dict, strict=False)
            print("✔ Checkpoint loaded successfully.")
        except Exception as e:
            print(f"⚠ Failed to load checkpoint: {e}")

    if config.compile:
        print("⚡ Compiling model with torch.compile...")
        model = torch.compile(model, mode="max-autotune")
        
    return model

def train_phase8():
    config = parse_args()
    
    # Enable High Precision MatMul for Ampere (RTX 30xx)
    torch.set_float32_matmul_precision('high')
    # Enable CuDNN Benchmark
    torch.backends.cudnn.benchmark = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 8 Training (10B Scale) on {device}")

    # Initialize Triton Mode (Strict) if on CUDA
    if device.type == "cuda":
        # Force Triton to be active and strict
        from src.models.bk_core import set_triton_mode
        set_triton_mode(True)
        print("✔ Triton Mode Enforced: STRICT")

    # Log VRAM
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Detected VRAM: {vram:.2f} GB")

    print(f"Config: d_model={config.d_model}, n_layers={config.n_layers}")
    print(f"Compression: LowRankFFN={config.low_rank_ffn}, LowRankAttn={config.low_rank_attention}, BitNet={config.use_bitnet}")
    print(f"Rank: {config.low_rank_rank}, Grad Accum Steps: {config.grad_accum_steps}")
    print(f"Optimizer: {config.optimizer_type.upper()}")
    
    # Create model
    model = create_model(config, config.vocab_size, device)
    
    # Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters (Dense Equivalent): {total_params:,}")
    
    # Estimate Memory
    mem_params = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Parameter Memory (Approx): {mem_params / 1024**2:.2f} MB")
    
    # Optimizer Selection
    if config.optimizer_type == 'muon':
        print("⚛ Using Muon Optimizer (Momentum Orthogonal) for Stability")
        optimizer = Muon(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.95,
            adamw_lr=1e-4 # AdamW learning rate for 1D params
        )
    else:
        use_fused = (device.type == 'cuda') and hasattr(optim.AdamW, 'fused')
        if use_fused:
            print("🚀 Using Fused AdamW")
            optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, fused=True)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Dataset
    if config.dry_run:
        print("Dry Run: Skipping dataset loading.")
    else:
        print(f"Loading dataset from {config.dataset_path}...")
        try:
            dataset, vocab, steps_per_epoch = get_mixed_data_loader(
                config.dataset_path,
                batch_size=config.batch_size,
                n_seq=config.n_seq,
                total_tokens=config.data_limit,
                seed=config.seed,
                vocab_size=config.vocab_size
            )
            print(f"Dataset loaded. Steps per epoch: {steps_per_epoch}")
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            if not config.dry_run:
                sys.exit(1)
            dataset = None

    # Training Loop
    model.train()
    
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_mixed_precision)
    
    step = 0
    total_loss = 0.0
    
    # Custom Gradient Clipping Schedule
    # Start very strict, relax later
    initial_clip = 0.1
    final_clip = 1.0

    if config.dry_run:
        steps_to_run = 10
        print(f"Dry Run: Running {steps_to_run} steps with dummy data...")
        # Mock dataset iterator for dry run
        class MockDataset:
            def iter_epoch(self, epoch):
                 for _ in range(steps_to_run):
                     # Random tokens: (B, N)
                     x = torch.randint(0, config.vocab_size, (config.batch_size, config.n_seq))
                     # Random targets: (B*N)
                     y = torch.randint(0, config.vocab_size, (config.batch_size * config.n_seq,))
                     yield x, y

        dataset = MockDataset()
        steps_per_epoch = steps_to_run
        # Use dry-run optimized tqdm
        pbar = tqdm(total=steps_to_run, disable=not TQDM_AVAILABLE)
    else:
        # Real Training
        pbar = tqdm(total=steps_per_epoch * config.epochs, disable=not TQDM_AVAILABLE)

    for epoch in range(config.epochs):
        for x, y in dataset.iter_epoch(epoch):
            step += 1
            x, y = x.to(device), y.to(device)

            with torch.cuda.amp.autocast(enabled=config.use_mixed_precision):
                logits, diagnostics = model(x, return_diagnostics=True)
                logits = logits.view(-1, config.vocab_size)

                # Check for NaNs in logits before loss
                if torch.isnan(logits).any():
                     print(f"🚨 NaN detected in logits at step {step}!")
                     # Try to recover or skip
                     optimizer.zero_grad()
                     continue

                loss = torch.nn.functional.cross_entropy(logits, y)
                loss = loss / config.grad_accum_steps

            if torch.isnan(loss):
                print(f"🚨 NaN Loss detected at step {step}!")
                optimizer.zero_grad()
                continue

            # Backward
            scaler.scale(loss).backward()

            total_loss += loss.item() * config.grad_accum_steps

            if step % config.grad_accum_steps == 0:
                # Dynamic Clipping
                current_clip = initial_clip if step < config.warmup_steps else final_clip

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), current_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                avg_loss = total_loss / config.grad_accum_steps
                
                # Diagnostics Logging
                diag_str = ""
                if step % 100 == 0:
                    # Detailed diagnostics
                    diag_keys = ['hybrid_gate_mean', 'scattering_energy_mean']
                    vals = [f"{k}={diagnostics.get(k, 0):.2f}" for k in diag_keys if k in diagnostics]
                    diag_str = " | ".join(vals)

                pbar.set_description(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Clip: {current_clip} {diag_str}")
                total_loss = 0.0

            pbar.update(1)
            
            if config.max_steps and step >= config.max_steps:
                break

    print("Training Complete.")

    # Save Final
    os.makedirs(config.save_dir, exist_ok=True)
    save_path = os.path.join(config.save_dir, "phase8_10b_final.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_phase8()
