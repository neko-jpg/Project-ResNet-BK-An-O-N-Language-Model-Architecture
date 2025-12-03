#!/usr/bin/env python3
"""
Phase 7 Training Script - Hybrid Hyperbolic Attention Model

物理的直観:
Phase 7は双曲空間アテンションとSSMを組み合わせたハイブリッドモデルです。
- ローカルコンテキスト: 双曲空間での階層的関係を捉える
- グローバルコンテキスト: SSMによる効率的な長距離依存性

Usage:
    make train-phase7                    # デフォルト設定で学習開始
    make train-phase7 EPOCHS=10          # エポック数を指定
    python scripts/train_phase7.py --help  # 全オプション表示

Requirements:
    - NVIDIA GPU with CUDA support (RTX 3080 10GB recommended)
    - Triton for optimized kernels (REQUIRED - no CPU fallback)
    - Dataset prepared via `make recipe`
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not installed. Install with: pip install tqdm")

# ============================================================================
# STRICT REQUIREMENTS CHECK - Phase 7 requires CUDA + Triton
# ============================================================================

def check_phase7_requirements():
    """
    Phase 7はTritonカーネルによるO(N)計算が核心技術。
    CUDA + Tritonがない環境では意味がないため、厳格にチェックする。
    """
    import torch
    
    # 1. CUDA Check
    if not torch.cuda.is_available():
        print("\n" + "="*60)
        print("ERROR: Phase 7 requires NVIDIA CUDA GPU")
        print("="*60)
        print("\nPhase 7のハイブリッド双曲アテンションは、")
        print("Tritonカーネルによる高速化が必須です。")
        print("\n必要な環境:")
        print("  - NVIDIA GPU (RTX 3080 10GB推奨)")
        print("  - CUDA Toolkit 11.8+")
        print("  - Triton 2.0+")
        print("\nCPUでのトレーニングはサポートされていません。")
        print("="*60 + "\n")
        sys.exit(1)
    
    # 2. Triton Check
    try:
        import triton
        import triton.language as tl
        triton_version = getattr(triton, '__version__', 'unknown')
        print(f"OK: Triton {triton_version} detected")
    except ImportError:
        print("\n" + "="*60)
        print("ERROR: Phase 7 requires Triton")
        print("="*60)
        print("\nTritonがインストールされていません。")
        print("\nインストール方法:")
        print("  pip install triton")
        print("\nまたは:")
        print("  pip install triton==2.1.0")
        print("\nTritonなしでのPhase 7トレーニングは")
        print("O(N)の計算効率を達成できないため、サポートされていません。")
        print("="*60 + "\n")
        sys.exit(1)
    
    # 3. Verify Triton kernel can be loaded
    try:
        from src.kernels.hyperbolic_attention_fast import fast_hyperbolic_attention
        print("✓ Hyperbolic attention Triton kernel loaded")
    except Exception as e:
        print("\n" + "="*60)
        print("❌ ERROR: Failed to load Triton kernel")
        print("="*60)
        print(f"\nエラー詳細: {e}")
        print("\nTritonカーネルのロードに失敗しました。")
        print("以下を確認してください:")
        print("  1. CUDAドライバが正しくインストールされている")
        print("  2. PyTorchがCUDA対応版である")
        print("  3. Tritonのバージョンが互換性がある")
        print("="*60 + "\n")
        sys.exit(1)
    
    # 4. GPU Info
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✓ GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    
    return True

# Run check immediately
check_phase7_requirements()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

# Suppress noisy warnings
warnings.filterwarnings("ignore", message=".*_register_pytree_node is deprecated.*")
warnings.filterwarnings("ignore", message=".*torch.utils._pytree.*deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Triton kernel failed.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*use_reentrant.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*CuDNN issue.*nvrtc.so.*")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.phase7.integrated_model import Phase7IntegratedModel, Phase7Config
from src.models.config import ResNetBKConfig
from src.utils.data_utils import get_mixed_data_loader, get_data_loader
from src.training.curvature_scheduler import create_curvature_scheduler


@dataclass
class Phase7TrainingConfig:
    """Phase 7 Training Configuration"""
    # Model Architecture
    d_model: int = 512
    n_layers: int = 6
    n_seq: int = 512
    num_heads: int = 8
    htt_rank: int = 16
    hyperbolic_window_size: int = 64
    ar_ssm_max_rank: int = 32
    ar_ssm_min_rank: int = 4
    
    # Training
    batch_size: int = 4
    epochs: int = 5
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    max_steps: Optional[int] = None  # If set, stop training after this many steps
    
    # Mixed Precision & Memory
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    
    # Triton Kernels
    use_triton_kernel: bool = True
    triton_kernel_version: str = 'fast'  # 'fast', 'v2', 'v1'
    use_bitnet: bool = False
    
    # Curvature Scheduler
    curvature_warmup_steps: int = 5000
    target_curvature: float = 1.0
    
    # Data
    data_limit: int = 100_000_000  # 100M tokens
    vocab_size: int = 50257  # GPT-2 default
    
    # Logging & Checkpointing
    log_interval: int = 50
    save_interval: int = 1000
    eval_interval: int = 500
    save_dir: str = "checkpoints/phase7"
    
    # Device
    device: str = "auto"
    seed: int = 42


def parse_args() -> Phase7TrainingConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 7 Training Script - Hybrid Hyperbolic Attention",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model Architecture
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--n-seq", type=int, default=512, help="Sequence length")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--htt-rank", type=int, default=16, help="HTT embedding rank")
    parser.add_argument("--hyperbolic-window-size", type=int, default=64, help="Local attention window size")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum training steps (overrides epochs)")
    
    # Memory Optimization
    parser.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision")
    parser.add_argument("--no-gradient-checkpointing", action="store_true", help="Disable gradient checkpointing")
    
    # Triton
    parser.add_argument("--no-triton", action="store_true", help="Disable Triton kernels")
    parser.add_argument("--triton-kernel", type=str, default="fast", choices=["fast", "v2", "v1"], help="Triton kernel version")
    parser.add_argument("--use-bitnet", action="store_true", help="Enable BitNet 1.58-bit quantization")
    
    # Data
    parser.add_argument("--dataset", type=str, default="configs/dataset_mixing.yaml", help="Dataset config path")
    parser.add_argument("--data-limit", type=int, default=100_000_000, help="Max tokens to use")
    parser.add_argument("--vocab-size", type=int, default=50257, help="Vocabulary size")
    
    # Logging
    parser.add_argument("--log-interval", type=int, default=50, help="Log every N steps")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--eval-interval", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--save-dir", type=str, default="checkpoints/phase7", help="Checkpoint directory")
    
    # Device
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Resume
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint")
    
    # Config file
    parser.add_argument("--config", type=str, default=None, help="Load config from YAML file")
    
    # Dry run
    parser.add_argument("--dry-run", action="store_true", help="Test with dummy data (no dataset required)")
    
    args = parser.parse_args()
    
    # Load config from YAML if specified
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
        
        print(f"📄 Loading config from: {args.config}")
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Override defaults with YAML config
        for key, value in yaml_config.items():
            # Convert YAML keys to arg names (e.g., d_model -> d_model)
            arg_name = key.replace('-', '_')
            if hasattr(args, arg_name) and getattr(args, arg_name) == parser.get_default(arg_name):
                setattr(args, arg_name, value)
    
    config = Phase7TrainingConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_seq=args.n_seq,
        num_heads=args.num_heads,
        htt_rank=args.htt_rank,
        hyperbolic_window_size=args.hyperbolic_window_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        use_mixed_precision=not args.no_mixed_precision,
        use_gradient_checkpointing=not args.no_gradient_checkpointing,
        use_triton_kernel=not args.no_triton,
        triton_kernel_version=args.triton_kernel,
        use_bitnet=args.use_bitnet,
        data_limit=args.data_limit,
        vocab_size=args.vocab_size,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        save_dir=args.save_dir,
        device=args.device,
        seed=args.seed,
    )
    
    # Store dataset path, resume path, and dry-run flag
    config.dataset_path = args.dataset
    config.resume_from = args.resume_from
    config.dry_run = args.dry_run
    
    return config


def create_model(config: Phase7TrainingConfig, vocab_size: int, device: torch.device) -> Phase7IntegratedModel:
    """Create Phase 7 model from training config."""
    model_config = Phase7Config(
        vocab_size=vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_seq=config.n_seq,
        num_heads=config.num_heads,
        htt_rank=config.htt_rank,
        hyperbolic_window_size=config.hyperbolic_window_size,
        ar_ssm_max_rank=config.ar_ssm_max_rank,
        ar_ssm_min_rank=config.ar_ssm_min_rank,
        use_hybrid_attention=True,
        use_triton_kernel=config.use_triton_kernel,
        triton_kernel_version=config.triton_kernel_version,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        use_mixed_precision=config.use_mixed_precision,
        use_bitnet=config.use_bitnet,
    )
    
    model = Phase7IntegratedModel(model_config)
    return model.to(device)


def get_device(device_str: str) -> torch.device:
    """Get torch device from string."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def estimate_memory_usage(config: Phase7TrainingConfig) -> float:
    """Estimate VRAM usage in GB."""
    # Rough estimation based on model size and batch
    params_estimate = (
        config.vocab_size * config.d_model +  # Embedding
        config.n_layers * config.d_model * config.d_model * 4 +  # Attention + FFN
        config.vocab_size * config.d_model  # LM head
    )
    
    # Activations (batch * seq * d_model * layers * 2 for forward/backward)
    activations = config.batch_size * config.n_seq * config.d_model * config.n_layers * 2
    
    # Total in bytes (float16 for mixed precision)
    bytes_per_param = 2 if config.use_mixed_precision else 4
    total_bytes = (params_estimate + activations) * bytes_per_param
    
    # Add optimizer states (Adam: 2x params)
    total_bytes += params_estimate * 4 * 2
    
    return total_bytes / (1024 ** 3)  # Convert to GB


def train_phase7():
    """Main training function for Phase 7."""
    config = parse_args()
    
    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Get device
    device = get_device(config.device)
    print(f"\n{'='*60}")
    print("Phase 7 Training - Hybrid Hyperbolic Attention Model")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    # Check VRAM
    if device.type == "cuda":
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_estimate = estimate_memory_usage(config)
        print(f"VRAM Available: {vram_total:.1f} GB")
        print(f"VRAM Estimated: {vram_estimate:.1f} GB")
        
        if vram_estimate > vram_total * 0.9:
            print("\n⚠️  Warning: Estimated VRAM usage exceeds 90% of available memory.")
            print("   Consider reducing batch_size or d_model.")
            
            # Auto-adjust for RTX 3080 (10GB)
            if vram_total < 12:
                print("\n🔧 Auto-adjusting for RTX 3080 (10GB)...")
                config.batch_size = min(config.batch_size, 2)
                config.d_model = min(config.d_model, 384)
                config.n_seq = min(config.n_seq, 384)
                print(f"   New config: batch_size={config.batch_size}, d_model={config.d_model}, n_seq={config.n_seq}")
    
    # Load data
    print(f"\n📊 Loading data...")
    
    # Dry run mode: use dummy data
    if config.dry_run:
        print("   Dry-run mode: Using dummy data for testing")
        actual_vocab_size = config.vocab_size
        steps_per_epoch = 100  # Small number for testing
        use_mixed = False
        
        # Create dummy data generator
        def get_dummy_batch():
            x = torch.randint(0, actual_vocab_size, (config.batch_size, config.n_seq), device=device)
            y = torch.randint(0, actual_vocab_size, (config.batch_size * config.n_seq,), device=device)
            return x, y
        
        batch_iter_func = lambda epoch: [get_dummy_batch() for _ in range(steps_per_epoch)]
    else:
        print(f"   Loading from: {config.dataset_path}")
        dataset_path = Path(config.dataset_path)
        if dataset_path.exists() and dataset_path.suffix in ['.yaml', '.yml']:
            # Mixed dataset
            mixed_loader, vocab, steps_per_epoch = get_mixed_data_loader(
            config_path=str(dataset_path),
            batch_size=config.batch_size,
            n_seq=config.n_seq,
            total_tokens=config.data_limit,
            seed=config.seed,
            vocab_size=config.vocab_size,
            split='train'
            )
            actual_vocab_size = vocab['vocab_size']
            use_mixed = True
            print(f"   Mixed dataset loaded. Steps per epoch: {steps_per_epoch}")
        else:
            # Fallback to wikitext-2
            print("   Dataset config not found. Using wikitext-2 for testing...")
            train_data, vocab, get_batch = get_data_loader(
            batch_size=config.batch_size,
            n_seq=config.n_seq,
            dataset_name='wikitext-2',
            data_limit=config.data_limit
            )
            actual_vocab_size = vocab['vocab_size']
            steps_per_epoch = train_data.size(0) // config.n_seq
            use_mixed = False
    
    print(f"   Vocabulary size: {actual_vocab_size}")
    
    # Create model
    print(f"\n🏗️  Creating Phase 7 model...")
    model = create_model(config, actual_vocab_size, device)
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embedding_params = model.get_embedding_parameter_count()
    
    print(f"   Total parameters: {total_params / 1e6:.2f}M")
    print(f"   Trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"   HTT Embedding parameters: {embedding_params / 1e6:.4f}M")
    print(f"   Embedding compression: {(1 - embedding_params / (actual_vocab_size * config.d_model)) * 100:.1f}%")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Calculate total steps considering max_steps limit
    total_steps = steps_per_epoch * config.epochs
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)
        # Also adjust steps_per_epoch for display purposes
        effective_steps_per_epoch = min(steps_per_epoch, config.max_steps)
    else:
        effective_steps_per_epoch = steps_per_epoch
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=config.learning_rate / 10
    )
    
    # Mixed precision
    use_amp = config.use_mixed_precision and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Resume from checkpoint
    start_epoch = 1
    global_step = 0
    if config.resume_from:
        print(f"\n📂 Resuming from: {config.resume_from}")
        checkpoint = torch.load(config.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
        print(f"   Resumed from epoch {start_epoch}, step {global_step}")
    
    # Create save directory
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create JSON log file
    json_log_file = save_dir / 'training_log.json'
    print(f"📝 Training log: {json_log_file}")
    
    # Training loop
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {config.epochs}")
    if config.max_steps is not None:
        print(f"   Steps per epoch: {steps_per_epoch} (limited to {effective_steps_per_epoch} by max_steps)")
        print(f"   Max steps: {config.max_steps} ⚠️  Will stop early")
    else:
        print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Total steps: {total_steps}")
    print(f"   Mixed precision: {use_amp}")
    print(f"   Gradient checkpointing: {config.use_gradient_checkpointing}")
    print()
    
    model.train()
    best_loss = float('inf')
    training_log = []
    
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0
        epoch_step_times = []  # Track step times for ETA calculation
        
        if config.dry_run:
            batch_iter = batch_iter_func(epoch)
        elif use_mixed:
            batch_iter = mixed_loader.iter_epoch(epoch)
        else:
            batch_iter = range(0, train_data.size(0) - 1, config.n_seq)
        
        # Create progress bar
        if TQDM_AVAILABLE:
            # Calculate remaining steps for this epoch
            if config.max_steps is not None:
                remaining_steps = config.max_steps - global_step
                epoch_total = min(steps_per_epoch, remaining_steps)
            else:
                epoch_total = steps_per_epoch
            
            pbar = tqdm(
                total=epoch_total,
                desc=f"Epoch {epoch}/{config.epochs}",
                ncols=100,
                unit='step',
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}',
            )
            use_pbar = True
        else:
            use_pbar = False
        
        # Track steps in this epoch for max_steps limit
        epoch_step_count = 0
        
        batch_iterator = enumerate(batch_iter, start=1)
        
        for step_idx, batch_item in batch_iterator:
            # Check if we've reached max_steps before processing this batch
            if config.max_steps is not None and global_step >= config.max_steps:
                if use_pbar:
                    pbar.write(f"\n✅ Reached max_steps={config.max_steps}. Stopping training.")
                else:
                    print(f"\n✅ Reached max_steps={config.max_steps}. Stopping training.")
                break
            
            step_start = time.time()
            
            # Get batch
            if config.dry_run:
                x_batch, y_batch = batch_item
            elif use_mixed:
                x_batch, y_batch = batch_item
            else:
                x_batch, y_batch = get_batch(train_data, batch_item)
                x_batch = x_batch.t().contiguous()
                if x_batch.size(1) != config.n_seq:
                    continue
            
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            try:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(x_batch)
                    loss = criterion(logits.view(-1, logits.size(-1)), y_batch)
                
                # Skip NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                # Backward pass
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()
                
                scheduler.step()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n⚠️  CUDA OOM at step {global_step}. Clearing cache...")
                    torch.cuda.empty_cache()
                    continue
                raise
            
            global_step += 1
            epoch_step_count += 1
            epoch_loss += loss.item()
            num_batches += 1
            step_time = time.time() - step_start
            epoch_step_times.append(step_time)
            
            # Calculate ETA and metrics
            avg_loss = epoch_loss / num_batches
            perplexity = math.exp(min(avg_loss, 20))
            lr = scheduler.get_last_lr()[0]
            
            if len(epoch_step_times) > 0:
                avg_step_time = sum(epoch_step_times) / len(epoch_step_times)
                
                # Calculate remaining steps considering max_steps
                if config.max_steps is not None:
                    remaining_steps_total = config.max_steps - global_step
                    remaining_steps_in_epoch = min(steps_per_epoch - step_idx, remaining_steps_total)
                    eta_epoch = remaining_steps_in_epoch * avg_step_time
                    eta_total = remaining_steps_total * avg_step_time
                else:
                    remaining_steps_in_epoch = steps_per_epoch - step_idx
                    eta_epoch = remaining_steps_in_epoch * avg_step_time
                    remaining_epochs = config.epochs - epoch
                    eta_total = eta_epoch + (remaining_epochs * steps_per_epoch * avg_step_time)
            else:
                avg_step_time = step_time
                eta_epoch = 0
                eta_total = 0
            
            # Update progress bar every step
            if use_pbar:
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'ppl': f'{perplexity:.1f}',
                    'lr': f'{lr:.2e}',
                    'grad': f'{grad_norm:.2f}' if isinstance(grad_norm, torch.Tensor) else f'{grad_norm:.2f}',
                    'eta': f'{int(eta_epoch//60)}m'
                })
            
            # Detailed logging to file
            if global_step % config.log_interval == 0:
                # Get memory usage
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3
                else:
                    memory_allocated = 0
                    memory_reserved = 0
                
                log_entry = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'step': global_step,
                    'epoch': epoch,
                    'batch': step_idx,
                    'steps_per_epoch': steps_per_epoch,
                    'loss': loss.item(),
                    'avg_loss': avg_loss,
                    'perplexity': perplexity,
                    'lr': lr,
                    'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    'step_time': step_time,
                    'avg_step_time': avg_step_time if len(epoch_step_times) > 0 else step_time,
                    'eta_epoch_seconds': eta_epoch,
                    'eta_total_seconds': eta_total,
                    'eta_epoch_formatted': f"{int(eta_epoch//3600):02d}:{int((eta_epoch%3600)//60):02d}:{int(eta_epoch%60):02d}",
                    'eta_total_formatted': f"{int(eta_total//3600):02d}:{int((eta_total%3600)//60):02d}:{int(eta_total%60):02d}",
                    'memory_allocated_gb': memory_allocated,
                    'memory_reserved_gb': memory_reserved,
                }
                training_log.append(log_entry)
                
                # Write to JSON log file (append mode)
                with open(json_log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
                
                # Print detailed log if no progress bar
                if not use_pbar:
                    eta_str = f"{int(eta_epoch//3600):02d}:{int((eta_epoch%3600)//60):02d}:{int(eta_epoch%60):02d}"
                    print(f"  Step {global_step:6d}/{total_steps} | Epoch {step_idx}/{steps_per_epoch} | "
                          f"Loss: {loss.item():.4f} | PPL: {perplexity:.2f} | "
                          f"LR: {lr:.2e} | Grad: {grad_norm:.3f} | "
                          f"Time: {step_time:.2f}s | ETA: {eta_str}")
            
            # Save checkpoint
            if global_step % config.save_interval == 0:
                checkpoint_path = save_dir / f"phase7_step_{global_step}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                    'config': config.__dict__,
                    'loss': loss.item(),
                }, checkpoint_path)
                if use_pbar:
                    pbar.write(f"  💾 Checkpoint saved: {checkpoint_path}")
                else:
                    print(f"  💾 Checkpoint saved: {checkpoint_path}")
        
        # Close progress bar
        if use_pbar:
            pbar.close()
        
        # Check if we stopped early due to max_steps
        if config.max_steps is not None and global_step >= config.max_steps:
            print(f"\n✅ Training stopped at step {global_step} (max_steps={config.max_steps})")
            break
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / max(1, num_batches)
        epoch_ppl = math.exp(min(avg_epoch_loss, 20))
        
        # Log epoch summary to JSON
        epoch_summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'epoch_summary',
            'epoch': epoch,
            'step': global_step,
            'avg_loss': avg_epoch_loss,
            'perplexity': epoch_ppl,
            'epoch_time': epoch_time,
            'num_batches': num_batches,
        }
        with open(json_log_file, 'a') as f:
            f.write(json.dumps(epoch_summary) + '\n')
        
        print(f"\n📈 Epoch {epoch}/{config.epochs} Summary:")
        print(f"   Time: {epoch_time:.1f}s")
        print(f"   Avg Loss: {avg_epoch_loss:.4f}")
        print(f"   Perplexity: {epoch_ppl:.2f}")
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = save_dir / "phase7_best.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'config': config.__dict__,
                'loss': avg_epoch_loss,
            }, best_path)
            print(f"   🏆 New best model saved: {best_path}")
        
        print()
    
    # Save final model
    final_path = save_dir / "phase7_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': config.epochs,
        'global_step': global_step,
        'config': config.__dict__,
        'loss': avg_epoch_loss,
    }, final_path)
    
    # Save training log
    log_path = save_dir / "training_log.json"
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✅ Training Complete!")
    print(f"{'='*60}")
    print(f"Final model: {final_path}")
    print(f"Best model: {save_dir / 'phase7_best.pt'}")
    print(f"Training log: {log_path}")
    print(f"Final perplexity: {epoch_ppl:.2f}")


if __name__ == "__main__":
    train_phase7()
