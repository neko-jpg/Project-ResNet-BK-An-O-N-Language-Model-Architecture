#!/usr/bin/env python3
"""
Phase 8 Training Script - 10B Parameter Scale Model (Optimized)
(ResNetBK + HTT + Hybrid Hyperbolic Attention + Low-Rank BitNet)

Features:
- 10B Parameter Scale Support via >99% Compression
- Low-Rank FFN & Attention with BitNet 1.58-bit Quantization
- BK-Core Hyperbolic Integration
- AR-SSM Hyperbolic Fusion
- Gradient Accumulation for Low-VRAM Training
- LR Scheduler (Linear Warmup + Cosine Decay)
- EMA (Exponential Moving Average)
- Label Smoothing
- Proper Checkpoint Saving

Usage:
    python scripts/train_phase8.py --config configs/phase8_10b_japanese.yaml
"""

import argparse
import copy
import json
import math
import os
import sys
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import datetime
from collections import OrderedDict

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.phase8.integrated_model import Phase8IntegratedModel, Phase8Config
from src.optimizers.muon import Muon

# Lazy import for data_utils (requires datasets library)
def get_mixed_data_loader(*args, **kwargs):
    from src.utils.data_utils import get_mixed_data_loader as _loader
    return _loader(*args, **kwargs)

# Import Phase 8 optimization kernels
try:
    from src.kernels.resonance_adaptive_curvature import ResonanceAdaptiveCurvature, StabilityMonitor
    _RESONANCE_AVAILABLE = True
except ImportError:
    _RESONANCE_AVAILABLE = False
    ResonanceAdaptiveCurvature = None
    StabilityMonitor = None

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# EMA (Exponential Moving Average)
# =============================================================================
class EMA:
    """
    Exponential Moving Average of model weights.
    Updates shadow weights as: shadow = decay * shadow + (1 - decay) * model_weights
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._init_shadow()
    
    def _init_shadow(self):
        """Initialize shadow weights from model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow weights with current model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Apply shadow weights to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights after evaluation."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self) -> Dict:
        """Return EMA state for checkpointing."""
        return {
            'decay': self.decay,
            'shadow': {k: v.cpu() for k, v in self.shadow.items()}
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load EMA state from checkpoint."""
        self.decay = state_dict['decay']
        device = next(self.model.parameters()).device
        self.shadow = {k: v.to(device) for k, v in state_dict['shadow'].items()}


# =============================================================================
# Learning Rate Scheduler (Linear Warmup + Cosine Decay)
# =============================================================================
class CosineWarmupScheduler:
    """
    Learning rate scheduler with linear warmup and cosine decay.
    
    LR schedule:
    - Warmup (0 to warmup_steps): Linear from 0 to peak_lr
    - Decay (warmup_steps to total_steps): Cosine from peak_lr to min_lr
    """
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        peak_lr: float,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.current_step = 0
        self._last_lr = 0.0
    
    def get_lr(self, step: int) -> float:
        """Calculate learning rate for given step."""
        if step < self.warmup_steps:
            # Linear warmup
            return self.peak_lr * step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)  # Clamp to [0, 1]
            return self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
    
    def step(self):
        """Update learning rate for current step."""
        self.current_step += 1
        lr = self.get_lr(self.current_step)
        self._last_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self) -> float:
        """Return last computed learning rate."""
        return self._last_lr
    
    def state_dict(self) -> Dict:
        """Return scheduler state for checkpointing."""
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'peak_lr': self.peak_lr,
            'min_lr': self.min_lr
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load scheduler state from checkpoint."""
        self.current_step = state_dict['current_step']
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.peak_lr = state_dict['peak_lr']
        self.min_lr = state_dict['min_lr']


# =============================================================================
# Training Configuration
# =============================================================================
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
    batch_size: int = 1
    grad_accum_steps: int = 16
    epochs: int = 1
    learning_rate: float = 0.02
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    warmup_steps: int = 500  # Reduced from 2000
    max_steps: Optional[int] = None
    
    # Optimizer (AdamW settings)
    optimizer_type: str = 'adamw'
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # Gradient
    grad_clip_warmup: float = 0.1
    grad_clip_train: float = 1.0
    grad_skip_threshold: float = 10.0
    
    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999
    
    # Regularization
    label_smoothing: float = 0.1
    dropout: float = 0.1
    
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
    compile: bool = False


def parse_args() -> Phase8TrainingConfig:
    parser = argparse.ArgumentParser(description="Phase 8 Training Script (Optimized)")
    
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
    parser.add_argument("--no-ema", action="store_false", dest="use_ema")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    
    # Optimizer
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"])
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    
    # Regularization
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    
    # Gradient
    parser.add_argument("--grad-clip-warmup", type=float, default=0.1)
    parser.add_argument("--grad-clip-train", type=float, default=1.0)
    parser.add_argument("--grad-skip-threshold", type=float, default=10.0)
    
    # Optimization
    parser.add_argument("--extreme-compression", action="store_true")
    parser.add_argument("--ultra-compression", action="store_true")
    parser.add_argument("--compile", action="store_true")
    
    # Config file support
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=500)

    parser.set_defaults(
        low_rank_ffn=True, 
        low_rank_attention=True, 
        use_bitnet=True, 
        use_bk_hyperbolic=True, 
        use_ar_ssm_fusion=True,
        use_ema=True
    )
    
    args = parser.parse_args()

    # Load YAML config if provided
    yaml_config = {}
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
    
    # Helper to get config value (CLI > YAML > default)
    def get_val(name, default):
        cli_val = getattr(args, name, None)
        yaml_val = yaml_config.get(name, yaml_config.get(name.replace('_', '-'), None))
        if cli_val is not None and cli_val != default:
            return cli_val
        if yaml_val is not None:
            return yaml_val
        return default if cli_val is None else cli_val
    
    # Extreme Compression Logic
    if args.extreme_compression:
        print("🚀 Extreme Compression Enabled (Target: 8GB VRAM)")
        args.low_rank_rank = 16
    
    if args.ultra_compression:
        print("🌌 Ultra Compression Enabled (Target: <3GB VRAM)")
        args.low_rank_rank = 8
    
    # Build config
    config = Phase8TrainingConfig(
        d_model=get_val('d_model', 4096),
        n_layers=get_val('n_layers', 48),
        n_seq=get_val('n_seq', 512),
        num_heads=get_val('num_heads', 32),
        low_rank_rank=get_val('low_rank_rank', 64),
        low_rank_ffn=get_val('low_rank_ffn', True),
        low_rank_attention=get_val('low_rank_attention', True),
        use_bitnet=get_val('use_bitnet', True),
        use_bk_hyperbolic=get_val('use_bk_hyperbolic', True),
        use_ar_ssm_fusion=get_val('use_ar_ssm_fusion', True),
        batch_size=get_val('batch_size', 1),
        grad_accum_steps=get_val('grad_accum_steps', get_val('gradient_accumulation_steps', 16)),
        epochs=get_val('epochs', 1),
        learning_rate=get_val('learning_rate', 0.02),
        min_lr=get_val('min_lr', 1e-6),
        warmup_steps=get_val('warmup_steps', 500),
        max_steps=get_val('max_steps', None),
        optimizer_type=get_val('optimizer_type', 'adamw'),
        beta1=get_val('beta1', 0.9),
        beta2=get_val('beta2', 0.95),
        eps=get_val('eps', 1e-8),
        weight_decay=get_val('weight_decay', 0.01),
        grad_clip_warmup=get_val('grad_clip_warmup', get_val('max_grad_norm', 0.1)),
        grad_clip_train=get_val('grad_clip_train', 1.0),
        grad_skip_threshold=get_val('grad_skip_threshold', 10.0),
        use_ema=get_val('use_ema', True),
        ema_decay=get_val('ema_decay', 0.999),
        label_smoothing=get_val('label_smoothing', 0.1),
        dropout=get_val('dropout', 0.1),
        use_mixed_precision=get_val('use_mixed_precision', True),
        use_gradient_checkpointing=get_val('use_gradient_checkpointing', True),
        use_triton_kernel=get_val('use_triton_kernel', True),
        vocab_size=get_val('vocab_size', 50257),
        save_interval=get_val('save_interval', 500),
        save_dir=get_val('save_dir', 'checkpoints/phase8'),
        dry_run=args.dry_run,
        dataset_path=args.dataset if args.dataset else get_val('dataset_path', 'configs/dataset_mixing.yaml'),
        resume_from=args.resume_from,
        compile=args.compile,
    )
    
    return config


def init_weights(model: nn.Module):
    """
    Initialize model weights with proper schemes.
    - Attention: Xavier
    - FFN: He (Kaiming)
    - LayerNorm: gamma=1, beta=0
    - Embedding: N(0, 0.02)
    """
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'embedding' in name.lower() or 'embed' in name.lower():
            nn.init.normal_(param, mean=0.0, std=0.02)
        elif 'layernorm' in name.lower() or 'layer_norm' in name.lower():
            if 'weight' in name or 'gamma' in name:
                nn.init.ones_(param)
            elif 'bias' in name or 'beta' in name:
                nn.init.zeros_(param)
        elif 'attn' in name.lower() or 'attention' in name.lower():
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
        elif 'ffn' in name.lower() or 'mlp' in name.lower() or 'fc' in name.lower():
            if param.dim() >= 2:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        elif param.dim() >= 2:
            nn.init.xavier_uniform_(param)


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
    
    # Apply weight initialization
    init_weights(model)
    
    model = model.to(device)
    
    return model


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: CosineWarmupScheduler,
    scaler: torch.cuda.amp.GradScaler,
    ema: Optional[EMA],
    step: int,
    epoch: int,
    loss: float,
    config: Phase8TrainingConfig
):
    """Save complete checkpoint including all training state."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'step': step,
        'epoch': epoch,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'config': asdict(config),
    }
    
    if ema is not None:
        checkpoint['ema_state_dict'] = ema.state_dict()
    
    torch.save(checkpoint, path)
    print(f"\n💾 Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: CosineWarmupScheduler,
    scaler: torch.cuda.amp.GradScaler,
    ema: Optional[EMA],
    device: torch.device
) -> Tuple[int, int, float]:
    """Load checkpoint and return (step, epoch, loss)."""
    print(f"Loading checkpoint from {path}...")
    
    checkpoint = torch.load(path, map_location=device)
    
    # Load model
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    # Load optimizer
    if 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print(f"⚠ Could not load optimizer state: {e}")
    
    # Load scheduler
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Load scaler
    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # Load EMA
    if ema is not None and 'ema_state_dict' in checkpoint:
        ema.load_state_dict(checkpoint['ema_state_dict'])
    
    step = checkpoint.get('step', 0)
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)
    
    print(f"✔ Checkpoint loaded: step={step}, epoch={epoch}, loss={loss:.4f}")
    
    return step, epoch, loss


def save_training_log(log: Dict, path: str):
    """Save training log to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False, default=str)


def train_phase8():
    config = parse_args()
    
    # Set precision and CUDA options
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 8 Training (10B Scale - Optimized) on {device}")

    # Initialize Triton Mode
    if device.type == "cuda" and config.use_triton_kernel:
        try:
            from src.models.bk_core import set_triton_mode
            set_triton_mode(True)
            print("✔ Triton Mode Enabled")
        except ImportError:
            print("⚠ Triton mode not available")

    # Log VRAM
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Detected VRAM: {vram:.2f} GB")

    # Print config summary
    print(f"Config: d_model={config.d_model}, n_layers={config.n_layers}")
    print(f"Compression: LowRankFFN={config.low_rank_ffn}, LowRankAttn={config.low_rank_attention}, BitNet={config.use_bitnet}")
    print(f"Rank: {config.low_rank_rank}, Grad Accum: {config.grad_accum_steps}")
    print(f"Optimizer: {config.optimizer_type.upper()}, LR: {config.learning_rate}, Warmup: {config.warmup_steps}")
    print(f"EMA: {config.use_ema}, Label Smoothing: {config.label_smoothing}")
    
    # Create model
    model = create_model(config, config.vocab_size, device)
    
    # Apply torch.compile() for additional speedup
    if config.compile and hasattr(torch, 'compile'):
        try:
            print("🔧 Applying torch.compile()...")
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            print("✔ torch.compile() applied successfully")
        except Exception as e:
            print(f"⚠ torch.compile() failed (continuing without): {e}")
    
    # Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Optimizer
    if config.optimizer_type == 'muon':
        print("⚛ Using Muon Optimizer")
        optimizer = Muon(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.95,
            adamw_lr=1e-4
        )
    else:
        print(f"⚡ Using AdamW (β1={config.beta1}, β2={config.beta2})")
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
            fused=device.type == 'cuda'
        )
    
    # Scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_mixed_precision)
    
    # EMA
    ema = EMA(model, decay=config.ema_decay) if config.use_ema else None
    if ema:
        print(f"✔ EMA Enabled (decay={config.ema_decay})")
    
    # Dataset
    steps_per_epoch = 100  # Default for dry run
    if config.dry_run:
        print("Dry Run: Using dummy data")
        dataset = None
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
            print(f"⚠ Failed to load dataset: {e}")
            sys.exit(1)
    
    # Calculate total steps
    total_steps = config.max_steps if config.max_steps else steps_per_epoch * config.epochs
    
    # LR Scheduler
    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps,
        peak_lr=config.learning_rate,
        min_lr=config.min_lr
    )
    print(f"✔ LR Scheduler: Warmup {config.warmup_steps} steps, Cosine decay to {config.min_lr}")
    
    # Resume from checkpoint if specified
    start_step = 0
    start_epoch = 0
    if config.resume_from and os.path.exists(config.resume_from):
        start_step, start_epoch, _ = load_checkpoint(
            config.resume_from, model, optimizer, scheduler, scaler, ema, device
        )
    
    # Training state
    model.train()
    step = start_step
    optimizer_step = 0
    total_loss = 0.0
    skip_count = 0
    
    # Resonance-Adaptive Curvature Optimizer (Phase 8 optimization)
    resonance_curvature = None
    stability_monitor = None
    if _RESONANCE_AVAILABLE:
        try:
            resonance_curvature = ResonanceAdaptiveCurvature(
                model=model,
                initial_curvature=1.0,
                min_curvature=0.1,
                max_curvature=2.0,
                resonance_threshold=0.5,
                adjustment_rate=0.01,
            )
            stability_monitor = StabilityMonitor(
                window_size=100,
                nan_threshold=5,
                exploding_grad_threshold=100.0,
            )
            print("✔ Resonance-Adaptive Curvature & Stability Monitor Enabled")
        except Exception as e:
            print(f"⚠ Resonance optimizers not available: {e}")
    
    # JSON Log
    training_log = {
        'config': asdict(config),
        'start_time': datetime.now().isoformat(),
        'total_params': total_params,
        'steps': [],
    }
    
    # Dry run mock dataset
    if config.dry_run:
        steps_to_run = 10
        print(f"Dry Run: Running {steps_to_run} steps...")
        
        class MockDataset:
            def iter_epoch(self, epoch):
                for _ in range(steps_to_run):
                    x = torch.randint(0, config.vocab_size, (config.batch_size, config.n_seq))
                    y = torch.randint(0, config.vocab_size, (config.batch_size * config.n_seq,))
                    yield x, y
        
        dataset = MockDataset()
        steps_per_epoch = steps_to_run
        total_steps = steps_to_run
    
    # Progress bar
    pbar = tqdm(
        total=total_steps - start_step,
        initial=0,
        disable=not TQDM_AVAILABLE,
        desc="Training"
    )
    
    # Training loop
    for epoch in range(start_epoch, config.epochs):
        for x, y in dataset.iter_epoch(epoch):
            step += 1
            
            if step <= start_step:
                continue
            
            x, y = x.to(device), y.to(device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=config.use_mixed_precision):
                logits, diagnostics = model(x, return_diagnostics=True)
                logits = logits.view(-1, config.vocab_size)
                
                # Check for NaN in logits
                if torch.isnan(logits).any():
                    print(f"🚨 NaN in logits at step {step}!")
                    optimizer.zero_grad()
                    skip_count += 1
                    continue
                
                # Loss with label smoothing
                loss = F.cross_entropy(logits, y, label_smoothing=config.label_smoothing)
                loss = loss / config.grad_accum_steps
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"🚨 NaN loss at step {step}!")
                optimizer.zero_grad()
                skip_count += 1
                continue
            
            # Backward
            scaler.scale(loss).backward()
            total_loss += loss.item() * config.grad_accum_steps
            
            # Optimizer step (every grad_accum_steps)
            if step % config.grad_accum_steps == 0:
                optimizer_step += 1
                
                # Unscale gradients
                scaler.unscale_(optimizer)
                
                # Compute gradient norm
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    float('inf')  # Just compute, don't clip yet
                ).item()
                
                # Gradient skip if too large
                if grad_norm > config.grad_skip_threshold:
                    print(f"⚠ Grad norm {grad_norm:.2f} > {config.grad_skip_threshold}, skipping step")
                    optimizer.zero_grad()
                    skip_count += 1
                    total_loss = 0.0
                    pbar.update(1)
                    continue
                
                # Dynamic gradient clipping
                current_clip = config.grad_clip_warmup if step < config.warmup_steps else config.grad_clip_train
                torch.nn.utils.clip_grad_norm_(model.parameters(), current_clip)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # LR scheduler step
                scheduler.step()
                
                # EMA update
                if ema is not None:
                    ema.update()
                
                # Resonance-Adaptive Curvature step (Phase 8 optimization)
                if resonance_curvature is not None and 'phase8' in (diagnostics or {}):
                    phase8_diag = diagnostics.get('phase8', {})
                    if 'G_ii_mean' in phase8_diag:
                        # Create a dummy G_ii tensor from diagnostics
                        g_ii_val = phase8_diag.get('G_ii_mean', 0.0)
                        g_ii_tensor = torch.tensor([g_ii_val], device=device, dtype=torch.complex64)
                        res_diag = resonance_curvature.step(g_ii_tensor)
                
                # Stability Monitor update
                had_nan = False
                if stability_monitor is not None:
                    status = stability_monitor.update(
                        loss=avg_loss,
                        grad_norm=grad_norm,
                        had_nan=had_nan
                    )
                    if status.get('warning'):
                        print(f"⚠ Stability: {status['warning']}")
                
                # Compute metrics
                avg_loss = total_loss / config.grad_accum_steps
                ppl = math.exp(min(avg_loss, 20.0))
                current_lr = scheduler.get_last_lr()
                
                # Update progress bar
                pbar.set_description(
                    f"Epoch {epoch} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | "
                    f"LR: {current_lr:.2e} | Clip: {current_clip} | GradNorm: {grad_norm:.2f}"
                )
                
                # Log step
                step_log = {
                    'step': step,
                    'optimizer_step': optimizer_step,
                    'epoch': epoch,
                    'loss': avg_loss,
                    'ppl': ppl,
                    'lr': current_lr,
                    'grad_norm': grad_norm,
                    'grad_clip': current_clip,
                    'skip_count': skip_count,
                }
                
                # Add diagnostics
                for k, v in diagnostics.items():
                    if isinstance(v, torch.Tensor):
                        step_log[k] = v.item() if v.numel() == 1 else v.mean().item()
                    elif isinstance(v, (int, float)):
                        step_log[k] = v
                
                training_log['steps'].append(step_log)
                
                # Save checkpoint at save_interval
                if optimizer_step > 0 and optimizer_step % (config.save_interval // config.grad_accum_steps) == 0:
                    ckpt_path = os.path.join(config.save_dir, f"step_{step}.pt")
                    save_checkpoint(
                        ckpt_path, model, optimizer, scheduler, scaler, ema,
                        step, epoch, avg_loss, config
                    )
                    
                    # Also save training log
                    log_path = os.path.join(config.save_dir, "training_log.json")
                    training_log['last_update'] = datetime.now().isoformat()
                    save_training_log(training_log, log_path)
                
                total_loss = 0.0
            
            pbar.update(1)
            
            if config.max_steps and step >= config.max_steps:
                break
        
        if config.max_steps and step >= config.max_steps:
            break
    
    pbar.close()
    print("\n✅ Training Complete!")
    print(f"Total steps: {step}, Optimizer steps: {optimizer_step}, Skipped: {skip_count}")
    
    # Save final checkpoint
    final_path = os.path.join(config.save_dir, "phase8_10b_final.pt")
    save_checkpoint(
        final_path, model, optimizer, scheduler, scaler, ema,
        step, epoch, avg_loss if 'avg_loss' in dir() else 0.0, config
    )
    
    # Save final training log
    training_log['end_time'] = datetime.now().isoformat()
    training_log['final_step'] = step
    training_log['final_loss'] = training_log['steps'][-1]['loss'] if training_log['steps'] else None
    log_path = os.path.join(config.save_dir, "training_log.json")
    save_training_log(training_log, log_path)
    print(f"📝 Training log saved to {log_path}")


if __name__ == "__main__":
    train_phase8()
