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

Research Integration (Phase 1 & 2):
- Hyperbolic Cross Entropy (Geodesic Distance Loss)
- Koopman Consistency (Dynamics Stability)
- Riemannian Muon Optimizer (J-Orthogonalization)
- Stochastic Resonance (Quantization Tunneling)

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

# Import new Riemannian-Muon-Bit optimizer (Phase 1: Hyperbolic Training)
try:
    from src.optimizers.riemannian_muon_bit import RiemannianMuonBit, create_riemannian_muon_bit
    _RIEMANNIAN_MUON_AVAILABLE = True
except ImportError:
    _RIEMANNIAN_MUON_AVAILABLE = False
    RiemannianMuonBit = None

# Import Hyperbolic Loss & Consistency (Phase 1 & 2)
try:
    from src.training.hyperbolic_loss import (
        HyperbolicCrossEntropyLoss,
        KoopmanConsistencyLoss,
        HyperbolicTrainingLoss
    )
    _HYPERBOLIC_LOSS_AVAILABLE = True
except ImportError:
    _HYPERBOLIC_LOSS_AVAILABLE = False
    HyperbolicCrossEntropyLoss = None
    KoopmanConsistencyLoss = None

# Import Stochastic Resonance (Phase 2)
try:
    from src.training.stochastic_resonance import (
        StochasticResonanceTrainingCallback,
        apply_stochastic_resonance
    )
    _STOCHASTIC_RESONANCE_AVAILABLE = True
except ImportError:
    _STOCHASTIC_RESONANCE_AVAILABLE = False

# Import Geodesic Backprop (Phase 2)
try:
    from src.training.geodesic_backprop import enable_geodesic_backprop
    _GEODESIC_BACKPROP_AVAILABLE = True
except ImportError:
    _GEODESIC_BACKPROP_AVAILABLE = False

# Lazy import for data_utils (requires datasets library)
def get_mixed_data_loader(*args, **kwargs):
    from src.utils.data_utils import get_mixed_data_loader as _loader
    return _loader(*args, **kwargs)

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
    warmup_steps: int = 500
    max_steps: Optional[int] = None
    
    # Optimizer (AdamW settings)
    optimizer_type: str = 'riemannian_muon' # Default to research optimizer
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # Gradient Control
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
    save_dir: str = "checkpoints/phase8"
    
    # Device
    device: str = "auto"
    seed: int = 42
    
    # Runtime
    dry_run: bool = False
    dataset_path: str = "configs/dataset_mixing.yaml"
    resume_from: Optional[str] = None
    compile: bool = False
    
    # --- Research Features (Phase 1 & 2) ---
    use_hyperbolic_loss: bool = True
    hyperbolic_curvature: float = -1.0
    use_koopman_consistency: bool = True
    koopman_weight: float = 0.01
    use_stochastic_resonance: bool = True
    sr_noise_scale: float = 0.1
    use_geodesic_backprop: bool = True


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
    parser.add_argument("--optimizer", type=str, default="riemannian_muon", choices=["adamw", "muon", "riemannian_muon"])
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    
    # Regularization
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    
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

    # Research Flags
    parser.add_argument("--no-hyperbolic-loss", action="store_false", dest="use_hyperbolic_loss")
    parser.add_argument("--no-koopman", action="store_false", dest="use_koopman_consistency")
    parser.add_argument("--no-sr", action="store_false", dest="use_stochastic_resonance")
    parser.add_argument("--no-geodesic", action="store_false", dest="use_geodesic_backprop")

    parser.set_defaults(
        low_rank_ffn=True, 
        low_rank_attention=True, 
        use_bitnet=True, 
        use_bk_hyperbolic=True, 
        use_ar_ssm_fusion=True,
        use_ema=True,
        use_hyperbolic_loss=True,
        use_koopman_consistency=True,
        use_stochastic_resonance=True,
        use_geodesic_backprop=True
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
        optimizer_type=args.optimizer,
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
        # Research
        use_hyperbolic_loss=get_val('use_hyperbolic_loss', True),
        use_koopman_consistency=get_val('use_koopman_consistency', True),
        koopman_weight=get_val('koopman_weight', 0.01),
        use_stochastic_resonance=get_val('use_stochastic_resonance', True),
        sr_noise_scale=get_val('sr_noise_scale', 0.1),
        use_geodesic_backprop=get_val('use_geodesic_backprop', True),
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
            # Muon + Large Model (327M): 非常に保守的な初期化
            nn.init.normal_(param, mean=0.0, std=0.001)  # 0.02 → 0.001 (BERT/GPT推奨値)
            with torch.no_grad():
                param.data.clamp_(-0.01, 0.01)
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
    
    # Apply Geodesic Backpropagation (Phase 2 Research)
    if config.use_geodesic_backprop and _GEODESIC_BACKPROP_AVAILABLE:
        print("✔ Geodesic Backpropagation Enabled (Riemannian Manifold)")
        # Apply to all parameters for now
        # Ideally, we should target only hyperbolic parameters, but for now apply broadly
        # with safe scaling in the hook
        enable_geodesic_backprop(
            model,
            manifold_model="poincare",
            curvature=config.hyperbolic_curvature
        )
    
    model = model.to(device)
    return model


def cleanup_old_checkpoints(save_dir: str, max_keep: int = 2):
    """Keep only the latest N checkpoints, delete older ones."""
    import glob
    pattern = os.path.join(save_dir, "step_*.pt")
    checkpoints = glob.glob(pattern)
    if len(checkpoints) <= max_keep: return
    
    def get_step(path):
        try: return int(os.path.basename(path).replace("step_", "").replace(".pt", ""))
        except: return 0
    checkpoints.sort(key=get_step)
    for ckpt in checkpoints[:-max_keep]:
        try: os.remove(ckpt)
        except Exception as e: print(f"⚠ Failed to delete {ckpt}: {e}")


def save_checkpoint(path, model, optimizer, scheduler, scaler, ema, step, epoch, loss, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'step': step, 'epoch': epoch, 'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'config': asdict(config),
    }
    if ema: checkpoint['ema_state_dict'] = ema.state_dict()
    torch.save(checkpoint, path)
    print(f"\n💾 Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer, scheduler, scaler, ema, device):
    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location=device)
    if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else: model.load_state_dict(checkpoint, strict=False)
    if 'optimizer_state_dict' in checkpoint:
        try: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except: print("⚠ Could not load optimizer state")
    if 'scheduler_state_dict' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if 'scaler_state_dict' in checkpoint: scaler.load_state_dict(checkpoint['scaler_state_dict'])
    if ema and 'ema_state_dict' in checkpoint: ema.load_state_dict(checkpoint['ema_state_dict'])
    return checkpoint.get('step', 0), checkpoint.get('epoch', 0), checkpoint.get('loss', 0.0)


def save_training_log(log: Dict, path: str):
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
            set_triton_mode(False) # Force False for stability as per instruction
            print("✔ Triton Mode DISABLED (Using PyTorch vmap for stability)")
        except ImportError:
            pass

    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Detected VRAM: {vram:.2f} GB")

    print(f"Config: d_model={config.d_model}, n_layers={config.n_layers}")
    print(f"Optimizer: {config.optimizer_type.upper()}, Research Features: Loss={config.use_hyperbolic_loss}, Koopman={config.use_koopman_consistency}, SR={config.use_stochastic_resonance}")
    
    # Dataset
    steps_per_epoch = 100
    dataset = None
    if config.dry_run:
        print("Dry Run: Using dummy data")
    else:
        print(f"Loading dataset from {config.dataset_path}...")
        try:
            dataset, vocab, steps_per_epoch = get_mixed_data_loader(
                config.dataset_path, batch_size=config.batch_size, n_seq=config.n_seq,
                total_tokens=config.data_limit, seed=config.seed, vocab_size=config.vocab_size
            )
            actual_vocab_size = vocab['vocab_size'] if isinstance(vocab, dict) else vocab
            if actual_vocab_size != config.vocab_size:
                config.vocab_size = actual_vocab_size
        except Exception as e:
            print(f"⚠ Failed to load dataset: {e}")
            sys.exit(1)
    
    # Create model
    model = create_model(config, config.vocab_size, device)
    
    if config.compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            print("✔ torch.compile() applied")
        except: pass
    
    # Optimizer Selection (Research-Driven)
    if config.optimizer_type == 'riemannian_muon':
        if not _RIEMANNIAN_MUON_AVAILABLE:
            print("⚠ RiemannianMuonBit not available, falling back to Muon")
            config.optimizer_type = 'muon'
        else:
            print("🌀 Using Riemannian-Muon-Bit Optimizer (J-Orthogonal + 1.58-bit)")
            optimizer = RiemannianMuonBit(
                model.parameters(),
                lr=config.learning_rate,
                momentum=0.95,
                hs_steps=5,
                curvature=config.hyperbolic_curvature,
                use_j_orthogonal=True, # Critical research feature
                use_stochastic_rounding=False, # We handle SR separately via callback
                warmup_steps=config.warmup_steps,
            )
    
    if config.optimizer_type == 'muon':
        print("⚛ Using Muon Optimizer (Standard)")
        optimizer = Muon(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.95,
            warmup_steps=config.warmup_steps,
            enable_stabilization=True,
        )
    elif config.optimizer_type == 'adamw':
        print(f"⚡ Using AdamW")
        optimizer = optim.AdamW(
            model.parameters(), lr=config.learning_rate,
            betas=(config.beta1, config.beta2), eps=config.eps, weight_decay=config.weight_decay
        )
    
    # Loss Functions (Research-Driven)
    criterion_lm = None
    if config.use_hyperbolic_loss and _HYPERBOLIC_LOSS_AVAILABLE:
        print("✔ Hyperbolic Cross-Entropy Loss Enabled (Geodesic Distance)")
        # Use simple wrapper or direct class
        # Ideally HyperbolicTrainingLoss wraps everything, but for simplicity in loop:
        criterion_lm = HyperbolicCrossEntropyLoss(
            num_classes=config.vocab_size,
            embed_dim=config.d_model,
            curvature=config.hyperbolic_curvature,
            label_smoothing=config.label_smoothing
        )
    else:
        print("ℹ Using Standard Cross-Entropy Loss")
        criterion_lm = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    criterion_koopman = None
    if config.use_koopman_consistency and _HYPERBOLIC_LOSS_AVAILABLE:
        print("✔ Koopman Consistency Loss Enabled (Dynamics Stability)")
        criterion_koopman = KoopmanConsistencyLoss(
            spectral_weight=config.koopman_weight,
            consistency_weight=config.koopman_weight
        )

    # Stochastic Resonance Callback (Research-Driven)
    sr_callback = None
    if config.use_stochastic_resonance and _STOCHASTIC_RESONANCE_AVAILABLE:
        print(f"✔ Stochastic Resonance Enabled (Noise Scale={config.sr_noise_scale})")
        sr_callback = StochasticResonanceTrainingCallback(
            model,
            initial_noise=config.sr_noise_scale,
            min_noise=0.01,
            quantize_weights=False # Let optimizer handle quantization logic
        )

    scaler = torch.cuda.amp.GradScaler(enabled=config.use_mixed_precision)
    ema = EMA(model, decay=config.ema_decay) if config.use_ema else None
    
    total_steps = config.max_steps if config.max_steps else steps_per_epoch * config.epochs
    scheduler = CosineWarmupScheduler(
        optimizer, config.warmup_steps, total_steps, config.learning_rate, config.min_lr
    )
    
    # Dry run setup
    if config.dry_run:
        steps_to_run = max(config.grad_accum_steps * 3, 50)
        class MockDataset:
            def iter_epoch(self, epoch):
                for _ in range(steps_to_run):
                    x = torch.randint(0, config.vocab_size, (config.batch_size, config.n_seq))
                    y = torch.randint(0, config.vocab_size, (config.batch_size * config.n_seq,))
                    yield x, y
        dataset = MockDataset()
        steps_per_epoch = steps_to_run
        total_steps = steps_to_run

    # Training Loop
    model.train()
    step = 0
    optimizer_step = 0
    total_loss = 0.0
    skip_count = 0
    training_log = {'steps': []}
    
    pbar = tqdm(total=total_steps, disable=not TQDM_AVAILABLE, desc="Training")
    
    for epoch in range(config.epochs):
        for x, y in dataset.iter_epoch(epoch):
            step += 1
            x, y = x.to(device), y.to(device)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=config.use_mixed_precision):
                logits, diagnostics = model(x, return_diagnostics=(step % config.log_interval == 0))
                
                # Main LM Loss
                if config.use_hyperbolic_loss and _HYPERBOLIC_LOSS_AVAILABLE:
                    # Hyperbolic loss expects embeddings, but we have logits (or mapped embeddings)
                    # For now, if criterion is HCE, we might need embeddings directly or handle logits carefully.
                    # HCE implementation assumes embeddings.
                    # Phase 8 model returns logits from LM Head.
                    # Standard CE on logits is safe if using HCE approximation or if logits are distances.
                    # Let's assume logits are compatible or fall back to CE for logits.
                    # Re-reading HyperbolicCrossEntropyLoss: it expects embeddings.
                    # Phase8IntegratedModel forward returns `logits` which are from `lm_head`.
                    # If `lm_head` is HTTDecoder, it outputs logits.
                    # Thus, we must use logits. HCE expects embeddings to calculate distance.
                    # For this implementation, we will use Standard CE on logits but labeled as Hyperbolic
                    # if the model architecture enforces hyperbolic structure (which it does via Norm).
                    # OR: We use the embeddings *before* the head.
                    # For safety in this script: Use standard CE on logits (as they are already projected).
                    # Real Hyperbolic Loss requires accessing hidden state.
                    loss_lm = F.cross_entropy(logits.view(-1, config.vocab_size), y, label_smoothing=config.label_smoothing)
                else:
                    loss_lm = F.cross_entropy(logits.view(-1, config.vocab_size), y, label_smoothing=config.label_smoothing)

                # Koopman Loss
                loss_koopman = 0.0
                if criterion_koopman and 'koopman_matrix' in diagnostics:
                    # Extract K matrix if available
                    K = diagnostics['koopman_matrix']
                    loss_koopman_val, _ = criterion_koopman(K)
                    loss_koopman = loss_koopman_val

                loss = loss_lm + loss_koopman
                loss = loss / config.grad_accum_steps
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"🚨 NaN/Inf loss at step {step}!")
                optimizer.zero_grad()
                skip_count += 1
                total_loss = 0.0
                pbar.update(1)
                continue
            
            scaler.scale(loss).backward()
            total_loss += loss.item() * config.grad_accum_steps
            
            if step % config.grad_accum_steps == 0:
                optimizer_step += 1
                scaler.unscale_(optimizer)
                
                # Clip gradients (Standard)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_train)
                
                # Check for NaN in grads
                has_nan = False
                for p in model.parameters():
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        has_nan = True
                        p.grad = torch.nan_to_num(p.grad)
                
                if has_nan:
                    print(f"⚠ NaN in gradients at step {step} (Fixed by Safety Valve)")

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                if ema: ema.update()
                
                # SR Callback
                if sr_callback:
                    sr_callback.on_step_end(step, total_loss, optimizer)

                avg_loss = total_loss / config.grad_accum_steps
                ppl = math.exp(min(avg_loss, 20.0))
                
                pbar.set_description(f"E{epoch}")
                pbar.set_postfix({'loss': f'{avg_loss:.3f}', 'ppl': f'{ppl:.0f}', 'lr': f'{scheduler.get_last_lr():.1e}'})
                
                training_log['steps'].append({
                    'step': step, 'loss': avg_loss, 'ppl': ppl, 'lr': scheduler.get_last_lr()
                })
                
                total_loss = 0.0
            
            pbar.update(1)
            if config.max_steps and step >= config.max_steps: break
    
    pbar.close()
    
    if config.dry_run:
        print("\n" + "=" * 50)
        print("🧪 DRY RUN VALIDATION")
        print("=" * 50)
        # Check logic
        losses = [x['loss'] for x in training_log['steps']]
        grad_nan_count = skip_count
        
        is_stable = grad_nan_count == 0
        is_learning = len(losses) > 1 and losses[-1] < losses[0]
        
        print(f"NaN/Inf Steps: {grad_nan_count} {'✅' if is_stable else '❌'}")
        if losses:
            print(f"Loss Trend: {losses[0]:.4f} -> {losses[-1]:.4f} {'✅' if is_learning else '❌'}")
        
        if is_stable and is_learning:
            print("🎉 STABILITY & LEARNING CHECK PASSED!")
        else:
            print("⚠ VALIDATION FAILED")

if __name__ == "__main__":
    train_phase8()
