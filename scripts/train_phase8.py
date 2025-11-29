#!/usr/bin/env python3
"""
Phase 8 Training Script - Hyperbolic Transcendence
双曲幾何学ベースのO(N)言語モデル訓練

Features:
- Tangent-Space Linear Attention (O(N) complexity)
- Hyperbolic SSM for sequential processing
- Adaptive computation based on hyperbolic distance
- Memory-efficient training on consumer GPUs
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import json
from tqdm import tqdm
import time
from datetime import datetime

# Phase 8 imports
try:
    from src.models.phase8.linear_attention import TangentSpaceLinearAttention, LinearAttentionConfig
    from src.models.phase8.hyperbolic_ssm import HyperbolicSSM, HyperbolicSSMConfig
    PHASE8_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Phase 8 modules not fully available: {e}")
    PHASE8_AVAILABLE = False


class Phase8Config:
    """Phase 8モデル設定"""
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 6,
        num_heads: int = 8,
        vocab_size: int = 50257,
        max_seq_len: int = 512,
        curvature: float = 0.01,  # 低曲率でO(N)を保証
        dropout: float = 0.1,
        use_hyperbolic_ssm: bool = False,  # SSMは重いのでデフォルトOFF
    ):
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.curvature = curvature
        self.dropout = dropout
        self.use_hyperbolic_ssm = use_hyperbolic_ssm


class Phase8Layer(nn.Module):
    """Phase 8 Transformer Layer"""
    def __init__(self, config: Phase8Config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        
        # Tangent-Space Linear Attention
        attn_config = LinearAttentionConfig(
            d_model=config.d_model,
            num_heads=config.num_heads,
            curvature=config.curvature,
            low_curvature_threshold=0.1,
            high_curvature_threshold=1.0,
            num_features=config.d_model // config.num_heads,
            kernel_type="elu"
        )
        self.attn = TangentSpaceLinearAttention(attn_config)
        
        # Optional: Hyperbolic SSM
        if config.use_hyperbolic_ssm:
            ssm_config = HyperbolicSSMConfig(
                d_model=config.d_model,
                d_state=config.d_model // 4,
                curvature=config.curvature
            )
            self.ssm = HyperbolicSSM(ssm_config)
        else:
            self.ssm = None
        
        # FFN with Low-Rank compression
        ffn_rank = config.d_model // 8
        ffn_hidden = config.d_model * 4
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, ffn_rank),
            nn.Linear(ffn_rank, ffn_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(ffn_hidden, ffn_rank),
            nn.Linear(ffn_rank, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        if self.ssm is not None:
            self.ln_ssm = nn.LayerNorm(config.d_model)
    
    def _forward_impl(self, x, mask=None):
        """実際のforward処理"""
        # SSM (optional)
        if self.ssm is not None:
            ssm_out = self.ssm(self.ln_ssm(x))
            if isinstance(ssm_out, tuple):
                ssm_out = ssm_out[0]
            x = x + ssm_out
        
        # Attention
        residual = x
        x = self.ln1(x)
        attn_out = self.attn(x, mask=mask)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        x = residual + attn_out
        
        # FFN
        x = x + self.ffn(self.ln2(x))
        return x
    
    def forward(self, x, mask=None):
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, mask, use_reentrant=False
            )
        else:
            return self._forward_impl(x, mask)


class Phase8Model(nn.Module):
    """Phase 8 Language Model"""
    def __init__(self, config: Phase8Config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        
        # Low-Rank Embedding (75% compression)
        embed_rank = config.d_model // 4
        self.embed_low = nn.Embedding(config.vocab_size, embed_rank)
        self.embed_high = nn.Linear(embed_rank, config.d_model)
        
        # Positional Encoding
        self.pos_embed = nn.Parameter(torch.randn(1, config.max_seq_len, config.d_model) * 0.02)
        
        # Transformer Layers
        self.layers = nn.ModuleList([
            Phase8Layer(config) for _ in range(config.n_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    
    def enable_gradient_checkpointing(self):
        """Gradient Checkpointingを有効化"""
        self.gradient_checkpointing = True
        for layer in self.layers:
            layer.gradient_checkpointing = True
    
    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        
        # Embedding
        x = self.embed_high(self.embed_low(input_ids))
        x = x + self.pos_embed[:, :T, :]
        
        # Transformer
        for layer in self.layers:
            x = layer(x)
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Loss
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        
        return logits, loss


def create_dummy_dataset(config: Phase8Config, num_samples: int = 1000):
    """ダミーデータセット作成（テスト用）"""
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples, seq_len, vocab_size):
            self.num_samples = num_samples
            self.seq_len = seq_len
            self.vocab_size = vocab_size
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
            labels = input_ids.clone()
            return {'input_ids': input_ids, 'labels': labels}
    
    return DummyDataset(num_samples, config.max_seq_len, config.vocab_size)


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, config, 
                gradient_accumulation_steps=1, use_amp=False, empty_cache_every=10):
    """1エポックの訓練"""
    model.train()
    total_loss = 0
    total_tokens = 0
    
    # AMP Scaler
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Memory monitoring
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Total VRAM: {total_vram:.2f} GB")
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        # Memory check before forward pass
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            if allocated > total_vram * 0.9:  # 90%以上使用で警告
                print(f"\nWarning: High memory usage! Allocated: {allocated:.2f}GB / {total_vram:.2f}GB")
                torch.cuda.empty_cache()
        
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward with AMP
        if use_amp:
            with torch.cuda.amp.autocast():
                logits, loss = model(input_ids, labels)
                loss = loss / gradient_accumulation_steps
        else:
            logits, loss = model(input_ids, labels)
            loss = loss / gradient_accumulation_steps
        
        # NaN検出
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n❌ NaN/Inf detected in loss at batch {batch_idx}! Skipping this batch...")
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            continue
        
        # Backward
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step (with gradient accumulation)
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # 勾配のNaNチェック
            grad_norm = 0.0
            has_nan_grad = False
            for p in model.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        has_nan_grad = True
                        break
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            if has_nan_grad:
                print(f"\n❌ NaN/Inf in gradients at batch {batch_idx}! Skipping optimizer step...")
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            else:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
                scheduler.step()
        
        # Stats
        total_loss += loss.item() * gradient_accumulation_steps * input_ids.numel()
        total_tokens += input_ids.numel()
        
        # Clear cache periodically
        if (batch_idx + 1) % empty_cache_every == 0:
            torch.cuda.empty_cache()
        
        # Update progress bar
        postfix = {
            'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
            'ppl': f'{torch.exp(loss * gradient_accumulation_steps).item():.2f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        }
        if torch.cuda.is_available():
            postfix['mem'] = f'{torch.cuda.memory_allocated() / 1024**3:.2f}GB'
        pbar.set_postfix(postfix)
    
    avg_loss = total_loss / total_tokens
    avg_ppl = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, avg_ppl


def main():
    parser = argparse.ArgumentParser(description='Phase 8 Training')
    
    # Model config
    parser.add_argument('--d-model', type=int, default=512)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--n-seq', type=int, default=512)
    parser.add_argument('--curvature', type=float, default=0.01)
    parser.add_argument('--use-ssm', action='store_true', help='Use Hyperbolic SSM')
    
    # Training config
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)  # 3e-4 -> 1e-4 (より安全)
    parser.add_argument('--warmup-steps', type=int, default=1000)
    
    # Data
    parser.add_argument('--dataset', type=str, help='Dataset config YAML')
    parser.add_argument('--dry-run', action='store_true', help='Use dummy data')
    
    # Checkpoint
    parser.add_argument('--resume-from', type=str, help='Resume from checkpoint')
    parser.add_argument('--save-dir', type=str, default='checkpoints/phase8')
    parser.add_argument('--save-every', type=int, default=1000)
    
    # Config file
    parser.add_argument('--config', type=str, help='Config YAML file')
    
    args = parser.parse_args()
    
    # Check Phase 8 availability
    if not PHASE8_AVAILABLE:
        print("Error: Phase 8 modules not available!")
        print("Please ensure src/models/phase8/ is properly installed.")
        sys.exit(1)
    
    # Load config from file if specified
    gradient_accumulation_steps = 1
    use_amp = False
    gradient_checkpointing = False
    empty_cache_every = 10
    use_8bit_adam = False
    
    if args.config:
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)
            # Load model config
            if 'model' in config_dict:
                model_cfg = config_dict['model']
                args.d_model = model_cfg.get('d_model', args.d_model)
                args.n_layers = model_cfg.get('n_layers', args.n_layers)
                args.num_heads = model_cfg.get('num_heads', args.num_heads)
                args.n_seq = model_cfg.get('max_seq_len', args.n_seq)
                args.curvature = model_cfg.get('curvature', args.curvature)
                args.use_ssm = model_cfg.get('use_hyperbolic_ssm', args.use_ssm)
            # Load training config
            if 'training' in config_dict:
                train_cfg = config_dict['training']
                args.batch_size = train_cfg.get('batch_size', args.batch_size)
                args.epochs = train_cfg.get('epochs', args.epochs)
                args.lr = train_cfg.get('learning_rate', args.lr)
                args.warmup_steps = train_cfg.get('warmup_steps', args.warmup_steps)
                gradient_accumulation_steps = train_cfg.get('gradient_accumulation_steps', 1)
                use_amp = train_cfg.get('mixed_precision', False) or train_cfg.get('fp16', False)
                gradient_checkpointing = train_cfg.get('gradient_checkpointing', False)
                use_8bit_adam = train_cfg.get('use_8bit_adam', False)
            # Load optimization config
            if 'optimization' in config_dict:
                opt_cfg = config_dict['optimization']
                empty_cache_every = opt_cfg.get('empty_cache_every', 10)
    
    # Create model config
    model_config = Phase8Config(
        d_model=args.d_model,
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        max_seq_len=args.n_seq,
        curvature=args.curvature,
        use_hyperbolic_ssm=args.use_ssm
    )
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Phase 8 Training - Hyperbolic Transcendence")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"\nModel Config:")
    print(f"  d_model: {model_config.d_model}")
    print(f"  n_layers: {model_config.n_layers}")
    print(f"  num_heads: {model_config.num_heads}")
    print(f"  max_seq_len: {model_config.max_seq_len}")
    print(f"  curvature: {model_config.curvature}")
    print(f"  use_ssm: {model_config.use_hyperbolic_ssm}")
    print(f"\nTraining Config:")
    print(f"  batch_size: {args.batch_size}")
    print(f"  epochs: {args.epochs}")
    print(f"  learning_rate: {args.lr}")
    print(f"  warmup_steps: {args.warmup_steps}")
    print(f"{'='*60}\n")
    
    # Create model
    model = Phase8Model(model_config)
    
    # Enable gradient checkpointing if requested
    if gradient_checkpointing:
        print("Enabling Gradient Checkpointing...")
        model.enable_gradient_checkpointing()
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Memory optimization info
    if use_amp:
        print(f"Mixed Precision (FP16): Enabled")
    if gradient_checkpointing:
        print(f"Gradient Checkpointing: Enabled")
    if gradient_accumulation_steps > 1:
        print(f"Gradient Accumulation: {gradient_accumulation_steps} steps (effective batch size: {args.batch_size * gradient_accumulation_steps})")
    print()
    
    # Create dataset
    if args.dry_run or not args.dataset:
        print("Using dummy dataset for testing...")
        dataset = create_dummy_dataset(model_config, num_samples=1000)
    else:
        print(f"Loading dataset from {args.dataset}...")
        # TODO: Implement real dataset loading
        dataset = create_dummy_dataset(model_config, num_samples=10000)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Optimizer & Scheduler
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.01)
            print("Using 8-bit AdamW optimizer (bitsandbytes)")
        except ImportError:
            print("Warning: bitsandbytes not available, falling back to standard AdamW")
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Warmup + Cosine Annealing Scheduler
    total_steps = len(dataloader) * args.epochs
    
    def lr_lambda(current_step):
        if current_step < args.warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, args.warmup_steps))
        else:
            # Cosine annealing
            progress = float(current_step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))))
    
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        avg_loss, avg_ppl = train_epoch(
            model, dataloader, optimizer, scheduler, device, epoch, model_config,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_amp=use_amp,
            empty_cache_every=empty_cache_every
        )
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Perplexity: {avg_ppl:.2f}")
        print(f"  Time: {time.time() - start_time:.2f}s\n")
        
        # Save checkpoint
        if epoch % (args.save_every // len(dataloader) + 1) == 0:
            checkpoint_path = save_dir / f"epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': vars(model_config),
                'loss': avg_loss,
                'perplexity': avg_ppl,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}\n")
    
    # Final save
    final_path = save_dir / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(model_config),
    }, final_path)
    print(f"\nTraining completed! Final model saved: {final_path}")
    
    # Save training summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': vars(model_config),
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'total_time': time.time() - start_time,
        },
        'final_metrics': {
            'loss': avg_loss,
            'perplexity': avg_ppl,
        }
    }
    
    summary_path = save_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved: {summary_path}")


if __name__ == '__main__':
    main()
