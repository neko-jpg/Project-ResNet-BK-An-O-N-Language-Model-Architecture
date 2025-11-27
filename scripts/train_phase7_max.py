#!/usr/bin/env python3
"""
Phase 7 Maximum Parameters Training Script (1.8B Monster)

Teppeiの夢: 8GB VRAMで1.8Bパラメータモデルを訓練する

物理的直観:
- 低ランク圧縮により、パラメータ数を維持しつつVRAM使用量を削減
- Gradient Checkpointingで活性化メモリを節約
- Mixed Precision (FP16) でメモリ効率を2倍に
- Tritonカーネルによる高速化

Usage:
    make train-chat                          # チャットAI訓練開始
    make train-chat-test                     # ダミーデータでテスト
    python scripts/train_phase7_max.py       # 直接実行

Requirements:
    - NVIDIA GPU with CUDA (8GB+ VRAM)
    - Triton for optimized kernels
    - Dataset prepared via `make recipe`
"""

import argparse
import gc
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.checkpoint import checkpoint

warnings.filterwarnings("ignore", category=UserWarning)


def check_requirements():
    """環境チェック"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Phase 7 Max requires GPU.")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✓ GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    
    triton_available = False
    try:
        import triton
        print(f"✓ Triton {getattr(triton, '__version__', 'unknown')} detected")
        triton_available = True
    except ImportError:
        print("⚠ Triton not found. Using PyTorch fallback (slower).")
    
    return vram_gb, triton_available


def clear_memory():
    """メモリクリア"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class LowRankLinear(nn.Module):
    """低ランク線形層 (パラメータ圧縮)"""
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class EfficientSSMBlock(nn.Module):
    """効率的SSMブロック (Phase 7 Hybrid Attention簡易版)"""
    def __init__(self, d_model: int, ffn_rank: int, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # SSM-like gating (O(N) complexity)
        self.ssm_proj = nn.Linear(d_model, d_model)
        self.ssm_gate = nn.Linear(d_model, d_model)
        
        # Low-rank FFN
        self.ffn = nn.Sequential(
            LowRankLinear(d_model, d_model * 4, ffn_rank),
            nn.GELU(),
            LowRankLinear(d_model * 4, d_model, ffn_rank),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SSM path
        h = self.norm1(x)
        gate = torch.sigmoid(self.ssm_gate(h))
        h = self.ssm_proj(h) * gate
        x = x + h
        
        # FFN path
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h
        
        return x


class Phase7MaxModel(nn.Module):
    """
    Phase 7 Maximum Parameters Model (1.8B)
    
    低ランク圧縮を活用した超大規模モデル
    - 埋め込み: 低ランク分解 (vocab_size × embed_rank + embed_rank × d_model)
    - FFN: 低ランク分解 (d_model × ffn_rank × 4 × 2)
    - 出力: 低ランク分解 (d_model × head_rank + head_rank × vocab_size)
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 4096,
        n_layers: int = 32,
        n_seq: int = 512,
        num_heads: int = 32,
        embed_rank: int = 512,
        ffn_rank: int = 512,
        head_rank: int = 512,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq
        self.use_checkpoint = use_checkpoint
        
        # Low-rank embedding
        self.embed_down = nn.Embedding(vocab_size, embed_rank)
        self.embed_up = nn.Linear(embed_rank, d_model, bias=False)
        self.pos_embed = nn.Embedding(n_seq, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            EfficientSSMBlock(d_model, ffn_rank, num_heads)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
        # Low-rank output head
        self.head_down = nn.Linear(d_model, head_rank, bias=False)
        self.head_up = nn.Linear(head_rank, vocab_size, bias=True)
        
        self._init_weights()
        self._count_params()
    
    def _init_weights(self):
        """重み初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _count_params(self):
        """パラメータ数カウント"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.total_params = total
        self.trainable_params = trainable
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        
        # Embedding
        x = self.embed_up(self.embed_down(input_ids))
        pos = torch.arange(L, device=input_ids.device)
        x = x + self.pos_embed(pos)
        
        # Blocks with gradient checkpointing
        for block in self.blocks:
            if self.training and self.use_checkpoint:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        # Output
        x = self.final_norm(x)
        logits = self.head_up(self.head_down(x))
        
        return logits


@dataclass
class TrainingConfig:
    """訓練設定"""
    # Model
    vocab_size: int = 50257
    d_model: int = 4096
    n_layers: int = 32
    n_seq: int = 512
    num_heads: int = 32
    embed_rank: int = 512
    ffn_rank: int = 512
    head_rank: int = 512
    
    # Training
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    max_steps: int = 100000
    
    # Memory optimization
    use_checkpoint: bool = True
    use_mixed_precision: bool = True
    
    # Logging
    log_interval: int = 50
    save_interval: int = 2000
    eval_interval: int = 500
    save_dir: str = "checkpoints/phase7_max_push"
    
    # Data
    data_limit: int = 100_000_000
    seed: int = 42


def load_config(config_path: str) -> TrainingConfig:
    """YAMLから設定読み込み"""
    config = TrainingConfig()
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        for key, value in yaml_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return config


def get_dummy_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device):
    """ダミーバッチ生成 (データセットがない場合のテスト用)"""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size * seq_len,), device=device)
    return input_ids, targets


def train_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    config: TrainingConfig,
    accumulation_step: int,
) -> Tuple[float, float]:
    """訓練ステップ"""
    
    with torch.cuda.amp.autocast(enabled=config.use_mixed_precision):
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets
        )
        loss = loss / config.gradient_accumulation_steps
    
    scaler.scale(loss).backward()
    
    # Gradient accumulation
    if (accumulation_step + 1) % config.gradient_accumulation_steps == 0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        return loss.item() * config.gradient_accumulation_steps, grad_norm.item()
    
    return loss.item() * config.gradient_accumulation_steps, 0.0


def main():
    """メイン訓練ループ"""
    parser = argparse.ArgumentParser(description="Phase 7 Max Training (1.8B)")
    parser.add_argument("--config", type=str, default="configs/phase7_max_push.yaml",
                        help="Config file path")
    parser.add_argument("--dataset", type=str, default="configs/dataset_mixing.yaml",
                        help="Dataset config path")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test with dummy data")
    parser.add_argument("--use-integrated", action="store_true",
                        help="Use Phase7IntegratedModel with Triton kernels")
    args = parser.parse_args()
    
    # 環境チェック
    vram_gb, triton_available = check_requirements()
    
    # 設定読み込み
    config = load_config(args.config)
    
    print("\n" + "=" * 60)
    print("Phase 7 Maximum Parameters Training (1.8B Monster)")
    print("=" * 60)
    print(f"d_model: {config.d_model}")
    print(f"n_layers: {config.n_layers}")
    print(f"n_seq: {config.n_seq}")
    print(f"embed_rank: {config.embed_rank}")
    print(f"ffn_rank: {config.ffn_rank}")
    print(f"batch_size: {config.batch_size}")
    print(f"gradient_accumulation: {config.gradient_accumulation_steps}")
    print(f"effective_batch_size: {config.batch_size * config.gradient_accumulation_steps}")
    
    # シード設定
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    device = torch.device("cuda")
    
    # モデル作成
    print("\n🏗️  Creating model...")
    clear_memory()
    
    use_integrated = args.use_integrated and triton_available
    
    if use_integrated:
        # Phase 7 統合モデル (Tritonカーネル + HTT Embedding)
        print("  Using Phase7IntegratedModel with Triton kernels...")
        try:
            from src.models.phase7.integrated_model import Phase7IntegratedModel, Phase7Config
            
            phase7_config = Phase7Config(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                n_layers=config.n_layers,
                n_seq=config.n_seq,
                num_heads=config.num_heads,
                htt_rank=getattr(config, 'htt_rank', 64),
                use_hybrid_attention=True,
                hyperbolic_window_size=getattr(config, 'hyperbolic_window_size', 64),
                use_triton_kernel=triton_available,
                triton_kernel_version='fast',
                use_gradient_checkpointing=config.use_checkpoint,
                use_mixed_precision=config.use_mixed_precision,
            )
            model = Phase7IntegratedModel(phase7_config).to(device)
            total_params = model.get_total_parameter_count()
            model.total_params = total_params
            model.trainable_params = total_params
        except Exception as e:
            print(f"  ⚠ Failed to load integrated model: {e}")
            print("  Falling back to standalone model...")
            use_integrated = False
    
    if not use_integrated:
        # スタンドアロンモデル (Tritonなしでも動作)
        model = Phase7MaxModel(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_seq=config.n_seq,
            num_heads=config.num_heads,
            embed_rank=config.embed_rank,
            ffn_rank=config.ffn_rank,
            head_rank=config.head_rank,
            use_checkpoint=config.use_checkpoint,
        ).to(device)
    
    if config.use_mixed_precision:
        model = model.half()
    
    print(f"✓ Total parameters: {model.total_params / 1e9:.2f}B ({model.total_params / 1e6:.1f}M)")
    print(f"✓ Trainable parameters: {model.trainable_params / 1e9:.2f}B")
    print(f"✓ Model type: {'Phase7Integrated (Triton)' if use_integrated else 'Phase7Max (Standalone)'}")
    
    # VRAM使用量確認
    torch.cuda.synchronize()
    model_vram = torch.cuda.memory_allocated() / 1024**3
    print(f"✓ Model VRAM: {model_vram:.2f} GB")
    
    # データローダー
    print("\n📊 Loading data...")
    use_real_data = False
    mixed_loader = None
    steps_per_epoch = 1000
    
    dataset_path = Path(args.dataset)
    if not args.dry_run and dataset_path.exists():
        try:
            from src.utils.data_utils import get_mixed_data_loader
            mixed_loader, vocab, steps_per_epoch = get_mixed_data_loader(
                config_path=str(dataset_path),
                batch_size=config.batch_size,
                n_seq=config.n_seq,
                total_tokens=config.data_limit,
                seed=config.seed,
                vocab_size=config.vocab_size,
                split='train'
            )
            use_real_data = True
            print(f"✓ Dataset loaded. Steps per epoch: {steps_per_epoch}")
        except Exception as e:
            print(f"⚠ Failed to load dataset: {e}")
            print("  Using dummy data for testing...")
    else:
        print("  Using dummy data (dry-run mode or no dataset)")
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    total_steps = config.max_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=config.learning_rate / 10)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_mixed_precision)
    
    # Resume
    start_step = 0
    if args.resume_from and Path(args.resume_from).exists():
        print(f"\n📂 Resuming from: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'step' in ckpt:
            start_step = ckpt['step']
        print(f"✓ Resumed from step {start_step}")
    
    # Save directory
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\n🚀 Starting training...")
    print(f"   Max steps: {total_steps}")
    print(f"   Mixed precision: {config.use_mixed_precision}")
    print(f"   Gradient checkpointing: {config.use_checkpoint}")
    print()
    
    model.train()
    optimizer.zero_grad()
    
    training_log = []
    best_loss = float('inf')
    running_loss = 0.0
    step_count = 0
    
    global_step = start_step
    epoch = 0
    
    while global_step < total_steps:
        epoch += 1
        epoch_start = time.time()
        
        if use_real_data:
            batch_iter = mixed_loader.iter_epoch(epoch)
        else:
            batch_iter = range(steps_per_epoch * config.gradient_accumulation_steps)
        
        for batch_idx, batch_item in enumerate(batch_iter):
            if global_step >= total_steps:
                break
            
            step_start = time.time()
            
            # Get batch
            if use_real_data:
                input_ids, targets = batch_item
                input_ids = input_ids.to(device)
                targets = targets.to(device)
            else:
                input_ids, targets = get_dummy_batch(
                    config.batch_size, config.n_seq, config.vocab_size, device
                )
            
            # Training step
            try:
                loss, grad_norm = train_step(
                    model, optimizer, scaler,
                    input_ids, targets, config,
                    batch_idx
                )
                running_loss += loss
                step_count += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n⚠ OOM at step {global_step}. Clearing cache...")
                    clear_memory()
                    optimizer.zero_grad()
                    continue
                raise
            
            # Update step counter after gradient accumulation
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                global_step += 1
                scheduler.step()
                
                # Logging
                if global_step % config.log_interval == 0:
                    avg_loss = running_loss / step_count if step_count > 0 else 0
                    ppl = math.exp(min(avg_loss, 20))
                    lr = scheduler.get_last_lr()[0]
                    vram = torch.cuda.memory_allocated() / 1024**3
                    step_time = time.time() - step_start
                    
                    log_entry = {
                        'step': global_step,
                        'loss': avg_loss,
                        'ppl': ppl,
                        'lr': lr,
                        'grad_norm': grad_norm,
                        'vram_gb': vram,
                    }
                    training_log.append(log_entry)
                    
                    print(f"  Step {global_step:6d} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | "
                          f"LR: {lr:.2e} | VRAM: {vram:.2f}GB")
                    
                    running_loss = 0.0
                    step_count = 0
                
                # Save checkpoint
                if global_step % config.save_interval == 0:
                    ckpt_path = save_dir / f"step_{global_step}.pt"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step': global_step,
                        'config': config.__dict__,
                    }, ckpt_path)
                    print(f"  💾 Saved: {ckpt_path}")
                    
                    # Save best
                    if avg_loss < best_loss and avg_loss > 0:
                        best_loss = avg_loss
                        best_path = save_dir / "best.pt"
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'step': global_step,
                            'loss': avg_loss,
                        }, best_path)
                        print(f"  🏆 New best: {best_path}")
        
        epoch_time = time.time() - epoch_start
        print(f"\n📈 Epoch {epoch} completed in {epoch_time:.1f}s")
    
    # Final save
    final_path = save_dir / "final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': global_step,
        'config': config.__dict__,
    }, final_path)
    
    # Save training log
    log_path = save_dir / "training_log.json"
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"Final model: {final_path}")
    print(f"Training log: {log_path}")
    print(f"Total steps: {global_step}")


if __name__ == "__main__":
    main()
