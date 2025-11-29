#!/usr/bin/env python3
"""
ResNet-BK 8GB VRAM対応訓練スクリプト
BK-Coreを使ったO(N)言語モデルの訓練
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

from src.models.resnet_bk import LanguageModel
from src.models.config import ResNetBKConfig


def create_dummy_dataset(config: ResNetBKConfig, num_samples: int = 1000):
    """ダミーデータセット作成"""
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
    
    return DummyDataset(num_samples, config.n_seq, config.vocab_size)


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, 
                gradient_accumulation_steps=1, use_amp=False, empty_cache_every=10):
    """1エポックの訓練"""
    model.train()
    total_loss = 0
    total_tokens = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Total VRAM: {total_vram:.2f} GB")
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        # Memory check
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            if allocated > total_vram * 0.9:
                print(f"\nWarning: High memory usage! {allocated:.2f}GB / {total_vram:.2f}GB")
                torch.cuda.empty_cache()
        
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward with AMP
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
                loss = loss / gradient_accumulation_steps
        else:
            logits = model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            loss = loss / gradient_accumulation_steps
        
        # NaN検出
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n❌ NaN/Inf detected at batch {batch_idx}! Skipping...")
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            continue
        
        # Backward
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient check
            has_nan_grad = False
            for p in model.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                print(f"\n❌ NaN in gradients at batch {batch_idx}! Skipping...")
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
        
        # Clear cache
        if (batch_idx + 1) % empty_cache_every == 0:
            torch.cuda.empty_cache()
        
        # Update progress
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
    parser = argparse.ArgumentParser(description='ResNet-BK Training (8GB VRAM)')
    
    # Model config
    parser.add_argument('--d-model', type=int, default=512)
    parser.add_argument('--n-layers', type=int, default=12)
    parser.add_argument('--n-seq', type=int, default=512)
    parser.add_argument('--num-experts', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=2)
    parser.add_argument('--use-birman-schwinger', action='store_true')
    parser.add_argument('--use-hybrid-attention', action='store_true')
    
    # Training config
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--gradient-accumulation', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup-steps', type=int, default=1000)
    parser.add_argument('--use-amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--gradient-checkpointing', action='store_true')
    
    # Data
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--num-samples', type=int, default=10000)
    
    # Checkpoint
    parser.add_argument('--save-dir', type=str, default='checkpoints/resnet_bk_8gb')
    parser.add_argument('--save-every', type=int, default=1000)
    
    # Config file
    parser.add_argument('--config', type=str, help='Config YAML file')
    
    args = parser.parse_args()
    
    # Load config from file if specified
    if args.config:
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)
            if 'model' in config_dict:
                model_cfg = config_dict['model']
                args.d_model = model_cfg.get('d_model', args.d_model)
                args.n_layers = model_cfg.get('n_layers', args.n_layers)
                args.n_seq = model_cfg.get('n_seq', args.n_seq)
                args.num_experts = model_cfg.get('num_experts', args.num_experts)
                args.top_k = model_cfg.get('top_k', args.top_k)
                args.use_birman_schwinger = model_cfg.get('use_birman_schwinger', args.use_birman_schwinger)
                args.use_hybrid_attention = model_cfg.get('use_hybrid_attention', args.use_hybrid_attention)
            if 'training' in config_dict:
                train_cfg = config_dict['training']
                args.batch_size = train_cfg.get('batch_size', args.batch_size)
                args.gradient_accumulation = train_cfg.get('gradient_accumulation_steps', args.gradient_accumulation)
                args.epochs = train_cfg.get('epochs', args.epochs)
                args.lr = train_cfg.get('learning_rate', args.lr)
                args.warmup_steps = train_cfg.get('warmup_steps', args.warmup_steps)
                args.use_amp = train_cfg.get('mixed_precision', args.use_amp) or train_cfg.get('fp16', args.use_amp)
                args.gradient_checkpointing = train_cfg.get('gradient_checkpointing', args.gradient_checkpointing)
    
    # Create model config
    model_config = ResNetBKConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_seq=args.n_seq,
        num_experts=args.num_experts,
        top_k=args.top_k,
        use_birman_schwinger=args.use_birman_schwinger,
        use_hybrid_attention=args.use_hybrid_attention,
        use_gradient_checkpointing=args.gradient_checkpointing,
        use_mixed_precision=args.use_amp,
    )
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"ResNet-BK Training - 8GB VRAM Optimized")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"\nModel Config:")
    print(f"  d_model: {model_config.d_model}")
    print(f"  n_layers: {model_config.n_layers}")
    print(f"  n_seq: {model_config.n_seq}")
    print(f"  num_experts: {model_config.num_experts}")
    print(f"  top_k: {model_config.top_k}")
    print(f"  use_birman_schwinger: {model_config.use_birman_schwinger}")
    print(f"  use_hybrid_attention: {model_config.use_hybrid_attention}")
    print(f"\nTraining Config:")
    print(f"  batch_size: {args.batch_size}")
    print(f"  gradient_accumulation: {args.gradient_accumulation}")
    print(f"  epochs: {args.epochs}")
    print(f"  learning_rate: {args.lr}")
    print(f"  warmup_steps: {args.warmup_steps}")
    print(f"  mixed_precision: {args.use_amp}")
    print(f"  gradient_checkpointing: {args.gradient_checkpointing}")
    print(f"{'='*60}\n")
    
    # Create model
    model = LanguageModel(model_config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)\n")
    
    # Create dataset
    if args.dry_run:
        print("Using dummy dataset...")
        dataset = create_dummy_dataset(model_config, num_samples=args.num_samples)
    else:
        print("Loading real dataset...")
        # TODO: Implement real dataset loading
        dataset = create_dummy_dataset(model_config, num_samples=args.num_samples)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(dataloader) * args.epochs
    
    def lr_lambda(current_step):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        else:
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
            model, dataloader, optimizer, scheduler, device, epoch,
            gradient_accumulation_steps=args.gradient_accumulation,
            use_amp=args.use_amp,
            empty_cache_every=10
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


if __name__ == '__main__':
    main()
