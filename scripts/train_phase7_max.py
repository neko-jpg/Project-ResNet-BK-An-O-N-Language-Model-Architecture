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
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.checkpoint import checkpoint

warnings.filterwarnings("ignore", category=UserWarning)

# Rich imports for beautiful progress display
try:
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
    )
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    from rich.layout import Layout
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠ rich not installed. Install with: pip install rich")
    print("  Falling back to basic output.")

console = Console() if RICH_AVAILABLE else None


def check_requirements():
    """環境チェック"""
    if not torch.cuda.is_available():
        if console:
            console.print("[red]❌ CUDA not available. Phase 7 Max requires GPU.[/red]")
        else:
            print("❌ CUDA not available. Phase 7 Max requires GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    triton_available = False
    try:
        import triton
        triton_version = getattr(triton, '__version__', 'unknown')
        triton_available = True
    except ImportError:
        if console:
            console.print("[red]❌ Triton not found![/red]")
            console.print("   Phase 7 requires Triton for optimal performance.")
            console.print("   Install with: [cyan]pip install triton[/cyan]")
        else:
            print("❌ Triton not found!")
            print("   Install with: pip install triton")
        sys.exit(1)

    if console:
        table = Table(title="🖥️ System Check", box=box.ROUNDED)
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_row("GPU", f"{gpu_name}")
        table.add_row("VRAM", f"{vram_gb:.1f} GB")
        table.add_row("Triton", f"v{triton_version}")
        table.add_row("CUDA", f"v{torch.version.cuda}")
        console.print(table)
    else:
        print(f"✓ GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        print(f"✓ Triton {triton_version} detected")

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

        self.ssm_proj = nn.Linear(d_model, d_model)
        self.ssm_gate = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            LowRankLinear(d_model, d_model * 4, ffn_rank),
            nn.GELU(),
            LowRankLinear(d_model * 4, d_model, ffn_rank),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        gate = torch.sigmoid(self.ssm_gate(h))
        h = self.ssm_proj(h) * gate
        x = x + h

        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h
        return x


class Phase7MaxModel(nn.Module):
    """Phase 7 Maximum Parameters Model (1.8B)"""

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

        self.embed_down = nn.Embedding(vocab_size, embed_rank)
        self.embed_up = nn.Linear(embed_rank, d_model, bias=False)
        self.pos_embed = nn.Embedding(n_seq, d_model)

        self.blocks = nn.ModuleList([
            EfficientSSMBlock(d_model, ffn_rank, num_heads)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.head_down = nn.Linear(d_model, head_rank, bias=False)
        self.head_up = nn.Linear(head_rank, vocab_size, bias=True)

        self._init_weights()
        self._count_params()

    def _init_weights(self):
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
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.total_params = total
        self.trainable_params = trainable

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        x = self.embed_up(self.embed_down(input_ids))
        pos = torch.arange(L, device=input_ids.device)
        x = x + self.pos_embed(pos)

        for block in self.blocks:
            if self.training and self.use_checkpoint:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.final_norm(x)
        logits = self.head_up(self.head_down(x))
        return logits


@dataclass
class TrainingConfig:
    """訓練設定"""
    vocab_size: int = 50257
    d_model: int = 3072      # 1B狙い
    n_layers: int = 24       # 24層
    n_seq: int = 512
    num_heads: int = 24
    embed_rank: int = 384    # d_model/8
    ffn_rank: int = 384
    head_rank: int = 384

    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    min_epochs: int = 5       # ★ 最低5エポックは回す
    max_steps: int = 100000   # ステップ数上限

    use_checkpoint: bool = True
    use_mixed_precision: bool = True

    log_interval: int = 50
    save_interval: int = 2000
    eval_interval: int = 500
    save_dir: str = "checkpoints/phase7_max_push"

    data_limit: int = 100_000_000
    seed: int = 42


def load_config(config_path: str) -> TrainingConfig:
    config = TrainingConfig()
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        for key, value in yaml_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    return config


def get_dummy_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device):
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
    with torch.cuda.amp.autocast(enabled=config.use_mixed_precision):
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets
        )
        loss = loss / config.gradient_accumulation_steps

    scaler.scale(loss).backward()

    if (accumulation_step + 1) % config.gradient_accumulation_steps == 0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        return loss.item() * config.gradient_accumulation_steps, grad_norm.item()

    return loss.item() * config.gradient_accumulation_steps, 0.0


def create_training_table(step, total_steps, loss, ppl, lr, vram, grad_norm, tokens_per_sec, best_loss):
    """リアルタイム訓練状況テーブル"""
    table = Table(box=box.ROUNDED, expand=True)
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="green", width=25)
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="green", width=25)

    progress_pct = (step / total_steps) * 100

    table.add_row(
        "Step", f"{step:,} / {total_steps:,}",
        "Progress", f"{progress_pct:.1f}%"
    )
    table.add_row(
        "Loss", f"{loss:.4f}",
        "Best Loss", f"{best_loss:.4f}" if best_loss < float('inf') else "N/A"
    )
    table.add_row(
        "Perplexity", f"{ppl:.2f}",
        "Learning Rate", f"{lr:.2e}"
    )
    table.add_row(
        "VRAM", f"{vram:.2f} GB",
        "Grad Norm", f"{grad_norm:.4f}" if grad_norm > 0 else "N/A"
    )
    table.add_row(
        "Tokens/sec", f"{tokens_per_sec:.1f}",
        "ETA", estimate_eta(step, total_steps, tokens_per_sec)
    )
    return table


def estimate_eta(current_step, total_steps, tokens_per_sec):
    """残り時間推定"""
    if tokens_per_sec <= 0 or current_step >= total_steps:
        return "N/A"
    remaining_steps = total_steps - current_step
    # 1ステップあたりのトークン数（batch * seq * accum）
    remaining_secs = remaining_steps * 512 / max(tokens_per_sec, 1)
    hours = int(remaining_secs // 3600)
    mins = int((remaining_secs % 3600) // 60)
    if hours > 0:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def main():
    """メイン訓練ループ"""
    parser = argparse.ArgumentParser(description="Phase 7 Max Training (1.8B)")
    parser.add_argument("--config", type=str, default="configs/phase7_max_push.yaml")
    parser.add_argument("--dataset", type=str, default="configs/dataset_mixing.yaml")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-integrated", action="store_true")
    args = parser.parse_args()

    # 環境チェック
    vram_gb, triton_available = check_requirements()

    # 設定読み込み
    config = load_config(args.config)

    # ヘッダー表示
    if console:
        console.print(Panel.fit(
            "[bold cyan]🚀 Phase 7 Maximum Parameters Training[/bold cyan]\n"
            "[yellow]1.8B Monster - Teppeiの夢[/yellow]",
            border_style="cyan"
        ))

        config_table = Table(title="📋 Configuration", box=box.ROUNDED)
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="green")
        config_table.add_row("d_model", str(config.d_model))
        config_table.add_row("n_layers", str(config.n_layers))
        config_table.add_row("n_seq", str(config.n_seq))
        config_table.add_row("embed_rank", str(config.embed_rank))
        config_table.add_row("ffn_rank", str(config.ffn_rank))
        config_table.add_row("batch_size", str(config.batch_size))
        config_table.add_row("gradient_accumulation", str(config.gradient_accumulation_steps))
        config_table.add_row("effective_batch", str(config.batch_size * config.gradient_accumulation_steps))
        config_table.add_row("min_epochs", str(config.min_epochs))
        config_table.add_row("max_steps", f"{config.max_steps:,}")
        console.print(config_table)
    else:
        print("\n" + "=" * 60)
        print("Phase 7 Maximum Parameters Training (1.8B Monster)")
        print("=" * 60)
        print(f"d_model: {config.d_model}, n_layers: {config.n_layers}")

    # シード設定
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda")

    # モデル作成
    if console:
        console.print("\n[bold]🏗️  Creating model...[/bold]")
    else:
        print("\n🏗️  Creating model...")

    clear_memory()

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

    if console:
        console.print(f"[green]✓ Parameters: {model.total_params / 1e9:.2f}B ({model.total_params / 1e6:.1f}M)[/green]")
    else:
        print(f"✓ Parameters: {model.total_params / 1e9:.2f}B")

    # データローダー
    if console:
        console.print("\n[bold]📊 Loading data...[/bold]")

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
            if console:
                console.print(f"[green]✓ Dataset loaded. Steps/epoch: {steps_per_epoch:,}[/green]")
        except Exception as e:
            if console:
                console.print(f"[yellow]⚠ Dataset error: {e}[/yellow]")
    else:
        if console:
            console.print("[yellow]Using dummy data (dry-run mode)[/yellow]")

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = config.max_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=config.learning_rate / 10)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_mixed_precision)

    # Resume
    start_step = 0
    if args.resume_from and Path(args.resume_from).exists():
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'step' in ckpt:
            start_step = ckpt['step']
        if console:
            console.print(f"[green]✓ Resumed from step {start_step}[/green]")

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop with rich progress
    model.train()
    optimizer.zero_grad()

    training_log = []
    best_loss = float('inf')
    running_loss = 0.0
    step_count = 0
    global_step = start_step
    epoch = 0
    tokens_processed = 0
    start_time = time.time()

    if console and RICH_AVAILABLE:
        console.print("\n[bold green]🚀 Training Started![/bold green]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=2,
        ) as progress:
            train_task = progress.add_task(
                f"[cyan]Training Phase 7 (1.8B)",
                total=total_steps - start_step
            )

            # 終了条件: min_epochs完了 AND (max_steps到達 OR 十分な訓練)
            while epoch < config.min_epochs or global_step < total_steps:
                epoch += 1
                if epoch > config.min_epochs and global_step >= total_steps:
                    break

                if use_real_data:
                    batch_iter = mixed_loader.iter_epoch(epoch)
                else:
                    batch_iter = range(steps_per_epoch * config.gradient_accumulation_steps)

                for batch_idx, batch_item in enumerate(batch_iter):
                    if global_step >= total_steps:
                        break

                    if use_real_data:
                        input_ids, targets = batch_item
                        input_ids = input_ids.to(device)
                        targets = targets.to(device)
                    else:
                        input_ids, targets = get_dummy_batch(
                            config.batch_size, config.n_seq, config.vocab_size, device
                        )

                    try:
                        loss, grad_norm = train_step(
                            model, optimizer, scaler,
                            input_ids, targets, config, batch_idx
                        )
                        running_loss += loss
                        step_count += 1
                        tokens_processed += config.batch_size * config.n_seq

                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            console.print(f"[red]⚠ OOM at step {global_step}[/red]")
                            clear_memory()
                            optimizer.zero_grad()
                            continue
                        raise

                    if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                        global_step += 1
                        scheduler.step()
                        progress.update(train_task, advance=1)

                        if global_step % config.log_interval == 0:
                            avg_loss = running_loss / step_count if step_count > 0 else 0
                            ppl = math.exp(min(avg_loss, 20))
                            lr = scheduler.get_last_lr()[0]
                            vram = torch.cuda.memory_allocated() / 1024**3
                            elapsed = time.time() - start_time
                            tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0

                            log_entry = {
                                'step': global_step, 'loss': avg_loss, 'ppl': ppl,
                                'lr': lr, 'grad_norm': grad_norm, 'vram_gb': vram,
                            }
                            training_log.append(log_entry)

                            # Update progress description
                            progress.update(
                                train_task,
                                description=f"[cyan]Loss: {avg_loss:.4f} | PPL: {ppl:.1f} | VRAM: {vram:.1f}GB"
                            )

                            running_loss = 0.0
                            step_count = 0

                        if global_step % config.save_interval == 0:
                            ckpt_path = save_dir / f"step_{global_step}.pt"
                            torch.save({
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'step': global_step,
                                'config': config.__dict__,
                            }, ckpt_path)
                            console.print(f"[green]💾 Saved: {ckpt_path}[/green]")

                            if avg_loss < best_loss and avg_loss > 0:
                                best_loss = avg_loss
                                best_path = save_dir / "best.pt"
                                torch.save({
                                    'model_state_dict': model.state_dict(),
                                    'step': global_step, 'loss': avg_loss,
                                }, best_path)
                                console.print(f"[yellow]🏆 New best: {best_path}[/yellow]")

    else:
        # Fallback without rich
        print("\n🚀 Training Started!\n")
        print(f"   Min epochs: {config.min_epochs}, Max steps: {total_steps:,}\n")
        while epoch < config.min_epochs or global_step < total_steps:
            epoch += 1
            if epoch > config.min_epochs and global_step >= total_steps:
                break
            if use_real_data:
                batch_iter = mixed_loader.iter_epoch(epoch)
            else:
                batch_iter = range(steps_per_epoch * config.gradient_accumulation_steps)

            for batch_idx, batch_item in enumerate(batch_iter):
                if epoch > config.min_epochs and global_step >= total_steps:
                    break

                if use_real_data:
                    input_ids, targets = batch_item
                    input_ids = input_ids.to(device)
                    targets = targets.to(device)
                else:
                    input_ids, targets = get_dummy_batch(
                        config.batch_size, config.n_seq, config.vocab_size, device
                    )

                loss, grad_norm = train_step(
                    model, optimizer, scaler,
                    input_ids, targets, config, batch_idx
                )
                running_loss += loss
                step_count += 1

                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    global_step += 1
                    scheduler.step()

                    if global_step % config.log_interval == 0:
                        avg_loss = running_loss / step_count if step_count > 0 else 0
                        ppl = math.exp(min(avg_loss, 20))
                        lr = scheduler.get_last_lr()[0]
                        vram = torch.cuda.memory_allocated() / 1024**3
                        print(f"Step {global_step:6d} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | VRAM: {vram:.2f}GB")
                        running_loss = 0.0
                        step_count = 0

                    if global_step % config.save_interval == 0:
                        ckpt_path = save_dir / f"step_{global_step}.pt"
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'step': global_step, 'config': config.__dict__,
                        }, ckpt_path)
                        print(f"💾 Saved: {ckpt_path}")

    # Final save
    final_path = save_dir / "final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': global_step, 'config': config.__dict__,
    }, final_path)

    log_path = save_dir / "training_log.json"
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)

    if console:
        console.print(Panel.fit(
            f"[bold green]✅ Training Complete![/bold green]\n\n"
            f"Final model: [cyan]{final_path}[/cyan]\n"
            f"Training log: [cyan]{log_path}[/cyan]\n"
            f"Total steps: [yellow]{global_step:,}[/yellow]",
            border_style="green"
        ))
    else:
        print("\n✅ Training Complete!")
        print(f"Final model: {final_path}")


if __name__ == "__main__":
    main()
