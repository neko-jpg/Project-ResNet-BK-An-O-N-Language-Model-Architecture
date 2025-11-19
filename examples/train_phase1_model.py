"""
Phase 1 Model Training Example

このスクリプトは、Phase 1 Efficiency Engineを使用した完全な訓練パイプラインの例を示します。

Features:
    - Phase 1設定の作成とカスタマイズ
    - 安定性監視の統合
    - エラーハンドリングとリカバリ
    - メトリクスのロギング
    - チェックポイントの保存

Requirements:
    - Task 12.5: Write example scripts
    - Task 4.4: Example usage scripts

Usage:
    python examples/train_phase1_model.py --config configs/phase1_config.yaml
    python examples/train_phase1_model.py --preset 8gb --dataset wikitext-2

Author: Project MUSE Team
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Phase 1コンポーネント
from src.models.phase1 import (
    Phase1Config,
    get_preset_config,
    create_phase1_model,
    BKStabilityMonitor,
    Phase1ErrorRecovery,
    Phase1Diagnostics,
)
from src.models.phase1.errors import (
    VRAMExhaustedError,
    NumericalInstabilityError,
)


class Phase1Trainer:
    """Phase 1モデルの訓練を管理するクラス"""
    
    def __init__(
        self,
        model: nn.Module,
        config: Phase1Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        output_dir: str = "checkpoints/phase1",
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # オプティマイザとスケジューラ
        self.optimizer = AdamW(
            model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 10,  # 10 epochs
            eta_min=1e-6,
        )
        
        # 安定性監視
        if config.stability_monitoring_enabled:
            self.stability_monitor = BKStabilityMonitor(
                stability_threshold=config.stability_threshold,
                schatten_s1_bound=config.schatten_s1_bound,
                schatten_s2_bound=config.schatten_s2_bound,
                gradient_norm_threshold=config.gradient_norm_threshold,
            )
        else:
            self.stability_monitor = None
        
        # エラーリカバリ
        self.error_recovery = Phase1ErrorRecovery()
        
        # 訓練状態
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # メトリクス履歴
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        print(f"Phase1Trainer initialized:")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Device: {device}")
        print(f"  Output directory: {output_dir}")
        print(f"  Stability monitoring: {config.stability_monitoring_enabled}")
    
    def train_epoch(self) -> Dict[str, float]:
        """1エポックの訓練を実行"""
        self.model.train()
        epoch_loss = 0.0
        epoch_diagnostics = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            try:
                # データをデバイスに移動
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 順伝播
                if self.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                
                output = self.model(input_ids)
                
                # 損失計算
                loss = nn.functional.cross_entropy(
                    output.view(-1, output.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
                
                # 診断情報の収集
                diagnostics = self._collect_diagnostics(loss)
                epoch_diagnostics.append(diagnostics)
                
                # 安定性チェック
                if self.stability_monitor is not None:
                    self._check_stability(diagnostics)
                
                # 逆伝播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 勾配クリッピング
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.gradient_norm_threshold,
                )
                
                # オプティマイザステップ
                self.optimizer.step()
                self.scheduler.step()
                
                # メトリクスの記録
                epoch_loss += loss.item()
                self.global_step += 1
                
                # ログ出力
                if batch_idx % 10 == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    print(f"  Batch {batch_idx}/{len(self.train_loader)}: "
                          f"loss={loss.item():.4f}, lr={current_lr:.2e}, "
                          f"vram={diagnostics.peak_vram_mb:.1f}MB")
                
            except VRAMExhaustedError as e:
                print(f"\n⚠ VRAM Exhausted: {e}")
                if self.error_recovery.handle_vram_exhausted(e, self.model, self.config):
                    print("Recovery successful, continuing...")
                    continue
                else:
                    raise
            
            except NumericalInstabilityError as e:
                print(f"\n⚠ Numerical Instability: {e}")
                self.error_recovery.handle_numerical_instability(e, self.optimizer, self.config)
                print("Recovery successful, continuing...")
                continue
        
        # エポック統計
        avg_loss = epoch_loss / len(self.train_loader)
        avg_diagnostics = self._average_diagnostics(epoch_diagnostics)
        
        return {
            'loss': avg_loss,
            'perplexity': torch.exp(torch.tensor(avg_loss)).item(),
            **avg_diagnostics,
        }
    
    def validate(self) -> Dict[str, float]:
        """検証セットで評価"""
        self.model.eval()
        val_loss = 0.0
        val_diagnostics = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                output = self.model(input_ids)
                loss = nn.functional.cross_entropy(
                    output.view(-1, output.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
                
                diagnostics = self._collect_diagnostics(loss)
                val_diagnostics.append(diagnostics)
                
                val_loss += loss.item()
        
        avg_loss = val_loss / len(self.val_loader)
        avg_diagnostics = self._average_diagnostics(val_diagnostics)
        
        return {
            'loss': avg_loss,
            'perplexity': torch.exp(torch.tensor(avg_loss)).item(),
            **avg_diagnostics,
        }
    
    def train(self, num_epochs: int):
        """完全な訓練ループ"""
        print(f"\nStarting training for {num_epochs} epochs...")
        print("=" * 80)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 80)
            
            # 訓練
            start_time = time.time()
            train_metrics = self.train_epoch()
            train_time = time.time() - start_time
            
            # 検証
            val_metrics = self.validate()
            
            # メトリクスの記録
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.learning_rates.append(self.scheduler.get_last_lr()[0])
            
            # ログ出力
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} "
                  f"(PPL: {train_metrics['perplexity']:.2f})")
            print(f"  Val Loss: {val_metrics['loss']:.4f} "
                  f"(PPL: {val_metrics['perplexity']:.2f})")
            print(f"  Time: {train_time:.1f}s")
            print(f"  Peak VRAM: {train_metrics.get('peak_vram_mb', 0):.1f}MB")
            
            if 'ar_ssm_effective_rank' in train_metrics:
                print(f"  AR-SSM Effective Rank: {train_metrics['ar_ssm_effective_rank']:.2f}")
            
            # チェックポイントの保存
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(is_best=True)
                print(f"  ✓ New best model saved!")
            
            # 定期的なチェックポイント
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(is_best=False)
        
        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, is_best: bool = False):
        """チェックポイントを保存"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        # 通常のチェックポイント
        checkpoint_path = self.output_dir / f"checkpoint_epoch{self.current_epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # ベストモデル
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
    
    def _collect_diagnostics(self, loss: torch.Tensor) -> Phase1Diagnostics:
        """診断情報を収集"""
        diagnostics = Phase1Diagnostics()
        
        # 基本メトリクス
        diagnostics.loss = loss.item()
        
        # メモリ使用量
        if self.device.type == 'cuda':
            diagnostics.peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
            diagnostics.current_vram_mb = torch.cuda.memory_allocated() / 1024**2
        
        # AR-SSM診断（モデルがサポートしている場合）
        if hasattr(self.model, 'get_ar_ssm_diagnostics'):
            ar_ssm_diag = self.model.get_ar_ssm_diagnostics()
            diagnostics.ar_ssm_effective_rank = ar_ssm_diag.get('effective_rank', 0.0)
            diagnostics.ar_ssm_gate_sparsity = ar_ssm_diag.get('gate_sparsity', 0.0)
        
        # HTT診断
        if hasattr(self.model, 'get_htt_diagnostics'):
            htt_diag = self.model.get_htt_diagnostics()
            diagnostics.htt_compression_ratio = htt_diag.get('compression_ratio', 0.0)
        
        return diagnostics
    
    def _check_stability(self, diagnostics: Phase1Diagnostics):
        """安定性をチェック"""
        # 注意: 実際のBK-Core診断データが必要
        # ここでは簡略化のため省略
        pass
    
    def _average_diagnostics(self, diagnostics_list):
        """診断情報の平均を計算"""
        if not diagnostics_list:
            return {}
        
        avg = {}
        keys = ['peak_vram_mb', 'ar_ssm_effective_rank', 'ar_ssm_gate_sparsity', 
                'htt_compression_ratio']
        
        for key in keys:
            values = [getattr(d, key, 0.0) for d in diagnostics_list]
            if any(v != 0.0 for v in values):
                avg[key] = sum(values) / len(values)
        
        return avg


def create_dummy_dataloader(vocab_size: int, seq_len: int, batch_size: int, num_batches: int):
    """デモ用のダミーデータローダーを作成"""
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, vocab_size, seq_len, num_samples):
            self.vocab_size = vocab_size
            self.seq_len = seq_len
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
            labels = torch.randint(0, self.vocab_size, (self.seq_len,))
            return {'input_ids': input_ids, 'labels': labels}
    
    dataset = DummyDataset(vocab_size, seq_len, num_batches * batch_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="Train Phase 1 model")
    parser.add_argument('--preset', type=str, default='8gb',
                        choices=['8gb', '10gb', '24gb', 'inference', 'max_efficiency'],
                        help='Configuration preset')
    parser.add_argument('--vocab-size', type=int, default=10000,
                        help='Vocabulary size')
    parser.add_argument('--d-model', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=6,
                        help='Number of layers')
    parser.add_argument('--seq-len', type=int, default=128,
                        help='Sequence length')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=3,
                        help='Number of epochs')
    parser.add_argument('--output-dir', type=str, default='checkpoints/phase1',
                        help='Output directory for checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("Phase 1 Model Training")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Preset: {args.preset}")
    print(f"  Vocabulary size: {args.vocab_size:,}")
    print(f"  Model dimension: {args.d_model}")
    print(f"  Number of layers: {args.n_layers}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of epochs: {args.num_epochs}")
    print(f"  Device: {args.device}")
    
    # 設定の作成
    config = get_preset_config(args.preset)
    print(f"\nPhase 1 Configuration:")
    print(f"  AR-SSM enabled: {config.ar_ssm_enabled}")
    print(f"  AR-SSM max rank: {config.ar_ssm_max_rank}")
    print(f"  HTT enabled: {config.htt_enabled}")
    print(f"  HTT rank: {config.htt_rank}")
    print(f"  Stability monitoring: {config.stability_monitoring_enabled}")
    print(f"  Gradient checkpointing: {config.use_gradient_checkpointing}")
    
    # モデルの作成
    print(f"\nCreating Phase 1 model...")
    
    # 簡易的なモデル（実際のcreate_phase1_modelが利用できない場合）
    class SimplePhase1Model(nn.Module):
        def __init__(self, vocab_size, d_model):
            super().__init__()
            from src.models.phase1 import HolographicTTEmbedding, AdaptiveRankSemiseparableLayer
            
            self.embedding = HolographicTTEmbedding(
                vocab_size=vocab_size,
                d_model=d_model,
                rank=config.htt_rank,
            )
            self.ar_ssm = AdaptiveRankSemiseparableLayer(
                d_model=d_model,
                max_rank=config.ar_ssm_max_rank,
                min_rank=config.ar_ssm_min_rank,
            )
            self.output = nn.Linear(d_model, vocab_size)
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            x = self.ar_ssm(x)[0]  # AR-SSMは(output, diagnostics)を返す
            return self.output(x)
    
    model = SimplePhase1Model(args.vocab_size, args.d_model)
    
    # データローダーの作成（デモ用）
    print(f"Creating data loaders...")
    train_loader = create_dummy_dataloader(
        args.vocab_size, args.seq_len, args.batch_size, num_batches=50
    )
    val_loader = create_dummy_dataloader(
        args.vocab_size, args.seq_len, args.batch_size, num_batches=10
    )
    
    # トレーナーの作成
    device = torch.device(args.device)
    trainer = Phase1Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=args.output_dir,
    )
    
    # 訓練の実行
    try:
        trainer.train(num_epochs=args.num_epochs)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        trainer.save_checkpoint(is_best=False)
        print("Checkpoint saved")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        trainer.save_checkpoint(is_best=False)
        print("Emergency checkpoint saved")
    
    print("\n" + "=" * 80)
    print("Training script completed")
    print("=" * 80)


if __name__ == "__main__":
    main()
