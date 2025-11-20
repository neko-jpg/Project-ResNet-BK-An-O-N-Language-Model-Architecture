"""
Phase 2 Model Training Script

このスクリプトは、Phase 2 "Breath of Life"モデルの完全な訓練パイプラインを提供します。

Features:
    - Phase 2統合モデルの学習
    - 診断情報のリアルタイムロギング（Γ、SNR、共鳴、安定性）
    - WandBによるリアルタイム可視化
    - チェックポイントの自動保存
    - エラーハンドリングとリカバリ

Requirements:
    - Task 12: 学習スクリプトの実装
    - Task 12.1: 学習ループの実装
    - Task 12.2: 診断情報ロギングとリアルタイム可視化の実装
    - Task 12.3: チェックポイント保存の実装

Usage:
    # 基本的な使用方法
    python scripts/train_phase2.py --preset small --num-epochs 10
    
    # WandBを有効化
    python scripts/train_phase2.py --preset base --use-wandb --wandb-project phase2-training
    
    # カスタム設定
    python scripts/train_phase2.py --d-model 512 --n-layers 6 --batch-size 4

Author: Project MUSE Team
Date: 2024
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Phase 2コンポーネント
from src.models.phase2 import (
    Phase2IntegratedModel,
    create_phase2_model,
)

# ロギング
from src.utils.wandb_logger import WandBLogger


class Phase2Trainer:
    """
    Phase 2モデルの訓練を管理するクラス
    
    このクラスは以下の機能を提供します:
    - 学習ループの実行
    - 診断情報の収集とロギング
    - WandBによるリアルタイム可視化
    - チェックポイントの保存
    - エラーハンドリング
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        output_dir: str = "checkpoints/phase2",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        gradient_clip_norm: float = 1.0,
        use_wandb: bool = False,
        wandb_project: str = "phase2-training",
        wandb_name: Optional[str] = None,
    ):
        """
        Args:
            model: Phase2IntegratedModel
            train_loader: 訓練データローダー
            val_loader: 検証データローダー
            device: 計算デバイス
            output_dir: チェックポイント保存ディレクトリ
            learning_rate: 学習率
            weight_decay: 重み減衰
            gradient_clip_norm: 勾配クリッピングのノルム
            use_wandb: WandBを使用するか
            wandb_project: WandBプロジェクト名
            wandb_name: WandB実験名
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gradient_clip_norm = gradient_clip_norm
        
        # オプティマイザとスケジューラ
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )
        
        total_steps = len(train_loader) * 100  # 最大100エポックを想定
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-6,
        )
        
        # WandBロガー
        self.wandb_logger = WandBLogger(
            project=wandb_project,
            name=wandb_name,
            config={
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'gradient_clip_norm': gradient_clip_norm,
            },
            enabled=use_wandb,
        )
        
        if use_wandb:
            self.wandb_logger.log_model(model)
        
        # 訓練状態
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # メトリクス履歴
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.gamma_history = []  # Γの履歴
        self.snr_history = []  # SNRの履歴
        self.resonance_history = []  # 共鳴情報の履歴
        self.stability_history = []  # 安定性メトリクスの履歴
        
        print(f"\n{'='*80}")
        print("Phase2Trainer initialized")
        print(f"{'='*80}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Device: {device}")
        print(f"Output directory: {output_dir}")
        print(f"WandB logging: {use_wandb}")
        if use_wandb:
            print(f"WandB project: {wandb_project}")
            print(f"WandB name: {wandb_name}")
        print(f"{'='*80}\n")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        1エポックの訓練を実行
        
        Returns:
            エポック統計の辞書
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_diagnostics = []
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            try:
                # データをデバイスに移動
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # メモリ統計のリセット
                if self.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                
                # 順伝播（診断情報付き）
                logits, diagnostics = self.model(
                    input_ids,
                    return_diagnostics=True
                )
                
                # 損失計算
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
                
                # 診断情報の収集
                batch_diagnostics = self._collect_diagnostics(loss, diagnostics)
                epoch_diagnostics.append(batch_diagnostics)
                
                # 逆伝播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 勾配クリッピング
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.gradient_clip_norm,
                )
                
                # オプティマイザステップ
                self.optimizer.step()
                self.scheduler.step()
                
                # メトリクスの記録
                epoch_loss += loss.item()
                self.global_step += 1
                
                # バッチごとのロギング
                if batch_idx % 10 == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    self._log_batch(batch_idx, loss.item(), current_lr, batch_diagnostics)
                
                # WandBへのリアルタイムロギング
                if batch_idx % 5 == 0:
                    self._log_to_wandb_realtime(batch_diagnostics)
                
            except Exception as e:
                print(f"\n⚠ Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # エポック統計
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(self.train_loader)
        avg_diagnostics = self._average_diagnostics(epoch_diagnostics)
        
        return {
            'loss': avg_loss,
            'perplexity': torch.exp(torch.tensor(avg_loss)).item(),
            'epoch_time': epoch_time,
            **avg_diagnostics,
        }
    
    def validate(self) -> Dict[str, float]:
        """
        検証セットで評価
        
        Returns:
            検証統計の辞書
        """
        self.model.eval()
        val_loss = 0.0
        val_diagnostics = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 順伝播（診断情報付き）
                logits, diagnostics = self.model(
                    input_ids,
                    return_diagnostics=True
                )
                
                # 損失計算
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
                
                # 診断情報の収集
                batch_diagnostics = self._collect_diagnostics(loss, diagnostics)
                val_diagnostics.append(batch_diagnostics)
                
                val_loss += loss.item()
        
        avg_loss = val_loss / len(self.val_loader)
        avg_diagnostics = self._average_diagnostics(val_diagnostics)
        
        return {
            'loss': avg_loss,
            'perplexity': torch.exp(torch.tensor(avg_loss)).item(),
            **avg_diagnostics,
        }
    
    def train(self, num_epochs: int):
        """
        完全な訓練ループ
        
        Args:
            num_epochs: エポック数
        """
        print(f"\n{'='*80}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*80}\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*80}")
            
            # 訓練
            train_metrics = self.train_epoch()
            
            # 検証
            val_metrics = self.validate()
            
            # メトリクスの記録
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.learning_rates.append(self.scheduler.get_last_lr()[0])
            
            # Γ、SNR、共鳴、安定性の履歴を記録
            if 'mean_gamma' in train_metrics:
                self.gamma_history.append(train_metrics['mean_gamma'])
            if 'mean_snr' in train_metrics:
                self.snr_history.append(train_metrics['mean_snr'])
            if 'num_resonant_modes' in train_metrics:
                self.resonance_history.append(train_metrics['num_resonant_modes'])
            if 'lyapunov_stable_ratio' in train_metrics:
                self.stability_history.append(train_metrics['lyapunov_stable_ratio'])
            
            # エポックサマリーの表示
            self._print_epoch_summary(epoch, num_epochs, train_metrics, val_metrics)
            
            # WandBへのエポックロギング
            self._log_to_wandb_epoch(epoch, train_metrics, val_metrics)
            
            # チェックポイントの保存
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(is_best=True)
                print(f"  ✓ New best model saved! (val_loss: {val_metrics['loss']:.4f})")
            
            # 定期的なチェックポイント
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(is_best=False)
                print(f"  ✓ Checkpoint saved (epoch {epoch + 1})")
        
        print(f"\n{'='*80}")
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*80}\n")
        
        # 訓練履歴の保存
        self._save_training_history()
        
        # WandBの終了
        self.wandb_logger.finish()
    
    def save_checkpoint(self, is_best: bool = False):
        """
        チェックポイントを保存
        
        Args:
            is_best: ベストモデルとして保存するか
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'gamma_history': self.gamma_history,
            'snr_history': self.snr_history,
            'resonance_history': self.resonance_history,
            'stability_history': self.stability_history,
        }
        
        # 通常のチェックポイント
        checkpoint_path = self.output_dir / f"checkpoint_epoch{self.current_epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # ベストモデル
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
    
    def _collect_diagnostics(
        self,
        loss: torch.Tensor,
        model_diagnostics: Dict
    ) -> Dict[str, Any]:
        """
        診断情報を収集
        
        Args:
            loss: 損失値
            model_diagnostics: モデルからの診断情報
        
        Returns:
            統合された診断情報
        """
        diagnostics = {
            'loss': loss.item(),
        }
        
        # メモリ使用量
        if self.device.type == 'cuda':
            diagnostics['peak_vram_mb'] = torch.cuda.max_memory_allocated() / 1024**2
            diagnostics['current_vram_mb'] = torch.cuda.memory_allocated() / 1024**2
        
        # Γ（忘却率）の統計
        if 'gamma_values' in model_diagnostics:
            gamma_values = model_diagnostics['gamma_values']
            if len(gamma_values) > 0:
                # 各層のΓを収集
                all_gamma = []
                for layer_gamma in gamma_values:
                    if isinstance(layer_gamma, torch.Tensor):
                        all_gamma.append(layer_gamma.detach().cpu())
                
                if all_gamma:
                    all_gamma = torch.cat([g.flatten() for g in all_gamma])
                    diagnostics['mean_gamma'] = all_gamma.mean().item()
                    diagnostics['std_gamma'] = all_gamma.std().item()
                    diagnostics['min_gamma'] = all_gamma.min().item()
                    diagnostics['max_gamma'] = all_gamma.max().item()
        
        # SNR統計
        if 'snr_stats' in model_diagnostics:
            snr_stats = model_diagnostics['snr_stats']
            if len(snr_stats) > 0:
                # 各層のSNR統計を平均
                mean_snrs = [s.get('mean_snr', 0.0) for s in snr_stats if s]
                if mean_snrs:
                    diagnostics['mean_snr'] = sum(mean_snrs) / len(mean_snrs)
                
                low_snr_ratios = [s.get('low_snr_ratio', 0.0) for s in snr_stats if s]
                if low_snr_ratios:
                    diagnostics['low_snr_ratio'] = sum(low_snr_ratios) / len(low_snr_ratios)
        
        # 共鳴情報
        if 'resonance_info' in model_diagnostics:
            resonance_info = model_diagnostics['resonance_info']
            if len(resonance_info) > 0:
                # 各層の共鳴情報を平均
                num_resonant = [r.get('num_resonant', 0.0) for r in resonance_info if r]
                if num_resonant:
                    diagnostics['num_resonant_modes'] = sum(num_resonant) / len(num_resonant)
                
                total_energy = [r.get('total_energy', 0.0) for r in resonance_info if r]
                if total_energy:
                    diagnostics['total_resonance_energy'] = sum(total_energy) / len(total_energy)
        
        # 安定性メトリクス
        if 'stability_metrics' in model_diagnostics:
            stability_metrics = model_diagnostics['stability_metrics']
            if len(stability_metrics) > 0:
                # 各層の安定性を集計
                is_stable = [s.get('is_stable', True) for s in stability_metrics if s]
                if is_stable:
                    diagnostics['lyapunov_stable_ratio'] = sum(is_stable) / len(is_stable)
                
                energies = [s.get('energy', 0.0) for s in stability_metrics if s]
                if energies:
                    diagnostics['mean_fast_weight_energy'] = sum(energies) / len(energies)
        
        return diagnostics
    
    def _average_diagnostics(self, diagnostics_list: List[Dict]) -> Dict[str, float]:
        """
        診断情報の平均を計算
        
        Args:
            diagnostics_list: 診断情報のリスト
        
        Returns:
            平均された診断情報
        """
        if not diagnostics_list:
            return {}
        
        avg = {}
        keys = [
            'peak_vram_mb', 'current_vram_mb',
            'mean_gamma', 'std_gamma', 'min_gamma', 'max_gamma',
            'mean_snr', 'low_snr_ratio',
            'num_resonant_modes', 'total_resonance_energy',
            'lyapunov_stable_ratio', 'mean_fast_weight_energy',
        ]
        
        for key in keys:
            values = [d.get(key, 0.0) for d in diagnostics_list]
            if any(v != 0.0 for v in values):
                avg[key] = sum(values) / len(values)
        
        return avg
    
    def _log_batch(
        self,
        batch_idx: int,
        loss: float,
        lr: float,
        diagnostics: Dict
    ):
        """
        バッチごとのログ出力
        
        Args:
            batch_idx: バッチインデックス
            loss: 損失値
            lr: 学習率
            diagnostics: 診断情報
        """
        print(f"  Batch {batch_idx}/{len(self.train_loader)}: "
              f"loss={loss:.4f}, lr={lr:.2e}, "
              f"vram={diagnostics.get('peak_vram_mb', 0):.1f}MB", end="")
        
        if 'mean_gamma' in diagnostics:
            print(f", Γ={diagnostics['mean_gamma']:.4f}", end="")
        
        if 'mean_snr' in diagnostics:
            print(f", SNR={diagnostics['mean_snr']:.2f}", end="")
        
        print()  # 改行
    
    def _print_epoch_summary(
        self,
        epoch: int,
        num_epochs: int,
        train_metrics: Dict,
        val_metrics: Dict
    ):
        """
        エポックサマリーを表示
        
        Args:
            epoch: 現在のエポック
            num_epochs: 総エポック数
            train_metrics: 訓練メトリクス
            val_metrics: 検証メトリクス
        """
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{num_epochs} Summary")
        print(f"{'='*80}")
        print(f"Train Loss: {train_metrics['loss']:.4f} (PPL: {train_metrics['perplexity']:.2f})")
        print(f"Val Loss:   {val_metrics['loss']:.4f} (PPL: {val_metrics['perplexity']:.2f})")
        print(f"Time:       {train_metrics.get('epoch_time', 0):.1f}s")
        print(f"Peak VRAM:  {train_metrics.get('peak_vram_mb', 0):.1f}MB")
        
        # Γ（忘却率）の統計
        if 'mean_gamma' in train_metrics:
            print(f"\nΓ (Forgetting Rate) Statistics:")
            print(f"  Mean: {train_metrics['mean_gamma']:.4f}")
            print(f"  Std:  {train_metrics.get('std_gamma', 0):.4f}")
            print(f"  Min:  {train_metrics.get('min_gamma', 0):.4f}")
            print(f"  Max:  {train_metrics.get('max_gamma', 0):.4f}")
        
        # SNR統計
        if 'mean_snr' in train_metrics:
            print(f"\nSNR (Signal-to-Noise Ratio) Statistics:")
            print(f"  Mean SNR:       {train_metrics['mean_snr']:.2f}")
            print(f"  Low SNR Ratio:  {train_metrics.get('low_snr_ratio', 0)*100:.1f}%")
        
        # 共鳴情報
        if 'num_resonant_modes' in train_metrics:
            print(f"\nMemory Resonance:")
            print(f"  Resonant Modes: {train_metrics['num_resonant_modes']:.1f}")
            print(f"  Total Energy:   {train_metrics.get('total_resonance_energy', 0):.4f}")
        
        # 安定性メトリクス
        if 'lyapunov_stable_ratio' in train_metrics:
            print(f"\nLyapunov Stability:")
            print(f"  Stable Ratio:   {train_metrics['lyapunov_stable_ratio']*100:.1f}%")
            print(f"  Mean Energy:    {train_metrics.get('mean_fast_weight_energy', 0):.4f}")
        
        print(f"{'='*80}\n")
    
    def _log_to_wandb_realtime(self, diagnostics: Dict):
        """
        WandBへのリアルタイムロギング（バッチごと）
        
        Args:
            diagnostics: 診断情報
        """
        metrics = {
            'batch/loss': diagnostics.get('loss', 0.0),
            'batch/vram_mb': diagnostics.get('peak_vram_mb', 0.0),
        }
        
        # Γの時間変化をリアルタイムに可視化
        if 'mean_gamma' in diagnostics:
            metrics['batch/gamma_mean'] = diagnostics['mean_gamma']
            metrics['batch/gamma_std'] = diagnostics.get('std_gamma', 0.0)
            metrics['batch/gamma_min'] = diagnostics.get('min_gamma', 0.0)
            metrics['batch/gamma_max'] = diagnostics.get('max_gamma', 0.0)
        
        # SNRのリアルタイム可視化
        if 'mean_snr' in diagnostics:
            metrics['batch/snr_mean'] = diagnostics['mean_snr']
            metrics['batch/snr_low_ratio'] = diagnostics.get('low_snr_ratio', 0.0)
        
        # 共鳴のリアルタイム可視化
        if 'num_resonant_modes' in diagnostics:
            metrics['batch/resonant_modes'] = diagnostics['num_resonant_modes']
            metrics['batch/resonance_energy'] = diagnostics.get('total_resonance_energy', 0.0)
        
        # 安定性のリアルタイム可視化
        if 'lyapunov_stable_ratio' in diagnostics:
            metrics['batch/stability_ratio'] = diagnostics['lyapunov_stable_ratio']
            metrics['batch/fast_weight_energy'] = diagnostics.get('mean_fast_weight_energy', 0.0)
        
        self.wandb_logger.log(metrics, step=self.global_step)
    
    def _log_to_wandb_epoch(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict
    ):
        """
        WandBへのエポックロギング
        
        Args:
            epoch: エポック番号
            train_metrics: 訓練メトリクス
            val_metrics: 検証メトリクス
        """
        metrics = {
            'epoch': epoch + 1,
            'train/loss': train_metrics['loss'],
            'train/perplexity': train_metrics['perplexity'],
            'val/loss': val_metrics['loss'],
            'val/perplexity': val_metrics['perplexity'],
            'train/epoch_time': train_metrics.get('epoch_time', 0.0),
            'train/peak_vram_mb': train_metrics.get('peak_vram_mb', 0.0),
            'learning_rate': self.scheduler.get_last_lr()[0],
        }
        
        # Γの統計
        if 'mean_gamma' in train_metrics:
            metrics['train/gamma_mean'] = train_metrics['mean_gamma']
            metrics['train/gamma_std'] = train_metrics.get('std_gamma', 0.0)
        
        # SNR統計
        if 'mean_snr' in train_metrics:
            metrics['train/snr_mean'] = train_metrics['mean_snr']
            metrics['train/snr_low_ratio'] = train_metrics.get('low_snr_ratio', 0.0)
        
        # 共鳴統計
        if 'num_resonant_modes' in train_metrics:
            metrics['train/resonant_modes'] = train_metrics['num_resonant_modes']
            metrics['train/resonance_energy'] = train_metrics.get('total_resonance_energy', 0.0)
        
        # 安定性統計
        if 'lyapunov_stable_ratio' in train_metrics:
            metrics['train/stability_ratio'] = train_metrics['lyapunov_stable_ratio']
            metrics['train/fast_weight_energy'] = train_metrics.get('mean_fast_weight_energy', 0.0)
        
        self.wandb_logger.log(metrics, step=epoch)
    
    def _save_training_history(self):
        """訓練履歴をJSONファイルに保存"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'gamma_history': self.gamma_history,
            'snr_history': self.snr_history,
            'resonance_history': self.resonance_history,
            'stability_history': self.stability_history,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch + 1,
        }
        
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training history saved to {history_path}")



def create_dummy_dataloader(
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    num_batches: int
) -> DataLoader:
    """
    デモ用のダミーデータローダーを作成
    
    Args:
        vocab_size: 語彙サイズ
        seq_len: シーケンス長
        batch_size: バッチサイズ
        num_batches: バッチ数
    
    Returns:
        DataLoader
    """
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, vocab_size, seq_len, num_samples):
            self.vocab_size = vocab_size
            self.seq_len = seq_len
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
            # ラベルは入力を1つシフト
            labels = torch.cat([input_ids[1:], torch.tensor([0])])
            return {'input_ids': input_ids, 'labels': labels}
    
    dataset = DummyDataset(vocab_size, seq_len, num_batches * batch_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(
        description="Train Phase 2 'Breath of Life' model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # モデル設定
    parser.add_argument('--preset', type=str, default=None,
                        choices=['small', 'base', 'large'],
                        help='Model preset configuration')
    parser.add_argument('--vocab-size', type=int, default=10000,
                        help='Vocabulary size')
    parser.add_argument('--d-model', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=6,
                        help='Number of layers')
    parser.add_argument('--n-seq', type=int, default=512,
                        help='Sequence length')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--head-dim', type=int, default=64,
                        help='Head dimension')
    
    # 訓練設定
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--gradient-clip-norm', type=float, default=1.0,
                        help='Gradient clipping norm')
    
    # データ設定
    parser.add_argument('--num-train-batches', type=int, default=100,
                        help='Number of training batches (for dummy data)')
    parser.add_argument('--num-val-batches', type=int, default=20,
                        help='Number of validation batches (for dummy data)')
    
    # 出力設定
    parser.add_argument('--output-dir', type=str, default='checkpoints/phase2',
                        help='Output directory for checkpoints')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    # WandB設定
    parser.add_argument('--use-wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='phase2-training',
                        help='WandB project name')
    parser.add_argument('--wandb-name', type=str, default=None,
                        help='WandB experiment name')
    
    # Phase 2固有の設定
    parser.add_argument('--use-triton', action='store_true', default=True,
                        help='Use Triton kernels for BK-Core')
    parser.add_argument('--base-decay', type=float, default=0.01,
                        help='Base decay rate for Non-Hermitian potential')
    parser.add_argument('--hebbian-eta', type=float, default=0.1,
                        help='Hebbian learning rate')
    parser.add_argument('--snr-threshold', type=float, default=2.0,
                        help='SNR threshold for memory filtering')
    parser.add_argument('--resonance-threshold', type=float, default=0.1,
                        help='Energy threshold for memory resonance')
    
    args = parser.parse_args()
    
    # ヘッダー表示
    print("\n" + "="*80)
    print("Phase 2 'Breath of Life' Model Training")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Preset: {args.preset if args.preset else 'Custom'}")
    print(f"  Vocabulary size: {args.vocab_size:,}")
    print(f"  Model dimension: {args.d_model}")
    print(f"  Number of layers: {args.n_layers}")
    print(f"  Sequence length: {args.n_seq}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of epochs: {args.num_epochs}")
    print(f"  Device: {args.device}")
    print(f"  Use Triton: {args.use_triton}")
    print(f"  WandB logging: {args.use_wandb}")
    
    # モデルの作成
    print(f"\nCreating Phase 2 model...")
    
    if args.preset:
        # プリセット設定を使用
        model = create_phase2_model(
            preset=args.preset,
            vocab_size=args.vocab_size,
        )
        print(f"  Using preset: {args.preset}")
    else:
        # カスタム設定を使用
        model = Phase2IntegratedModel(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_seq=args.n_seq,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            use_triton=args.use_triton,
        )
        print(f"  Using custom configuration")
    
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # データローダーの作成
    print(f"\nCreating data loaders...")
    train_loader = create_dummy_dataloader(
        args.vocab_size, args.n_seq, args.batch_size, args.num_train_batches
    )
    val_loader = create_dummy_dataloader(
        args.vocab_size, args.n_seq, args.batch_size, args.num_val_batches
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # トレーナーの作成
    device = torch.device(args.device)
    trainer = Phase2Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
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
    
    print("\n" + "="*80)
    print("Training script completed")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
