"""
SNRベースの記憶選択機構

このモジュールは、信号対雑音比（SNR）に基づいて重要な記憶を選択的に保持する機構を提供します。

生物学的動機:
- 脳は重要な記憶だけを長期保持する
- ノイズは自動的に忘却される
- 記憶の重要度は信号強度とノイズレベルの比で評価される

物理的背景:
- SNR = |Signal| / σ_noise
- 高SNR: 明確な信号 → 保持・強化
- 低SNR: ノイズ優勢 → 急速忘却

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import warnings


class SNRMemoryFilter(nn.Module):
    """
    SNRベースの記憶フィルタリング
    
    生物学的動機:
    - 脳は重要な記憶だけを長期保持する
    - ノイズは自動的に忘却される
    
    SNR定義:
    SNR_i = |W_i| / σ_noise
    
    Args:
        threshold: SNR閾値（デフォルト: 2.0）
        gamma_boost: 低SNR成分のΓ増加率（デフォルト: 2.0）
        eta_boost: 高SNR成分のη増加率（デフォルト: 1.5）
    
    Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6
    """
    
    def __init__(
        self,
        threshold: float = 2.0,
        gamma_boost: float = 2.0,
        eta_boost: float = 1.5,
    ):
        super().__init__()
        self.threshold = threshold
        self.gamma_boost = gamma_boost
        self.eta_boost = eta_boost
        
        # 統計追跡用バッファ (Requirement 9.6)
        self.register_buffer('snr_history', torch.zeros(1000))
        self.history_idx = 0
    
    def forward(
        self,
        weights: torch.Tensor,
        gamma: torch.Tensor,
        eta: float
    ) -> Tuple[torch.Tensor, float]:
        """
        SNRに基づいてΓとηを調整する
        
        Args:
            weights: (B, H, D, D) Fast Weights
            gamma: (B,) 現在の減衰率
            eta: 現在のHebbian学習率
        
        Returns:
            adjusted_gamma: (B,) 調整された減衰率
            adjusted_eta: 調整された学習率
        
        Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
        """
        # ノイズ標準偏差推定（全体の分布から） (Requirement 9.2)
        sigma_noise = torch.std(weights) + 1e-6
        
        # 各成分のSNR計算 (Requirement 9.1)
        # SNR = |W_i| / σ_noise
        snr = torch.abs(weights) / sigma_noise  # (B, H, D, D)
        
        # 平均SNR（バッチ・ヘッド・次元で平均）
        mean_snr = snr.mean(dim=[1, 2, 3])  # (B,)
        
        # SNRに基づく調整 (Requirements 9.4, 9.5)
        # 低SNR → Γ増加（急速忘却）
        # 高SNR → η増加（強化学習）
        gamma_adjust = torch.where(
            mean_snr < self.threshold,  # Requirement 9.3: threshold = 2.0
            gamma * self.gamma_boost,  # 低SNR: 忘却促進 (Requirement 9.4)
            gamma  # 高SNR: 現状維持
        )
        
        eta_adjust = eta * torch.where(
            mean_snr > self.threshold,
            torch.tensor(self.eta_boost, device=weights.device),  # Requirement 9.5
            torch.tensor(1.0, device=weights.device)
        ).mean()
        
        # 統計記録 (Requirement 9.6)
        if self.training:
            idx = self.history_idx % 1000
            self.snr_history[idx] = mean_snr.mean()
            self.history_idx += 1
        
        return gamma_adjust, eta_adjust
    
    def get_statistics(self) -> Dict[str, float]:
        """
        SNR統計を取得
        
        Returns:
            統計情報の辞書
        
        Requirement: 9.6
        """
        valid_history = self.snr_history[:min(self.history_idx, 1000)]
        
        if len(valid_history) == 0:
            return {
                'mean_snr': 0.0,
                'std_snr': 0.0,
                'min_snr': 0.0,
                'max_snr': 0.0,
            }
        
        return {
            'mean_snr': valid_history.mean().item(),
            'std_snr': valid_history.std().item(),
            'min_snr': valid_history.min().item(),
            'max_snr': valid_history.max().item(),
        }


class MemoryImportanceEstimator(nn.Module):
    """
    記憶の重要度を推定するモジュール
    
    重要度の定義:
    - SNRベース: 信号対雑音比が高いほど重要
    - エネルギーベース: ||W_i||² が大きいほど重要
    - 時間ベース: 最近更新された記憶ほど重要
    
    Args:
        snr_weight: SNRの重み（デフォルト: 0.5）
        energy_weight: エネルギーの重み（デフォルト: 0.3）
        recency_weight: 最近性の重み（デフォルト: 0.2）
    
    Requirement: 9.7
    """
    
    def __init__(
        self,
        snr_weight: float = 0.5,
        energy_weight: float = 0.3,
        recency_weight: float = 0.2,
    ):
        super().__init__()
        
        # 重みの正規化
        total = snr_weight + energy_weight + recency_weight
        self.snr_weight = snr_weight / total
        self.energy_weight = energy_weight / total
        self.recency_weight = recency_weight / total
        
        # 時間追跡用バッファ
        self.register_buffer('update_timestamps', torch.zeros(1))
        self.current_time = 0
    
    def forward(
        self,
        weights: torch.Tensor,
        snr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        記憶の重要度を計算
        
        Args:
            weights: (B, H, D, D) Fast Weights
            snr: (B, H, D, D) SNR値（オプション、Noneの場合は計算）
        
        Returns:
            importance: (B, H, D, D) 重要度スコア [0, 1]
        
        Requirement: 9.7
        """
        B, H, D, _ = weights.shape
        
        # 1. SNRベースの重要度
        if snr is None:
            sigma_noise = torch.std(weights) + 1e-6
            snr = torch.abs(weights) / sigma_noise
        
        # 正規化 [0, 1]
        snr_norm = torch.sigmoid(snr - 2.0)  # threshold=2.0を中心に
        
        # 2. エネルギーベースの重要度
        energy = torch.abs(weights) ** 2
        energy_norm = energy / (energy.max() + 1e-6)
        
        # 3. 最近性ベースの重要度
        # タイムスタンプの初期化
        if self.update_timestamps.shape[0] != B * H * D * D:
            self.update_timestamps = torch.zeros(
                B * H * D * D,
                device=weights.device,
                dtype=weights.dtype
            )
        
        # 更新検出（重みが変化した場所）
        if hasattr(self, '_prev_weights'):
            changed = torch.abs(weights - self._prev_weights) > 1e-6
            changed_flat = changed.view(-1)
            self.update_timestamps[changed_flat] = self.current_time
        
        self._prev_weights = weights.detach().clone()
        self.current_time += 1
        
        # 最近性スコア（指数減衰）
        time_since_update = self.current_time - self.update_timestamps
        time_since_update = time_since_update.view(B, H, D, D)
        recency = torch.exp(-0.1 * time_since_update)
        
        # 総合重要度
        importance = (
            self.snr_weight * snr_norm +
            self.energy_weight * energy_norm +
            self.recency_weight * recency
        )
        
        return importance
    
    def get_top_k_memories(
        self,
        weights: torch.Tensor,
        k: int,
        snr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        上位k個の重要な記憶を取得
        
        Args:
            weights: (B, H, D, D) Fast Weights
            k: 保持する記憶の数
            snr: (B, H, D, D) SNR値（オプション）
        
        Returns:
            top_weights: (B, H, k) 上位k個の重み値
            top_indices: (B, H, k) 上位k個のインデックス
        
        Requirement: 9.7
        """
        importance = self.forward(weights, snr)
        
        # 各ヘッドごとに上位k個を選択
        B, H, D, _ = weights.shape
        importance_flat = importance.view(B, H, -1)  # (B, H, D*D)
        weights_flat = weights.view(B, H, -1)
        
        # 上位k個のインデックスを取得
        top_k_values, top_k_indices = torch.topk(
            importance_flat, k=min(k, D * D), dim=-1
        )
        
        # 対応する重み値を取得
        top_weights = torch.gather(weights_flat, -1, top_k_indices)
        
        return top_weights, top_k_indices
