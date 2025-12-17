#!/usr/bin/env python3
"""
Gradient Aligner - 勾配方向整合器 (v2)

異なる方向を向く勾配ベクトルを「Loss減少方向」に揃え、
勾配エネルギーを効率的に学習進行に向ける。

Key Features:
- optimizer state (exp_avg) を参照方向として再利用 → VRAM +0
- ソフト射影: min_alignment + strength で調整可能
- warmup期間は観測のみ

挿入位置: sanitizer直後、clip前
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple


@dataclass
class GradientAlignerConfig:
    """Gradient Aligner の設定"""
    
    enabled: bool = True
    
    # 参照方向のソース
    # "optimizer_exp_avg": AdamW/BK-HyperSGD の state['exp_avg'] を使用 (推奨)
    # "none": 統計のみ取得（比較実験用）
    ref_source: str = "optimizer_exp_avg"
    
    # Warmup: この期間は観測のみ（勾配は変更しない）
    warmup_steps: int = 100
    
    # ソフト射影パラメータ
    min_alignment: float = 0.0      # cos類似度の下限 (0.0 = 逆向きのみ除去)
    strength: float = 0.3           # 補正強度 (0.0-1.0, 0.1から開始推奨)
    
    # スキップ条件
    ref_norm_min: float = 1e-8      # 参照ベクトルがこれより小さいとスキップ
    grad_norm_min: float = 0.0      # 勾配がこれより小さいとスキップ
    
    # 除外パターン（bias/LayerNorm は外すのが無難）
    include_bias: bool = False
    include_norm: bool = False
    
    # ログ設定
    log_tensor_stats: bool = False  # 重い: 各テンソルの統計をログ


class GradientAligner:
    """
    Gradient Aligner - 勾配を参照方向に整合させる
    
    optimizer の exp_avg を参照方向として使用し、
    逆向き成分を除去（または低減）することで、
    全勾配がLoss減少方向に寄与するようにする。
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[GradientAlignerConfig] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config or GradientAlignerConfig()
        
        # パラメータ名のマッピング（optimizer param_groups との対応）
        self._param_to_name: Dict[int, str] = {}
        for name, param in model.named_parameters():
            self._param_to_name[id(param)] = name
        
        # 統計追跡
        self._step_count = 0
        self._cumulative_stats = {
            'aligned_tensors': 0,
            'total_tensors': 0,
            'total_time_ms': 0.0,
        }
    
    def _should_align_param(self, name: str) -> bool:
        """このパラメータを整合対象にするか"""
        if not self.config.include_bias and '.bias' in name:
            return False
        if not self.config.include_norm and ('norm' in name.lower() or 'ln' in name.lower()):
            return False
        return True
    
    def _get_reference_for_param(self, param: nn.Parameter) -> Optional[torch.Tensor]:
        """パラメータの参照方向を取得"""
        if self.config.ref_source == "none":
            return None
        
        if self.config.ref_source == "optimizer_exp_avg":
            # optimizer state から exp_avg を取得
            state = self.optimizer.state.get(param)
            if state is None:
                return None
            
            # AdamW系: 'exp_avg' / BK-HyperSGD: 'momentum_buffer' or 'exp_avg'
            for key in ['exp_avg', 'momentum_buffer']:
                if key in state:
                    return state[key]
            return None
        
        return None
    
    def _soft_project(
        self,
        grad: torch.Tensor,
        ref: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, bool]:
        """
        ソフト射影: 勾配を参照方向に整合させる
        
        Returns:
            (aligned_grad, cos_sim, was_aligned)
        """
        # フラット化
        g = grad.flatten().float()
        r = ref.flatten().float()
        
        # ノルム計算
        g_norm = g.norm()
        r_norm = r.norm()
        
        # スキップ条件
        if r_norm < self.config.ref_norm_min:
            return grad, 0.0, False
        if g_norm < self.config.grad_norm_min:
            return grad, 0.0, False
        
        # 内積とcos類似度
        dot = torch.dot(g, r)
        cos_sim = (dot / (g_norm * r_norm + 1e-12)).item()
        
        # 目標内積値
        target_dot = self.config.min_alignment * g_norm * r_norm
        
        # 補正が必要かどうか
        if dot >= target_dot:
            # 既に十分整合している
            return grad, cos_sim, False
        
        # ソフト射影: 逆向き成分を除去
        # g' = g - strength * (dot - target_dot) / ||r||^2 * r
        correction = self.config.strength * (dot - target_dot) / (r_norm ** 2 + 1e-12)
        g_aligned = g - correction * r
        
        # 元の形状に戻す
        aligned_grad = g_aligned.view_as(grad).to(grad.dtype)
        
        return aligned_grad, cos_sim, True
    
    def maybe_align(self, step: int) -> Dict[str, float]:
        """
        勾配の整合を実行（メインエントリポイント）
        
        Args:
            step: 現在のoptimizer step
            
        Returns:
            統計情報の辞書
        """
        start_time = time.perf_counter()
        self._step_count = step
        
        stats = {
            'ga_enabled': 1 if self.config.enabled else 0,
            'ga_warmup': 1 if step < self.config.warmup_steps else 0,
            'ga_total_tensors': 0,
            'ga_aligned_tensors': 0,
            'ga_neg_tensors': 0,
            'ga_cos_sum': 0.0,
            'ga_cos_min': 1.0,
            'ga_energy_removed': 0.0,
        }
        
        if not self.config.enabled:
            stats['ga_time_ms'] = 0.0
            return stats
        
        is_warmup = step < self.config.warmup_steps
        
        for param in self.model.parameters():
            if param.grad is None:
                continue
            
            param_id = id(param)
            name = self._param_to_name.get(param_id, 'unknown')
            
            # 除外チェック
            if not self._should_align_param(name):
                continue
            
            stats['ga_total_tensors'] += 1
            
            # 参照方向を取得
            ref = self._get_reference_for_param(param)
            if ref is None:
                continue
            
            # 整合処理
            aligned_grad, cos_sim, was_aligned = self._soft_project(param.grad, ref)
            
            # 統計更新
            stats['ga_cos_sum'] += cos_sim
            stats['ga_cos_min'] = min(stats['ga_cos_min'], cos_sim)
            
            if cos_sim < 0:
                stats['ga_neg_tensors'] += 1
            
            if was_aligned:
                stats['ga_aligned_tensors'] += 1
                
                # エネルギー除去量（補正前後のノルム差）
                energy_before = param.grad.norm().item()
                energy_after = aligned_grad.norm().item()
                if energy_before > 0:
                    stats['ga_energy_removed'] += (energy_before - energy_after) / energy_before
            
            # Warmup中は観測のみ（勾配は変更しない）
            if not is_warmup and was_aligned:
                param.grad.data.copy_(aligned_grad)
        
        # 統計の正規化
        n = max(stats['ga_total_tensors'], 1)
        stats['ga_mean_cos'] = stats['ga_cos_sum'] / n
        stats['ga_neg_frac'] = stats['ga_neg_tensors'] / n
        stats['ga_aligned_frac'] = stats['ga_aligned_tensors'] / n
        stats['ga_energy_removed'] /= max(stats['ga_aligned_tensors'], 1)
        
        # 不要なキーを削除
        del stats['ga_cos_sum']
        
        # 時間計測
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        stats['ga_time_ms'] = elapsed_ms
        
        # 累積統計更新
        self._cumulative_stats['aligned_tensors'] += stats['ga_aligned_tensors']
        self._cumulative_stats['total_tensors'] += stats['ga_total_tensors']
        self._cumulative_stats['total_time_ms'] += elapsed_ms
        
        return stats
    
    def get_cumulative_stats(self) -> Dict[str, float]:
        """累積統計を取得"""
        return dict(self._cumulative_stats)


def create_gradient_aligner(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    enabled: bool = True,
    strength: float = 0.3,
    min_alignment: float = 0.0,
    warmup_steps: int = 100,
) -> GradientAligner:
    """
    GradientAligner のファクトリ関数
    
    Args:
        model: 対象モデル
        optimizer: オプティマイザ（state['exp_avg']を参照方向として使用）
        enabled: 有効/無効
        strength: 補正強度 (0.0-1.0)
        min_alignment: cos類似度の下限
        warmup_steps: 観測のみ期間
        
    Returns:
        GradientAligner インスタンス
    """
    config = GradientAlignerConfig(
        enabled=enabled,
        strength=strength,
        min_alignment=min_alignment,
        warmup_steps=warmup_steps,
    )
    return GradientAligner(model, optimizer, config)


# =============================================================================
# Self-test
# =============================================================================
if __name__ == "__main__":
    print("=== Gradient Aligner Self-Test ===\n")
    
    # Simple model
    model = nn.Linear(10, 5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    
    # Create aligner
    aligner = create_gradient_aligner(model, optimizer, warmup_steps=2)
    
    # Simulate training steps
    for step in range(5):
        # Forward/backward
        x = torch.randn(2, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        # Optimizer step to create exp_avg
        optimizer.step()
        optimizer.zero_grad()
        
        # Another backward for alignment test
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        # Align
        stats = aligner.maybe_align(step)
        
        print(f"Step {step}:")
        print(f"  warmup={stats['ga_warmup']}, total={stats['ga_total_tensors']}")
        print(f"  mean_cos={stats['ga_mean_cos']:.3f}, neg_frac={stats['ga_neg_frac']:.3f}")
        print(f"  aligned={stats['ga_aligned_tensors']}, time={stats['ga_time_ms']:.2f}ms")
        
        optimizer.zero_grad()
    
    print("\n✅ Self-test passed!")
