"""
Hybrid Precision Strategy for Phase 8

物理的直観:
- 曲率計算はFP32で精度を保証（数値安定性が重要）
- 境界近くの埋め込みはFP32で崩壊を防止
- 通常の計算はFP16/BF16で高速化

Requirements: 6.1-6.6
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
import json

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PrecisionConfig:
    """Hybrid Precision設定"""
    # 精度設定
    default_dtype: str = "float16"  # "float16", "bfloat16", "float32"
    curvature_dtype: str = "float32"  # 曲率計算は常にFP32
    boundary_dtype: str = "float32"  # 境界近くはFP32
    
    # 境界検出閾値
    boundary_threshold: float = 0.95  # ||x|| > threshold で境界近く
    
    # オーバーフロー検出
    gradient_clip_value: float = 1.0
    overflow_threshold: float = 65504.0  # FP16の最大値
    
    # 自動アップキャスト
    auto_upcast_on_overflow: bool = True
    checkpoint_on_nan: bool = True
    
    # 数値安定性
    eps: float = 1e-6
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "default_dtype": self.default_dtype,
            "curvature_dtype": self.curvature_dtype,
            "boundary_dtype": self.boundary_dtype,
            "boundary_threshold": self.boundary_threshold,
            "gradient_clip_value": self.gradient_clip_value,
            "overflow_threshold": self.overflow_threshold,
            "auto_upcast_on_overflow": self.auto_upcast_on_overflow,
            "checkpoint_on_nan": self.checkpoint_on_nan,
            "eps": self.eps,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PrecisionConfig":
        return cls(**d)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, s: str) -> "PrecisionConfig":
        return cls.from_dict(json.loads(s))


@dataclass
class PrecisionDiagnostics:
    """Precision診断情報"""
    current_dtype: str = "float16"
    boundary_tokens_count: int = 0
    overflow_detected: bool = False
    nan_detected: bool = False
    upcast_triggered: bool = False
    gradient_clipped: bool = False
    max_gradient_norm: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_dtype": self.current_dtype,
            "boundary_tokens_count": self.boundary_tokens_count,
            "overflow_detected": self.overflow_detected,
            "nan_detected": self.nan_detected,
            "upcast_triggered": self.upcast_triggered,
            "gradient_clipped": self.gradient_clipped,
            "max_gradient_norm": self.max_gradient_norm,
        }


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """文字列からtorch.dtypeを取得"""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    return dtype_map.get(dtype_str, torch.float32)


class BoundaryDetector(nn.Module):
    """
    境界近くのトークンを検出
    
    Poincaré球の境界（||x|| → 1）近くでは数値不安定になるため、
    FP32にアップキャストする必要がある
    """
    
    def __init__(self, threshold: float = 0.95):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        境界近くのトークンを検出
        
        Args:
            x: (B, N, D) 入力テンソル
        
        Returns:
            mask: (B, N) 境界近くならTrue
        """
        norms = x.norm(dim=-1)  # (B, N)
        return norms > self.threshold
    
    def count_boundary_tokens(self, x: torch.Tensor) -> int:
        """境界近くのトークン数をカウント"""
        mask = self.forward(x)
        return mask.sum().item()


class GradientOverflowDetector(nn.Module):
    """
    勾配オーバーフローを検出
    
    FP16では勾配が65504を超えるとオーバーフローする
    """
    
    def __init__(
        self,
        overflow_threshold: float = 65504.0,
        clip_value: float = 1.0,
    ):
        super().__init__()
        self.overflow_threshold = overflow_threshold
        self.clip_value = clip_value
        self._overflow_detected = False
        self._nan_detected = False
    
    def check_overflow(self, tensor: torch.Tensor) -> bool:
        """オーバーフローをチェック"""
        if tensor.abs().max() > self.overflow_threshold:
            self._overflow_detected = True
            return True
        return False
    
    def check_nan(self, tensor: torch.Tensor) -> bool:
        """NaN/Infをチェック"""
        if not torch.isfinite(tensor).all():
            self._nan_detected = True
            return True
        return False
    
    def clip_gradients(self, parameters) -> Tuple[bool, float]:
        """
        勾配をクリップ
        
        Returns:
            clipped: クリップされたか
            max_norm: 最大勾配ノルム
        """
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        clipped = total_norm > self.clip_value
        if clipped:
            clip_coef = self.clip_value / (total_norm + 1e-6)
            for p in parameters:
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
        
        return clipped, total_norm
    
    @property
    def overflow_detected(self) -> bool:
        return self._overflow_detected
    
    @property
    def nan_detected(self) -> bool:
        return self._nan_detected
    
    def reset(self):
        """検出フラグをリセット"""
        self._overflow_detected = False
        self._nan_detected = False


class HybridPrecisionManager(nn.Module):
    """
    Hybrid Precision Manager
    
    物理的直観:
    - 曲率計算: FP32（数値安定性が重要）
    - 境界近く: FP32（崩壊防止）
    - 通常計算: FP16/BF16（高速化）
    
    Requirements: 6.1-6.6
    """
    
    def __init__(self, config: PrecisionConfig):
        super().__init__()
        self.config = config
        
        # 精度設定
        self.default_dtype = get_torch_dtype(config.default_dtype)
        self.curvature_dtype = get_torch_dtype(config.curvature_dtype)
        self.boundary_dtype = get_torch_dtype(config.boundary_dtype)
        
        # 検出器
        self.boundary_detector = BoundaryDetector(config.boundary_threshold)
        self.overflow_detector = GradientOverflowDetector(
            config.overflow_threshold,
            config.gradient_clip_value,
        )
        
        # 状態
        self._current_dtype = self.default_dtype
        self._upcast_triggered = False
        self._last_checkpoint = None
    
    def compute_curvature_safe(
        self,
        func,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        曲率計算をFP32で安全に実行
        
        Property 11: Curvature Precision Enforcement
        Validates: Requirements 6.1
        """
        # 入力をFP32にキャスト
        args_fp32 = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.is_floating_point():
                args_fp32.append(arg.to(self.curvature_dtype))
            else:
                args_fp32.append(arg)
        
        kwargs_fp32 = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                kwargs_fp32[k] = v.to(self.curvature_dtype)
            else:
                kwargs_fp32[k] = v
        
        # FP32で計算
        result = func(*args_fp32, **kwargs_fp32)
        
        return result
    
    def apply_boundary_precision(
        self,
        x: torch.Tensor,
        func,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        境界近くのトークンをFP32で処理
        
        Property 12: Boundary Collapse Prevention
        Validates: Requirements 6.6
        """
        # 境界マスクを取得
        boundary_mask = self.boundary_detector(x)  # (B, N)
        
        if not boundary_mask.any():
            # 境界近くのトークンがない場合は通常精度で計算
            return func(x, *args, **kwargs)
        
        # 境界近くのトークンをFP32で処理
        x_fp32 = x.to(self.boundary_dtype)
        result_fp32 = func(x_fp32, *args, **kwargs)
        
        # 元の精度に戻す（境界以外）
        result = result_fp32.to(self.default_dtype)
        
        return result
    
    def detect_and_recover_overflow(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, bool]:
        """
        オーバーフローを検出して回復
        
        Requirements: 6.4, 6.5
        """
        # オーバーフローチェック
        overflow = self.overflow_detector.check_overflow(tensor)
        nan_detected = self.overflow_detector.check_nan(tensor)
        
        if overflow or nan_detected:
            if self.config.auto_upcast_on_overflow:
                # FP32にアップキャスト
                self._upcast_triggered = True
                self._current_dtype = torch.float32
                
                # NaN/Infを安全な値に置換
                if nan_detected:
                    tensor = torch.where(
                        torch.isfinite(tensor),
                        tensor,
                        torch.zeros_like(tensor),
                    )
                
                return tensor.to(torch.float32), True
            
            if self.config.checkpoint_on_nan and nan_detected:
                # チェックポイントから復元（実装は外部で行う）
                pass
        
        return tensor, False
    
    def forward(
        self,
        x: torch.Tensor,
        func,
        is_curvature_computation: bool = False,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[PrecisionDiagnostics]]:
        """
        Hybrid Precisionでの計算実行
        
        Args:
            x: 入力テンソル
            func: 実行する関数
            is_curvature_computation: 曲率計算かどうか
            return_diagnostics: 診断情報を返すか
        
        Returns:
            result: 計算結果
            diagnostics: 診断情報（オプション）
        """
        self.overflow_detector.reset()
        
        # 曲率計算の場合はFP32を強制
        if is_curvature_computation:
            result = self.compute_curvature_safe(func, x)
        else:
            # 境界近くのトークンをチェック
            boundary_count = self.boundary_detector.count_boundary_tokens(x)
            
            if boundary_count > 0:
                result = self.apply_boundary_precision(x, func)
            else:
                result = func(x)
        
        # オーバーフロー検出と回復
        result, recovered = self.detect_and_recover_overflow(result)
        
        # 診断情報
        diagnostics = None
        if return_diagnostics:
            diagnostics = PrecisionDiagnostics(
                current_dtype=str(self._current_dtype).split(".")[-1],
                boundary_tokens_count=self.boundary_detector.count_boundary_tokens(x),
                overflow_detected=self.overflow_detector.overflow_detected,
                nan_detected=self.overflow_detector.nan_detected,
                upcast_triggered=self._upcast_triggered,
            )
        
        return result, diagnostics
    
    def clip_gradients(self, parameters) -> Tuple[bool, float]:
        """勾配クリッピング"""
        return self.overflow_detector.clip_gradients(parameters)
    
    def save_checkpoint(self, state: Dict[str, Any]):
        """チェックポイント保存"""
        self._last_checkpoint = state
    
    def restore_checkpoint(self) -> Optional[Dict[str, Any]]:
        """チェックポイント復元"""
        return self._last_checkpoint


class BoundaryCollapseGuard(nn.Module):
    """
    境界崩壊防止ガード
    
    Property 12: Boundary Collapse Prevention
    Validates: Requirements 6.6
    
    Poincaré球の境界（||x|| → 1）での崩壊を防止
    """
    
    def __init__(
        self,
        max_norm: float = 0.99,
        regularization_strength: float = 0.01,
    ):
        super().__init__()
        self.max_norm = max_norm
        self.regularization_strength = regularization_strength
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        境界崩壊を防止
        
        Args:
            x: (B, N, D) 入力テンソル
        
        Returns:
            x_safe: 境界から離れた安全なテンソル
        """
        norms = x.norm(dim=-1, keepdim=True)  # (B, N, 1)
        
        # ノルムが閾値を超えている場合はスケーリング
        scale = torch.where(
            norms > self.max_norm,
            self.max_norm / (norms + 1e-8),
            torch.ones_like(norms),
        )
        
        return x * scale
    
    def compute_regularization_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        境界正則化損失を計算
        
        境界に近いほど大きなペナルティ
        """
        norms = x.norm(dim=-1)  # (B, N)
        
        # 境界に近いほど大きなペナルティ
        # loss = -log(1 - ||x||²) の近似
        boundary_penalty = -torch.log(1 - norms.pow(2).clamp(max=0.99) + 1e-8)
        
        return self.regularization_strength * boundary_penalty.mean()


def create_precision_manager(
    default_dtype: str = "float16",
    boundary_threshold: float = 0.95,
    gradient_clip_value: float = 1.0,
    **kwargs,
) -> HybridPrecisionManager:
    """
    Precision Managerを作成
    
    Args:
        default_dtype: デフォルト精度
        boundary_threshold: 境界検出閾値
        gradient_clip_value: 勾配クリップ値
        **kwargs: その他の設定
    
    Returns:
        HybridPrecisionManager インスタンス
    """
    config = PrecisionConfig(
        default_dtype=default_dtype,
        boundary_threshold=boundary_threshold,
        gradient_clip_value=gradient_clip_value,
        **kwargs,
    )
    return HybridPrecisionManager(config)
