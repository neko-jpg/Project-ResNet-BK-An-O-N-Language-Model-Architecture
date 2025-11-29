"""
Tangent-Space Linear Attention for Phase 8

物理的直観:
- 双曲空間での距離計算はO(N²)だが、接空間では線形近似が可能
- 低曲率(c < 0.1)では接空間近似が高精度
- 高曲率(c > 1.0)では正確な双曲計算が必要

Requirements: 5.1-5.6, 70.1-70.6
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import json

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LinearAttentionConfig:
    """Tangent-Space Linear Attention設定"""
    d_model: int = 256
    num_heads: int = 8
    curvature: float = 1.0
    # 自動モード切替閾値
    low_curvature_threshold: float = 0.1  # これ以下で線形モード
    high_curvature_threshold: float = 1.0  # これ以上で正確モード
    # カーネル近似
    num_features: int = 64  # Random Fourier Features数
    kernel_type: str = "elu"  # "elu", "relu", "softmax"
    # 数値安定性
    eps: float = 1e-6
    max_norm: float = 0.99
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "curvature": self.curvature,
            "low_curvature_threshold": self.low_curvature_threshold,
            "high_curvature_threshold": self.high_curvature_threshold,
            "num_features": self.num_features,
            "kernel_type": self.kernel_type,
            "eps": self.eps,
            "max_norm": self.max_norm,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LinearAttentionConfig":
        return cls(**d)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, s: str) -> "LinearAttentionConfig":
        return cls.from_dict(json.loads(s))


@dataclass
class LinearAttentionDiagnostics:
    """Linear Attention診断情報"""
    mode: str = "linear"  # "linear", "exact", "hybrid"
    effective_curvature: float = 1.0
    approximation_error: Optional[float] = None
    computation_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    correlation_with_exact: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "effective_curvature": self.effective_curvature,
            "approximation_error": self.approximation_error,
            "computation_time_ms": self.computation_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "correlation_with_exact": self.correlation_with_exact,
        }


def log_map_at_origin(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    原点での対数写像: 双曲空間 → 接空間
    
    Args:
        x: (B, N, D) Poincaré球内の点
        c: 曲率パラメータ
    
    Returns:
        v: (B, N, D) 接空間のベクトル
    """
    sqrt_c = torch.sqrt(c.clamp(min=1e-6))
    x_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-6, max=0.999)
    
    # arctanh(sqrt(c) * ||x||) / (sqrt(c) * ||x||) * x
    scale = torch.arctanh(sqrt_c * x_norm) / (sqrt_c * x_norm + 1e-8)
    return scale * x


def exp_map_at_origin(v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    原点での指数写像: 接空間 → 双曲空間
    
    Args:
        v: (B, N, D) 接空間のベクトル
        c: 曲率パラメータ
    
    Returns:
        x: (B, N, D) Poincaré球内の点
    """
    sqrt_c = torch.sqrt(c.clamp(min=1e-6))
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-6, max=15.0)
    
    # tanh(sqrt(c) * ||v||) / (sqrt(c) * ||v||) * v
    tanh_arg = (sqrt_c * v_norm).clamp(max=15.0)
    scale = torch.tanh(tanh_arg) / (sqrt_c * v_norm + 1e-8)
    return scale * v


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Poincaré距離の計算
    
    Args:
        x: (B, N, D) 点1
        y: (B, M, D) 点2
        c: 曲率
    
    Returns:
        dist: (B, N, M) 距離行列
    """
    sqrt_c = torch.sqrt(c.clamp(min=1e-6))
    
    # ||x - y||²
    diff = x.unsqueeze(2) - y.unsqueeze(1)  # (B, N, M, D)
    diff_sq = (diff ** 2).sum(dim=-1)  # (B, N, M)
    
    # (1 - c||x||²)(1 - c||y||²)
    x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)  # (B, N, 1)
    y_norm_sq = (y ** 2).sum(dim=-1).unsqueeze(1)  # (B, 1, M)
    
    denom = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)
    denom = denom.clamp(min=1e-6)
    
    # arcosh(1 + 2c * ||x-y||² / denom)
    arg = 1 + 2 * c * diff_sq / denom
    arg = arg.clamp(min=1.0 + 1e-6)
    
    dist = (1 / sqrt_c) * torch.acosh(arg)
    return dist


class KernelFeatureMap(nn.Module):
    """
    カーネル特徴写像（Linear Attention用）
    
    φ(x) を計算し、Attention(Q, K, V) ≈ φ(Q)(φ(K)^T V) を実現
    """
    
    def __init__(
        self,
        d_model: int,
        num_features: int = 64,
        kernel_type: str = "elu",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_features = num_features
        self.kernel_type = kernel_type
        
        if kernel_type == "random_fourier":
            # Random Fourier Features
            self.register_buffer(
                "omega",
                torch.randn(d_model, num_features) / math.sqrt(d_model)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        特徴写像を適用
        
        Args:
            x: (B, N, D) 入力
        
        Returns:
            phi_x: (B, N, F) 特徴ベクトル
        """
        if self.kernel_type == "elu":
            # ELU + 1 (非負保証)
            return F.elu(x) + 1
        elif self.kernel_type == "relu":
            return F.relu(x)
        elif self.kernel_type == "softmax":
            # Softmax kernel (Performer style)
            return F.softmax(x, dim=-1)
        elif self.kernel_type == "random_fourier":
            # Random Fourier Features
            proj = torch.matmul(x, self.omega)  # (B, N, F)
            return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) / math.sqrt(self.num_features)
        else:
            return F.elu(x) + 1


class TangentSpaceLinearAttention(nn.Module):
    """
    Tangent-Space Linear Attention
    
    物理的直観:
    - 双曲空間の点を接空間に写像してLinear Attentionを実行
    - O(N)複雑度を達成
    - 低曲率では高精度、高曲率では正確な双曲計算にフォールバック
    
    Requirements: 5.1-5.6, 70.1-70.6
    """
    
    def __init__(self, config: LinearAttentionConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_head = config.d_model // config.num_heads
        
        # 曲率パラメータ
        self.curvature = nn.Parameter(torch.tensor(config.curvature))
        
        # Q, K, V投影
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        
        # カーネル特徴写像
        self.feature_map = KernelFeatureMap(
            self.d_head,
            config.num_features,
            config.kernel_type,
        )
        
        # モード切替用
        self.low_threshold = config.low_curvature_threshold
        self.high_threshold = config.high_curvature_threshold
    
    def _get_mode(self, c: torch.Tensor) -> str:
        """曲率に基づいてモードを決定"""
        c_val = c.item() if c.numel() == 1 else c.mean().item()
        if c_val < self.low_threshold:
            return "linear"
        elif c_val > self.high_threshold:
            return "exact"
        else:
            return "hybrid"
    
    def _linear_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Linear Attention: O(N)複雑度
        
        Attention(Q, K, V) = φ(Q)(φ(K)^T V) / φ(Q)(φ(K)^T 1)
        """
        B, H, N, D = q.shape
        
        # 特徴写像適用
        phi_q = self.feature_map(q)  # (B, H, N, F)
        phi_k = self.feature_map(k)  # (B, H, N, F)
        
        # φ(K)^T V を計算: (B, H, F, D)
        kv = torch.einsum("bhnd,bhnf->bhfd", v, phi_k)
        
        # φ(Q)(φ(K)^T V): (B, H, N, D)
        qkv = torch.einsum("bhnf,bhfd->bhnd", phi_q, kv)
        
        # 正規化: φ(Q)(φ(K)^T 1)
        k_sum = phi_k.sum(dim=2)  # (B, H, F)
        normalizer = torch.einsum("bhnf,bhf->bhn", phi_q, k_sum)  # (B, H, N)
        normalizer = normalizer.clamp(min=self.config.eps)
        
        output = qkv / normalizer.unsqueeze(-1)
        return output
    
    def _exact_hyperbolic_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        c: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        正確な双曲アテンション: O(N²)複雑度
        """
        B, H, N, D = q.shape
        
        # Poincaré距離計算
        q_flat = q.reshape(B * H, N, D)
        k_flat = k.reshape(B * H, N, D)
        
        dist = poincare_distance(q_flat, k_flat, c)  # (B*H, N, N)
        dist = dist.reshape(B, H, N, N)
        
        # 距離をスコアに変換（負の距離）
        scores = -dist
        
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.einsum("bhij,bhjd->bhid", attn_weights, v)
        
        return output
    
    def _hybrid_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        c: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        ハイブリッドアテンション: 線形と正確の補間
        """
        # 両方計算
        linear_out = self._linear_attention(q, k, v, mask)
        
        # 曲率に基づく補間係数
        c_val = c.item() if c.numel() == 1 else c.mean().item()
        alpha = (c_val - self.low_threshold) / (self.high_threshold - self.low_threshold)
        alpha = max(0.0, min(1.0, alpha))
        
        if alpha > 0.5:
            # 正確な計算も実行
            exact_out = self._exact_hyperbolic_attention(q, k, v, c, mask)
            output = (1 - alpha) * linear_out + alpha * exact_out
        else:
            output = linear_out
        
        return output, alpha
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[LinearAttentionDiagnostics]]:
        """
        Forward pass
        
        Args:
            x: (B, N, D) 入力（Poincaré球内）
            mask: (B, N, N) アテンションマスク
            return_diagnostics: 診断情報を返すか
        
        Returns:
            output: (B, N, D) 出力
            diagnostics: 診断情報（オプション）
        """
        B, N, D = x.shape
        c = self.curvature.abs().clamp(min=1e-6)
        
        # 接空間に写像
        x_tan = log_map_at_origin(x, c)
        
        # Q, K, V投影
        q = self.q_proj(x_tan).reshape(B, N, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x_tan).reshape(B, N, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x_tan).reshape(B, N, self.num_heads, self.d_head).transpose(1, 2)
        
        # モード決定
        mode = self._get_mode(c)
        
        if mode == "linear":
            output = self._linear_attention(q, k, v, mask)
            alpha = 0.0
        elif mode == "exact":
            output = self._exact_hyperbolic_attention(q, k, v, c, mask)
            alpha = 1.0
        else:
            output, alpha = self._hybrid_attention(q, k, v, c, mask)
        
        # 形状を戻す
        output = output.transpose(1, 2).reshape(B, N, D)
        output = self.out_proj(output)
        
        # 双曲空間に戻す
        output = exp_map_at_origin(output, c)
        
        # 診断情報
        diagnostics = None
        if return_diagnostics:
            diagnostics = LinearAttentionDiagnostics(
                mode=mode,
                effective_curvature=c.item() if c.numel() == 1 else c.mean().item(),
            )
        
        return output, diagnostics
    
    def compute_correlation_with_exact(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> float:
        """
        線形近似と正確な計算の相関を計算
        
        Property 21: Linear Attention Correlation検証用
        """
        B, N, D = x.shape
        c = self.curvature.abs().clamp(min=1e-6)
        
        x_tan = log_map_at_origin(x, c)
        
        q = self.q_proj(x_tan).reshape(B, N, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x_tan).reshape(B, N, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x_tan).reshape(B, N, self.num_heads, self.d_head).transpose(1, 2)
        
        linear_out = self._linear_attention(q, k, v, mask)
        exact_out = self._exact_hyperbolic_attention(q, k, v, c, mask)
        
        # Pearson相関
        linear_flat = linear_out.flatten()
        exact_flat = exact_out.flatten()
        
        linear_centered = linear_flat - linear_flat.mean()
        exact_centered = exact_flat - exact_flat.mean()
        
        correlation = (linear_centered * exact_centered).sum() / (
            linear_centered.norm() * exact_centered.norm() + 1e-8
        )
        
        return correlation.item()


def create_linear_attention(
    d_model: int = 256,
    num_heads: int = 8,
    curvature: float = 1.0,
    **kwargs,
) -> TangentSpaceLinearAttention:
    """
    Linear Attentionモジュールを作成
    
    Args:
        d_model: モデル次元
        num_heads: ヘッド数
        curvature: 初期曲率
        **kwargs: その他の設定
    
    Returns:
        TangentSpaceLinearAttention インスタンス
    """
    config = LinearAttentionConfig(
        d_model=d_model,
        num_heads=num_heads,
        curvature=curvature,
        **kwargs,
    )
    return TangentSpaceLinearAttention(config)
