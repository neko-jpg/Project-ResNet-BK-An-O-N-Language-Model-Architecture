"""
Entailment Cones Module - Phase 8 Implementation

双曲空間でのエンテイルメント（含意）関係をモデル化。
Poincaré球モデルでのコーン構造を使用して階層的な概念関係を表現。

物理的直観:
- 一般的な概念（例：「動物」）は原点に近い
- 具体的な概念（例：「犬」）は境界に近い
- エンテイルメント: A → B は B が A のコーン内にあることを意味

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import json
import math

EPS = 1e-6


@dataclass
class EntailmentConeConfig:
    """Entailment Cone設定"""
    d_model: int = 256
    initial_aperture: float = 0.5
    aperture_min: float = 0.1
    aperture_max: float = 2.0
    curvature: float = 1.0
    use_learnable_aperture: bool = True
    use_aperture_network: bool = False
    aperture_hidden_dim: int = 64
    
    def to_json(self) -> str:
        """JSON形式にシリアライズ"""
        return json.dumps(asdict(self), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'EntailmentConeConfig':
        """JSONからデシリアライズ"""
        data = json.loads(json_str)
        return cls(**data)
    
    def pretty_print(self) -> str:
        """人間が読みやすい形式で出力"""
        lines = [
            "EntailmentConeConfig:",
            f"  d_model: {self.d_model}",
            f"  initial_aperture: {self.initial_aperture}",
            f"  aperture_range: [{self.aperture_min}, {self.aperture_max}]",
            f"  curvature: {self.curvature}",
            f"  use_learnable_aperture: {self.use_learnable_aperture}",
            f"  use_aperture_network: {self.use_aperture_network}",
        ]
        return "\n".join(lines)


class ApertureNetwork(nn.Module):
    """
    学習可能なアパーチャネットワーク
    
    入力ベクトルに基づいて動的にアパーチャを計算。
    原点に近いベクトルほど大きなアパーチャ（より一般的な概念）。
    
    Requirements: 1.2
    """
    
    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 64,
        aperture_min: float = 0.1,
        aperture_max: float = 2.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.aperture_min = aperture_min
        self.aperture_max = aperture_max
        
        # アパーチャ予測ネットワーク
        self.network = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # [0, 1]に正規化
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        アパーチャを計算
        
        Args:
            x: 入力ベクトル [batch, d_model]
        
        Returns:
            aperture: アパーチャ角度 [batch]
        """
        # ネットワーク出力を[aperture_min, aperture_max]にスケール
        raw = self.network(x).squeeze(-1)
        aperture = self.aperture_min + raw * (self.aperture_max - self.aperture_min)
        return aperture


class EntailmentCones(nn.Module):
    """
    Entailment Cones for Hyperbolic Entailment
    
    双曲空間でのエンテイルメント関係をモデル化。
    A → B は B が A のコーン内にあることを意味。
    
    物理的直観:
    - コーンの頂点は概念の位置
    - コーンの開き角（アパーチャ）は概念の一般性を表す
    - 一般的な概念ほど大きなアパーチャを持つ
    
    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
    """
    
    def __init__(self, config: EntailmentConeConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        # 学習可能なアパーチャ
        if config.use_aperture_network:
            self.aperture_network = ApertureNetwork(
                d_model=config.d_model,
                hidden_dim=config.aperture_hidden_dim,
                aperture_min=config.aperture_min,
                aperture_max=config.aperture_max,
            )
            self.aperture_param = None
        elif config.use_learnable_aperture:
            self.aperture_param = nn.Parameter(
                torch.tensor([config.initial_aperture])
            )
            self.aperture_network = None
        else:
            self.register_buffer(
                'aperture_param',
                torch.tensor([config.initial_aperture])
            )
            self.aperture_network = None
        
        self.softplus = nn.Softplus()
    
    def compute_aperture(
        self,
        x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        アパーチャを計算
        
        Requirements: 1.1, 1.2
        
        Args:
            x: 入力ベクトル（アパーチャネットワーク使用時）
        
        Returns:
            aperture: アパーチャ角度
        """
        if self.aperture_network is not None and x is not None:
            return self.aperture_network(x)
        else:
            # 固定または学習可能なスカラーアパーチャ
            aperture = self.softplus(self.aperture_param)
            # 範囲制限
            aperture = aperture.clamp(
                min=self.config.aperture_min,
                max=self.config.aperture_max
            )
            return aperture
    
    def _mobius_add(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        c: float = 1.0,
    ) -> torch.Tensor:
        """
        Möbius加算（Poincaré球）
        
        x ⊕ y = ((1 + 2c<x,y> + c|y|²)x + (1 - c|x|²)y) / (1 + 2c<x,y> + c²|x|²|y|²)
        
        Requirements: 1.4 (OR演算の基礎)
        """
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        
        num = (1 + 2*c*xy + c*y2) * x + (1 - c*x2) * y
        denom = 1 + 2*c*xy + c**2 * x2 * y2
        
        return num / denom.clamp(min=EPS)
    
    def _poincare_dist(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        c: float = 1.0,
    ) -> torch.Tensor:
        """
        Poincaré距離
        
        d(u, v) = (2/√c) * arctanh(√c * |−u ⊕ v|)
        """
        sqrt_c = math.sqrt(c)
        v_minus_u = self._mobius_add(-u, v, c=c)
        norm = v_minus_u.norm(dim=-1, keepdim=True)
        
        # 数値安定性のためのクランプ
        arg = (sqrt_c * norm).clamp(max=1.0 - EPS)
        dist = (2 / sqrt_c) * torch.atanh(arg)
        
        return dist.squeeze(-1)
    
    def _exp_map(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        c: float = 1.0,
    ) -> torch.Tensor:
        """
        指数写像: 接空間から多様体へ
        
        exp_x(v) = x ⊕ (tanh(√c|v|/2 * (1-c|x|²)) * v / (√c|v|))
        """
        sqrt_c = math.sqrt(c)
        x_norm_sq = x.pow(2).sum(dim=-1, keepdim=True)
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=EPS)
        
        lambda_x = 2 / (1 - c * x_norm_sq).clamp(min=EPS)
        
        tanh_arg = (sqrt_c * lambda_x * v_norm / 2).clamp(max=15.0)
        direction = v / v_norm
        
        result = self._mobius_add(
            x,
            torch.tanh(tanh_arg) * direction / sqrt_c,
            c=c
        )
        
        return result
    
    def _log_map(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        c: float = 1.0,
    ) -> torch.Tensor:
        """
        対数写像: 多様体から接空間へ
        
        log_x(y) = (2/(√c * λ_x)) * arctanh(√c|−x ⊕ y|) * (−x ⊕ y) / |−x ⊕ y|
        """
        sqrt_c = math.sqrt(c)
        x_norm_sq = x.pow(2).sum(dim=-1, keepdim=True)
        lambda_x = 2 / (1 - c * x_norm_sq).clamp(min=EPS)
        
        minus_x_plus_y = self._mobius_add(-x, y, c=c)
        norm = minus_x_plus_y.norm(dim=-1, keepdim=True).clamp(min=EPS)
        
        arg = (sqrt_c * norm).clamp(max=1.0 - EPS)
        
        result = (2 / (sqrt_c * lambda_x)) * torch.atanh(arg) * minus_x_plus_y / norm
        
        return result
    
    def check_entailment(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        c: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        エンテイルメントをチェック: u → v
        
        Requirements: 1.1, 1.5
        
        Args:
            u: 前提ベクトル [batch, d_model]
            v: 仮説ベクトル [batch, d_model]
            c: 曲率（Noneの場合はconfig値を使用）
        
        Returns:
            score: エンテイルメントスコア [0, 1] (1 = 完全なエンテイルメント)
            penalty: 違反ペナルティ
            diagnostics: 診断情報
        """
        c = c or self.config.curvature
        
        # ノルムを計算（境界に近すぎないようにクランプ）
        u_norm = u.norm(dim=-1, keepdim=True).clamp(max=0.99)
        v_norm = v.norm(dim=-1, keepdim=True).clamp(max=0.99)
        
        # 角度を計算
        dot_prod = (u * v).sum(dim=-1, keepdim=True)
        cos_theta = dot_prod / (u_norm * v_norm).clamp(min=EPS)
        theta = torch.acos(cos_theta.clamp(-1.0 + EPS, 1.0 - EPS))
        
        # アパーチャを計算
        aperture = self.compute_aperture(u)
        if aperture.dim() == 0:
            aperture = aperture.unsqueeze(0).expand(u.shape[0])
        
        # 角度違反: theta > aperture
        angle_violation = F.relu(theta.squeeze(-1) - aperture)
        
        # 順序違反: u は v より原点に近いべき（一般→具体）
        order_violation = F.relu(u_norm.squeeze(-1) - v_norm.squeeze(-1))
        
        # 総ペナルティ
        penalty = angle_violation + order_violation
        
        # エンテイルメントスコア: ペナルティが0なら1、大きいほど0に近づく
        score = torch.exp(-penalty)
        
        diagnostics = {
            'theta': theta.squeeze(-1),
            'aperture': aperture,
            'u_norm': u_norm.squeeze(-1),
            'v_norm': v_norm.squeeze(-1),
            'angle_violation': angle_violation,
            'order_violation': order_violation,
        }
        
        return score, penalty, diagnostics
    
    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        c: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: エンテイルメントペナルティを計算
        
        Args:
            u: 前提ベクトル [batch, d_model]
            v: 仮説ベクトル [batch, d_model]
            c: 曲率
        
        Returns:
            penalty: エンテイルメント違反ペナルティ
            aperture: 使用されたアパーチャ
        """
        score, penalty, _ = self.check_entailment(u, v, c)
        aperture = self.compute_aperture(u)
        return penalty, aperture
    
    def logical_and(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        c: Optional[float] = None,
    ) -> torch.Tensor:
        """
        論理AND演算（接空間での交差）
        
        Requirements: 1.3
        
        物理的直観:
        - 2つの概念の共通部分を表す
        - 接空間での平均として近似
        
        Args:
            x: 第1ベクトル [batch, d_model]
            y: 第2ベクトル [batch, d_model]
            c: 曲率
        
        Returns:
            result: AND結果 [batch, d_model]
        """
        c = c or self.config.curvature
        
        # 接空間での交差を平均として近似
        # より厳密には、両方のコーンの交差を計算すべき
        
        # x を基準点として y を接空間に写像
        y_tangent = self._log_map(x, y, c)
        
        # 接空間での平均（原点方向に移動）
        avg_tangent = y_tangent / 2
        
        # 多様体に戻す
        result = self._exp_map(x, avg_tangent, c)
        
        return result
    
    def logical_or(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        c: Optional[float] = None,
    ) -> torch.Tensor:
        """
        論理OR演算（Möbius加算）
        
        Requirements: 1.4
        
        物理的直観:
        - 2つの概念の和集合を表す
        - Möbius加算として実装
        
        Args:
            x: 第1ベクトル [batch, d_model]
            y: 第2ベクトル [batch, d_model]
            c: 曲率
        
        Returns:
            result: OR結果 [batch, d_model]
        """
        c = c or self.config.curvature
        
        # Möbius加算でOR演算を近似
        # より一般的な概念（原点に近い方）を結果とする
        result = self._mobius_add(x, y, c)
        
        # 結果が境界を超えないようにクランプ
        result_norm = result.norm(dim=-1, keepdim=True)
        max_norm = 1.0 / math.sqrt(c) - EPS
        scale = torch.where(
            result_norm > max_norm,
            max_norm / result_norm,
            torch.ones_like(result_norm)
        )
        result = result * scale
        
        return result
    
    def compute_energy(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """物理エネルギーとしてのペナルティを返す"""
        penalty, _ = self.forward(u, v)
        return penalty
    
    def get_config(self) -> EntailmentConeConfig:
        """設定を取得"""
        return self.config
    
    def to_json(self) -> str:
        """
        設定をJSONにシリアライズ
        
        Requirements: 1.6
        """
        return self.config.to_json()
    
    @classmethod
    def from_json(cls, json_str: str) -> 'EntailmentCones':
        """
        JSONから復元
        
        Requirements: 1.7
        """
        config = EntailmentConeConfig.from_json(json_str)
        return cls(config)


def create_entailment_cones(
    d_model: int = 256,
    initial_aperture: float = 0.5,
    use_aperture_network: bool = False,
    **kwargs,
) -> EntailmentCones:
    """
    Entailment Conesのファクトリ関数
    
    Args:
        d_model: モデル次元
        initial_aperture: 初期アパーチャ
        use_aperture_network: アパーチャネットワークを使用するか
        **kwargs: その他の設定
    
    Returns:
        EntailmentCones instance
    """
    config = EntailmentConeConfig(
        d_model=d_model,
        initial_aperture=initial_aperture,
        use_aperture_network=use_aperture_network,
        **kwargs,
    )
    return EntailmentCones(config)
