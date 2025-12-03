"""
AR-SSM Hyperbolic Fusion - Phase 8 Implementation

AR-SSM（自己回帰状態空間モデル）と双曲空間の融合。
双曲距離を複雑性シグナルとして使用し、動的にランクを調整。

物理的直観:
- 双曲距離が大きい = 情報が複雑 = 高ランクが必要
- 原点に近い = 単純な情報 = 低ランクで十分
- BK-CoreのG_iiを物理ベースのゲーティングに使用

Requirements: 21.1, 21.2, 21.3, 21.4, 21.5, 21.6
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import math

EPS = 1e-6


@dataclass
class ARSSMFusionConfig:
    """AR-SSM Fusion設定"""
    d_model: int = 256
    d_state: int = 64
    max_rank: int = 32
    min_rank: int = 4
    curvature: float = 1.0
    distance_threshold: float = 1.0
    curvature_adjustment_rate: float = 0.1
    use_physics_gating: bool = True
    use_adaptive_rank: bool = True


class HyperbolicRankGating(nn.Module):
    """
    双曲距離に基づくランクゲーティング
    
    双曲距離を複雑性シグナルとして使用し、
    必要なランクを動的に決定。
    
    Requirements: 21.1, 21.2
    """
    
    def __init__(
        self,
        d_model: int,
        max_rank: int = 32,
        min_rank: int = 4,
        curvature: float = 1.0,
        distance_threshold: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_rank = max_rank
        self.min_rank = min_rank
        self.curvature = curvature
        self.distance_threshold = distance_threshold
        
        # 距離からランクへの変換ネットワーク
        self.rank_predictor = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def compute_hyperbolic_distance(
        self,
        x: torch.Tensor,
        c: Optional[float] = None,
    ) -> torch.Tensor:
        """
        原点からの双曲距離を計算
        
        d(0, x) = (2/√c) * arctanh(√c * |x|)
        
        Args:
            x: 入力テンソル [batch, seq_len, d_model]
            c: 曲率
        
        Returns:
            distance: 双曲距離 [batch, seq_len]
        """
        c = c or self.curvature
        sqrt_c = math.sqrt(c)
        
        # ノルムを計算（境界を超えないようにクランプ）
        norm = x.norm(dim=-1).clamp(max=1.0 / sqrt_c - EPS)
        
        # 双曲距離
        arg = (sqrt_c * norm).clamp(max=1.0 - EPS)
        distance = (2 / sqrt_c) * torch.atanh(arg)
        
        return distance
    
    def forward(
        self,
        x: torch.Tensor,
        c: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        双曲距離に基づいてランクを決定
        
        Args:
            x: 入力テンソル [batch, seq_len, d_model]
            c: 曲率
        
        Returns:
            rank_weights: ランク重み [batch, seq_len]
            effective_rank: 有効ランク [batch, seq_len]
            diagnostics: 診断情報
        """
        # 双曲距離を計算
        distance = self.compute_hyperbolic_distance(x, c)
        
        # 距離からランク重みを予測
        distance_normalized = distance / (self.distance_threshold + EPS)
        rank_weights = self.rank_predictor(distance_normalized.unsqueeze(-1)).squeeze(-1)
        
        # 有効ランクを計算
        rank_range = self.max_rank - self.min_rank
        effective_rank = self.min_rank + rank_weights * rank_range
        
        diagnostics = {
            'distance_mean': distance.mean(),
            'distance_max': distance.max(),
            'rank_weights_mean': rank_weights.mean(),
            'effective_rank_mean': effective_rank.mean(),
        }
        
        return rank_weights, effective_rank, diagnostics


class PhysicsInformedGating(nn.Module):
    """
    物理ベースのゲーティング（BK-Core G_ii使用）
    
    BK-CoreのG_iiを使用して物理的に意味のあるゲーティングを実現。
    
    Requirements: 21.4
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # G_iiからゲート値への変換
        self.gate_proj = nn.Linear(2, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.ones_(self.gate_proj.bias)
    
    def forward(
        self,
        G_ii: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        G_iiからゲート値を計算
        
        Args:
            G_ii: グリーン関数対角成分 [batch, seq_len] (complex)
        
        Returns:
            gate: ゲート値 [batch, seq_len]
            diagnostics: 診断情報
        """
        # G_iiの実部と虚部を特徴量として使用
        G_features = torch.stack([G_ii.real, G_ii.imag], dim=-1)
        
        # ゲート値を計算
        gate = torch.sigmoid(self.gate_proj(G_features)).squeeze(-1)
        
        diagnostics = {
            'physics_gate_mean': gate.mean(),
            'physics_gate_std': gate.std(),
        }
        
        return gate, diagnostics


class AdaptiveRankSSM(nn.Module):
    """
    適応的ランクSSM
    
    入力の複雑性に応じてランクを動的に調整するSSM。
    
    Requirements: 21.1, 21.2, 21.3
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        max_rank: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.max_rank = max_rank
        
        # 状態遷移行列（低ランク分解）
        self.A_low = nn.Parameter(torch.randn(d_state, max_rank) * 0.01)
        self.A_high = nn.Parameter(torch.randn(max_rank, d_state) * 0.01)
        
        # 入力射影
        self.B = nn.Linear(d_model, d_state)
        
        # 出力射影
        self.C = nn.Linear(d_state, d_model)
        
        # 直接パススルー
        self.D = nn.Parameter(torch.zeros(d_model))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.B.weight)
        nn.init.zeros_(self.B.bias)
        nn.init.xavier_uniform_(self.C.weight)
        nn.init.zeros_(self.C.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        rank_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        適応的ランクSSMの前向き計算
        
        Args:
            x: 入力テンソル [batch, seq_len, d_model]
            rank_weights: ランク重み [batch, seq_len]
        
        Returns:
            output: 出力テンソル [batch, seq_len, d_model]
            diagnostics: 診断情報
        """
        batch_size, seq_len, _ = x.shape
        
        # 状態遷移行列を構築
        A = self.A_low @ self.A_high  # [d_state, d_state]
        
        # ランク重みによるスケーリング
        if rank_weights is not None:
            # ランク重みを状態に適用
            rank_scale = rank_weights.unsqueeze(-1)  # [B, N, 1]
        else:
            rank_scale = torch.ones(batch_size, seq_len, 1, device=x.device)
        
        # 入力を状態空間に射影
        u = self.B(x)  # [B, N, d_state]
        
        # 状態を初期化
        h = torch.zeros(batch_size, self.d_state, device=x.device)
        
        outputs = []
        for t in range(seq_len):
            # 状態更新: h = A @ h + B @ u
            h = h @ A.T + u[:, t] * rank_scale[:, t]
            
            # 出力計算: y = C @ h + D @ u
            y = self.C(h) + self.D * x[:, t]
            outputs.append(y)
        
        output = torch.stack(outputs, dim=1)  # [B, N, d_model]
        
        diagnostics = {
            'state_norm_mean': h.norm(dim=-1).mean(),
            'A_spectral_norm': torch.linalg.matrix_norm(A.float(), ord=2),
        }
        
        return output, diagnostics




class ARSSMHyperbolicFusion(nn.Module):
    """
    AR-SSM Hyperbolic Fusion
    
    AR-SSMと双曲空間を融合したモジュール。
    
    主要機能:
    1. 双曲距離に基づくランクゲーティング
    2. 物理ベースのゲーティング（BK-Core使用）
    3. 適応的ランクSSM
    4. 高ランク時の曲率自動増加
    
    Requirements: 21.1, 21.2, 21.3, 21.4, 21.5, 21.6
    """
    
    def __init__(self, config: ARSSMFusionConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.curvature = config.curvature
        
        # 双曲ランクゲーティング
        if config.use_adaptive_rank:
            self.rank_gating = HyperbolicRankGating(
                d_model=config.d_model,
                max_rank=config.max_rank,
                min_rank=config.min_rank,
                curvature=config.curvature,
                distance_threshold=config.distance_threshold,
            )
        else:
            self.rank_gating = None
        
        # 物理ベースゲーティング
        if config.use_physics_gating:
            self.physics_gating = PhysicsInformedGating(d_model=config.d_model)
        else:
            self.physics_gating = None
        
        # 適応的ランクSSM
        self.ssm = AdaptiveRankSSM(
            d_model=config.d_model,
            d_state=config.d_state,
            max_rank=config.max_rank,
        )
        
        # 曲率調整率
        self.curvature_adjustment_rate = config.curvature_adjustment_rate
    
    def forward(
        self,
        x: torch.Tensor,
        G_ii: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass
        
        Args:
            x: 入力テンソル [batch, seq_len, d_model]
            G_ii: グリーン関数対角成分（オプション）
        
        Returns:
            output: 出力テンソル [batch, seq_len, d_model]
            diagnostics: 診断情報
        """
        diagnostics = {}
        
        # ランクゲーティング
        if self.rank_gating is not None:
            rank_weights, effective_rank, rank_diag = self.rank_gating(x, self.curvature)
            diagnostics.update(rank_diag)
            
            # 高ランク時の曲率調整
            high_rank_ratio = (effective_rank > (self.config.max_rank * 0.7)).float().mean()
            if high_rank_ratio > 0.5:
                suggested_curvature = self.curvature * (1 + self.curvature_adjustment_rate)
                diagnostics['suggested_curvature'] = torch.tensor(suggested_curvature)
        else:
            rank_weights = None
        
        # 物理ベースゲーティング
        if self.physics_gating is not None and G_ii is not None:
            physics_gate, physics_diag = self.physics_gating(G_ii)
            diagnostics.update(physics_diag)
            
            # ランク重みと物理ゲートを組み合わせ
            if rank_weights is not None:
                rank_weights = rank_weights * physics_gate
        
        # SSM計算
        output, ssm_diag = self.ssm(x, rank_weights)
        diagnostics.update(ssm_diag)
        
        return output, diagnostics
    
    def compute_throughput_metrics(
        self,
        x: torch.Tensor,
        num_iterations: int = 10,
    ) -> Dict[str, float]:
        """
        スループットメトリクスを計算
        
        Requirements: 21.5 (Property 14の検証用)
        
        Args:
            x: 入力テンソル
            num_iterations: 測定イテレーション数
        
        Returns:
            metrics: スループットメトリクス
        """
        import time
        
        # ウォームアップ
        with torch.no_grad():
            for _ in range(3):
                _ = self.forward(x)
        
        # 測定
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.forward(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed_time = time.time() - start_time
        
        batch_size, seq_len, _ = x.shape
        total_tokens = batch_size * seq_len * num_iterations
        tokens_per_second = total_tokens / elapsed_time
        
        return {
            'tokens_per_second': tokens_per_second,
            'elapsed_time': elapsed_time,
            'iterations': num_iterations,
        }


def create_ar_ssm_fusion(
    d_model: int = 256,
    d_state: int = 64,
    max_rank: int = 32,
    curvature: float = 1.0,
    use_physics_gating: bool = True,
    use_adaptive_rank: bool = True,
    **kwargs,
) -> ARSSMHyperbolicFusion:
    """
    AR-SSM Hyperbolic Fusionのファクトリ関数
    
    Args:
        d_model: モデル次元
        d_state: 状態次元
        max_rank: 最大ランク
        curvature: 曲率
        use_physics_gating: 物理ベースゲーティングを使用するか
        use_adaptive_rank: 適応的ランクを使用するか
        **kwargs: その他の設定
    
    Returns:
        ARSSMHyperbolicFusion instance
    """
    config = ARSSMFusionConfig(
        d_model=d_model,
        d_state=d_state,
        max_rank=max_rank,
        curvature=curvature,
        use_physics_gating=use_physics_gating,
        use_adaptive_rank=use_adaptive_rank,
        **kwargs,
    )
    return ARSSMHyperbolicFusion(config)
