"""
Hyperbolic Persistent Homology Module for Phase 8.

物理的直観:
- β₀が大きい → 思考が断片化している
- β₁が大きい → 循環論理が存在する
- 曲率を上げることで概念を分離できる

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import json
import math


@dataclass
class PersistentHomologyConfig:
    """Configuration for Persistent Homology module."""
    d_model: int = 256
    max_dimension: int = 1  # 計算するホモロジーの最大次元
    threshold_beta1: int = 3  # β₁の警告閾値
    num_landmarks: int = 64  # Witness Complexのランドマーク数
    filtration_steps: int = 20  # フィルトレーションのステップ数
    use_sparse_filtration: bool = True  # スパースフィルトレーションを使用
    curvature_adjustment_rate: float = 0.1  # 曲率調整率
    
    def to_json(self) -> str:
        """Serialize configuration to JSON."""
        return json.dumps({
            'd_model': self.d_model,
            'max_dimension': self.max_dimension,
            'threshold_beta1': self.threshold_beta1,
            'num_landmarks': self.num_landmarks,
            'filtration_steps': self.filtration_steps,
            'use_sparse_filtration': self.use_sparse_filtration,
            'curvature_adjustment_rate': self.curvature_adjustment_rate,
        }, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'PersistentHomologyConfig':
        """Parse configuration from JSON."""
        data = json.loads(json_str)
        return cls(**data)


@dataclass
class PersistentHomologyDiagnostics:
    """Diagnostics from Persistent Homology computation."""
    beta_0: torch.Tensor  # 連結成分数
    beta_1: torch.Tensor  # ループ数
    fragmentation_score: torch.Tensor  # 断片化スコア
    circular_reasoning_detected: torch.Tensor  # 循環論理検出フラグ
    persistence_diagram: Optional[torch.Tensor] = None  # 永続図
    computation_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'beta_0': self.beta_0.tolist() if isinstance(self.beta_0, torch.Tensor) else self.beta_0,
            'beta_1': self.beta_1.tolist() if isinstance(self.beta_1, torch.Tensor) else self.beta_1,
            'fragmentation_score': self.fragmentation_score.tolist() if isinstance(self.fragmentation_score, torch.Tensor) else self.fragmentation_score,
            'circular_reasoning_detected': self.circular_reasoning_detected.tolist() if isinstance(self.circular_reasoning_detected, torch.Tensor) else self.circular_reasoning_detected,
            'computation_time_ms': self.computation_time_ms,
        }


class HyperbolicPersistentHomology(nn.Module):
    """
    Hyperbolic Persistent Homology for Reasoning Analysis.
    
    物理的直観:
    - β₀が大きい → 思考が断片化している
    - β₁が大きい → 循環論理が存在する
    - 曲率を上げることで概念を分離できる
    
    Witness Complexを使用してO(N log N)の計算量を実現。
    
    Args:
        config: PersistentHomologyConfig
    """
    
    def __init__(
        self,
        d_model: int = 256,
        max_dimension: int = 1,
        threshold_beta1: int = 3,
        num_landmarks: int = 64,
        filtration_steps: int = 20,
        use_sparse_filtration: bool = True,
        curvature_adjustment_rate: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_dimension = max_dimension
        self.threshold_beta1 = threshold_beta1
        self.num_landmarks = num_landmarks
        self.filtration_steps = filtration_steps
        self.use_sparse_filtration = use_sparse_filtration
        self.curvature_adjustment_rate = curvature_adjustment_rate
        
        # 学習可能なパラメータ
        self.curvature_scale = nn.Parameter(torch.tensor(1.0))
        
    def _compute_hyperbolic_distance(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        curvature: float = 1.0
    ) -> torch.Tensor:
        """
        Poincaré球での双曲距離を計算。
        
        d(x, y) = (2/√c) * arctanh(√c * ||−x ⊕ y||)
        
        Args:
            x: (B, N, D) or (B, D)
            y: (B, M, D) or (B, D)
            curvature: 曲率パラメータ
            
        Returns:
            distance: (B, N, M) or (B,)
        """
        c = abs(curvature)
        sqrt_c = math.sqrt(c) if c > 0 else 1.0
        
        # ノルムの計算
        x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)
        y_norm_sq = (y ** 2).sum(dim=-1, keepdim=True)
        
        # Möbius加算 -x ⊕ y の計算
        # ||−x ⊕ y||² = ||x - y||² / ((1 - c||x||²)(1 - c||y||²) + c||x - y||²)
        if x.dim() == 3 and y.dim() == 3:
            # (B, N, D) と (B, M, D) の場合
            xy = torch.bmm(x, y.transpose(1, 2))  # (B, N, M)
            diff_sq = x_norm_sq + y_norm_sq.transpose(1, 2) - 2 * xy
        else:
            diff_sq = ((x - y) ** 2).sum(dim=-1)
        
        # 数値安定性のためのクランプ
        diff_sq = diff_sq.clamp(min=1e-10)
        
        # 簡略化された双曲距離（近似）
        # 完全な式は計算コストが高いため、近似を使用
        factor = 1.0 + c * (x_norm_sq.squeeze(-1) if x.dim() == 3 else x_norm_sq)
        if y.dim() == 3:
            factor = factor.unsqueeze(-1) + c * y_norm_sq.squeeze(-1).unsqueeze(1)
        
        dist = torch.sqrt(diff_sq) * torch.sqrt(factor.clamp(min=1e-6))
        
        return dist
    
    def _select_landmarks(
        self,
        embeddings: torch.Tensor,
        num_landmarks: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Witness Complexのためのランドマーク選択（MaxMin法）。
        
        Args:
            embeddings: (B, N, D)
            num_landmarks: ランドマーク数
            
        Returns:
            landmarks: (B, L, D)
            landmark_indices: (B, L)
        """
        B, N, D = embeddings.shape
        L = min(num_landmarks, N)
        
        device = embeddings.device
        landmark_indices = torch.zeros(B, L, dtype=torch.long, device=device)
        
        # 最初のランドマークはランダムに選択
        landmark_indices[:, 0] = torch.randint(0, N, (B,), device=device)
        
        # 各点からランドマークまでの最小距離
        min_distances = torch.full((B, N), float('inf'), device=device)
        
        for i in range(1, L):
            # 現在のランドマークからの距離を計算
            current_landmark = embeddings[
                torch.arange(B, device=device).unsqueeze(1),
                landmark_indices[:, i-1:i]
            ]  # (B, 1, D)
            
            distances = self._compute_hyperbolic_distance(
                embeddings, current_landmark, self.curvature_scale.item()
            ).squeeze(-1)  # (B, N)
            
            # 最小距離を更新
            min_distances = torch.minimum(min_distances, distances)
            
            # 最小距離が最大の点を次のランドマークとして選択
            landmark_indices[:, i] = min_distances.argmax(dim=1)
        
        # ランドマークを抽出
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, L)
        landmarks = embeddings[batch_indices, landmark_indices]
        
        return landmarks, landmark_indices

    def _compute_witness_complex(
        self,
        embeddings: torch.Tensor,
        landmarks: torch.Tensor,
        curvature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Witness Complexを構築。
        
        Args:
            embeddings: (B, N, D) 全点
            landmarks: (B, L, D) ランドマーク
            curvature: 曲率
            
        Returns:
            edge_weights: (B, L, L) エッジの重み（距離）
            witness_counts: (B, L, L) 各エッジのwitness数
        """
        B, N, D = embeddings.shape
        L = landmarks.shape[1]
        device = embeddings.device
        
        # 各点から各ランドマークへの距離
        # (B, N, L)
        point_to_landmark = self._compute_hyperbolic_distance(
            embeddings, landmarks, curvature
        )
        
        # 各点の最近傍ランドマーク2つを見つける
        _, nearest_two = point_to_landmark.topk(2, dim=-1, largest=False)  # (B, N, 2)
        
        # エッジの重みとwitness数を初期化
        edge_weights = torch.full((B, L, L), float('inf'), device=device)
        witness_counts = torch.zeros(B, L, L, device=device)
        
        # 各点がwitnessするエッジを記録
        for b in range(B):
            for n in range(N):
                i, j = nearest_two[b, n, 0].item(), nearest_two[b, n, 1].item()
                if i > j:
                    i, j = j, i
                
                # このエッジのwitness距離（2番目に近いランドマークまでの距離）
                witness_dist = point_to_landmark[b, n, nearest_two[b, n, 1]].item()
                
                # 最小witness距離を記録
                edge_weights[b, i, j] = min(edge_weights[b, i, j].item(), witness_dist)
                edge_weights[b, j, i] = edge_weights[b, i, j]
                witness_counts[b, i, j] += 1
                witness_counts[b, j, i] += 1
        
        return edge_weights, witness_counts
    
    def _compute_betti_numbers_sparse(
        self,
        edge_weights: torch.Tensor,
        filtration_steps: int = 20
    ) -> torch.Tensor:
        """
        スパースフィルトレーションを使用してBetti数を計算。
        O(N log N)の計算量を実現。
        
        Args:
            edge_weights: (B, L, L) エッジの重み
            filtration_steps: フィルトレーションのステップ数
            
        Returns:
            betti_numbers: (B, 2) [β₀, β₁]
        """
        B, L, _ = edge_weights.shape
        device = edge_weights.device
        
        # 有限のエッジ重みのみを考慮
        finite_mask = edge_weights < float('inf')
        
        # フィルトレーション閾値を決定
        finite_weights = edge_weights[finite_mask]
        if finite_weights.numel() == 0:
            return torch.zeros(B, 2, device=device)
        
        max_weight = finite_weights.max().item()
        min_weight = finite_weights.min().item()
        
        thresholds = torch.linspace(min_weight, max_weight, filtration_steps, device=device)
        
        betti_numbers = torch.zeros(B, 2, device=device)
        
        for b in range(B):
            # Union-Findを使用して連結成分を追跡
            parent = list(range(L))
            rank = [0] * L
            
            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]
            
            def union(x, y):
                px, py = find(x), find(y)
                if px == py:
                    return False  # 既に同じ成分
                if rank[px] < rank[py]:
                    px, py = py, px
                parent[py] = px
                if rank[px] == rank[py]:
                    rank[px] += 1
                return True
            
            # エッジをソート
            edges = []
            for i in range(L):
                for j in range(i + 1, L):
                    w = edge_weights[b, i, j].item()
                    if w < float('inf'):
                        edges.append((w, i, j))
            edges.sort()
            
            # β₀: 初期は全点が独立 = L
            # β₁: ループ数
            num_components = L
            num_loops = 0
            
            for w, i, j in edges:
                if not union(i, j):
                    # 既に連結 → ループを形成
                    num_loops += 1
                else:
                    num_components -= 1
            
            betti_numbers[b, 0] = num_components
            betti_numbers[b, 1] = num_loops
        
        return betti_numbers
    
    def forward(
        self,
        embeddings: torch.Tensor,
        curvature: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        トポロジカル解析を実行。
        
        Args:
            embeddings: (B, N, D) 埋め込み
            curvature: 双曲空間の曲率（オプション）
            
        Returns:
            diagnostics: {
                'beta_0': (B,) 連結成分数,
                'beta_1': (B,) ループ数,
                'fragmentation_score': (B,) 断片化スコア,
                'circular_reasoning_detected': (B,) bool
            }
        """
        B, N, D = embeddings.shape
        device = embeddings.device
        
        c = curvature.item() if curvature is not None else self.curvature_scale.item()
        
        # ランドマーク選択
        num_landmarks = min(self.num_landmarks, N)
        landmarks, _ = self._select_landmarks(embeddings, num_landmarks)
        
        # Witness Complex構築
        edge_weights, _ = self._compute_witness_complex(embeddings, landmarks, c)
        
        # Betti数計算
        betti_numbers = self._compute_betti_numbers_sparse(
            edge_weights, self.filtration_steps
        )
        
        beta_0 = betti_numbers[:, 0]
        beta_1 = betti_numbers[:, 1] if self.max_dimension >= 1 else torch.zeros(B, device=device)
        
        # 断片化スコア: β₀ / L（正規化）
        fragmentation_score = beta_0 / num_landmarks
        
        # 循環論理検出
        circular_reasoning = beta_1 > self.threshold_beta1
        
        return {
            'beta_0': beta_0,
            'beta_1': beta_1,
            'fragmentation_score': fragmentation_score,
            'circular_reasoning_detected': circular_reasoning,
        }
    
    def suggest_curvature_adjustment(
        self,
        beta_1: torch.Tensor,
        current_curvature: torch.Tensor
    ) -> torch.Tensor:
        """
        トポロジカル複雑性に基づく曲率調整を提案。
        
        β₁が高い → 曲率を上げて概念を分離
        
        Args:
            beta_1: (B,) ループ数
            current_curvature: 現在の曲率
            
        Returns:
            adjusted_curvature: 調整後の曲率
        """
        adjustment = 1.0 + self.curvature_adjustment_rate * beta_1
        return current_curvature * adjustment
    
    def get_diagnostics(
        self,
        embeddings: torch.Tensor,
        curvature: Optional[torch.Tensor] = None
    ) -> PersistentHomologyDiagnostics:
        """
        詳細な診断情報を取得。
        
        Args:
            embeddings: (B, N, D)
            curvature: 曲率
            
        Returns:
            PersistentHomologyDiagnostics
        """
        import time
        start_time = time.time()
        
        result = self.forward(embeddings, curvature)
        
        computation_time = (time.time() - start_time) * 1000  # ms
        
        return PersistentHomologyDiagnostics(
            beta_0=result['beta_0'],
            beta_1=result['beta_1'],
            fragmentation_score=result['fragmentation_score'],
            circular_reasoning_detected=result['circular_reasoning_detected'],
            computation_time_ms=computation_time,
        )


def create_persistent_homology(
    d_model: int = 256,
    max_dimension: int = 1,
    threshold_beta1: int = 3,
    **kwargs
) -> HyperbolicPersistentHomology:
    """
    Factory function for creating HyperbolicPersistentHomology.
    
    Args:
        d_model: Model dimension
        max_dimension: Maximum homology dimension
        threshold_beta1: Threshold for β₁ warning
        **kwargs: Additional arguments
        
    Returns:
        HyperbolicPersistentHomology instance
    """
    return HyperbolicPersistentHomology(
        d_model=d_model,
        max_dimension=max_dimension,
        threshold_beta1=threshold_beta1,
        **kwargs
    )
