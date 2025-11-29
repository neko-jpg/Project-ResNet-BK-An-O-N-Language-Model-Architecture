"""
Block-wise Distance Computation for Phase 8

物理的直観:
- 双曲距離行列を128x128ブロックで計算
- 各ブロック計算後すぐにsoftmaxとV乗算を実行
- 距離ブロックは使用後すぐに破棄してメモリを節約
- Causalマスクの場合、上三角ブロックをスキップ

Requirements: 7.1-7.6
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
class BlockDistanceConfig:
    """Block-wise Distance設定"""
    d_model: int = 256
    num_heads: int = 8
    curvature: float = 1.0
    
    # ブロックサイズ
    block_size_m: int = 128  # Queryブロックサイズ
    block_size_n: int = 128  # Key/Valueブロックサイズ
    
    # Causalマスク
    causal: bool = False
    
    # 数値安定性
    eps: float = 1e-6
    max_norm: float = 0.99
    
    # 共有メモリ最適化
    use_shared_memory: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "curvature": self.curvature,
            "block_size_m": self.block_size_m,
            "block_size_n": self.block_size_n,
            "causal": self.causal,
            "eps": self.eps,
            "max_norm": self.max_norm,
            "use_shared_memory": self.use_shared_memory,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BlockDistanceConfig":
        return cls(**d)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, s: str) -> "BlockDistanceConfig":
        return cls.from_dict(json.loads(s))


@dataclass
class BlockDistanceDiagnostics:
    """Block-wise Distance診断情報"""
    num_blocks_computed: int = 0
    num_blocks_skipped: int = 0  # Causalでスキップされたブロック
    peak_memory_mb: float = 0.0
    total_distance_computations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_blocks_computed": self.num_blocks_computed,
            "num_blocks_skipped": self.num_blocks_skipped,
            "peak_memory_mb": self.peak_memory_mb,
            "total_distance_computations": self.total_distance_computations,
        }


def poincare_distance_block(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    c: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    ブロック単位でPoincaré距離を計算
    
    Args:
        q_block: (B, H, M, D) Queryブロック
        k_block: (B, H, N, D) Keyブロック
        c: 曲率
        eps: 数値安定性のための小さな値
    
    Returns:
        dist: (B, H, M, N) 距離行列
    """
    sqrt_c = torch.sqrt(c.clamp(min=eps))
    
    # ||q - k||²
    # q_block: (B, H, M, D), k_block: (B, H, N, D)
    diff = q_block.unsqueeze(3) - k_block.unsqueeze(2)  # (B, H, M, N, D)
    diff_sq = (diff ** 2).sum(dim=-1)  # (B, H, M, N)
    
    # (1 - c||q||²)(1 - c||k||²)
    q_norm_sq = (q_block ** 2).sum(dim=-1, keepdim=True)  # (B, H, M, 1)
    k_norm_sq = (k_block ** 2).sum(dim=-1).unsqueeze(2)  # (B, H, 1, N)
    
    denom = (1 - c * q_norm_sq) * (1 - c * k_norm_sq)
    denom = denom.clamp(min=eps)
    
    # arcosh(1 + 2c * ||q-k||² / denom)
    arg = 1 + 2 * c * diff_sq / denom
    arg = arg.clamp(min=1.0 + eps)
    
    dist = (1 / sqrt_c) * torch.acosh(arg)
    return dist


class BlockWiseDistanceComputation(nn.Module):
    """
    Block-wise Distance Computation
    
    物理的直観:
    - 128x128ブロックで距離を計算
    - 各ブロック計算後すぐにsoftmaxとV乗算
    - 距離ブロックは使用後すぐに破棄
    - O(N)メモリスケーリングを達成
    
    Requirements: 7.1-7.6
    """
    
    def __init__(self, config: BlockDistanceConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_head = config.d_model // config.num_heads
        self.block_size_m = config.block_size_m
        self.block_size_n = config.block_size_n
        
        # 曲率パラメータ
        self.curvature = nn.Parameter(torch.tensor(config.curvature))
        
        # Q, K, V投影
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
    
    def _should_skip_block(
        self,
        block_row: int,
        block_col: int,
        causal: bool,
    ) -> bool:
        """
        Causalマスクでブロックをスキップすべきか判定
        
        上三角ブロック（block_col > block_row）はスキップ
        """
        if not causal:
            return False
        return block_col > block_row
    
    def _compute_block_attention(
        self,
        q_block: torch.Tensor,
        k_block: torch.Tensor,
        v_block: torch.Tensor,
        c: torch.Tensor,
        block_row: int,
        block_col: int,
        causal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        単一ブロックのアテンション計算
        
        Args:
            q_block: (B, H, M, D) Queryブロック
            k_block: (B, H, N, D) Keyブロック
            v_block: (B, H, N, D) Valueブロック
            c: 曲率
            block_row: ブロック行インデックス
            block_col: ブロック列インデックス
            causal: Causalマスクを使用するか
        
        Returns:
            output: (B, H, M, D) アテンション出力
            normalizer: (B, H, M) 正規化係数
        """
        B, H, M, D = q_block.shape
        N = k_block.shape[2]
        
        # 距離計算
        dist = poincare_distance_block(q_block, k_block, c, self.config.eps)
        
        # 距離をスコアに変換（負の距離）
        scores = -dist
        
        # Causalマスク（対角ブロックの場合）
        if causal and block_row == block_col:
            # 対角ブロック内のCausalマスク
            mask = torch.triu(
                torch.ones(M, N, device=scores.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        
        # Softmax（数値安定性のためにmax減算）
        scores_max = scores.max(dim=-1, keepdim=True).values
        scores_exp = torch.exp(scores - scores_max)
        
        # 正規化係数
        normalizer = scores_exp.sum(dim=-1)  # (B, H, M)
        
        # V乗算
        output = torch.einsum("bhmn,bhnd->bhmd", scores_exp, v_block)
        
        return output, normalizer
    
    def forward(
        self,
        x: torch.Tensor,
        causal: Optional[bool] = None,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[BlockDistanceDiagnostics]]:
        """
        Forward pass
        
        Args:
            x: (B, N, D) 入力
            causal: Causalマスクを使用するか（Noneの場合はconfig設定を使用）
            return_diagnostics: 診断情報を返すか
        
        Returns:
            output: (B, N, D) 出力
            diagnostics: 診断情報（オプション）
        """
        B, N, D = x.shape
        c = self.curvature.abs().clamp(min=1e-6)
        
        if causal is None:
            causal = self.config.causal
        
        # Q, K, V投影
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.d_head).transpose(1, 2)
        
        # ブロック数
        num_blocks_m = (N + self.block_size_m - 1) // self.block_size_m
        num_blocks_n = (N + self.block_size_n - 1) // self.block_size_n
        
        # 出力と正規化係数の初期化
        output = torch.zeros_like(q)
        normalizer = torch.zeros(B, self.num_heads, N, device=x.device, dtype=x.dtype)
        
        # 診断情報
        blocks_computed = 0
        blocks_skipped = 0
        
        # ブロック単位で計算
        for block_row in range(num_blocks_m):
            # Queryブロックの範囲
            q_start = block_row * self.block_size_m
            q_end = min(q_start + self.block_size_m, N)
            q_block = q[:, :, q_start:q_end, :]
            
            for block_col in range(num_blocks_n):
                # Causalでスキップ判定
                if self._should_skip_block(block_row, block_col, causal):
                    blocks_skipped += 1
                    continue
                
                # Key/Valueブロックの範囲
                k_start = block_col * self.block_size_n
                k_end = min(k_start + self.block_size_n, N)
                k_block = k[:, :, k_start:k_end, :]
                v_block = v[:, :, k_start:k_end, :]
                
                # ブロックアテンション計算
                block_output, block_normalizer = self._compute_block_attention(
                    q_block, k_block, v_block, c,
                    block_row, block_col, causal,
                )
                
                # 累積
                output[:, :, q_start:q_end, :] += block_output
                normalizer[:, :, q_start:q_end] += block_normalizer
                
                blocks_computed += 1
        
        # 正規化
        normalizer = normalizer.clamp(min=self.config.eps)
        output = output / normalizer.unsqueeze(-1)
        
        # 形状を戻す
        output = output.transpose(1, 2).reshape(B, N, D)
        output = self.out_proj(output)
        
        # 診断情報
        diagnostics = None
        if return_diagnostics:
            diagnostics = BlockDistanceDiagnostics(
                num_blocks_computed=blocks_computed,
                num_blocks_skipped=blocks_skipped,
                total_distance_computations=blocks_computed * self.block_size_m * self.block_size_n,
            )
        
        return output, diagnostics
    
    def estimate_memory_usage(self, seq_len: int) -> float:
        """
        メモリ使用量を推定（MB）
        
        Property 13: Block-wise Memory Scaling検証用
        """
        # ブロックサイズ分のメモリのみ使用
        block_memory = self.block_size_m * self.block_size_n * 4  # float32
        
        # Q, K, Vのメモリ
        qkv_memory = 3 * seq_len * self.d_model * 4
        
        # 出力のメモリ
        output_memory = seq_len * self.d_model * 4
        
        total_bytes = block_memory + qkv_memory + output_memory
        return total_bytes / (1024 * 1024)


class SharedMemoryBlockDistance(nn.Module):
    """
    共有メモリ最適化版Block-wise Distance
    
    Requirements: 7.4
    
    注: これはPyTorch実装のシミュレーション。
    実際のTriton実装では共有メモリを使用。
    """
    
    def __init__(self, config: BlockDistanceConfig):
        super().__init__()
        self.config = config
        self.base_module = BlockWiseDistanceComputation(config)
    
    def forward(
        self,
        x: torch.Tensor,
        causal: Optional[bool] = None,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[BlockDistanceDiagnostics]]:
        """
        共有メモリ最適化版forward
        
        PyTorchでは通常のブロック計算と同じ。
        Triton実装では共有メモリを使用。
        """
        return self.base_module(x, causal, return_diagnostics)


def create_block_distance(
    d_model: int = 256,
    num_heads: int = 8,
    curvature: float = 1.0,
    block_size: int = 128,
    causal: bool = False,
    **kwargs,
) -> BlockWiseDistanceComputation:
    """
    Block-wise Distanceモジュールを作成
    
    Args:
        d_model: モデル次元
        num_heads: ヘッド数
        curvature: 初期曲率
        block_size: ブロックサイズ
        causal: Causalマスクを使用するか
        **kwargs: その他の設定
    
    Returns:
        BlockWiseDistanceComputation インスタンス
    """
    config = BlockDistanceConfig(
        d_model=d_model,
        num_heads=num_heads,
        curvature=curvature,
        block_size_m=block_size,
        block_size_n=block_size,
        causal=causal,
        **kwargs,
    )
    return BlockWiseDistanceComputation(config)
