"""
MERA Router for Phase 3: Physics Transcendence

このモジュールは、Multiscale Entanglement Renormalization Ansatz (MERA) に基づく
階層的情報集約ルーターを実装します。

物理的直観:
    MERAは、量子多体系の波動関数を効率的に表現するテンソルネットワークです。
    シーケンス長Nに対して Log(N) 層の階層構造を持ち、
    遠く離れたトークン同士を短絡することで、超長距離依存関係を解決します。

主要コンポーネント:
    1. Disentangler: 局所的な短距離相関を除去（量子もつれの解消）
    2. Isometry: 2トークンを1トークンに粗視化（情報の圧縮）
    3. Global Context: 最上層の単一トークン（全体要約）

Requirements:
    - Requirement 5.1: Log(N)層の階層構造
    - Requirement 5.2: Disentanglerによる局所相関除去
    - Requirement 5.3: Isometryによる粗視化
    - Requirement 5.4: Global Contextの取得
    - Requirement 5.5: パディングの自動処理
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

class MERADisentangler(nn.Module):
    """
    MERA Disentangler Layer (Unitary)

    物理的意味:
        隣接するトークン間の短距離相関（量子もつれ）を解消し、
        情報をより純粋な形に変換します。

    Args:
        d_model: モデル次元
    """
    def __init__(self, d_model: int):
        super().__init__()
        # 2つのトークンを入力とし、2つのトークンを出力
        self.linear = nn.Linear(2 * d_model, 2 * d_model, bias=False)

        # 初期化: 恒等写像に近い形にする（学習の安定性のため）
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(2 * d_model) + 0.01 * torch.randn(2 * d_model, 2 * d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, N, D)
        Returns:
            out: Disentangled tensor (B, N, D)
        """
        B, N, D = x.shape

        # 偶数番目と奇数番目のペアを作成
        # (B, N/2, 2D)
        x_pairs = x.view(B, N // 2, 2 * D)

        # Disentangle
        out_pairs = self.linear(x_pairs)

        # 元の形状に戻す
        out = out_pairs.view(B, N, D)

        return out

class MERAIsometry(nn.Module):
    """
    MERA Isometry Layer (Coarse-graining)

    物理的意味:
        2つのトークンを1つの有効トークンに圧縮（粗視化）します。
        情報の重要度に基づいて、次元を維持したまま情報を集約します。

    Args:
        d_model: モデル次元
    """
    def __init__(self, d_model: int):
        super().__init__()
        # 2つのトークンを入力とし、1つのトークンを出力
        self.linear = nn.Linear(2 * d_model, d_model, bias=False)

        # 初期化
        nn.init.orthogonal_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, N, D)
        Returns:
            out: Coarse-grained tensor (B, N/2, D)
        """
        B, N, D = x.shape

        # ペアを作成 (B, N/2, 2D)
        x_pairs = x.view(B, N // 2, 2 * D)

        # Coarse-grain (Isometry)
        out = self.linear(x_pairs)  # (B, N/2, D)

        return out

class MERARouter(nn.Module):
    """
    MERA (Multiscale Entanglement Renormalization Ansatz) Router

    階層的な情報集約を行い、グローバルコンテキストを抽出します。

    Architecture:
        Layer 0 (Bottom): Input (N tokens)
        ↓ Disentangler
        ↓ Isometry (N -> N/2)
        Layer 1: Coarse (N/2 tokens)
        ↓ Disentangler
        ↓ Isometry (N/2 -> N/4)
        ...
        Top: Global Context (1 token)

    Args:
        d_model: モデル次元
        max_seq_len: 最大シーケンス長
    """
    def __init__(self, d_model: int, max_seq_len: int = 2048):
        super().__init__()
        self.d_model = d_model

        # 階層の深さを計算: ceil(log2(max_seq_len))
        self.num_layers = math.ceil(math.log2(max_seq_len))

        # 各階層のDisentanglerとIsometry
        # パラメータ共有するか、層ごとに独立にするかは選択可能だが、
        # 表現力を高めるために層ごとに独立にする
        self.disentanglers = nn.ModuleList([
            MERADisentangler(d_model) for _ in range(self.num_layers)
        ])
        self.isometries = nn.ModuleList([
            MERAIsometry(d_model) for _ in range(self.num_layers)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Input tensor (B, N, D)

        Returns:
            global_context: (B, 1, D)
            hierarchy: List of tensors at each scale
        """
        B, N, D = x.shape

        # Requirement 5.5: パディング処理
        # Nが2の累乗でない場合、パディングを行う
        next_power_of_2 = 2 ** math.ceil(math.log2(N))
        if N < next_power_of_2:
            padding_size = next_power_of_2 - N
            # (left, right, top, bottom) padding for last 2 dims
            # But here we pad sequence dimension (dim 1)
            # F.pad pads last dim first, so need to look carefully
            # x is (B, N, D), we want to pad N.
            # F.pad input: (B, C, L) usually for 1D, or (B, H, W) for 2D
            # Here we use tensor-level padding manually or F.pad with correct indices
            # F.pad(input, (pad_last_dim_left, pad_last_dim_right, pad_2nd_last_dim_left, ...))
            # Pad D dim (0, 0), Pad N dim (0, padding_size)
            x = F.pad(x, (0, 0, 0, padding_size))

        current_x = x
        hierarchy = [current_x]

        # Bottom-up pass
        for layer_idx in range(self.num_layers):
            # 現在のシーケンス長が1なら終了
            if current_x.shape[1] == 1:
                break

            # Disentangle
            current_x = self.disentanglers[layer_idx](current_x)

            # Isometry (Coarse-graining)
            current_x = self.isometries[layer_idx](current_x)

            hierarchy.append(current_x)

        global_context = current_x # (B, 1, D)

        return global_context, hierarchy

    def broadcast(self, global_context: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        グローバルコンテキストを元のシーケンス長にブロードキャスト

        Args:
            global_context: (B, 1, D)
            target_len: N

        Returns:
            broadcasted: (B, N, D)
        """
        return global_context.expand(-1, target_len, -1)
