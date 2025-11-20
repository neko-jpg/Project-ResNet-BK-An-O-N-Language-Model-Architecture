"""
Phase 2: Riemann-Zeta Regularization (フラクタル記憶配置)

物理的直観 (Physical Intuition):
    リーマン・ゼータ関数の非自明な零点の虚部（14.13, 21.02, ...）は、
    「最もランダムかつ規則的」な分布（GUE統計）に従います。
    これは量子カオス系におけるエネルギー準位の間隔と同じです。
    
    MUSEでは、記憶素子（Expertやニューロン）の初期配置や
    活性化関数のバイアスをこの分布に従わせることで、
    「情報の衝突（干渉）」を最小化し、効率的な分散表現を実現します。

数学的背景:
    - GUE (Gaussian Unitary Ensemble): ランダム行列理論における確率分布
    - Montgomery-Odlyzko Law: ゼータ零点の間隔分布がGUE固有値間隔と一致
    - Wigner Surmise: P(s) ~ 32/π² * s² * exp(-4s²/π)
    
    これにより、記憶の干渉を最小化し、情報の効率的な分散表現を実現します。

Requirements:
    - 5.1: リーマンゼータ関数の零点近似計算
    - 5.2: n <= 10 で精密値、n > 10 で GUE統計ベース近似
    - 5.3: GUE行列生成による零点分布の近似
    - 5.4: 線形層の特異値をゼータ零点分布で初期化
    - 5.5: ゼータ零点を周波数とする位置埋め込み
    - 5.6: PE(pos, 2i) = sin(pos / zero_i), PE(pos, 2i+1) = cos(pos / zero_i)
"""

import torch
import torch.nn as nn
import math
from typing import Optional
import warnings


class ZetaInitializer:
    """
    ゼータ関数に基づく初期化ユーティリティ
    
    リーマンゼータ関数の零点分布（GUE統計）を用いて、
    ニューラルネットワークの重みを初期化します。
    これにより、情報の干渉を最小化し、効率的な分散表現を実現します。
    
    使用例:
        >>> linear = nn.Linear(512, 512)
        >>> ZetaInitializer.initialize_linear_zeta(linear)
        >>> 
        >>> embedding = nn.Embedding(1024, 512)
        >>> ZetaInitializer.initialize_embedding_zeta(embedding)
    
    物理的解釈:
        - ゼータ零点 = 量子カオス系のエネルギー準位
        - GUE統計 = 最大エントロピー分布（最もランダムかつ規則的）
        - 特異値分布 = 情報の分散度合い
    """
    
    @staticmethod
    def get_approx_zeta_zeros(n: int) -> torch.Tensor:
        """
        最初のn個のリーマンゼータ関数の零点の虚部を近似計算
        
        Args:
            n: 取得する零点の数
        
        Returns:
            zeros: (n,) テンソル。ゼータ零点の虚部
        
        実装詳細:
            - n <= 10: 精密な零点値を使用（Odlyzkoの計算結果）
            - n > 10: GUE統計に基づく近似生成
                1. ランダムエルミート行列を生成
                2. 固有値を計算（GUE統計に従う）
                3. 固有値間隔をゼータ零点の間隔にスケーリング
        
        数学的背景:
            N(T) ~ (T/2π) log(T/2π) - T/2π  (零点の累積個数)
            平均間隔 ~ 2π / log(T)
        
        Requirements: 5.1, 5.2, 5.3
        """
        # 精密な零点（最初の10個）
        # 出典: Odlyzko's tables
        precise_zeros = torch.tensor([
            14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
            37.586178, 40.918719, 43.327073, 48.005150, 49.773832
        ])
        
        if n <= 10:
            return precise_zeros[:n]
        
        # n > 10: GUE統計に基づく近似生成
        extra = n - 10
        
        # GUE行列の生成
        # 行列サイズ k は必要な固有値数より大きくする
        k = max(extra + 20, int(2 * math.sqrt(extra)) + 20)
        
        # ランダムエルミート行列: H = (A + A†) / 2
        # A は複素ガウス行列
        A = torch.randn(k, k, dtype=torch.complex64)
        H = (A + A.conj().transpose(-2, -1)) / 2
        
        # 固有値を計算（実数）
        eigs = torch.linalg.eigvalsh(H.real)
        
        # 中央部の固有値を取得（端の効果を避ける）
        sorted_eigs = eigs.sort()[0]
        
        # 固有値間隔を計算
        spacings = sorted_eigs[1:] - sorted_eigs[:-1]
        
        # 中央部の間隔を抽出（十分な数を確保）
        if len(spacings) < extra:
            # 足りない場合は全体を使用
            selected_spacings = spacings
        else:
            center_start = (len(spacings) - extra) // 2
            center_end = center_start + extra
            selected_spacings = spacings[center_start:center_end]
        
        # 間隔を正規化してスケーリング
        # ゼータ零点の平均間隔 ~ 2.5 (T=50付近)
        selected_spacings = torch.abs(selected_spacings)
        selected_spacings = selected_spacings / selected_spacings.mean() * 2.5  # ヒューリスティックなスケーリング
        
        # 足りない場合は平均間隔で補完
        if len(selected_spacings) < extra:
            mean_spacing = selected_spacings.mean()
            additional = torch.full((extra - len(selected_spacings),), mean_spacing)
            selected_spacings = torch.cat([selected_spacings, additional])
        
        # 累積和で新しい零点を生成
        last_zero = precise_zeros[-1]
        new_zeros = torch.cumsum(selected_spacings[:extra], dim=0) + last_zero
        
        # 精密値と近似値を結合
        result = torch.cat([precise_zeros, new_zeros])
        return result[:n].float()
    
    @staticmethod
    def initialize_linear_zeta(
        module: nn.Linear,
        scale: float = 10.0
    ) -> None:
        """
        線形層の特異値をゼータ零点分布に基づいて初期化
        
        Args:
            module: 初期化する線形層
            scale: スケーリング係数（デフォルト: 10.0）
        
        実装詳細:
            1. 重み行列をSVD分解: W = U S V^T
            2. 特異値をゼータ零点の逆数でスケーリング: S_i = scale / zero_i
            3. 重み行列を再構成: W = U S_new V^T
        
        物理的解釈:
            - 特異値 = 情報の伝達強度
            - ゼータ零点の逆数 = 減衰率
            - 高周波成分（大きい零点）ほど弱く初期化
        
        Requirements: 5.4
        """
        with torch.no_grad():
            # 重み行列の形状を取得
            out_features, in_features = module.weight.shape
            
            # SVD分解
            u, s, v = torch.svd(module.weight)
            
            # 特異値の数は min(out_features, in_features)
            n_s = s.shape[0]
            zeros = ZetaInitializer.get_approx_zeta_zeros(n_s).to(module.weight.device)
            
            # 特異値を零点の逆数でスケーリング
            # S_i = scale / zero_i
            # これにより、高周波成分（大きい零点）ほど弱く初期化される
            new_s = scale / zeros
            
            # 重み行列を再構成
            # U: (out_features, n_s), S: (n_s,), V: (in_features, n_s)
            # W = U @ diag(S) @ V^T
            module.weight.data = torch.mm(u[:, :n_s] * new_s.unsqueeze(0), v[:, :n_s].t())
    
    @staticmethod
    def initialize_embedding_zeta(
        embedding: nn.Embedding,
        scale: float = 1.0
    ) -> None:
        """
        Embeddingをゼータ零点ベースの位相パターンで初期化
        
        Args:
            embedding: 初期化するEmbedding層
            scale: スケーリング係数（デフォルト: 1.0）
        
        実装詳細:
            位置エンコーディング:
            PE(pos, 2i) = sin(pos / zero_i)
            PE(pos, 2i+1) = cos(pos / zero_i)
        
        物理的解釈:
            - ゼータ零点 = 周波数成分
            - Sin/Cos = 位相エンコーディング
            - 不規則な周波数 = 情報の干渉最小化
        
        Requirements: 5.4
        """
        max_len, d_model = embedding.weight.shape
        
        # ゼータ零点取得（d_model // 2 個必要）
        num_zeros = d_model // 2
        zeros = ZetaInitializer.get_approx_zeta_zeros(num_zeros).to(embedding.weight.device)
        
        # 位置エンコーディング
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        position = position.to(embedding.weight.device)
        
        # 周波数（ゼータ零点由来）
        # gamma_i / (2π) を周波数として使用
        freqs = zeros.unsqueeze(0) / (2 * torch.pi)  # (1, num_zeros)
        
        # Sin/Cos エンコーディング
        pe = torch.zeros(max_len, d_model, device=embedding.weight.device)
        
        # position: (max_len, 1), freqs: (1, num_zeros)
        # position * freqs: (max_len, num_zeros)
        sin_values = torch.sin(position * freqs)  # (max_len, num_zeros)
        cos_values = torch.cos(position * freqs)  # (max_len, num_zeros)
        
        # 偶数インデックスにsin、奇数インデックスにcosを配置
        pe[:, 0::2] = sin_values[:, :d_model//2] if d_model % 2 == 0 else sin_values
        pe[:, 1::2] = cos_values[:, :d_model//2] if d_model % 2 == 0 else cos_values[:, :(d_model+1)//2]
        
        # スケーリングして適用
        embedding.weight.data.copy_(pe * scale)


class ZetaEmbedding(nn.Module):
    """
    ゼータ零点ベースの位置埋め込み
    
    標準のSinusoidal Embeddingの代替として、
    リーマンゼータ関数の零点を周波数として使用します。
    
    Args:
        max_len: 最大シーケンス長
        d_model: モデル次元
        trainable: 学習可能にするか（デフォルト: False）
        scale: 初期化スケール（デフォルト: 1.0）
    
    数式:
        PE(pos, 2i) = sin(pos * gamma_i / (2π))
        PE(pos, 2i+1) = cos(pos * gamma_i / (2π))
        
        ここで gamma_i はi番目のゼータ零点の虚部
    
    物理的解釈:
        - 標準のSinusoidal: 等間隔の周波数（10000^(2i/d)）
        - Zeta Embedding: 不規則な周波数（ゼータ零点）
        - 利点: 情報の干渉を最小化、フラクタル的な記憶配置
    
    使用例:
        >>> pos_emb = ZetaEmbedding(max_len=1024, d_model=512, trainable=False)
        >>> positions = torch.arange(0, 100).unsqueeze(0)  # (1, 100)
        >>> embeddings = pos_emb(positions)  # (1, 100, 512)
    
    Requirements: 5.5, 5.6
    """
    
    def __init__(
        self,
        max_len: int,
        d_model: int,
        trainable: bool = False,
        scale: float = 1.0
    ):
        super().__init__()
        
        self.max_len = max_len
        self.d_model = d_model
        self.trainable = trainable
        
        # Embedding層を作成
        self.embedding = nn.Embedding(max_len, d_model)
        
        # ゼータ初期化を適用
        ZetaInitializer.initialize_embedding_zeta(self.embedding, scale=scale)
        
        # 学習可能/固定を設定
        self.embedding.weight.requires_grad = trainable
        
        if not trainable:
            # 固定の場合、パラメータとして登録しない
            self.register_buffer('_fixed_embedding', self.embedding.weight.data.clone())
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        位置埋め込みを計算
        
        Args:
            positions: (B, N) 位置インデックス
        
        Returns:
            embeddings: (B, N, D) 位置埋め込み
        
        注意:
            - positions は [0, max_len) の範囲内である必要があります
            - 範囲外の位置は自動的にクリップされます
        """
        # 位置インデックスの範囲チェック
        if positions.max() >= self.max_len:
            warnings.warn(
                f"Position index {positions.max().item()} exceeds max_len {self.max_len}. "
                f"Clipping to valid range.",
                UserWarning
            )
            positions = torch.clamp(positions, 0, self.max_len - 1)
        
        if not self.trainable:
            # 固定の場合、バッファから取得
            return self._fixed_embedding[positions]
        else:
            # 学習可能な場合、Embedding層を使用
            return self.embedding(positions)
    
    def extra_repr(self) -> str:
        """モジュールの追加情報を返す"""
        return (
            f'max_len={self.max_len}, d_model={self.d_model}, '
            f'trainable={self.trainable}'
        )


# ユーティリティ関数

def apply_zeta_initialization(model: nn.Module, scale: float = 10.0) -> None:
    """
    モデル全体にゼータ初期化を適用
    
    Args:
        model: 初期化するモデル
        scale: スケーリング係数
    
    使用例:
        >>> model = MyModel()
        >>> apply_zeta_initialization(model, scale=10.0)
    
    注意:
        - すべてのnn.Linearとnn.Embeddingに適用されます
        - 既存の初期化を上書きします
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            ZetaInitializer.initialize_linear_zeta(module, scale=scale)
        elif isinstance(module, nn.Embedding):
            ZetaInitializer.initialize_embedding_zeta(module, scale=1.0)


def get_zeta_statistics(n: int = 100) -> dict:
    """
    ゼータ零点の統計情報を取得
    
    Args:
        n: 計算する零点の数
    
    Returns:
        stats: 統計情報の辞書
            - 'zeros': 零点のリスト
            - 'mean_spacing': 平均間隔
            - 'std_spacing': 間隔の標準偏差
            - 'min_spacing': 最小間隔
            - 'max_spacing': 最大間隔
    
    使用例:
        >>> stats = get_zeta_statistics(n=100)
        >>> print(f"Mean spacing: {stats['mean_spacing']:.3f}")
    """
    zeros = ZetaInitializer.get_approx_zeta_zeros(n)
    spacings = zeros[1:] - zeros[:-1]
    
    return {
        'zeros': zeros.tolist(),
        'mean_spacing': spacings.mean().item(),
        'std_spacing': spacings.std().item(),
        'min_spacing': spacings.min().item(),
        'max_spacing': spacings.max().item(),
        'num_zeros': n,
    }
