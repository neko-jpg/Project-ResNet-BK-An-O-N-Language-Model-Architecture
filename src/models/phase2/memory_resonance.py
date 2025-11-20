"""
Memory Resonance Layer - 記憶共鳴層

Phase 2: Breath of Life

物理的背景:
- 量子系の固有状態は互いに直交する
- ゼータ零点は「最も規則的なランダム性」を持つ
- この基底で対角化すると、記憶の干渉が最小化される

数学的定式化:
W' = U^(-1) W U
ここで U はゼータ零点由来の周波数基底
"""

import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class ZetaBasisTransform:
    """
    ゼータ零点に基づく周波数基底の生成
    
    基底行列 U の構成:
    U[i, j] = exp(2πi * gamma_j * i / N)
    ここで gamma_j はj番目のゼータ零点の虚部
    
    重要: 基底行列は **モデル固定** であり、入力に依存しない。
    これにより、事前計算とキャッシュが可能となり、
    毎ステップの対角化コストを大幅に削減できる。
    
    Args:
        max_dim: 最大次元（デフォルト: 512）
    """
    
    def __init__(self, max_dim: int = 512):
        self.max_dim = max_dim
        # ゼータ零点をキャッシュ
        self._zeta_zeros_cache = {}
        # 基底行列をキャッシュ（デバイスごと）
        self._basis_cache = {}
    
    def get_zeta_zeros(self, n: int) -> torch.Tensor:
        """
        最初のn個のゼータ零点の虚部を取得
        
        n <= 10: 精密値を使用
        n > 10: GUE統計に基づく近似生成
        
        Args:
            n: 取得する零点の数
        
        Returns:
            zeros: (n,) ゼータ零点の虚部
        """
        if n in self._zeta_zeros_cache:
            return self._zeta_zeros_cache[n]
        
        # 精密な零点（最初の10個）
        # リーマンゼータ関数の非自明な零点の虚部
        precise_zeros = torch.tensor([
            14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
            37.586178, 40.918719, 43.327073, 48.005150, 49.773832
        ], dtype=torch.float32)
        
        if n <= 10:
            zeros = precise_zeros[:n]
        else:
            # GUE統計に基づく近似
            # ランダムエルミート行列の固有値を生成
            extra = n - 10
            
            # 十分な数の固有値を生成するため、行列サイズを大きくする
            k = max(extra * 2, 50)
            
            # エルミート行列を生成
            A = torch.randn(k, k, dtype=torch.complex64)
            H = (A + A.conj().T) / 2
            eigs = torch.linalg.eigvalsh(H.real)
            
            # スケーリングして零点分布に合わせる
            sorted_eigs = torch.sort(eigs)[0]
            spacings = sorted_eigs[1:] - sorted_eigs[:-1]
            
            # 中央部分の間隔を使用（端は統計的に異なる）
            mid_start = len(spacings) // 4
            mid_end = mid_start + extra
            if mid_end > len(spacings):
                # 足りない場合は等間隔で補完
                available = len(spacings) - mid_start
                spacings_used = spacings[mid_start:]
                spacings_used = torch.abs(spacings_used)
                
                # 平均間隔を計算
                mean_spacing = spacings_used.mean() if spacings_used.numel() > 0 else 2.5
                
                # 不足分を等間隔で補完
                additional = torch.full((extra - available,), mean_spacing, dtype=torch.float32)
                spacings = torch.cat([spacings_used, additional])
            else:
                spacings = spacings[mid_start:mid_end]
                spacings = torch.abs(spacings)
            
            # 平均間隔を2.5にスケーリング（ゼータ零点の典型的な間隔）
            if spacings.numel() > 0 and spacings.mean() > 0:
                spacings = spacings / spacings.mean() * 2.5
            else:
                # フォールバック: 等間隔
                spacings = torch.full((extra,), 2.5, dtype=torch.float32)
            
            # 累積和で新しい零点を生成
            new_zeros = torch.cumsum(spacings[:extra], dim=0) + precise_zeros[-1]
            zeros = torch.cat([precise_zeros, new_zeros])
        
        self._zeta_zeros_cache[n] = zeros
        return zeros
    
    def get_basis_matrix(self, dim: int, device: torch.device) -> torch.Tensor:
        """
        ゼータ基底行列を生成（キャッシュ付き）
        
        重要: この基底行列は入力に依存せず、モデル固定である。
        一度計算すれば、同じ次元・デバイスの組み合わせでは
        キャッシュから取得できる。
        
        Args:
            dim: 行列次元
            device: デバイス
        
        Returns:
            U: (dim, dim) complex64 基底行列
        """
        cache_key = (dim, str(device))
        if cache_key in self._basis_cache:
            return self._basis_cache[cache_key]
        
        # 正確にdim個の零点を取得
        zeros = self.get_zeta_zeros(dim).to(device)
        
        # zerosがdim個あることを確認
        assert zeros.shape[0] == dim, f"Expected {dim} zeros, got {zeros.shape[0]}"
        
        # 周波数行列の構築
        i_indices = torch.arange(dim, device=device, dtype=torch.float32).unsqueeze(1)
        
        # U[i, j] = exp(2πi * gamma_j * i / dim)
        phase = 2 * torch.pi * zeros.unsqueeze(0) * i_indices / dim
        U = torch.exp(1j * phase)
        
        # 正規化
        U = U / torch.sqrt(torch.tensor(dim, dtype=torch.float32, device=device))
        
        # キャッシュに保存
        self._basis_cache[cache_key] = U
        
        return U
    
    def clear_cache(self):
        """キャッシュをクリア（メモリ節約用）"""
        self._zeta_zeros_cache.clear()
        self._basis_cache.clear()


class MemoryResonanceLayer(nn.Module):
    """
    記憶共鳴層
    
    物理的背景:
    - 量子系の固有状態は互いに直交する
    - ゼータ零点は「最も規則的なランダム性」を持つ
    - この基底で対角化すると、記憶の干渉が最小化される
    
    数学的定式化:
    W' = U^(-1) W U
    ここで U はゼータ零点由来の周波数基底
    
    Args:
        d_model: モデル次元
        head_dim: ヘッド次元（デフォルト: 64）
        num_heads: ヘッド数（デフォルト: 8）
        energy_threshold: 共鳴エネルギー閾値（デフォルト: 0.1）
    """
    
    def __init__(
        self,
        d_model: int,
        head_dim: int = 64,
        num_heads: int = 8,
        energy_threshold: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.energy_threshold = energy_threshold
        
        # ゼータ基底変換
        self.zeta_basis = ZetaBasisTransform(max_dim=head_dim)
        
        # 共鳴検出用パラメータ（オプション）
        self.resonance_gate = nn.Linear(head_dim, 1)
    
    def forward(
        self,
        weights: torch.Tensor,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            weights: (B, H, D_h, D_h) Fast Weights
            x: (B, N, D) 入力（共鳴検出用）
        
        Returns:
            filtered_weights: (B, H, D_h, D_h) フィルタ後の重み
            resonance_info: 共鳴情報の辞書
        """
        B, H, D_h, _ = weights.shape
        
        try:
            # ゼータ基底への変換
            U = self.zeta_basis.get_basis_matrix(D_h, device=weights.device)
            U_inv = torch.linalg.inv(U)
            
            # Ensure weights are complex for proper basis transformation
            # Use torch.view_as_complex for safer conversion
            if not weights.is_complex():
                # Stack real and zero imaginary parts
                weights_real_imag = torch.stack([weights, torch.zeros_like(weights)], dim=-1)
                weights_complex = torch.view_as_complex(weights_real_imag.contiguous())
            else:
                weights_complex = weights
            
            # 対角化: W' = U^(-1) W U
            # Use matmul instead of einsum for better CUDA compatibility
            # Reshape for batch matrix multiplication
            weights_complex_reshaped = weights_complex.view(B * H, D_h, D_h)
            U_inv_expanded = U_inv.unsqueeze(0).expand(B * H, -1, -1)
            U_expanded = U.unsqueeze(0).expand(B * H, -1, -1)
            
            # W' = U^(-1) @ W @ U
            temp = torch.bmm(U_inv_expanded, weights_complex_reshaped)
            weights_diag = torch.bmm(temp, U_expanded)
            weights_diag = weights_diag.view(B, H, D_h, D_h)
            
            # 対角成分のエネルギー
            diag_energy = torch.abs(torch.diagonal(weights_diag, dim1=-2, dim2=-1))  # (B, H, D_h)
            
            # エネルギー閾値でフィルタリング
            mask = diag_energy > self.energy_threshold  # (B, H, D_h)
            
            # マスク適用（対角成分のみ保持）
            # 対角行列を作成してマスクを適用
            mask_expanded = mask.unsqueeze(-1)  # (B, H, D_h, 1)
            weights_diag_filtered = weights_diag * mask_expanded
            
            # 元の基底に戻す: W_filtered = U @ W' @ U^(-1)
            weights_diag_filtered_reshaped = weights_diag_filtered.view(B * H, D_h, D_h)
            temp2 = torch.bmm(U_expanded, weights_diag_filtered_reshaped)
            filtered_weights_complex = torch.bmm(temp2, U_inv_expanded)
            filtered_weights_complex = filtered_weights_complex.view(B, H, D_h, D_h)
            
            # Convert back to real if input was real
            if not weights.is_complex():
                filtered_weights = filtered_weights_complex.real
            else:
                filtered_weights = filtered_weights_complex
            
            # 共鳴情報
            resonance_info = {
                'diag_energy': diag_energy.detach(),
                'resonance_mask': mask.detach(),
                'num_resonant': mask.sum(dim=-1).float().mean().item(),  # 平均共鳴成分数
                'total_energy': diag_energy.sum(dim=-1).mean().item(),
                'sparsity_ratio': (1.0 - mask.float().mean()).item(),  # フィルタリング率
            }
            
        except Exception as e:
            # エラー時はフィルタリングをスキップ
            warnings.warn(
                f"Memory resonance computation failed: {e}. Skipping filtering.",
                UserWarning
            )
            filtered_weights = weights
            resonance_info = {
                'diag_energy': torch.zeros(B, H, D_h, device=weights.device),
                'resonance_mask': torch.ones(B, H, D_h, dtype=torch.bool, device=weights.device),
                'num_resonant': float(D_h),
                'total_energy': 0.0,
                'sparsity_ratio': 0.0,
                'error': str(e),
            }
        
        return filtered_weights, resonance_info
    
    def get_resonance_strength(
        self,
        weights: torch.Tensor,
        mode_i: int,
        mode_j: int
    ) -> torch.Tensor:
        """
        2つの記憶モード間の共鳴強度を計算
        
        Args:
            weights: (B, H, D_h, D_h) Fast Weights
            mode_i: モードiのインデックス
            mode_j: モードjのインデックス
        
        Returns:
            strength: (B, H) 共鳴強度
        """
        B, H, D_h, _ = weights.shape
        
        # ゼータ基底への変換
        U = self.zeta_basis.get_basis_matrix(D_h, device=weights.device)
        U_inv = torch.linalg.inv(U)
        
        # 対角化
        weights_diag = torch.einsum('ij,bhjk,kl->bhil', U_inv, weights, U)
        
        # 非対角成分が共鳴を表す
        strength = torch.abs(weights_diag[:, :, mode_i, mode_j])
        
        return strength


class MemoryImportanceEstimator(nn.Module):
    """
    記憶の重要度を推定するモジュール
    
    共鳴エネルギーとSNRを組み合わせて、
    記憶の重要度を総合的に評価する。
    
    Args:
        head_dim: ヘッド次元
        num_heads: ヘッド数
    """
    
    def __init__(self, head_dim: int = 64, num_heads: int = 8):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        
        # 重要度スコア計算用の線形層
        self.importance_proj = nn.Linear(2, 1)  # [energy, snr] -> importance
    
    def forward(
        self,
        resonance_energy: torch.Tensor,
        snr: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            resonance_energy: (B, H, D_h) 共鳴エネルギー
            snr: (B, H, D_h) 信号対雑音比
        
        Returns:
            importance: (B, H, D_h) 重要度スコア
        """
        # 特徴を結合
        features = torch.stack([resonance_energy, snr], dim=-1)  # (B, H, D_h, 2)
        
        # 重要度スコアを計算
        importance = self.importance_proj(features).squeeze(-1)  # (B, H, D_h)
        
        # Sigmoidで0-1に正規化
        importance = torch.sigmoid(importance)
        
        return importance
