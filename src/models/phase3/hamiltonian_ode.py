"""
Hamiltonian Neural ODE with Automatic Fallback for Phase 3: Physics Transcendence

このモジュールは、3段階フォールバック機構を持つハミルトニアンODEを実装します。

フォールバック戦略:
    1. Default: Symplectic Adjoint (O(1) memory)
    2. Fallback: Gradient Checkpointing (再構成誤差 > threshold)
    3. Emergency: Full Backprop (チェックポイント失敗時)

物理的直観:
    - Symplectic Adjoint: 最もメモリ効率が良いが、数値不安定性のリスク
    - Checkpointing: メモリとスピードのバランス
    - Full Backprop: 最も安定だが、メモリ使用量が大きい

Requirements: 2.13, 2.14, 2.15, 2.16, 2.17
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, Union
import warnings

from .hamiltonian import HamiltonianFunction, symplectic_leapfrog_step
from .symplectic_adjoint import SymplecticAdjoint, ReconstructionError


class HamiltonianNeuralODE(nn.Module):
    """
    Hamiltonian Neural ODE with Automatic Fallback
    
    3段階フォールバック機構:
        1. Symplectic Adjoint (デフォルト): O(1)メモリ
        2. Gradient Checkpointing (フォールバック): O(√T)メモリ
        3. Full Backprop (緊急): O(T)メモリ
    
    Args:
        d_model: モデルの次元数
        potential_type: ポテンシャルネットワークの種類
            - 'bk_core': Phase 2のBK-Coreを使用（推奨）
            - 'mlp': 標準MLP
        potential_hidden_dim: MLPの隠れ層次元数（potential_type='mlp'の場合）
        dt: 時間刻み（デフォルト: 0.1）
        recon_threshold: 再構成誤差の閾値（デフォルト: 1e-5）
        checkpoint_interval: チェックポイント間隔（デフォルト: 10ステップ）
    
    Requirements: 2.13
    """
    
    def __init__(
        self,
        d_model: int,
        potential_type: str = 'bk_core',
        potential_hidden_dim: Optional[int] = None,
        dt: float = 0.1,
        recon_threshold: float = 1e-5,
        checkpoint_interval: int = 10
    ):
        super().__init__()
        
        # ハミルトニアン関数（Requirement 2.13）
        self.h_func = HamiltonianFunction(
            d_model=d_model,
            potential_type=potential_type,
            potential_hidden_dim=potential_hidden_dim
        )
        
        # パラメータ
        self.d_model = d_model
        self.dt = dt
        self.recon_threshold = recon_threshold
        self.checkpoint_interval = checkpoint_interval
        
        # フォールバック状態管理（Requirement 2.13）
        self.mode = 'symplectic_adjoint'  # 'symplectic_adjoint', 'checkpointing', 'full_backprop'
        self.recon_error_history = []
        self.fallback_count = 0
        
        print(f"HamiltonianNeuralODE initialized:")
        print(f"  - d_model: {d_model}")
        print(f"  - potential_type: {potential_type}")
        print(f"  - dt: {dt}")
        print(f"  - recon_threshold: {recon_threshold}")
        print(f"  - mode: {self.mode}")
    
    def forward(
        self,
        x: torch.Tensor,
        t_span: Tuple[float, float] = (0, 1),
        return_energy: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        """
        Forward pass with automatic fallback
        
        Args:
            x: 初期状態 (B, N, 2D) = [q₀, p₀]
                - 前半D次元: 位置 q
                - 後半D次元: 運動量 p
            t_span: 時間範囲 (t0, t1)
            return_energy: エネルギー変動を返すかどうか (LOGOS Layer 2)
        
        Returns:
            x_final: 最終状態 (B, N, 2D) = [q_T, p_T]
            energy_drift: (Optional) |H(T) - H(0)| の平均値
        
        物理的直観:
            初期状態からハミルトン方程式に従って時間発展させる。
            数値不安定性が検出された場合、自動的にフォールバックする。
            LOGOS: エネルギー保存則 dH/dt ≈ 0 を監視する。
        """
        # モードに応じて適切な方法を選択
        if self.mode == 'symplectic_adjoint':
            try:
                result = self._forward_symplectic_adjoint(x, t_span)
            except ReconstructionError as e:
                warnings.warn(
                    f"Symplectic Adjoint failed (recon_error={e.error:.2e} at step {e.step}). "
                    f"Falling back to checkpointing.",
                    UserWarning
                )
                self.mode = 'checkpointing'
                self.fallback_count += 1
                self.recon_error_history.append({
                    'error': e.error,
                    'threshold': e.threshold,
                    'step': e.step
                })
                result = self._forward_with_checkpointing(x, t_span)
        
        elif self.mode == 'checkpointing':
            result = self._forward_with_checkpointing(x, t_span)
        
        elif self.mode == 'full_backprop':
            result = self._forward_full_backprop(x, t_span)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if return_energy:
            # LOGOS Layer 2: Energy-Based Consistency Check
            # Calculate initial and final energy
            with torch.no_grad():
                # Note: H calculation might be expensive, so we do it only if requested.
                # Assuming Hamiltonian is separable H = T(p) + V(q)
                q0, p0 = x.split(self.d_model, dim=-1)
                qT, pT = result.split(self.d_model, dim=-1)

                # Use h_func to calculate H
                # We need to reconstruct full state for h_func
                h0 = self.h_func(0.0, x).mean().item() # Average energy across batch/seq
                hT = self.h_func(t_span[1], result).mean().item()

                energy_drift = abs(hT - h0)
                return result, energy_drift

        return result
    
    def _forward_symplectic_adjoint(
        self,
        x: torch.Tensor,
        t_span: Tuple[float, float]
    ) -> torch.Tensor:
        """
        Symplectic Adjoint（メモリ効率優先）（Requirement 2.14）
        
        メモリ使用量: O(1)
        計算時間: O(T) (forward) + O(T) (backward) = O(2T)
        
        物理的直観:
            最終状態のみを保存し、逆伝播時に時間を逆再生する。
            ハミルトン系の時間反転対称性を利用。
        """
        return SymplecticAdjoint.apply(
            self.h_func,
            x,
            t_span,
            self.dt,
            self.recon_threshold,
            *self.h_func.parameters()
        )
    
    def _forward_with_checkpointing(
        self,
        x: torch.Tensor,
        t_span: Tuple[float, float]
    ) -> torch.Tensor:
        """
        Gradient Checkpointing（フォールバック）（Requirement 2.15）
        
        メモリ使用量: O(√T) または O(T/checkpoint_interval)
        計算時間: O(T) (forward) + O(2T) (backward with recomputation)
        
        戦略:
            checkpoint_intervalステップごとにチェックポイントを保存。
            逆伝播時は、チェックポイント間を再計算する。
        
        物理的直観:
            完全な軌跡を保存せず、要所要所でスナップショットを取る。
            メモリとスピードのバランスを取る。
        """
        from torch.utils.checkpoint import checkpoint
        
        def forward_chunk(x_in: torch.Tensor, n_steps: int) -> torch.Tensor:
            """n_stepsだけ積分"""
            current = x_in
            for _ in range(n_steps):
                current = symplectic_leapfrog_step(self.h_func, current, self.dt)
            return current
        
        t0, t1 = t_span
        total_steps = int((t1 - t0) / self.dt)
        
        current = x
        for i in range(0, total_steps, self.checkpoint_interval):
            steps_in_chunk = min(self.checkpoint_interval, total_steps - i)
            # チェックポイント機能を使用（自動的に再計算）
            current = checkpoint(forward_chunk, current, steps_in_chunk, use_reentrant=False)
        
        return current
    
    def _forward_full_backprop(
        self,
        x: torch.Tensor,
        t_span: Tuple[float, float]
    ) -> torch.Tensor:
        """
        Full Backpropagation（緊急フォールバック）（Requirement 2.16）
        
        メモリ使用量: O(T)
        計算時間: O(T) (forward) + O(T) (backward)
        
        警告:
            全ステップの状態を保存するため、メモリ使用量が大きい。
            数値不安定性が深刻な場合の最終手段。
        
        物理的直観:
            通常のニューラルネットワークと同じく、全ての中間状態を保存。
            最も安定だが、メモリ効率は最悪。
        """
        warnings.warn(
            "Using full backprop mode. Memory usage will be O(T).",
            UserWarning
        )
        
        t0, t1 = t_span
        steps = int((t1 - t0) / self.dt)
        
        current = x
        for _ in range(steps):
            current = symplectic_leapfrog_step(self.h_func, current, self.dt)
        
        return current
    
    def reset_to_symplectic(self):
        """
        Symplectic Adjointモードにリセット
        
        Usage:
            エポック開始時に呼び出すことで、再度Symplectic Adjointを試行する。
            学習が進むにつれて数値安定性が改善される可能性がある。
        """
        if self.fallback_count > 0:
            print(
                f"Resetting to Symplectic Adjoint mode "
                f"(fallback_count={self.fallback_count})"
            )
        self.mode = 'symplectic_adjoint'
        self.fallback_count = 0
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        診断情報の取得
        
        Returns:
            diagnostics: 診断情報
                - mode: 現在のモード
                - fallback_count: フォールバック回数
                - recon_error_history: 再構成誤差の履歴
        """
        return {
            'mode': self.mode,
            'fallback_count': self.fallback_count,
            'recon_error_history': self.recon_error_history
        }
    
    def set_mode(self, mode: str):
        """
        モードを手動で設定
        
        Args:
            mode: 'symplectic_adjoint', 'checkpointing', 'full_backprop'
        
        Usage:
            デバッグやベンチマーク時に特定のモードを強制する。
        """
        valid_modes = ['symplectic_adjoint', 'checkpointing', 'full_backprop']
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid mode: {mode}. Choose from {valid_modes}"
            )
        
        print(f"Manually setting mode to: {mode}")
        self.mode = mode
