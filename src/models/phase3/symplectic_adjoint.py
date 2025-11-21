"""
Symplectic Adjoint Method for Phase 3: Physics Transcendence

このモジュールは、O(1)メモリでハミルトニアンODEを学習するための
シンプレクティック随伴法を実装します。

物理的直観:
- 順伝播: Leapfrog積分で時間発展（最終状態のみ保存）
- 逆伝播: 時間を逆再生して随伴状態を更新（O(1)メモリ）
- 再構成誤差: 逆時間積分の数値誤差を監視

メモリ効率:
- 通常のBackprop: O(T) メモリ（全ステップの状態を保存）
- Symplectic Adjoint: O(1) メモリ（最終状態のみ保存）

数値安定性:
- 再構成誤差が閾値（1e-5）を超えた場合、自動的にフォールバック
- フォールバック先: Gradient Checkpointing → Full Backprop

Requirements: 2.8, 2.9, 2.10, 2.11, 2.12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import warnings

from .hamiltonian import HamiltonianFunction, symplectic_leapfrog_step


class ReconstructionError(Exception):
    """
    再構成誤差が閾値を超えた場合の例外（Requirement 2.11）
    
    物理的意味:
        逆時間積分時の数値誤差が大きすぎる場合に発生。
        カオス的挙動や数値不安定性を示唆する。
    
    Attributes:
        error: 再構成誤差の値
        threshold: 閾値
        step: エラーが発生したステップ
    """
    
    def __init__(self, error: float, threshold: float, step: int):
        self.error = error
        self.threshold = threshold
        self.step = step
        super().__init__(
            f"Reconstruction error {error:.2e} > threshold {threshold:.2e} "
            f"at step {step}. Consider using checkpointing."
        )


class SymplecticAdjoint(torch.autograd.Function):
    """
    シンプレクティック随伴法（torch.autograd.Function）
    
    O(1)メモリでハミルトニアンODEの勾配を計算する。
    
    Forward Pass（Requirement 2.8）:
        - Leapfrog積分で時間発展
        - 最終状態のみを保存（中間状態は破棄）
        - メモリ使用量: O(1)
    
    Backward Pass（Requirement 2.9）:
        - 時間を逆再生して随伴状態を更新
        - パラメータ勾配を累積
        - 再構成誤差を監視（Requirement 2.10）
    
    物理的直観:
        ハミルトン系は時間反転対称性を持つため、
        逆時間積分により元の状態を再構成できる。
        ただし、数値誤差により完全な再構成は不可能。
    """
    
    @staticmethod
    def forward(
        ctx: Any,
        h_func: HamiltonianFunction,
        x0: torch.Tensor,
        t_span: Tuple[float, float],
        dt: float,
        recon_threshold: float,
        *params: torch.Tensor
    ) -> torch.Tensor:
        """
        順伝播: Leapfrog積分（Requirement 2.8）
        
        Args:
            ctx: コンテキスト（backward用の情報を保存）
            h_func: ハミルトニアン関数
            x0: 初期状態 (B, N, 2D) = [q₀, p₀]
            t_span: 時間範囲 (t0, t1)
            dt: 時間刻み
            recon_threshold: 再構成誤差の閾値
            *params: h_funcのパラメータ（勾配計算用）
        
        Returns:
            x_final: 最終状態 (B, N, 2D) = [q_T, p_T]
        
        物理的直観:
            初期状態からLeapfrog積分で時間発展させる。
            中間状態は保存せず、最終状態のみを返す（O(1)メモリ）。
        """
        t0, t1 = t_span
        steps = int((t1 - t0) / dt)
        
        # Leapfrog積分
        x = x0.clone()
        for _ in range(steps):
            x = symplectic_leapfrog_step(h_func, x, dt)
        
        # 最終状態のみ保存（O(1)メモリ）
        ctx.save_for_backward(x, x0)
        ctx.h_func = h_func
        ctx.t_span = t_span
        ctx.dt = dt
        ctx.recon_threshold = recon_threshold
        
        return x
    
    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """
        逆伝播: 随伴法（Requirement 2.9）
        
        Args:
            ctx: コンテキスト（forward時に保存した情報）
            grad_output: 出力に対する勾配 ∂L/∂x_T
        
        Returns:
            勾配のタプル:
                - None: h_funcに対する勾配（不要）
                - adj: x0に対する勾配 ∂L/∂x₀
                - None: t_spanに対する勾配（不要）
                - None: dtに対する勾配（不要）
                - None: recon_thresholdに対する勾配（不要）
                - *param_grads: パラメータに対する勾配
        
        物理的直観:
            時間を逆再生して随伴状態を更新する。
            随伴状態 a(t) は「出力の変化が入力に与える影響」を表す。
        
        アルゴリズム:
            1. 随伴状態の初期化: a_T = ∂L/∂x_T
            2. 時間を逆再生: t = T → T-1 → ... → 0
            3. 各ステップで:
                - 状態を逆積分: x_{t-1} = Leapfrog⁻¹(x_t)
                - 随伴状態を更新: a_{t-1} = a_t + a_t·∂f/∂x·dt
                - パラメータ勾配を累積: ∂L/∂θ += a_t·∂f/∂θ·dt
            4. 再構成誤差を監視（Requirement 2.10）
        """
        x_final, x0 = ctx.saved_tensors
        h_func = ctx.h_func
        t0, t1 = ctx.t_span
        dt = ctx.dt
        recon_threshold = ctx.recon_threshold
        steps = int((t1 - t0) / dt)
        
        # 随伴状態の初期化
        adj = grad_output.clone()
        x = x_final.clone()
        
        # パラメータ勾配の累積器
        param_grads = [torch.zeros_like(p) for p in h_func.parameters()]
        
        # 再構成誤差の監視（Requirement 2.10）
        reconstruction_errors = []
        
        # 逆時間積分
        for step in range(steps):
            # 1. 状態の逆積分（近似）
            # 物理的直観: ハミルトン系は時間反転対称性を持つため、
            #            dt → -dt で逆時間積分が可能
            x_prev = symplectic_leapfrog_step(h_func, x, -dt)
            
            # 再構成誤差のチェック（10ステップごと）
            if step % 10 == 0:
                # Forward再計算で検証
                x_check = symplectic_leapfrog_step(h_func, x_prev, dt)
                recon_error = (x_check - x).abs().max().item()
                reconstruction_errors.append(recon_error)
                
                # 閾値チェック（Requirement 2.10）
                if recon_error > recon_threshold:
                    # 例外を投げてフォールバックをトリガー
                    raise ReconstructionError(recon_error, recon_threshold, step)
            
            # 2. ベクトル場の計算と随伴状態更新
            with torch.enable_grad():
                x_prev_grad = x_prev.requires_grad_(True)
                f = h_func.hamiltonian_vector_field(0, x_prev_grad)
                
                # VJP (Vector-Jacobian Product): adj^T · ∂f/∂x
                # 物理的直観: 随伴状態は「出力の変化が入力に与える影響」
                adj_grad = torch.autograd.grad(
                    f, x_prev_grad,
                    grad_outputs=adj,
                    retain_graph=True
                )[0]
                
                # 随伴状態の更新
                adj = adj + adj_grad * dt
                
                # パラメータ勾配の累積
                for i, param in enumerate(h_func.parameters()):
                    if param.requires_grad:
                        param_grad = torch.autograd.grad(
                            f, param,
                            grad_outputs=adj,
                            retain_graph=True,
                            allow_unused=True
                        )[0]
                        if param_grad is not None:
                            param_grads[i] += param_grad * dt
            
            x = x_prev
        
        # 再構成誤差の統計
        if reconstruction_errors:
            max_error = max(reconstruction_errors)
            mean_error = sum(reconstruction_errors) / len(reconstruction_errors)
            if max_error > recon_threshold * 0.5:
                # 閾値の50%を超えたら警告
                warnings.warn(
                    f"Symplectic Adjoint: max_recon_error={max_error:.2e}, "
                    f"mean={mean_error:.2e}. Close to threshold {recon_threshold:.2e}.",
                    UserWarning
                )
        
        # 勾配を返す
        # (h_func, x0, t_span, dt, recon_threshold, *params)
        return (None, adj, None, None, None, *param_grads)


