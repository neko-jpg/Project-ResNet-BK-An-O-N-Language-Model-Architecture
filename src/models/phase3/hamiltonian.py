"""
Hamiltonian Neural ODE for Phase 3: Physics Transcendence

このモジュールは、エネルギー保存則に従う思考プロセスを実現するための
ハミルトニアン力学系を実装します。

物理的直観:
- ハミルトニアン H(q, p) = T(p) + V(q) は系の全エネルギーを表す
- q: 位置（思考の状態）
- p: 運動量（思考の変化率）
- T(p): 運動エネルギー（思考の勢い）
- V(q): ポテンシャルエネルギー（思考の安定性）

エネルギー保存により、長時間の推論でも論理的矛盾や幻覚を防ぐ。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import warnings


class HamiltonianFunction(nn.Module):
    """
    ハミルトニアン関数 H(q, p) = T(p) + V(q)
    
    物理的意味:
    - T(p) = ½|p|²: 運動エネルギー（思考の勢い）
    - V(q) = Potential_Net(q): ポテンシャルエネルギー（思考の安定性）
    
    Args:
        d_model: モデルの次元数
        potential_type: ポテンシャルネットワークの種類
            - 'bk_core': Phase 2のBK-Coreを使用（推奨）
            - 'mlp': 標準MLP
        potential_hidden_dim: MLPの隠れ層次元数（potential_type='mlp'の場合）
    
    Requirements: 2.1, 2.2
    """
    
    def __init__(
        self,
        d_model: int,
        potential_type: str = 'bk_core',
        potential_hidden_dim: Optional[int] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.potential_type = potential_type
        
        # ポテンシャルネットワークの構築（Requirement 2.2）
        if potential_type == 'bk_core':
            # Phase 2のBK-Coreを使用
            try:
                from src.models.bk_core import BKCore
                self.potential_net = BKCore(d_model)
                print(f"HamiltonianFunction: Using BK-Core potential (d_model={d_model})")
            except ImportError:
                warnings.warn(
                    "BK-Core not available. Falling back to MLP potential.",
                    UserWarning
                )
                self.potential_type = 'mlp'
                self._build_mlp_potential(d_model, potential_hidden_dim)
        
        elif potential_type == 'mlp':
            self._build_mlp_potential(d_model, potential_hidden_dim)
        
        else:
            raise ValueError(
                f"Unknown potential_type: {potential_type}. "
                f"Choose from ['bk_core', 'mlp']"
            )
    
    def _build_mlp_potential(self, d_model: int, hidden_dim: Optional[int]):
        """標準MLPポテンシャルの構築"""
        if hidden_dim is None:
            hidden_dim = d_model * 4
        
        self.potential_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        print(f"HamiltonianFunction: Using MLP potential (d_model={d_model}, hidden={hidden_dim})")
    
    def forward(self, t: float, x: torch.Tensor) -> torch.Tensor:
        """
        ハミルトニアン H(q, p) = T(p) + V(q) を計算（Requirement 2.1）
        
        Args:
            t: 時刻（ODEソルバー用、実際には使用しない）
            x: 位相空間の状態 (B, N, 2D) = [q, p]
                - q: 位置（前半D次元）
                - p: 運動量（後半D次元）
        
        Returns:
            energy: ハミルトニアン値 (B, N)
        
        物理的直観:
            H = T + V は系の全エネルギー
            エネルギー保存則により、H(t) ≈ const
        """
        # 位置と運動量に分離
        n_dim = x.shape[-1] // 2
        q = x[..., :n_dim]  # 位置（思考の状態）
        p = x[..., n_dim:]  # 運動量（思考の変化率）
        
        # 運動エネルギー: T(p) = ½|p|²
        kinetic = 0.5 * (p ** 2).sum(dim=-1)  # (B, N)
        
        # ポテンシャルエネルギー: V(q) = Potential_Net(q)
        v_out = self.potential_net(q)  # (B, N, D) or (B, N, 1)
        
        if v_out.shape[-1] != 1:
            # BK-Coreの場合: 出力が(B, N, D)なので、ノルムを取る
            potential = 0.5 * (v_out ** 2).sum(dim=-1)  # (B, N)
        else:
            # MLPの場合: 出力が(B, N, 1)
            potential = v_out.squeeze(-1)  # (B, N)
        
        # 全エネルギー
        energy = kinetic + potential
        
        return energy
    
    def hamiltonian_vector_field(self, t: float, x: torch.Tensor) -> torch.Tensor:
        """
        ハミルトンベクトル場 J·∇H を計算（Requirement 2.3）
        
        ハミルトン方程式:
            dq/dt = ∂H/∂p = p  （運動量が位置の変化率）
            dp/dt = -∂H/∂q = -∇V(q)  （力が運動量の変化率）
        
        シンプレクティック構造:
            J = [[0, I], [-I, 0]]
            dx/dt = J·∇H
        
        Args:
            t: 時刻
            x: 位相空間の状態 (B, N, 2D) = [q, p]
        
        Returns:
            dx_dt: 時間微分 (B, N, 2D) = [dq/dt, dp/dt]
        
        物理的直観:
            - dq/dt = p: 運動量が大きいほど、位置が速く変化
            - dp/dt = -∇V: ポテンシャルの勾配が力となり、運動量を変化させる
        """
        # 勾配計算を有効化
        with torch.enable_grad():
            x = x.requires_grad_(True)
            
            # ハミルトニアンを計算
            h = self.forward(t, x).sum()  # スカラー化
            
            # ∇H を計算
            grad_h = torch.autograd.grad(h, x, create_graph=True)[0]
        
        # 位置と運動量の勾配に分離
        n_dim = x.shape[-1] // 2
        dH_dq = grad_h[..., :n_dim]  # ∂H/∂q
        dH_dp = grad_h[..., n_dim:]  # ∂H/∂p
        
        # シンプレクティック構造 J = [[0, I], [-I, 0]] を適用
        # dx/dt = J·∇H = [∂H/∂p, -∂H/∂q]
        dq_dt = dH_dp   # = p (if T = ½p²)
        dp_dt = -dH_dq  # = -∇V(q)
        
        # 結合
        dx_dt = torch.cat([dq_dt, dp_dt], dim=-1)
        
        return dx_dt


def symplectic_leapfrog_step(
    h_func: HamiltonianFunction,
    x: torch.Tensor,
    dt: float
) -> torch.Tensor:
    """
    Leapfrog法による1ステップのシンプレクティック積分
    
    アルゴリズム:
        1. p(t + dt/2) = p(t) - ∇V(q(t)) · dt/2  (Half-step momentum)
        2. q(t + dt)   = q(t) + p(t + dt/2) · dt  (Full-step position)
        3. p(t + dt)   = p(t + dt/2) - ∇V(q(t + dt)) · dt/2  (Half-step momentum)
    
    物理的直観:
        Leapfrog法はシンプレクティック積分器であり、エネルギー誤差が有界。
        長時間積分でもエネルギーが保存される。
    
    Args:
        h_func: ハミルトニアン関数
        x: 現在の状態 (B, N, 2D) = [q, p]
        dt: 時間刻み
    
    Returns:
        x_next: 次の状態 (B, N, 2D) = [q_new, p_new]
    
    Requirements: 2.5
    """
    n_dim = x.shape[-1] // 2
    q = x[..., :n_dim]
    p = x[..., n_dim:]
    
    # 力の計算: F = -∇V(q)
    with torch.enable_grad():
        q_grad = q.requires_grad_(True)
        v = h_func.potential_net(q_grad)
        
        if v.shape[-1] != 1:
            # BK-Coreの場合
            v = 0.5 * (v ** 2).sum(dim=-1)
        else:
            # MLPの場合
            v = v.squeeze(-1)
        
        force = -torch.autograd.grad(v.sum(), q_grad)[0]
    
    # Step 1: Half-step momentum
    p_half = p + force * (dt / 2)
    
    # Step 2: Full-step position
    q_new = q + p_half * dt
    
    # Step 3: Half-step momentum (with new position)
    with torch.enable_grad():
        q_new_grad = q_new.requires_grad_(True)
        v_new = h_func.potential_net(q_new_grad)
        
        if v_new.shape[-1] != 1:
            # BK-Coreの場合
            v_new = 0.5 * (v_new ** 2).sum(dim=-1)
        else:
            # MLPの場合
            v_new = v_new.squeeze(-1)
        
        force_new = -torch.autograd.grad(v_new.sum(), q_new_grad)[0]
    
    p_new = p_half + force_new * (dt / 2)
    
    # 結合
    x_next = torch.cat([q_new, p_new], dim=-1)
    
    return x_next


def monitor_energy_conservation(
    h_func: HamiltonianFunction,
    trajectory: torch.Tensor
) -> Dict[str, float]:
    """
    エネルギー保存則の検証（Requirement 2.6）
    
    Args:
        h_func: ハミルトニアン関数
        trajectory: 軌跡 (B, T, N, 2D)
    
    Returns:
        metrics: エネルギー統計
            - mean_energy: 平均エネルギー
            - energy_drift: エネルギー誤差（相対値）
            - max_drift: 最大エネルギー誤差
    
    物理的直観:
        エネルギー保存則により、H(t) ≈ const
        energy_drift = (E_max - E_min) / E_mean が小さいほど良い
    """
    energies = []
    
    for t in range(trajectory.shape[1]):
        e = h_func(0, trajectory[:, t, :, :])  # (B, N)
        energies.append(e)
    
    energies = torch.stack(energies, dim=1)  # (B, T, N)
    
    # エネルギー統計
    mean_energy = energies.mean()
    max_energy = energies.max(dim=1)[0]  # (B, N)
    min_energy = energies.min(dim=1)[0]  # (B, N)
    
    # エネルギー誤差（相対値）
    energy_drift = (max_energy - min_energy) / (mean_energy.abs() + 1e-8)
    
    return {
        'mean_energy': mean_energy.item(),
        'energy_drift': energy_drift.mean().item(),
        'max_drift': energy_drift.max().item()
    }
