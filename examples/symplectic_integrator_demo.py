"""
Symplectic Integrator Demo for Phase 3

このデモは、シンプレクティック積分器の動作を実演します。

実演内容:
1. Leapfrog法による積分
2. エネルギー保存の検証
3. 長時間積分の安定性
4. MLPとBK-Coreの比較

Requirements: 2.5, 2.6, 2.7
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import matplotlib.pyplot as plt

from src.models.phase3.hamiltonian import (
    HamiltonianFunction,
    symplectic_leapfrog_step,
    monitor_energy_conservation
)


def demo_basic_integration():
    """基本的な積分のデモ"""
    print("=" * 60)
    print("Demo 1: Basic Symplectic Integration")
    print("=" * 60)
    
    # ハミルトニアン関数の作成
    h_func = HamiltonianFunction(d_model=32, potential_type='mlp')
    
    # 初期状態
    x0 = torch.randn(1, 5, 64)  # (B=1, N=5, 2D=64)
    dt = 0.1
    n_steps = 10
    
    print(f"\nInitial state shape: {x0.shape}")
    print(f"Time step: {dt}")
    print(f"Number of steps: {n_steps}")
    
    # 積分
    trajectory = [x0]
    x = x0
    
    for step in range(n_steps):
        x = symplectic_leapfrog_step(h_func, x, dt)
        trajectory.append(x)
        
        # エネルギーを計算
        energy = h_func(0, x).mean()
        print(f"Step {step+1}: Energy = {energy:.4f}")
    
    trajectory = torch.stack(trajectory, dim=1)  # (B, T+1, N, 2D)
    
    # エネルギー保存を監視
    metrics = monitor_energy_conservation(h_func, trajectory)
    
    print(f"\nEnergy Conservation Metrics:")
    print(f"  Mean energy: {metrics['mean_energy']:.4f}")
    print(f"  Energy drift: {metrics['energy_drift']:.2e}")
    print(f"  Max drift: {metrics['max_drift']:.2e}")
    
    if metrics['max_drift'] < 1e-4:
        print("  ✅ Energy conservation: EXCELLENT")
    elif metrics['max_drift'] < 1e-3:
        print("  ✅ Energy conservation: GOOD")
    else:
        print("  ⚠️ Energy conservation: NEEDS IMPROVEMENT")


def demo_long_time_integration():
    """長時間積分のデモ"""
    print("\n" + "=" * 60)
    print("Demo 2: Long-Time Integration (100 steps)")
    print("=" * 60)
    
    # ハミルトニアン関数の作成
    h_func = HamiltonianFunction(d_model=32, potential_type='mlp')
    
    # 初期状態
    x0 = torch.randn(1, 5, 64)
    dt = 0.1
    n_steps = 100
    
    print(f"\nIntegrating for {n_steps} steps...")
    
    # 積分
    trajectory = [x0]
    energies = []
    x = x0
    
    for step in range(n_steps):
        x = symplectic_leapfrog_step(h_func, x, dt)
        trajectory.append(x)
        
        # エネルギーを記録
        energy = h_func(0, x).mean().item()
        energies.append(energy)
        
        if (step + 1) % 20 == 0:
            print(f"Step {step+1}: Energy = {energy:.4f}")
    
    trajectory = torch.stack(trajectory, dim=1)
    
    # エネルギー保存を監視
    metrics = monitor_energy_conservation(h_func, trajectory)
    
    print(f"\nEnergy Conservation Metrics (100 steps):")
    print(f"  Mean energy: {metrics['mean_energy']:.4f}")
    print(f"  Energy drift: {metrics['energy_drift']:.2e}")
    print(f"  Max drift: {metrics['max_drift']:.2e}")
    
    # エネルギーの時間変化をプロット
    plt.figure(figsize=(10, 6))
    plt.plot(energies, label='Energy')
    plt.axhline(y=metrics['mean_energy'], color='r', linestyle='--', label='Mean Energy')
    plt.xlabel('Step')
    plt.ylabel('Energy')
    plt.title('Energy Conservation over 100 Steps')
    plt.legend()
    plt.grid(True)
    
    # 保存
    output_dir = Path('results/phase3_demo_visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'symplectic_integrator_energy.png', dpi=150, bbox_inches='tight')
    print(f"\nEnergy plot saved to: {output_dir / 'symplectic_integrator_energy.png'}")
    plt.close()


def demo_potential_comparison():
    """MLPとBK-Coreの比較デモ"""
    print("\n" + "=" * 60)
    print("Demo 3: Potential Comparison (MLP vs BK-Core)")
    print("=" * 60)
    
    potentials = ['mlp']
    
    # BK-Coreが利用可能か確認
    try:
        from src.models.bk_core import BKCore
        potentials.append('bk_core')
        print("\nBK-Core is available. Comparing both potentials.")
    except ImportError:
        print("\nBK-Core not available. Testing MLP only.")
    
    results = {}
    
    for pot_type in potentials:
        print(f"\n--- Testing {pot_type.upper()} potential ---")
        
        # ハミルトニアン関数の作成
        h_func = HamiltonianFunction(d_model=32, potential_type=pot_type)
        
        # 初期状態
        x0 = torch.randn(1, 5, 64)
        dt = 0.1
        n_steps = 50
        
        # 積分
        trajectory = [x0]
        x = x0
        
        for _ in range(n_steps):
            x = symplectic_leapfrog_step(h_func, x, dt)
            trajectory.append(x)
        
        trajectory = torch.stack(trajectory, dim=1)
        
        # エネルギー保存を監視
        metrics = monitor_energy_conservation(h_func, trajectory)
        results[pot_type] = metrics
        
        print(f"  Mean energy: {metrics['mean_energy']:.4f}")
        print(f"  Energy drift: {metrics['energy_drift']:.2e}")
        print(f"  Max drift: {metrics['max_drift']:.2e}")
    
    # 比較結果
    print("\n" + "=" * 60)
    print("Comparison Summary:")
    print("=" * 60)
    
    for pot_type, metrics in results.items():
        status = "✅" if metrics['max_drift'] < 1e-4 else "⚠️"
        print(f"{status} {pot_type.upper()}: Energy drift = {metrics['energy_drift']:.2e}")


def demo_phase_space_trajectory():
    """位相空間軌跡のデモ"""
    print("\n" + "=" * 60)
    print("Demo 4: Phase Space Trajectory")
    print("=" * 60)
    
    # ハミルトニアン関数の作成
    h_func = HamiltonianFunction(d_model=2, potential_type='mlp')
    
    # 初期状態（2次元で可視化）
    x0 = torch.randn(1, 1, 4)  # (B=1, N=1, 2D=4) -> q=(2,), p=(2,)
    dt = 0.05
    n_steps = 200
    
    print(f"\nIntegrating 2D system for {n_steps} steps...")
    
    # 積分
    trajectory = []
    x = x0
    
    for _ in range(n_steps):
        x = symplectic_leapfrog_step(h_func, x, dt)
        trajectory.append(x)
    
    trajectory = torch.stack(trajectory, dim=0)  # (T, B, N, 2D)
    
    # 位置と運動量を抽出
    q = trajectory[:, 0, 0, :2].detach().numpy()  # (T, 2)
    p = trajectory[:, 0, 0, 2:].detach().numpy()  # (T, 2)
    
    # 位相空間プロット
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # q-p平面（第1成分）
    axes[0].plot(q[:, 0], p[:, 0], 'b-', alpha=0.7, linewidth=1)
    axes[0].scatter(q[0, 0], p[0, 0], c='g', s=100, marker='o', label='Start', zorder=5)
    axes[0].scatter(q[-1, 0], p[-1, 0], c='r', s=100, marker='x', label='End', zorder=5)
    axes[0].set_xlabel('Position q₁')
    axes[0].set_ylabel('Momentum p₁')
    axes[0].set_title('Phase Space Trajectory (Component 1)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # q-p平面（第2成分）
    axes[1].plot(q[:, 1], p[:, 1], 'b-', alpha=0.7, linewidth=1)
    axes[1].scatter(q[0, 1], p[0, 1], c='g', s=100, marker='o', label='Start', zorder=5)
    axes[1].scatter(q[-1, 1], p[-1, 1], c='r', s=100, marker='x', label='End', zorder=5)
    axes[1].set_xlabel('Position q₂')
    axes[1].set_ylabel('Momentum p₂')
    axes[1].set_title('Phase Space Trajectory (Component 2)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_dir = Path('results/phase3_demo_visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'symplectic_integrator_phase_space.png', dpi=150, bbox_inches='tight')
    print(f"\nPhase space plot saved to: {output_dir / 'symplectic_integrator_phase_space.png'}")
    plt.close()


def main():
    """メインデモ"""
    print("\n" + "=" * 60)
    print("Symplectic Integrator Demo - Phase 3")
    print("=" * 60)
    
    # デモ1: 基本的な積分
    demo_basic_integration()
    
    # デモ2: 長時間積分
    demo_long_time_integration()
    
    # デモ3: ポテンシャル比較
    demo_potential_comparison()
    
    # デモ4: 位相空間軌跡
    demo_phase_space_trajectory()
    
    print("\n" + "=" * 60)
    print("All demos completed successfully! ✅")
    print("=" * 60)


if __name__ == '__main__':
    main()
