"""
Memory Resonance Layer デモ

Phase 2: Breath of Life

このデモでは、Memory Resonance Layerの使用方法を示します:
1. ZetaBasisTransformによるゼータ零点基底の生成
2. MemoryResonanceLayerによる記憶の対角化とフィルタリング
3. 共鳴情報の可視化
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

from src.models.phase2.memory_resonance import (
    MemoryResonanceLayer,
    ZetaBasisTransform,
    MemoryImportanceEstimator,
)


def demo_zeta_basis_transform():
    """ZetaBasisTransformのデモ"""
    print("=" * 60)
    print("ZetaBasisTransform デモ")
    print("=" * 60)
    
    zeta = ZetaBasisTransform()
    
    # ゼータ零点の取得
    print("\n1. ゼータ零点の取得")
    zeros_10 = zeta.get_zeta_zeros(10)
    print(f"最初の10個の零点: {zeros_10}")
    
    zeros_50 = zeta.get_zeta_zeros(50)
    print(f"\n50個の零点（最初の10個 + GUE統計ベース）:")
    print(f"  平均間隔: {(zeros_50[1:] - zeros_50[:-1]).mean():.3f}")
    print(f"  標準偏差: {(zeros_50[1:] - zeros_50[:-1]).std():.3f}")
    
    # 基底行列の生成
    print("\n2. 基底行列の生成")
    dim = 32
    U = zeta.get_basis_matrix(dim, torch.device('cpu'))
    print(f"基底行列の形状: {U.shape}")
    print(f"基底行列のデータ型: {U.dtype}")
    
    # 逆行列の計算
    U_inv = torch.linalg.inv(U)
    identity = torch.mm(U, U_inv)
    error = torch.norm(identity - torch.eye(dim, dtype=torch.complex64)).item()
    print(f"逆行列の誤差: {error:.6f}")
    
    # キャッシュの確認
    print("\n3. キャッシュ機構")
    print(f"キャッシュされた零点の数: {len(zeta._zeta_zeros_cache)}")
    print(f"キャッシュされた基底行列の数: {len(zeta._basis_cache)}")


def demo_memory_resonance_layer():
    """MemoryResonanceLayerのデモ"""
    print("\n" + "=" * 60)
    print("MemoryResonanceLayer デモ")
    print("=" * 60)
    
    # パラメータ
    B, H, D_h = 2, 4, 32
    N, D = 16, 256
    
    # レイヤーの作成
    layer = MemoryResonanceLayer(
        d_model=D,
        head_dim=D_h,
        num_heads=H,
        energy_threshold=0.1,
    )
    
    print(f"\nレイヤー設定:")
    print(f"  モデル次元: {D}")
    print(f"  ヘッド次元: {D_h}")
    print(f"  ヘッド数: {H}")
    print(f"  エネルギー閾値: {layer.energy_threshold}")
    
    # Fast Weightsの生成（ランダム）
    weights = torch.randn(B, H, D_h, D_h, dtype=torch.complex64)
    x = torch.randn(B, N, D)
    
    print(f"\n入力:")
    print(f"  Fast Weights形状: {weights.shape}")
    print(f"  入力形状: {x.shape}")
    print(f"  元の重みのノルム: {torch.norm(weights).item():.4f}")
    
    # Forward pass
    filtered_weights, resonance_info = layer(weights, x)
    
    print(f"\n出力:")
    print(f"  フィルタ後の重み形状: {filtered_weights.shape}")
    print(f"  フィルタ後の重みのノルム: {torch.norm(filtered_weights).item():.4f}")
    
    print(f"\n共鳴情報:")
    print(f"  平均共鳴成分数: {resonance_info['num_resonant']:.1f} / {D_h}")
    print(f"  総エネルギー: {resonance_info['total_energy']:.4f}")
    print(f"  スパース率: {resonance_info['sparsity_ratio']:.2%}")
    
    return resonance_info


def demo_resonance_visualization(resonance_info):
    """共鳴情報の可視化"""
    print("\n" + "=" * 60)
    print("共鳴情報の可視化")
    print("=" * 60)
    
    # 対角エネルギーの取得
    diag_energy = resonance_info['diag_energy'].cpu().numpy()  # (B, H, D_h)
    resonance_mask = resonance_info['resonance_mask'].cpu().numpy()
    
    B, H, D_h = diag_energy.shape
    
    # プロット
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 対角エネルギーのヒートマップ（最初のバッチ）
    ax = axes[0, 0]
    im = ax.imshow(diag_energy[0], aspect='auto', cmap='viridis')
    ax.set_xlabel('Mode Index')
    ax.set_ylabel('Head Index')
    ax.set_title('Diagonal Energy (Batch 0)')
    plt.colorbar(im, ax=ax)
    
    # 2. 共鳴マスクのヒートマップ
    ax = axes[0, 1]
    im = ax.imshow(resonance_mask[0].astype(float), aspect='auto', cmap='RdYlGn')
    ax.set_xlabel('Mode Index')
    ax.set_ylabel('Head Index')
    ax.set_title('Resonance Mask (Batch 0)')
    plt.colorbar(im, ax=ax)
    
    # 3. エネルギー分布のヒストグラム
    ax = axes[1, 0]
    energies_flat = diag_energy.flatten()
    ax.hist(energies_flat, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0.1, color='r', linestyle='--', label='Threshold')
    ax.set_xlabel('Energy')
    ax.set_ylabel('Frequency')
    ax.set_title('Energy Distribution')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. ヘッドごとの共鳴成分数
    ax = axes[1, 1]
    num_resonant_per_head = resonance_mask.sum(axis=2).mean(axis=0)  # (H,)
    ax.bar(range(H), num_resonant_per_head)
    ax.set_xlabel('Head Index')
    ax.set_ylabel('Number of Resonant Modes')
    ax.set_title('Resonant Modes per Head')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/visualizations/memory_resonance_demo.png', dpi=150)
    print("\n可視化を保存しました: results/visualizations/memory_resonance_demo.png")
    plt.close()


def demo_memory_importance_estimator():
    """MemoryImportanceEstimatorのデモ"""
    print("\n" + "=" * 60)
    print("MemoryImportanceEstimator デモ")
    print("=" * 60)
    
    B, H, D_h = 2, 4, 32
    
    estimator = MemoryImportanceEstimator(head_dim=D_h, num_heads=H)
    
    # 共鳴エネルギーとSNRを生成
    resonance_energy = torch.rand(B, H, D_h) * 2.0
    snr = torch.rand(B, H, D_h) * 5.0
    
    print(f"\n入力:")
    print(f"  共鳴エネルギー範囲: [{resonance_energy.min():.3f}, {resonance_energy.max():.3f}]")
    print(f"  SNR範囲: [{snr.min():.3f}, {snr.max():.3f}]")
    
    # 重要度の計算
    importance = estimator(resonance_energy, snr)
    
    print(f"\n出力:")
    print(f"  重要度スコア範囲: [{importance.min():.3f}, {importance.max():.3f}]")
    print(f"  平均重要度: {importance.mean():.3f}")
    
    # 重要度の高い成分を抽出
    threshold = 0.7
    important_mask = importance > threshold
    num_important = important_mask.sum().item()
    total = B * H * D_h
    
    print(f"\n重要記憶の選択（閾値={threshold}）:")
    print(f"  重要な成分数: {num_important} / {total} ({num_important/total:.1%})")


def demo_performance_benchmark():
    """性能ベンチマークのデモ"""
    print("\n" + "=" * 60)
    print("性能ベンチマーク")
    print("=" * 60)
    
    import time
    
    # 様々なサイズでベンチマーク
    sizes = [
        (2, 4, 32, 16, 256),   # Small
        (4, 8, 64, 32, 512),   # Medium
        (8, 8, 64, 64, 512),   # Large
    ]
    
    print(f"\n{'Size':<20} {'Time (ms)':<15} {'Memory (MB)':<15}")
    print("-" * 50)
    
    for B, H, D_h, N, D in sizes:
        layer = MemoryResonanceLayer(
            d_model=D,
            head_dim=D_h,
            num_heads=H,
            energy_threshold=0.1,
        )
        
        weights = torch.randn(B, H, D_h, D_h, dtype=torch.complex64)
        x = torch.randn(B, N, D)
        
        # ウォームアップ
        for _ in range(5):
            layer(weights, x)
        
        # 計測
        num_runs = 20
        start = time.time()
        for _ in range(num_runs):
            filtered_weights, resonance_info = layer(weights, x)
        end = time.time()
        
        avg_time_ms = (end - start) / num_runs * 1000
        
        # メモリ使用量の推定
        memory_mb = (weights.numel() * 8 + x.numel() * 4) / (1024 * 1024)
        
        size_str = f"B={B}, H={H}, D_h={D_h}"
        print(f"{size_str:<20} {avg_time_ms:<15.2f} {memory_mb:<15.2f}")


def main():
    """メインデモ"""
    print("\n" + "=" * 60)
    print("Memory Resonance Layer デモ")
    print("Phase 2: Breath of Life")
    print("=" * 60)
    
    # 1. ZetaBasisTransformのデモ
    demo_zeta_basis_transform()
    
    # 2. MemoryResonanceLayerのデモ
    resonance_info = demo_memory_resonance_layer()
    
    # 3. 共鳴情報の可視化
    try:
        demo_resonance_visualization(resonance_info)
    except Exception as e:
        print(f"\n可視化のスキップ: {e}")
    
    # 4. MemoryImportanceEstimatorのデモ
    demo_memory_importance_estimator()
    
    # 5. 性能ベンチマーク
    demo_performance_benchmark()
    
    print("\n" + "=" * 60)
    print("デモ完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
