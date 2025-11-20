"""
SNRベースの記憶選択機構のデモ

このスクリプトは、SNRMemoryFilterとMemoryImportanceEstimatorの使用方法を示します。

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from src.models.phase2.memory_selection import SNRMemoryFilter, MemoryImportanceEstimator


def demo_snr_filter():
    """SNRMemoryFilterのデモ"""
    print("=" * 60)
    print("SNRMemoryFilter Demo")
    print("=" * 60)
    
    # フィルターの初期化
    filter = SNRMemoryFilter(
        threshold=2.0,
        gamma_boost=2.0,
        eta_boost=1.5
    )
    filter.train()
    
    # テスト用Fast Weights
    B, H, D = 4, 8, 16
    
    # 異なるSNRレベルの重みを生成
    print("\n1. 異なるSNRレベルの重みを生成...")
    
    # 低SNR（ノイズ優勢）
    low_snr_weights = torch.randn(B, H, D, D) * 0.01
    
    # 中SNR（バランス）
    mid_snr_weights = torch.randn(B, H, D, D) * 0.5
    
    # 高SNR（明確な信号）
    high_snr_weights = torch.randn(B, H, D, D) * 10.0
    
    # 初期パラメータ
    gamma = torch.ones(B) * 0.1
    eta = 0.1
    
    # 各ケースでフィルタリング
    print("\n2. SNRフィルタリングの実行...")
    
    results = []
    for name, weights in [
        ("Low SNR", low_snr_weights),
        ("Mid SNR", mid_snr_weights),
        ("High SNR", high_snr_weights)
    ]:
        adjusted_gamma, adjusted_eta = filter(weights, gamma, eta)
        
        # SNR計算
        sigma_noise = torch.std(weights) + 1e-6
        mean_snr = torch.abs(weights).mean() / sigma_noise
        
        results.append({
            'name': name,
            'mean_snr': mean_snr.item(),
            'gamma_ratio': (adjusted_gamma / gamma).mean().item(),
            'eta_ratio': adjusted_eta / eta,
        })
        
        print(f"\n{name}:")
        print(f"  Mean SNR: {mean_snr.item():.4f}")
        print(f"  Gamma adjustment: {(adjusted_gamma / gamma).mean().item():.2f}x")
        print(f"  Eta adjustment: {adjusted_eta / eta:.2f}x")
    
    # 統計情報の取得
    print("\n3. 統計情報の取得...")
    stats = filter.get_statistics()
    print(f"  Mean SNR (history): {stats['mean_snr']:.4f}")
    print(f"  Std SNR: {stats['std_snr']:.4f}")
    print(f"  Min SNR: {stats['min_snr']:.4f}")
    print(f"  Max SNR: {stats['max_snr']:.4f}")
    
    return results


def demo_importance_estimator():
    """MemoryImportanceEstimatorのデモ"""
    print("\n" + "=" * 60)
    print("MemoryImportanceEstimator Demo")
    print("=" * 60)
    
    # Estimatorの初期化
    estimator = MemoryImportanceEstimator(
        snr_weight=0.5,
        energy_weight=0.3,
        recency_weight=0.2
    )
    
    # テスト用Fast Weights
    B, H, D = 2, 4, 16
    
    print("\n1. 記憶の重要度計算...")
    
    # 重みを生成（一部を重要にする）
    weights = torch.randn(B, H, D, D) * 0.1
    
    # 特定の成分を重要にする
    important_positions = [
        (0, 0, 0, 0),
        (0, 1, 1, 1),
        (1, 0, 2, 2),
    ]
    
    for b, h, i, j in important_positions:
        weights[b, h, i, j] = 10.0
    
    # 重要度計算
    importance = estimator(weights)
    
    print(f"  Importance shape: {importance.shape}")
    print(f"  Importance range: [{importance.min().item():.4f}, {importance.max().item():.4f}]")
    
    # 重要な成分の重要度を確認
    print("\n2. 重要な成分の重要度:")
    for b, h, i, j in important_positions:
        imp = importance[b, h, i, j].item()
        print(f"  Position ({b}, {h}, {i}, {j}): {imp:.4f}")
    
    # 上位k個の記憶を取得
    print("\n3. 上位k個の記憶を取得...")
    k = 10
    top_weights, top_indices = estimator.get_top_k_memories(weights, k)
    
    print(f"  Top {k} weights shape: {top_weights.shape}")
    print(f"  Top {k} indices shape: {top_indices.shape}")
    
    # 重要な成分が上位に含まれているか確認
    print("\n4. 重要な成分が上位に含まれているか確認:")
    for b, h, i, j in important_positions:
        flat_idx = i * D + j
        is_in_top = flat_idx in top_indices[b, h].tolist()
        print(f"  Position ({b}, {h}, {i}, {j}): {'✓ In top-k' if is_in_top else '✗ Not in top-k'}")
    
    return importance, top_weights, top_indices


def demo_hebbian_update_suppression():
    """
    Hebbian更新量の抑制デモ
    
    Requirement 9.4の検証: SNR < 2.0 の信号に対して、
    Hebbian更新量が1/10以下に抑制されること
    """
    print("\n" + "=" * 60)
    print("Hebbian Update Suppression Demo")
    print("=" * 60)
    
    filter = SNRMemoryFilter(threshold=2.0, gamma_boost=10.0)
    
    B, H, D = 2, 4, 8
    
    # 低SNRの重み
    low_snr_weights = torch.randn(B, H, D, D) * 0.01
    
    gamma_original = torch.ones(B) * 0.1
    eta_original = 0.1
    
    adjusted_gamma, adjusted_eta = filter(low_snr_weights, gamma_original, eta_original)
    
    # Hebbian更新量の比較
    # 更新量 ∝ η / Γ
    update_ratio_original = eta_original / gamma_original.mean().item()
    update_ratio_adjusted = adjusted_eta / adjusted_gamma.mean().item()
    
    suppression_ratio = update_ratio_adjusted / update_ratio_original
    
    print(f"\n1. Hebbian更新量の抑制:")
    print(f"  Original update ratio: {update_ratio_original:.4f}")
    print(f"  Adjusted update ratio: {update_ratio_adjusted:.4f}")
    print(f"  Suppression ratio: {suppression_ratio:.4f}")
    print(f"  Target: ≤ 0.1 (1/10以下)")
    
    if suppression_ratio <= 0.1:
        print("  ✓ 抑制目標達成！")
    else:
        print("  ✗ 抑制目標未達成")
    
    return suppression_ratio


def visualize_snr_effects(results):
    """SNR効果の可視化"""
    print("\n" + "=" * 60)
    print("Visualization")
    print("=" * 60)
    
    names = [r['name'] for r in results]
    snrs = [r['mean_snr'] for r in results]
    gamma_ratios = [r['gamma_ratio'] for r in results]
    eta_ratios = [r['eta_ratio'] for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # SNR値
    axes[0].bar(names, snrs, color=['red', 'yellow', 'green'])
    axes[0].axhline(y=2.0, color='black', linestyle='--', label='Threshold')
    axes[0].set_ylabel('Mean SNR')
    axes[0].set_title('Signal-to-Noise Ratio')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Gamma調整
    axes[1].bar(names, gamma_ratios, color=['red', 'yellow', 'green'])
    axes[1].axhline(y=1.0, color='black', linestyle='--', label='No change')
    axes[1].set_ylabel('Gamma Adjustment Ratio')
    axes[1].set_title('Forgetting Rate Adjustment')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Eta調整
    axes[2].bar(names, eta_ratios, color=['red', 'yellow', 'green'])
    axes[2].axhline(y=1.0, color='black', linestyle='--', label='No change')
    axes[2].set_ylabel('Eta Adjustment Ratio')
    axes[2].set_title('Learning Rate Adjustment')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/visualizations/memory_selection_demo.png', dpi=150, bbox_inches='tight')
    print("\n可視化を保存しました: results/visualizations/memory_selection_demo.png")
    plt.close()


if __name__ == "__main__":
    print("SNRベースの記憶選択機構のデモ")
    print("=" * 60)
    
    # SNRフィルターのデモ
    results = demo_snr_filter()
    
    # 重要度推定のデモ
    importance, top_weights, top_indices = demo_importance_estimator()
    
    # Hebbian更新量抑制のデモ
    suppression_ratio = demo_hebbian_update_suppression()
    
    # 可視化
    try:
        visualize_snr_effects(results)
    except Exception as e:
        print(f"\n可視化のスキップ: {e}")
    
    print("\n" + "=" * 60)
    print("デモ完了！")
    print("=" * 60)
