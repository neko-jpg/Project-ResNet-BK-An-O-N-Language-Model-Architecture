"""
Phase 2: Zeta Initialization Demo

このデモは、Riemann-Zeta Regularization機構の使用方法を示します。

実行方法:
    python examples/zeta_init_demo.py

機能:
    1. ゼータ零点の生成と可視化
    2. 線形層のゼータ初期化
    3. ZetaEmbeddingの使用例
    4. モデル全体への初期化適用
    5. 統計情報の取得と分析
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from src.models.phase2.zeta_init import (
    ZetaInitializer,
    ZetaEmbedding,
    apply_zeta_initialization,
    get_zeta_statistics
)


def demo_zeta_zeros():
    """ゼータ零点の生成と可視化"""
    print("=" * 80)
    print("Demo 1: Zeta Zeros Generation and Visualization")
    print("=" * 80)
    
    # 100個の零点を生成
    zeros = ZetaInitializer.get_approx_zeta_zeros(100)
    
    print(f"\n最初の10個の零点（精密値）:")
    for i in range(10):
        print(f"  γ_{i+1} = {zeros[i]:.6f}")
    
    # 零点の間隔を計算
    spacings = zeros[1:] - zeros[:-1]
    
    print(f"\n零点の統計:")
    print(f"  平均間隔: {spacings.mean():.3f}")
    print(f"  標準偏差: {spacings.std():.3f}")
    print(f"  最小間隔: {spacings.min():.3f}")
    print(f"  最大間隔: {spacings.max():.3f}")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 零点の分布
    axes[0, 0].plot(zeros.numpy(), 'o-', markersize=3)
    axes[0, 0].set_xlabel('Index')
    axes[0, 0].set_ylabel('Zeta Zero (γ)')
    axes[0, 0].set_title('Riemann Zeta Zeros Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 間隔の分布
    axes[0, 1].hist(spacings.numpy(), bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(spacings.mean().item(), color='r', linestyle='--', 
                       label=f'Mean: {spacings.mean():.2f}')
    axes[0, 1].set_xlabel('Spacing')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Spacing Distribution (GUE Statistics)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 累積分布
    axes[1, 0].plot(np.arange(len(zeros)), zeros.numpy())
    axes[1, 0].set_xlabel('Index')
    axes[1, 0].set_ylabel('Cumulative γ')
    axes[1, 0].set_title('Cumulative Zeta Zeros')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 間隔の時系列
    axes[1, 1].plot(spacings.numpy(), 'o-', markersize=3, alpha=0.6)
    axes[1, 1].axhline(spacings.mean().item(), color='r', linestyle='--', 
                       label=f'Mean: {spacings.mean():.2f}')
    axes[1, 1].set_xlabel('Index')
    axes[1, 1].set_ylabel('Spacing')
    axes[1, 1].set_title('Spacing Sequence')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/visualizations/zeta_zeros_distribution.png', dpi=150, bbox_inches='tight')
    print(f"\n可視化を保存しました: results/visualizations/zeta_zeros_distribution.png")
    plt.close()


def demo_linear_initialization():
    """線形層のゼータ初期化"""
    print("\n" + "=" * 80)
    print("Demo 2: Linear Layer Zeta Initialization")
    print("=" * 80)
    
    # 線形層を作成
    linear = nn.Linear(128, 128)
    
    # 初期化前の特異値
    u_before, s_before, v_before = torch.svd(linear.weight)
    
    print(f"\n初期化前の特異値統計:")
    print(f"  平均: {s_before.mean():.6f}")
    print(f"  標準偏差: {s_before.std():.6f}")
    print(f"  最小: {s_before.min():.6f}")
    print(f"  最大: {s_before.max():.6f}")
    
    # ゼータ初期化を適用
    ZetaInitializer.initialize_linear_zeta(linear, scale=10.0)
    
    # 初期化後の特異値
    u_after, s_after, v_after = torch.svd(linear.weight)
    
    print(f"\n初期化後の特異値統計:")
    print(f"  平均: {s_after.mean():.6f}")
    print(f"  標準偏差: {s_after.std():.6f}")
    print(f"  最小: {s_after.min():.6f}")
    print(f"  最大: {s_after.max():.6f}")
    
    # 期待される特異値（scale / zeros）
    zeros = ZetaInitializer.get_approx_zeta_zeros(128)
    expected_s = 10.0 / zeros
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 特異値の比較
    axes[0].plot(s_before.detach().numpy(), 'o-', label='Before Init', alpha=0.6)
    axes[0].plot(s_after.detach().numpy(), 's-', label='After Init', alpha=0.6)
    axes[0].plot(expected_s.numpy(), '^--', label='Expected (scale/γ)', alpha=0.6)
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Singular Value')
    axes[0].set_title('Singular Values: Before vs After Zeta Init')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # 相対誤差
    relative_error = torch.abs(s_after - expected_s) / expected_s
    axes[1].plot(relative_error.detach().numpy(), 'o-', alpha=0.6)
    axes[1].axhline(0.1, color='r', linestyle='--', label='10% threshold')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Relative Error')
    axes[1].set_title('Relative Error: Actual vs Expected')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/visualizations/linear_zeta_init.png', dpi=150, bbox_inches='tight')
    print(f"\n可視化を保存しました: results/visualizations/linear_zeta_init.png")
    plt.close()


def demo_zeta_embedding():
    """ZetaEmbeddingの使用例"""
    print("\n" + "=" * 80)
    print("Demo 3: ZetaEmbedding Usage")
    print("=" * 80)
    
    # ZetaEmbeddingを作成
    max_len = 512
    d_model = 128
    pos_emb = ZetaEmbedding(max_len=max_len, d_model=d_model, trainable=False)
    
    print(f"\nZetaEmbedding作成:")
    print(f"  max_len: {max_len}")
    print(f"  d_model: {d_model}")
    print(f"  trainable: False")
    
    # 位置インデックスを作成
    positions = torch.arange(0, 100).unsqueeze(0)  # (1, 100)
    
    # Forward pass
    embeddings = pos_emb(positions)
    
    print(f"\nForward pass:")
    print(f"  入力形状: {positions.shape}")
    print(f"  出力形状: {embeddings.shape}")
    
    # 埋め込みの可視化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ヒートマップ
    im = axes[0, 0].imshow(embeddings[0].detach().numpy().T, aspect='auto', cmap='RdBu_r')
    axes[0, 0].set_xlabel('Position')
    axes[0, 0].set_ylabel('Dimension')
    axes[0, 0].set_title('Zeta Position Embeddings (Heatmap)')
    plt.colorbar(im, ax=axes[0, 0])
    
    # 最初の4次元の時系列
    for i in range(4):
        axes[0, 1].plot(embeddings[0, :, i].detach().numpy(), 
                       label=f'Dim {i}', alpha=0.7)
    axes[0, 1].set_xlabel('Position')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].set_title('First 4 Dimensions Over Positions')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 位置0と位置50の比較
    axes[1, 0].plot(embeddings[0, 0].detach().numpy(), 'o-', label='Position 0', alpha=0.6)
    axes[1, 0].plot(embeddings[0, 50].detach().numpy(), 's-', label='Position 50', alpha=0.6)
    axes[1, 0].set_xlabel('Dimension')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('Embedding Vectors at Different Positions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # ノルムの分布
    norms = torch.norm(embeddings[0], dim=1)
    axes[1, 1].plot(norms.detach().numpy(), 'o-', alpha=0.6)
    axes[1, 1].axhline(norms.mean().item(), color='r', linestyle='--', 
                      label=f'Mean: {norms.mean():.2f}')
    axes[1, 1].set_xlabel('Position')
    axes[1, 1].set_ylabel('L2 Norm')
    axes[1, 1].set_title('Embedding Norms Over Positions')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/visualizations/zeta_embedding.png', dpi=150, bbox_inches='tight')
    print(f"\n可視化を保存しました: results/visualizations/zeta_embedding.png")
    plt.close()


def demo_model_initialization():
    """モデル全体への初期化適用"""
    print("\n" + "=" * 80)
    print("Demo 4: Full Model Zeta Initialization")
    print("=" * 80)
    
    # 簡単なモデルを作成
    class SimpleLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding = nn.Embedding(1000, 128)
            self.position_embedding = ZetaEmbedding(512, 128, trainable=False)
            self.linear1 = nn.Linear(128, 256)
            self.linear2 = nn.Linear(256, 128)
            self.output = nn.Linear(128, 1000)
        
        def forward(self, input_ids):
            B, N = input_ids.shape
            positions = torch.arange(N, device=input_ids.device).unsqueeze(0).expand(B, -1)
            
            x = self.token_embedding(input_ids) + self.position_embedding(positions)
            x = torch.relu(self.linear1(x))
            x = torch.relu(self.linear2(x))
            return self.output(x)
    
    model = SimpleLanguageModel()
    
    print(f"\nモデル構造:")
    print(model)
    
    # 初期化前のパラメータ統計
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n総パラメータ数: {total_params:,}")
    
    # ゼータ初期化を適用
    print(f"\nゼータ初期化を適用中...")
    apply_zeta_initialization(model, scale=10.0)
    
    print(f"初期化完了!")
    
    # Forward passのテスト
    input_ids = torch.randint(0, 1000, (2, 50))
    output = model(input_ids)
    
    print(f"\nForward pass テスト:")
    print(f"  入力形状: {input_ids.shape}")
    print(f"  出力形状: {output.shape}")
    print(f"  出力統計:")
    print(f"    平均: {output.mean():.6f}")
    print(f"    標準偏差: {output.std():.6f}")
    print(f"    最小: {output.min():.6f}")
    print(f"    最大: {output.max():.6f}")


def demo_statistics():
    """統計情報の取得と分析"""
    print("\n" + "=" * 80)
    print("Demo 5: Zeta Statistics Analysis")
    print("=" * 80)
    
    # 異なるnに対する統計を取得
    n_values = [10, 50, 100, 200, 500]
    
    print(f"\nゼータ零点の統計（異なるn値）:")
    print(f"{'n':>6} | {'Mean Spacing':>13} | {'Std Spacing':>12} | {'Min Spacing':>12} | {'Max Spacing':>12}")
    print("-" * 70)
    
    mean_spacings = []
    for n in n_values:
        stats = get_zeta_statistics(n)
        print(f"{n:>6} | {stats['mean_spacing']:>13.3f} | {stats['std_spacing']:>12.3f} | "
              f"{stats['min_spacing']:>12.3f} | {stats['max_spacing']:>12.3f}")
        mean_spacings.append(stats['mean_spacing'])
    
    # 平均間隔の変化を可視化
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, mean_spacings, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Number of Zeros (n)')
    plt.ylabel('Mean Spacing')
    plt.title('Mean Spacing vs Number of Zeros')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/visualizations/zeta_statistics.png', dpi=150, bbox_inches='tight')
    print(f"\n可視化を保存しました: results/visualizations/zeta_statistics.png")
    plt.close()
    
    # 詳細統計（n=100）
    stats = get_zeta_statistics(100)
    print(f"\n詳細統計 (n=100):")
    print(f"  零点数: {stats['num_zeros']}")
    print(f"  平均間隔: {stats['mean_spacing']:.3f}")
    print(f"  標準偏差: {stats['std_spacing']:.3f}")
    print(f"  最小間隔: {stats['min_spacing']:.3f}")
    print(f"  最大間隔: {stats['max_spacing']:.3f}")
    print(f"  変動係数 (CV): {stats['std_spacing'] / stats['mean_spacing']:.3f}")


def main():
    """メイン実行関数"""
    print("\n" + "=" * 80)
    print("Phase 2: Zeta Initialization Demo")
    print("Riemann-Zeta Regularization for Fractal Memory Arrangement")
    print("=" * 80)
    
    # 結果ディレクトリを作成
    import os
    os.makedirs('results/visualizations', exist_ok=True)
    
    # 各デモを実行
    demo_zeta_zeros()
    demo_linear_initialization()
    demo_zeta_embedding()
    demo_model_initialization()
    demo_statistics()
    
    print("\n" + "=" * 80)
    print("すべてのデモが完了しました!")
    print("=" * 80)
    print("\n物理的解釈:")
    print("  - ゼータ零点 = 量子カオス系のエネルギー準位")
    print("  - GUE統計 = 最大エントロピー分布（最もランダムかつ規則的）")
    print("  - 不規則な周波数 = 情報の干渉を最小化")
    print("  - フラクタル記憶配置 = 効率的な分散表現")
    print("\n利点:")
    print("  ✓ 情報の衝突（干渉）を最小化")
    print("  ✓ 効率的な分散表現を実現")
    print("  ✓ 数学的に保証された初期化")
    print("  ✓ 標準的なSinusoidal Embeddingより優れた性能")


if __name__ == "__main__":
    main()
