"""
Phase 2 Basic Usage Example

このスクリプトは、Phase 2統合モデルの基本的な使用方法を示します。

主な内容:
1. Phase2IntegratedModelのインスタンス化
2. 簡単なforward pass
3. モデルの統計情報取得
4. 診断情報の取得

Requirements: 11.10
Author: Project MUSE Team
Date: 2025-01-20
"""

import torch
import torch.nn as nn
from src.models.phase2 import Phase2IntegratedModel, create_phase2_model, Phase2Config


def example_1_basic_instantiation():
    """
    例1: 基本的なインスタンス化
    
    Phase2IntegratedModelを最もシンプルな方法で作成します。
    """
    print("=" * 60)
    print("例1: 基本的なインスタンス化")
    print("=" * 60)
    
    # デフォルト設定でモデルを作成
    model = Phase2IntegratedModel(
        vocab_size=1000,  # 小さな語彙サイズ（デモ用）
        d_model=128,      # 小さなモデル次元（デモ用）
        n_layers=2,       # 2層
        n_seq=64,         # シーケンス長64
        num_heads=4,      # 4ヘッド
        head_dim=32,      # ヘッド次元32
    )
    
    print(f"\nモデルが正常に作成されました!")
    print(f"パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"学習可能パラメータ数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


def example_2_forward_pass(model: Phase2IntegratedModel):
    """
    例2: 簡単なforward pass
    
    ランダムな入力でforward passを実行します。
    """
    print("\n" + "=" * 60)
    print("例2: 簡単なforward pass")
    print("=" * 60)
    
    # ランダムな入力を生成
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    print(f"\n入力形状: {input_ids.shape}")
    print(f"入力例: {input_ids[0, :10].tolist()}")
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"\n出力形状: {logits.shape}")
    print(f"期待される形状: (batch_size={batch_size}, seq_len={seq_len}, vocab_size={model.vocab_size})")
    print(f"出力の統計:")
    print(f"  - 平均: {logits.mean().item():.4f}")
    print(f"  - 標準偏差: {logits.std().item():.4f}")
    print(f"  - 最小値: {logits.min().item():.4f}")
    print(f"  - 最大値: {logits.max().item():.4f}")
    
    return logits


def example_3_with_diagnostics(model: Phase2IntegratedModel):
    """
    例3: 診断情報付きforward pass
    
    return_diagnostics=Trueで詳細な診断情報を取得します。
    """
    print("\n" + "=" * 60)
    print("例3: 診断情報付きforward pass")
    print("=" * 60)
    
    # ランダムな入力を生成
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # 診断情報付きforward pass
    with torch.no_grad():
        logits, diagnostics = model(input_ids, return_diagnostics=True)
    
    print(f"\n診断情報のキー: {list(diagnostics.keys())}")
    
    # Γ値（忘却率）の確認
    if diagnostics['gamma_values']:
        gamma_layer_0 = diagnostics['gamma_values'][0]
        print(f"\nLayer 0のΓ値:")
        print(f"  - 形状: {gamma_layer_0.shape}")
        print(f"  - 平均: {gamma_layer_0.mean().item():.6f}")
        print(f"  - 標準偏差: {gamma_layer_0.std().item():.6f}")
        print(f"  - 最小値: {gamma_layer_0.min().item():.6f}")
        print(f"  - 最大値: {gamma_layer_0.max().item():.6f}")
    
    # SNR統計の確認
    if diagnostics['snr_stats']:
        snr_stats_layer_0 = diagnostics['snr_stats'][0]
        if snr_stats_layer_0:
            print(f"\nLayer 0のSNR統計:")
            for key, value in snr_stats_layer_0.items():
                print(f"  - {key}: {value:.4f}")
    
    # 共鳴情報の確認
    if diagnostics['resonance_info']:
        resonance_layer_0 = diagnostics['resonance_info'][0]
        if resonance_layer_0:
            print(f"\nLayer 0の共鳴情報:")
            for key, value in resonance_layer_0.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype == torch.bool:
                        print(f"  - {key}: 形状={value.shape}, True率={value.float().mean().item():.4f}")
                    elif value.numel() == 1:
                        print(f"  - {key}: {value.item():.4f}")
                    else:
                        print(f"  - {key}: 形状={value.shape}, 平均={value.mean().item():.4f}")
                else:
                    print(f"  - {key}: {value}")
    
    # 安定性メトリクスの確認
    if diagnostics['stability_metrics']:
        stability_layer_0 = diagnostics['stability_metrics'][0]
        if stability_layer_0:
            print(f"\nLayer 0の安定性メトリクス:")
            for key, value in stability_layer_0.items():
                print(f"  - {key}: {value}")
    
    return logits, diagnostics


def example_4_model_statistics(model: Phase2IntegratedModel):
    """
    例4: モデルの統計情報取得
    
    モデル全体の統計情報を取得します。
    """
    print("\n" + "=" * 60)
    print("例4: モデルの統計情報取得")
    print("=" * 60)
    
    stats = model.get_statistics()
    
    print(f"\nモデル全体の統計:")
    print(f"  - パラメータ数: {stats['num_parameters']:,}")
    print(f"  - 学習可能パラメータ数: {stats['num_trainable_parameters']:,}")
    print(f"  - レイヤー数: {stats['num_layers']}")
    print(f"  - モデル次元: {stats['d_model']}")
    print(f"  - 語彙サイズ: {stats['vocab_size']}")
    print(f"  - シーケンス長: {stats['n_seq']}")
    
    # 各ブロックの統計
    print(f"\n各ブロックの統計:")
    for block_stat in stats['block_stats']:
        layer_idx = block_stat['layer']
        print(f"\n  Layer {layer_idx}:")
        
        # Hebbian統計
        if 'hebbian' in block_stat:
            hebbian_stats = block_stat['hebbian']
            print(f"    Hebbian:")
            for key, value in hebbian_stats.items():
                print(f"      - {key}: {value:.6f}")
        
        # SNR統計
        if 'snr' in block_stat:
            snr_stats = block_stat['snr']
            if snr_stats:
                print(f"    SNR:")
                for key, value in snr_stats.items():
                    print(f"      - {key}: {value:.6f}")
        
        # Non-Hermitian統計
        if 'non_hermitian' in block_stat:
            nh_stats = block_stat['non_hermitian']
            print(f"    Non-Hermitian:")
            for key, value in nh_stats.items():
                print(f"      - {key}: {value:.6f}")


def example_5_factory_function():
    """
    例5: ファクトリ関数を使用したモデル作成
    
    create_phase2_model()を使用して、より簡単にモデルを作成します。
    """
    print("\n" + "=" * 60)
    print("例5: ファクトリ関数を使用したモデル作成")
    print("=" * 60)
    
    # プリセット設定を使用
    print("\n5-1: プリセット 'small' を使用")
    model_small = create_phase2_model(preset="small")
    
    # カスタム設定を使用
    print("\n5-2: カスタム設定を使用")
    config = Phase2Config(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_seq=64,
        num_heads=4,
        head_dim=32,
        base_decay=0.02,  # カスタム減衰率
        hebbian_eta=0.15,  # カスタムHebbian学習率
    )
    model_custom = create_phase2_model(config=config)
    
    # パラメータ直接指定
    print("\n5-3: パラメータ直接指定")
    model_direct = create_phase2_model(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_seq=64,
    )
    
    return model_small, model_custom, model_direct


def example_6_state_management(model: Phase2IntegratedModel):
    """
    例6: Fast Weight状態の管理
    
    Fast Weight状態のリセットと管理方法を示します。
    """
    print("\n" + "=" * 60)
    print("例6: Fast Weight状態の管理")
    print("=" * 60)
    
    # 最初のシーケンス
    input_ids_1 = torch.randint(0, 1000, (1, 16))
    print("\n最初のシーケンスを処理...")
    with torch.no_grad():
        logits_1 = model(input_ids_1)
    print(f"出力形状: {logits_1.shape}")
    
    # Fast Weight状態を確認
    print("\nFast Weight状態:")
    for i, block in enumerate(model.blocks):
        if block.fast_weight_state is not None:
            print(f"  Layer {i}: 形状={block.fast_weight_state.shape}, "
                  f"ノルム={torch.norm(block.fast_weight_state).item():.4f}")
        else:
            print(f"  Layer {i}: None")
    
    # 2番目のシーケンス（状態を保持）
    input_ids_2 = torch.randint(0, 1000, (1, 16))
    print("\n2番目のシーケンスを処理（状態を保持）...")
    with torch.no_grad():
        logits_2 = model(input_ids_2)
    print(f"出力形状: {logits_2.shape}")
    
    # Fast Weight状態を確認
    print("\nFast Weight状態（更新後）:")
    for i, block in enumerate(model.blocks):
        if block.fast_weight_state is not None:
            print(f"  Layer {i}: 形状={block.fast_weight_state.shape}, "
                  f"ノルム={torch.norm(block.fast_weight_state).item():.4f}")
        else:
            print(f"  Layer {i}: None")
    
    # 状態をリセット
    print("\n状態をリセット...")
    model.reset_state()
    
    # リセット後の状態を確認
    print("\nFast Weight状態（リセット後）:")
    for i, block in enumerate(model.blocks):
        if block.fast_weight_state is not None:
            print(f"  Layer {i}: 形状={block.fast_weight_state.shape}, "
                  f"ノルム={torch.norm(block.fast_weight_state).item():.4f}")
        else:
            print(f"  Layer {i}: None")


def main():
    """メイン関数"""
    print("\n" + "=" * 60)
    print("Phase 2 Basic Usage Examples")
    print("=" * 60)
    
    # 例1: 基本的なインスタンス化
    model = example_1_basic_instantiation()
    
    # 例2: 簡単なforward pass
    logits = example_2_forward_pass(model)
    
    # 例3: 診断情報付きforward pass
    logits, diagnostics = example_3_with_diagnostics(model)
    
    # 例4: モデルの統計情報取得
    example_4_model_statistics(model)
    
    # 例5: ファクトリ関数を使用したモデル作成
    model_small, model_custom, model_direct = example_5_factory_function()
    
    # 例6: Fast Weight状態の管理
    example_6_state_management(model)
    
    print("\n" + "=" * 60)
    print("すべての例が正常に完了しました!")
    print("=" * 60)


if __name__ == "__main__":
    # シード設定（再現性のため）
    torch.manual_seed(42)
    
    # メイン実行
    main()
