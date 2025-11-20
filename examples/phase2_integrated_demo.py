"""
Phase 2 Integrated Model - デモスクリプト

このスクリプトは、Phase2IntegratedModelの基本的な使用方法を示します:
1. モデルのインスタンス化
2. Forward passの実行
3. 診断情報の取得
4. 統計情報の表示

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
"""

import torch
import torch.nn as nn
from src.models.phase2.integrated_model import Phase2IntegratedModel


def demo_basic_usage():
    """基本的な使用例"""
    print("=" * 60)
    print("Phase2IntegratedModel - 基本的な使用例")
    print("=" * 60)
    
    # モデルの設定
    config = {
        'vocab_size': 1000,
        'd_model': 256,
        'n_layers': 4,
        'n_seq': 128,
        'num_heads': 8,
        'head_dim': 32,
        'use_triton': False,  # デモ環境ではTritonを無効化
        'ffn_dim': 1024,
        'dropout': 0.1,
        'zeta_embedding_trainable': False,
    }
    
    # モデルのインスタンス化
    print("\n1. モデルのインスタンス化...")
    model = Phase2IntegratedModel(**config)
    print(f"   ✓ モデル作成成功")
    print(f"   - パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - レイヤー数: {model.n_layers}")
    print(f"   - モデル次元: {model.d_model}")
    
    # サンプル入力
    batch_size = 4
    seq_len = 64
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    print(f"\n2. サンプル入力: shape={input_ids.shape}")
    
    # Forward pass
    print("\n3. Forward pass実行...")
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"   ✓ Forward pass成功")
    print(f"   - 出力形状: {logits.shape}")
    print(f"   - 出力範囲: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
    
    # 確率分布に変換
    probs = torch.softmax(logits, dim=-1)
    print(f"   - 確率分布: shape={probs.shape}")
    print(f"   - 確率合計: {probs.sum(dim=-1).mean().item():.4f} (should be ~1.0)")
    
    return model, input_ids


def demo_diagnostics():
    """診断情報の取得例"""
    print("\n" + "=" * 60)
    print("Phase2IntegratedModel - 診断情報の取得")
    print("=" * 60)
    
    # 小さめのモデルを作成
    config = {
        'vocab_size': 500,
        'd_model': 128,
        'n_layers': 2,
        'n_seq': 64,
        'num_heads': 4,
        'head_dim': 32,
        'use_triton': False,
    }
    
    model = Phase2IntegratedModel(**config)
    model.eval()
    
    # サンプル入力
    input_ids = torch.randint(0, config['vocab_size'], (2, 32))
    
    # 診断情報付きでForward pass
    print("\n1. 診断情報付きForward pass...")
    with torch.no_grad():
        logits, diagnostics = model(input_ids, return_diagnostics=True)
    
    print(f"   ✓ 診断情報取得成功")
    print(f"\n2. 診断情報の内容:")
    
    # 各レイヤーの情報
    for i in range(model.n_layers):
        print(f"\n   Layer {i}:")
        
        # Gamma値（減衰率）
        if diagnostics['gamma_values'][i] is not None:
            gamma = diagnostics['gamma_values'][i]
            print(f"   - Gamma (減衰率): mean={gamma.mean().item():.4f}, "
                  f"std={gamma.std().item():.4f}")
        
        # SNR統計
        snr_stats = diagnostics['snr_stats'][i]
        if snr_stats:
            print(f"   - SNR統計: {snr_stats}")
        
        # 共鳴情報
        resonance_info = diagnostics['resonance_info'][i]
        if resonance_info:
            if 'num_resonant' in resonance_info:
                print(f"   - 共鳴モード数: {resonance_info['num_resonant']:.2f}")
            if 'total_energy' in resonance_info:
                print(f"   - 総エネルギー: {resonance_info['total_energy']:.4f}")
        
        # 安定性メトリクス
        stability = diagnostics['stability_metrics'][i]
        if stability:
            if 'is_stable' in stability:
                status = "安定" if stability['is_stable'] else "不安定"
                print(f"   - Lyapunov安定性: {status}")
            if 'energy' in stability:
                print(f"   - エネルギー: {stability['energy']:.4f}")
    
    return model, diagnostics


def demo_statistics():
    """統計情報の表示例"""
    print("\n" + "=" * 60)
    print("Phase2IntegratedModel - 統計情報の表示")
    print("=" * 60)
    
    # モデルを作成
    config = {
        'vocab_size': 1000,
        'd_model': 256,
        'n_layers': 3,
        'n_seq': 128,
        'num_heads': 8,
        'head_dim': 32,
        'use_triton': False,
    }
    
    model = Phase2IntegratedModel(**config)
    
    # 統計情報を取得
    print("\n1. モデル統計情報:")
    stats = model.get_statistics()
    
    print(f"   - 総パラメータ数: {stats['num_parameters']:,}")
    print(f"   - 学習可能パラメータ数: {stats['num_trainable_parameters']:,}")
    print(f"   - レイヤー数: {stats['num_layers']}")
    print(f"   - モデル次元: {stats['d_model']}")
    print(f"   - 語彙サイズ: {stats['vocab_size']}")
    print(f"   - 最大シーケンス長: {stats['n_seq']}")
    
    # 各ブロックの統計
    print(f"\n2. 各ブロックの統計:")
    for block_stat in stats['block_stats']:
        layer_idx = block_stat['layer']
        print(f"\n   Layer {layer_idx}:")
        
        # Hebbian統計
        if 'hebbian' in block_stat:
            hebbian_stats = block_stat['hebbian']
            print(f"   - Hebbian:")
            for key, value in hebbian_stats.items():
                if isinstance(value, (int, float)):
                    print(f"     * {key}: {value:.4f}")
        
        # SNR統計
        if 'snr' in block_stat:
            snr_stats = block_stat['snr']
            print(f"   - SNR:")
            for key, value in snr_stats.items():
                if isinstance(value, (int, float)):
                    print(f"     * {key}: {value:.4f}")
        
        # Non-Hermitian統計
        if 'non_hermitian' in block_stat:
            nh_stats = block_stat['non_hermitian']
            print(f"   - Non-Hermitian:")
            for key, value in nh_stats.items():
                if isinstance(value, (int, float)):
                    print(f"     * {key}: {value:.4f}")
    
    return model, stats


def demo_training_loop():
    """簡単な学習ループの例"""
    print("\n" + "=" * 60)
    print("Phase2IntegratedModel - 学習ループの例")
    print("=" * 60)
    
    # 小さめのモデルを作成
    config = {
        'vocab_size': 500,
        'd_model': 128,
        'n_layers': 2,
        'n_seq': 64,
        'num_heads': 4,
        'head_dim': 32,
        'use_triton': False,
    }
    
    model = Phase2IntegratedModel(**config)
    model.train()
    
    # オプティマイザー
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # ダミーデータ
    batch_size = 4
    seq_len = 32
    
    print("\n1. 学習ループ開始...")
    print(f"   - バッチサイズ: {batch_size}")
    print(f"   - シーケンス長: {seq_len}")
    
    # 数ステップの学習
    num_steps = 5
    losses = []
    
    for step in range(num_steps):
        # 状態をリセット（各ステップで独立した処理）
        model.reset_state()
        
        # ダミーデータ生成
        input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
        target_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
        
        # Forward pass
        logits = model(input_ids)
        
        # Loss計算
        loss = nn.functional.cross_entropy(
            logits.view(-1, config['vocab_size']),
            target_ids.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # パラメータ更新
        optimizer.step()
        
        losses.append(loss.item())
        
        if (step + 1) % 1 == 0:
            print(f"   Step {step + 1}/{num_steps}: Loss = {loss.item():.4f}")
    
    print(f"\n2. 学習完了")
    print(f"   - 初期Loss: {losses[0]:.4f}")
    print(f"   - 最終Loss: {losses[-1]:.4f}")
    print(f"   - Loss変化: {losses[-1] - losses[0]:.4f}")
    
    return model, losses


def demo_state_management():
    """状態管理の例"""
    print("\n" + "=" * 60)
    print("Phase2IntegratedModel - 状態管理")
    print("=" * 60)
    
    # モデルを作成
    config = {
        'vocab_size': 500,
        'd_model': 128,
        'n_layers': 2,
        'n_seq': 64,
        'num_heads': 4,
        'head_dim': 32,
        'use_triton': False,
    }
    
    model = Phase2IntegratedModel(**config)
    model.eval()
    
    # サンプル入力
    input_ids = torch.randint(0, config['vocab_size'], (2, 32))
    
    print("\n1. 最初の推論...")
    with torch.no_grad():
        logits1 = model(input_ids)
    print(f"   ✓ 出力形状: {logits1.shape}")
    
    print("\n2. 状態をリセット...")
    model.reset_state()
    print(f"   ✓ 状態リセット完了")
    
    print("\n3. リセット後の推論...")
    with torch.no_grad():
        logits2 = model(input_ids)
    print(f"   ✓ 出力形状: {logits2.shape}")
    
    # 出力の差を確認
    diff = (logits1 - logits2).abs().mean().item()
    print(f"\n4. 出力の差: {diff:.6f}")
    print(f"   (Fast Weightsの状態がリセットされたため、出力が変わる可能性があります)")
    
    return model


def main():
    """メイン関数"""
    print("\n" + "=" * 60)
    print("Phase 2 Integrated Model - デモスクリプト")
    print("=" * 60)
    print("\nこのスクリプトは、Phase2IntegratedModelの使用方法を示します。")
    print("Phase 2は、動的な記憶機構と散逸的忘却を統合したモデルです。")
    
    # 各デモを実行
    try:
        # 1. 基本的な使用例
        model1, input_ids = demo_basic_usage()
        
        # 2. 診断情報の取得
        model2, diagnostics = demo_diagnostics()
        
        # 3. 統計情報の表示
        model3, stats = demo_statistics()
        
        # 4. 学習ループの例
        model4, losses = demo_training_loop()
        
        # 5. 状態管理
        model5 = demo_state_management()
        
        print("\n" + "=" * 60)
        print("すべてのデモが正常に完了しました！")
        print("=" * 60)
        
        print("\n次のステップ:")
        print("1. 実際のデータセットで学習を試す")
        print("2. 診断情報を使ってモデルの動作を分析する")
        print("3. Phase 1モデルからの変換を試す")
        print("4. より大きなモデルでスケーリングを確認する")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
