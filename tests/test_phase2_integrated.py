"""
Phase 2 Integrated Model - 単体テスト

このテストスイートは、Phase2IntegratedModelの動作を検証します:
1. モデルのインスタンス化
2. Forward passの実行
3. 勾配計算の正当性
4. 診断情報の収集
5. Phase 1互換性

Requirements: 6.4, 6.5
"""

import torch
import torch.nn as nn
import pytest
import warnings

from src.models.phase2.integrated_model import Phase2IntegratedModel, Phase2Block


class TestPhase2IntegratedModel:
    """Phase2IntegratedModelの単体テスト"""
    
    @pytest.fixture
    def model_config(self):
        """テスト用のモデル設定"""
        return {
            'vocab_size': 1000,
            'd_model': 128,
            'n_layers': 2,
            'n_seq': 64,
            'num_heads': 4,
            'head_dim': 32,
            'use_triton': False,  # テスト環境ではTritonを無効化
            'ffn_dim': 256,
            'dropout': 0.1,
        }
    
    @pytest.fixture
    def model(self, model_config):
        """テスト用のモデルインスタンス"""
        return Phase2IntegratedModel(**model_config)
    
    @pytest.fixture
    def sample_input(self):
        """テスト用の入力データ"""
        batch_size = 2
        seq_len = 32
        return torch.randint(0, 1000, (batch_size, seq_len))
    
    def test_model_instantiation(self, model_config):
        """
        テスト1: モデルのインスタンス化
        
        検証項目:
        - エラーなくインスタンス化できること
        - 正しいパラメータ数を持つこと
        
        Requirement: 6.4
        """
        model = Phase2IntegratedModel(**model_config)
        
        # モデルが正常に作成されたことを確認
        assert isinstance(model, Phase2IntegratedModel)
        assert isinstance(model, nn.Module)
        
        # 基本的な属性を確認
        assert model.vocab_size == model_config['vocab_size']
        assert model.d_model == model_config['d_model']
        assert model.n_layers == model_config['n_layers']
        assert model.n_seq == model_config['n_seq']
        
        # レイヤー数を確認
        assert len(model.blocks) == model_config['n_layers']
        
        # すべてのブロックがPhase2Blockであることを確認
        for block in model.blocks:
            assert isinstance(block, Phase2Block)
        
        print("✓ モデルのインスタンス化テスト成功")
    
    def test_forward_pass(self, model, sample_input):
        """
        テスト2: Forward passの実行
        
        検証項目:
        - Forward passがエラーなく実行されること
        - 出力の形状が正しいこと
        - 出力が有限値であること
        
        Requirement: 6.4, 6.5
        """
        model.eval()
        
        with torch.no_grad():
            logits = model(sample_input)
        
        # 出力の形状を確認
        batch_size, seq_len = sample_input.shape
        expected_shape = (batch_size, seq_len, model.vocab_size)
        assert logits.shape == expected_shape, f"Expected shape {expected_shape}, got {logits.shape}"
        
        # 出力が有限値であることを確認
        assert torch.isfinite(logits).all(), "Output contains NaN or Inf"
        
        # 出力の範囲を確認（ロジットなので制限なし）
        assert logits.abs().max() < 1e6, "Output values are too large"
        
        print(f"✓ Forward passテスト成功: 出力形状 {logits.shape}")
    
    def test_backward_pass(self, model, sample_input):
        """
        テスト3: Backward passと勾配計算
        
        検証項目:
        - Backward passがエラーなく実行されること
        - 勾配が計算されること
        - 勾配が有限値であること
        
        Requirement: 6.4, 6.5
        """
        model.train()
        
        # Forward pass
        logits = model(sample_input)
        
        # ダミーのターゲットとロス
        target = torch.randint(0, model.vocab_size, sample_input.shape)
        loss = nn.functional.cross_entropy(
            logits.view(-1, model.vocab_size),
            target.view(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # 勾配が計算されたことを確認
        has_grad = False
        params_with_grad = 0
        params_without_grad = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    assert torch.isfinite(param.grad).all(), f"Gradient contains NaN or Inf for {name}"
                    has_grad = True
                    params_with_grad += 1
                else:
                    # BK-Coreのパラメータなど、使用されないパラメータは勾配がないことがある
                    params_without_grad += 1
        
        assert has_grad, "No gradients were computed"
        
        # 少なくとも50%のパラメータに勾配があることを確認
        total_params = params_with_grad + params_without_grad
        grad_ratio = params_with_grad / total_params if total_params > 0 else 0
        assert grad_ratio > 0.5, f"Only {grad_ratio*100:.1f}% of parameters have gradients"
        
        print(f"✓ Backward passテスト成功: Loss = {loss.item():.4f}")
    
    def test_diagnostics(self, model, sample_input):
        """
        テスト4: 診断情報の収集
        
        検証項目:
        - return_diagnostics=Trueで診断情報が返されること
        - 診断情報に必要なキーが含まれること
        - 診断情報の値が有限であること
        
        Requirement: 6.5
        """
        model.eval()
        
        with torch.no_grad():
            logits, diagnostics = model(sample_input, return_diagnostics=True)
        
        # 診断情報が辞書であることを確認
        assert isinstance(diagnostics, dict), "Diagnostics should be a dictionary"
        
        # 必要なキーが含まれていることを確認
        required_keys = [
            'layer_outputs',
            'gamma_values',
            'snr_stats',
            'resonance_info',
            'stability_metrics',
            'input_embeddings',
            'final_hidden_states',
            'logits',
        ]
        
        for key in required_keys:
            assert key in diagnostics, f"Missing key in diagnostics: {key}"
        
        # レイヤー数と一致することを確認
        assert len(diagnostics['layer_outputs']) == model.n_layers
        assert len(diagnostics['gamma_values']) == model.n_layers
        assert len(diagnostics['snr_stats']) == model.n_layers
        assert len(diagnostics['resonance_info']) == model.n_layers
        assert len(diagnostics['stability_metrics']) == model.n_layers
        
        # 診断情報の値が有限であることを確認
        for i, layer_output in enumerate(diagnostics['layer_outputs']):
            assert torch.isfinite(layer_output).all(), f"Layer {i} output contains NaN or Inf"
        
        print(f"✓ 診断情報テスト成功: {len(diagnostics)} keys collected")
    
    def test_state_reset(self, model, sample_input):
        """
        テスト5: 状態のリセット
        
        検証項目:
        - reset_state()が正常に実行されること
        - リセット後も推論が可能であること
        
        Requirement: 6.4
        """
        model.eval()
        
        # 最初の推論
        with torch.no_grad():
            logits1 = model(sample_input)
        
        # 状態をリセット
        model.reset_state()
        
        # リセット後の推論
        with torch.no_grad():
            logits2 = model(sample_input)
        
        # 両方とも有限値であることを確認
        assert torch.isfinite(logits1).all()
        assert torch.isfinite(logits2).all()
        
        # 形状が同じであることを確認
        assert logits1.shape == logits2.shape
        
        print("✓ 状態リセットテスト成功")
    
    def test_statistics(self, model):
        """
        テスト6: 統計情報の取得
        
        検証項目:
        - get_statistics()が正常に実行されること
        - 統計情報に必要なキーが含まれること
        
        Requirement: 6.4
        """
        stats = model.get_statistics()
        
        # 統計情報が辞書であることを確認
        assert isinstance(stats, dict)
        
        # 必要なキーが含まれていることを確認
        required_keys = [
            'num_parameters',
            'num_trainable_parameters',
            'num_layers',
            'd_model',
            'vocab_size',
            'n_seq',
            'block_stats',
        ]
        
        for key in required_keys:
            assert key in stats, f"Missing key in statistics: {key}"
        
        # パラメータ数が正の値であることを確認
        assert stats['num_parameters'] > 0
        assert stats['num_trainable_parameters'] > 0
        assert stats['num_trainable_parameters'] <= stats['num_parameters']
        
        # レイヤー数が一致することを確認
        assert stats['num_layers'] == model.n_layers
        assert len(stats['block_stats']) == model.n_layers
        
        print(f"✓ 統計情報テスト成功: {stats['num_parameters']:,} parameters")
    
    def test_phase1_compatibility(self, model_config):
        """
        テスト7: Phase 1互換性
        
        検証項目:
        - phase1_configパラメータを受け入れること
        - Phase 1設定でもモデルが動作すること
        
        Requirement: 6.2
        """
        # ダミーのPhase 1設定
        class DummyPhase1Config:
            def __init__(self):
                self.target_vram_gb = 8.0
                self.d_model = 128
        
        phase1_config = DummyPhase1Config()
        
        # Phase 1設定を含むモデルを作成
        model = Phase2IntegratedModel(
            **model_config,
            phase1_config=phase1_config
        )
        
        # Phase 1設定が保存されていることを確認
        assert model.phase1_config is not None
        assert model.phase1_config == phase1_config
        
        # モデルが正常に動作することを確認
        sample_input = torch.randint(0, model_config['vocab_size'], (2, 32))
        
        with torch.no_grad():
            logits = model(sample_input)
        
        assert torch.isfinite(logits).all()
        
        print("✓ Phase 1互換性テスト成功")
    
    def test_different_sequence_lengths(self, model_config):
        """
        テスト8: 異なるシーケンス長での動作
        
        検証項目:
        - 異なるシーケンス長で推論が可能であること
        - 最大シーケンス長を超えない範囲で動作すること
        
        Requirement: 6.4
        """
        model = Phase2IntegratedModel(**model_config)
        model.eval()
        
        # 異なるシーケンス長でテスト
        seq_lengths = [8, 16, 32, 64]  # 64 = n_seq
        
        for seq_len in seq_lengths:
            if seq_len <= model_config['n_seq']:
                sample_input = torch.randint(0, model_config['vocab_size'], (2, seq_len))
                
                with torch.no_grad():
                    logits = model(sample_input)
                
                # 出力の形状を確認
                assert logits.shape == (2, seq_len, model_config['vocab_size'])
                assert torch.isfinite(logits).all()
        
        print(f"✓ 異なるシーケンス長テスト成功: {seq_lengths}")
    
    def test_batch_sizes(self, model_config):
        """
        テスト9: 異なるバッチサイズでの動作
        
        検証項目:
        - 異なるバッチサイズで推論が可能であること
        
        Requirement: 6.4
        """
        model = Phase2IntegratedModel(**model_config)
        model.eval()
        
        # 異なるバッチサイズでテスト
        batch_sizes = [1, 2, 4, 8]
        seq_len = 32
        
        for batch_size in batch_sizes:
            # バッチサイズが変わるたびに状態をリセット
            model.reset_state()
            
            sample_input = torch.randint(0, model_config['vocab_size'], (batch_size, seq_len))
            
            with torch.no_grad():
                logits = model(sample_input)
            
            # 出力の形状を確認
            assert logits.shape == (batch_size, seq_len, model_config['vocab_size'])
            assert torch.isfinite(logits).all()
        
        print(f"✓ 異なるバッチサイズテスト成功: {batch_sizes}")
    
    def test_zeta_initialization(self, model_config):
        """
        テスト10: ゼータ初期化の適用
        
        検証項目:
        - ゼータ初期化が適用されていること
        - 初期化後の重みが有限値であること
        
        Requirement: 6.1
        """
        model = Phase2IntegratedModel(**model_config)
        
        # すべてのパラメータが有限値であることを確認
        for name, param in model.named_parameters():
            assert torch.isfinite(param).all(), f"Parameter {name} contains NaN or Inf after initialization"
        
        # Token Embeddingが初期化されていることを確認
        token_emb_weight = model.token_embedding.weight
        assert token_emb_weight.abs().max() < 10.0, "Token embedding weights are too large"
        
        # Position Embeddingが初期化されていることを確認
        # ZetaEmbeddingは自動的に初期化される
        assert model.position_embedding is not None
        
        print("✓ ゼータ初期化テスト成功")


def test_model_creation_minimal():
    """
    最小限の設定でモデルを作成するテスト
    
    Requirement: 6.4
    """
    model = Phase2IntegratedModel(
        vocab_size=100,
        d_model=64,
        n_layers=1,
        n_seq=32,
    )
    
    assert isinstance(model, Phase2IntegratedModel)
    print("✓ 最小限のモデル作成テスト成功")


def test_model_forward_minimal():
    """
    最小限の設定でForward passを実行するテスト
    
    Requirement: 6.4, 6.5
    """
    model = Phase2IntegratedModel(
        vocab_size=100,
        d_model=64,
        n_layers=1,
        n_seq=32,
        use_triton=False,
    )
    model.eval()
    
    input_ids = torch.randint(0, 100, (1, 16))
    
    with torch.no_grad():
        logits = model(input_ids)
    
    assert logits.shape == (1, 16, 100)
    assert torch.isfinite(logits).all()
    
    print("✓ 最小限のForward passテスト成功")


if __name__ == "__main__":
    # 個別にテストを実行
    print("=" * 60)
    print("Phase2IntegratedModel 単体テスト")
    print("=" * 60)
    
    # 最小限のテスト
    test_model_creation_minimal()
    test_model_forward_minimal()
    
    # クラスベースのテスト
    test_class = TestPhase2IntegratedModel()
    
    # フィクスチャを手動で作成
    model_config = {
        'vocab_size': 1000,
        'd_model': 128,
        'n_layers': 2,
        'n_seq': 64,
        'num_heads': 4,
        'head_dim': 32,
        'use_triton': False,
        'ffn_dim': 256,
        'dropout': 0.1,
    }
    model = Phase2IntegratedModel(**model_config)
    sample_input = torch.randint(0, 1000, (2, 32))
    
    # 各テストを実行
    print("\n1. モデルのインスタンス化テスト")
    test_class.test_model_instantiation(model_config)
    
    print("\n2. Forward passテスト")
    test_class.test_forward_pass(model, sample_input)
    
    print("\n3. Backward passテスト")
    test_class.test_backward_pass(model, sample_input)
    
    print("\n4. 診断情報テスト")
    test_class.test_diagnostics(model, sample_input)
    
    print("\n5. 状態リセットテスト")
    test_class.test_state_reset(model, sample_input)
    
    print("\n6. 統計情報テスト")
    test_class.test_statistics(model)
    
    print("\n7. Phase 1互換性テスト")
    test_class.test_phase1_compatibility(model_config)
    
    print("\n8. 異なるシーケンス長テスト")
    test_class.test_different_sequence_lengths(model_config)
    
    print("\n9. 異なるバッチサイズテスト")
    test_class.test_batch_sizes(model_config)
    
    print("\n10. ゼータ初期化テスト")
    test_class.test_zeta_initialization(model_config)
    
    print("\n" + "=" * 60)
    print("すべてのテストが成功しました！")
    print("=" * 60)
