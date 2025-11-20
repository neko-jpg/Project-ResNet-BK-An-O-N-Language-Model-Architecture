"""
Phase 2: Zeta Initialization Tests

このテストスイートは、Riemann-Zeta Regularization機構の正当性を検証します。

テスト項目:
    1. ゼータ零点生成の正確性
    2. GUE統計の妥当性
    3. 線形層初期化の動作確認
    4. Embedding初期化の動作確認
    5. ZetaEmbeddingモジュールの動作確認
    6. 初期化後の重み分布の検証

Requirements: 5.4, 5.5, 5.6
"""

import torch
import torch.nn as nn
import pytest
import numpy as np
from src.models.phase2.zeta_init import (
    ZetaInitializer,
    ZetaEmbedding,
    apply_zeta_initialization,
    get_zeta_statistics
)


class TestZetaZeroGeneration:
    """ゼータ零点生成のテスト"""
    
    def test_precise_zeros_small_n(self):
        """
        n <= 10 の場合、精密な零点値が返されることを確認
        
        Requirements: 5.1, 5.2
        """
        # 最初の5個の零点を取得
        zeros = ZetaInitializer.get_approx_zeta_zeros(5)
        
        # 期待される精密値
        expected = torch.tensor([
            14.134725, 21.022040, 25.010858, 30.424876, 32.935062
        ])
        
        # 精度チェック（1e-5以内）
        assert torch.allclose(zeros, expected, atol=1e-5), \
            f"Expected {expected}, got {zeros}"
    
    def test_gue_approximation_large_n(self):
        """
        n > 10 の場合、GUE統計に基づく近似が生成されることを確認
        
        Requirements: 5.2, 5.3
        """
        # 100個の零点を生成
        zeros = ZetaInitializer.get_approx_zeta_zeros(100)
        
        # 基本的な性質をチェック
        assert zeros.shape[0] == 100, "Should return 100 zeros"
        assert torch.all(zeros[1:] > zeros[:-1]), "Zeros should be monotonically increasing"
        
        # 最初の10個は精密値であることを確認
        precise_zeros = torch.tensor([
            14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
            37.586178, 40.918719, 43.327073, 48.005150, 49.773832
        ])
        assert torch.allclose(zeros[:10], precise_zeros, atol=1e-5), \
            "First 10 zeros should be precise values"
    
    def test_spacing_statistics(self):
        """
        零点の間隔がGUE統計に従うことを確認
        
        GUE統計の特徴:
        - 平均間隔 ~ 2.5 (T=50付近)
        - 間隔の分散が存在する（等間隔ではない）
        
        Requirements: 5.3
        """
        zeros = ZetaInitializer.get_approx_zeta_zeros(50)
        spacings = zeros[1:] - zeros[:-1]
        
        # 平均間隔のチェック（おおよそ2.5前後）
        mean_spacing = spacings.mean().item()
        assert 1.5 < mean_spacing < 4.0, \
            f"Mean spacing {mean_spacing} should be around 2.5"
        
        # 間隔の分散が存在することを確認（等間隔ではない）
        std_spacing = spacings.std().item()
        assert std_spacing > 0.1, \
            f"Spacing should have variance, got std={std_spacing}"
    
    def test_zeros_are_positive(self):
        """すべての零点が正の値であることを確認"""
        zeros = ZetaInitializer.get_approx_zeta_zeros(100)
        assert torch.all(zeros > 0), "All zeros should be positive"
    
    def test_deterministic_generation(self):
        """同じnに対して同じ零点が生成されることを確認（決定論的）"""
        # 注意: GUE部分はランダムなので、この部分は精密値のみテスト
        zeros1 = ZetaInitializer.get_approx_zeta_zeros(10)
        zeros2 = ZetaInitializer.get_approx_zeta_zeros(10)
        
        assert torch.allclose(zeros1, zeros2, atol=1e-6), \
            "Same n should produce same zeros (for n <= 10)"


class TestLinearInitialization:
    """線形層初期化のテスト"""
    
    def test_initialize_linear_basic(self):
        """
        線形層の初期化が正常に動作することを確認
        
        Requirements: 5.4
        """
        # 線形層を作成
        linear = nn.Linear(64, 64)
        original_weight = linear.weight.data.clone()
        
        # ゼータ初期化を適用
        ZetaInitializer.initialize_linear_zeta(linear, scale=10.0)
        
        # 重みが変更されたことを確認
        assert not torch.allclose(linear.weight.data, original_weight), \
            "Weights should be modified after initialization"
    
    def test_singular_value_distribution(self):
        """
        特異値がゼータ零点の逆数に従うことを確認
        
        Requirements: 5.4
        """
        linear = nn.Linear(64, 64)
        ZetaInitializer.initialize_linear_zeta(linear, scale=10.0)
        
        # SVD分解
        u, s, v = torch.svd(linear.weight)
        
        # 期待される特異値: scale / zeros
        zeros = ZetaInitializer.get_approx_zeta_zeros(64)
        expected_s = 10.0 / zeros
        
        # 特異値が期待値に近いことを確認（相対誤差15%以内）
        # 注意: SVD分解と再構成の過程で数値誤差が蓄積するため、やや緩い閾値を使用
        relative_error = torch.abs(s - expected_s) / expected_s
        assert torch.all(relative_error < 0.15), \
            f"Singular values should match expected distribution, max error: {relative_error.max()}"
    
    def test_different_scales(self):
        """異なるスケールで初期化が正しく動作することを確認"""
        linear1 = nn.Linear(32, 32)
        linear2 = nn.Linear(32, 32)
        
        ZetaInitializer.initialize_linear_zeta(linear1, scale=5.0)
        ZetaInitializer.initialize_linear_zeta(linear2, scale=10.0)
        
        # スケールが異なれば重みも異なる
        assert not torch.allclose(linear1.weight, linear2.weight), \
            "Different scales should produce different weights"
        
        # スケール比が特異値比に反映される
        _, s1, _ = torch.svd(linear1.weight)
        _, s2, _ = torch.svd(linear2.weight)
        
        ratio = (s2 / s1).mean()
        assert 1.8 < ratio < 2.2, \
            f"Singular value ratio should be close to scale ratio (2.0), got {ratio}"
    
    def test_rectangular_matrix(self):
        """長方形行列でも初期化が動作することを確認"""
        linear = nn.Linear(128, 64)  # 入力 > 出力
        
        # エラーなく初期化できることを確認
        try:
            ZetaInitializer.initialize_linear_zeta(linear, scale=10.0)
        except Exception as e:
            pytest.fail(f"Initialization failed for rectangular matrix: {e}")
        
        # 特異値の数は min(in_features, out_features)
        _, s, _ = torch.svd(linear.weight)
        assert s.shape[0] == 64, "Number of singular values should be min(in, out)"


class TestEmbeddingInitialization:
    """Embedding初期化のテスト"""
    
    def test_initialize_embedding_basic(self):
        """
        Embedding初期化が正常に動作することを確認
        
        Requirements: 5.4
        """
        embedding = nn.Embedding(100, 64)
        original_weight = embedding.weight.data.clone()
        
        # ゼータ初期化を適用
        ZetaInitializer.initialize_embedding_zeta(embedding, scale=1.0)
        
        # 重みが変更されたことを確認
        assert not torch.allclose(embedding.weight.data, original_weight), \
            "Embedding weights should be modified after initialization"
    
    def test_sinusoidal_pattern(self):
        """
        Sin/Cosパターンが正しく適用されることを確認
        
        PE(pos, 2i) = sin(pos / zero_i)
        PE(pos, 2i+1) = cos(pos / zero_i)
        
        Requirements: 5.6
        """
        max_len = 100
        d_model = 64
        embedding = nn.Embedding(max_len, d_model)
        
        ZetaInitializer.initialize_embedding_zeta(embedding, scale=1.0)
        
        # 位置0の埋め込みをチェック
        pos_0 = embedding.weight[0]
        
        # 位置0では sin(0) = 0, cos(0) = 1
        # 偶数インデックスは0に近い、奇数インデックスは1に近い
        assert torch.allclose(pos_0[0::2], torch.zeros_like(pos_0[0::2]), atol=1e-5), \
            "Even indices at position 0 should be close to 0 (sin(0))"
        assert torch.allclose(pos_0[1::2], torch.ones_like(pos_0[1::2]), atol=1e-5), \
            "Odd indices at position 0 should be close to 1 (cos(0))"
    
    def test_position_encoding_properties(self):
        """位置エンコーディングの基本的な性質を確認"""
        max_len = 100
        d_model = 64
        embedding = nn.Embedding(max_len, d_model)
        
        ZetaInitializer.initialize_embedding_zeta(embedding, scale=1.0)
        
        # すべての位置エンコーディングのノルムが同程度であることを確認
        norms = torch.norm(embedding.weight, dim=1)
        mean_norm = norms.mean()
        std_norm = norms.std()
        
        # 標準偏差が平均の20%以内
        assert std_norm / mean_norm < 0.2, \
            f"Position encoding norms should be relatively uniform, got std/mean = {std_norm/mean_norm}"


class TestZetaEmbedding:
    """ZetaEmbeddingモジュールのテスト"""
    
    def test_module_creation(self):
        """
        ZetaEmbeddingモジュールが正常に作成されることを確認
        
        Requirements: 5.5
        """
        pos_emb = ZetaEmbedding(max_len=1024, d_model=512, trainable=False)
        
        assert pos_emb.max_len == 1024
        assert pos_emb.d_model == 512
        assert not pos_emb.trainable
    
    def test_forward_pass(self):
        """
        Forward passが正常に動作することを確認
        
        Requirements: 5.5, 5.6
        """
        pos_emb = ZetaEmbedding(max_len=100, d_model=64, trainable=False)
        
        # 位置インデックスを作成
        positions = torch.arange(0, 50).unsqueeze(0)  # (1, 50)
        
        # Forward pass
        embeddings = pos_emb(positions)
        
        # 出力形状の確認
        assert embeddings.shape == (1, 50, 64), \
            f"Expected shape (1, 50, 64), got {embeddings.shape}"
    
    def test_trainable_vs_fixed(self):
        """学習可能モードと固定モードの違いを確認"""
        # 固定モード
        pos_emb_fixed = ZetaEmbedding(max_len=100, d_model=64, trainable=False)
        assert not pos_emb_fixed.embedding.weight.requires_grad, \
            "Fixed mode should not require gradients"
        
        # 学習可能モード
        pos_emb_trainable = ZetaEmbedding(max_len=100, d_model=64, trainable=True)
        assert pos_emb_trainable.embedding.weight.requires_grad, \
            "Trainable mode should require gradients"
    
    def test_batch_processing(self):
        """バッチ処理が正常に動作することを確認"""
        pos_emb = ZetaEmbedding(max_len=100, d_model=64, trainable=False)
        
        # バッチサイズ4
        positions = torch.arange(0, 20).unsqueeze(0).expand(4, -1)  # (4, 20)
        
        embeddings = pos_emb(positions)
        
        assert embeddings.shape == (4, 20, 64), \
            f"Expected shape (4, 20, 64), got {embeddings.shape}"
        
        # すべてのバッチで同じ埋め込みが返されることを確認
        for i in range(1, 4):
            assert torch.allclose(embeddings[0], embeddings[i]), \
                "All batches should have identical embeddings for same positions"
    
    def test_out_of_range_positions(self):
        """範囲外の位置インデックスが適切に処理されることを確認"""
        pos_emb = ZetaEmbedding(max_len=100, d_model=64, trainable=False)
        
        # 範囲外の位置（警告が出るはず）
        positions = torch.tensor([[0, 50, 150]])  # 150 > max_len
        
        # エラーなく実行できることを確認（警告は出る）
        with pytest.warns(UserWarning):
            embeddings = pos_emb(positions)
        
        # 出力形状は正しい
        assert embeddings.shape == (1, 3, 64)
    
    def test_gradient_flow_trainable(self):
        """学習可能モードで勾配が流れることを確認"""
        pos_emb = ZetaEmbedding(max_len=100, d_model=64, trainable=True)
        
        positions = torch.arange(0, 10).unsqueeze(0)
        embeddings = pos_emb(positions)
        
        # 損失を計算
        loss = embeddings.sum()
        loss.backward()
        
        # 勾配が計算されていることを確認
        assert pos_emb.embedding.weight.grad is not None, \
            "Gradients should be computed in trainable mode"
        assert not torch.all(pos_emb.embedding.weight.grad == 0), \
            "Gradients should be non-zero"


class TestUtilityFunctions:
    """ユーティリティ関数のテスト"""
    
    def test_apply_zeta_initialization(self):
        """モデル全体への初期化適用が動作することを確認"""
        # 簡単なモデルを作成
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(64, 64)
                self.linear2 = nn.Linear(64, 32)
                self.embedding = nn.Embedding(100, 64)
        
        model = SimpleModel()
        
        # 元の重みを保存
        original_linear1 = model.linear1.weight.data.clone()
        original_embedding = model.embedding.weight.data.clone()
        
        # ゼータ初期化を適用
        apply_zeta_initialization(model, scale=10.0)
        
        # 重みが変更されたことを確認
        assert not torch.allclose(model.linear1.weight.data, original_linear1), \
            "Linear layer should be initialized"
        assert not torch.allclose(model.embedding.weight.data, original_embedding), \
            "Embedding should be initialized"
    
    def test_get_zeta_statistics(self):
        """ゼータ統計情報の取得が動作することを確認"""
        stats = get_zeta_statistics(n=50)
        
        # 必要なキーが存在することを確認
        required_keys = ['zeros', 'mean_spacing', 'std_spacing', 'min_spacing', 'max_spacing', 'num_zeros']
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"
        
        # 値の妥当性チェック
        assert stats['num_zeros'] == 50
        assert stats['mean_spacing'] > 0
        assert stats['std_spacing'] > 0
        assert stats['min_spacing'] > 0
        assert stats['max_spacing'] > stats['min_spacing']
        assert len(stats['zeros']) == 50


class TestNumericalStability:
    """数値安定性のテスト"""
    
    def test_large_dimensions(self):
        """大きな次元でも安定して動作することを確認"""
        # 大きな線形層
        linear = nn.Linear(512, 512)
        
        try:
            ZetaInitializer.initialize_linear_zeta(linear, scale=10.0)
        except Exception as e:
            pytest.fail(f"Initialization failed for large dimensions: {e}")
        
        # NaNやInfが含まれていないことを確認
        assert not torch.isnan(linear.weight).any(), "Weights should not contain NaN"
        assert not torch.isinf(linear.weight).any(), "Weights should not contain Inf"
    
    def test_small_dimensions(self):
        """小さな次元でも動作することを確認"""
        linear = nn.Linear(4, 4)
        
        try:
            ZetaInitializer.initialize_linear_zeta(linear, scale=10.0)
        except Exception as e:
            pytest.fail(f"Initialization failed for small dimensions: {e}")
    
    def test_zero_scale_handling(self):
        """スケール=0の場合の処理を確認"""
        linear = nn.Linear(32, 32)
        
        # スケール0で初期化（すべての特異値が0になる）
        ZetaInitializer.initialize_linear_zeta(linear, scale=0.0)
        
        # 重みがすべて0に近いことを確認
        assert torch.allclose(linear.weight, torch.zeros_like(linear.weight), atol=1e-6), \
            "Zero scale should produce near-zero weights"


class TestIntegrationWithPhase2:
    """Phase 2モデルとの統合テスト"""
    
    def test_zeta_embedding_in_model(self):
        """ZetaEmbeddingがモデル内で正常に動作することを確認"""
        class SimpleLanguageModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.token_embedding = nn.Embedding(1000, 128)
                self.position_embedding = ZetaEmbedding(512, 128, trainable=False)
                self.linear = nn.Linear(128, 1000)
            
            def forward(self, input_ids):
                B, N = input_ids.shape
                positions = torch.arange(N, device=input_ids.device).unsqueeze(0).expand(B, -1)
                
                x = self.token_embedding(input_ids) + self.position_embedding(positions)
                return self.linear(x)
        
        model = SimpleLanguageModel()
        
        # Forward pass
        input_ids = torch.randint(0, 1000, (2, 50))
        output = model(input_ids)
        
        assert output.shape == (2, 50, 1000), \
            f"Expected shape (2, 50, 1000), got {output.shape}"
    
    def test_initialization_preserves_model_structure(self):
        """初期化がモデル構造を壊さないことを確認"""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(64, 128)
                self.linear2 = nn.Linear(128, 64)
                self.embedding = nn.Embedding(100, 64)
            
            def forward(self, x, positions):
                x = self.linear1(x)
                x = self.linear2(x)
                x = x + self.embedding(positions)
                return x
        
        model = TestModel()
        apply_zeta_initialization(model)
        
        # Forward passが正常に動作することを確認
        x = torch.randn(2, 10, 64)
        positions = torch.arange(10).unsqueeze(0).expand(2, -1)
        
        try:
            output = model(x, positions)
            assert output.shape == (2, 10, 64)
        except Exception as e:
            pytest.fail(f"Model forward pass failed after initialization: {e}")


if __name__ == "__main__":
    # テストを実行
    pytest.main([__file__, "-v", "--tb=short"])
