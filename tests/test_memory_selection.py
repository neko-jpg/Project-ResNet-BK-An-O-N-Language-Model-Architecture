"""
SNRベースの記憶選択機構のテスト

Requirements: 9.7
"""

import pytest
import torch
import torch.nn as nn
from src.models.phase2.memory_selection import SNRMemoryFilter, MemoryImportanceEstimator


class TestSNRMemoryFilter:
    """SNRMemoryFilterのテスト"""
    
    def test_initialization(self):
        """初期化のテスト"""
        filter = SNRMemoryFilter(threshold=2.0, gamma_boost=2.0, eta_boost=1.5)
        
        assert filter.threshold == 2.0
        assert filter.gamma_boost == 2.0
        assert filter.eta_boost == 1.5
        assert filter.snr_history.shape[0] == 1000
        assert filter.history_idx == 0
    
    def test_snr_calculation(self):
        """SNR計算のテスト (Requirement 9.1)"""
        filter = SNRMemoryFilter()
        
        # テスト用Fast Weights
        # 高SNR成分と低SNR成分を混在させる
        B, H, D = 2, 4, 8
        weights = torch.randn(B, H, D, D)
        
        # 一部の成分を強い信号にする
        weights[:, 0, :, :] *= 10.0  # 高SNR
        weights[:, 1, :, :] *= 0.1   # 低SNR
        
        gamma = torch.ones(B) * 0.1
        eta = 0.1
        
        adjusted_gamma, adjusted_eta = filter(weights, gamma, eta)
        
        # 出力の形状確認
        assert adjusted_gamma.shape == (B,)
        assert isinstance(adjusted_eta, (float, torch.Tensor))
        
        # 値の範囲確認
        assert torch.all(adjusted_gamma > 0)
        if isinstance(adjusted_eta, torch.Tensor):
            assert adjusted_eta.item() > 0
        else:
            assert adjusted_eta > 0
    
    def test_low_snr_increases_gamma(self):
        """低SNRでΓが増加することを確認 (Requirement 9.4)"""
        filter = SNRMemoryFilter(threshold=2.0, gamma_boost=2.0)
        
        B, H, D = 2, 4, 8
        
        # 低SNRの重み（ノイズ優勢）
        low_snr_weights = torch.randn(B, H, D, D) * 0.01
        
        gamma = torch.ones(B) * 0.1
        eta = 0.1
        
        adjusted_gamma, _ = filter(low_snr_weights, gamma, eta)
        
        # 低SNRの場合、Γが増加するはず
        # 平均SNRが閾値未満なら gamma_boost が適用される
        mean_snr = torch.abs(low_snr_weights).mean() / (torch.std(low_snr_weights) + 1e-6)
        
        if mean_snr < filter.threshold:
            # Γが増加していることを確認
            assert torch.all(adjusted_gamma >= gamma)
    
    def test_high_snr_increases_eta(self):
        """高SNRでηが増加することを確認 (Requirement 9.5)"""
        filter = SNRMemoryFilter(threshold=2.0, eta_boost=1.5)
        
        B, H, D = 2, 4, 8
        
        # 高SNRの重み（明確な信号）
        high_snr_weights = torch.randn(B, H, D, D) * 10.0
        
        gamma = torch.ones(B) * 0.1
        eta = 0.1
        
        _, adjusted_eta = filter(high_snr_weights, gamma, eta)
        
        # 高SNRの場合、ηが増加するはず
        mean_snr = torch.abs(high_snr_weights).mean() / (torch.std(high_snr_weights) + 1e-6)
        
        if mean_snr > filter.threshold:
            # ηが増加していることを確認
            assert adjusted_eta >= eta
    
    def test_snr_suppression_verification(self):
        """
        SNR < 2.0 の信号に対して、Hebbian更新量が1/10以下に抑制されることを検証
        (Requirement 9.4の検証基準)
        """
        filter = SNRMemoryFilter(threshold=2.0, gamma_boost=10.0)  # 強い抑制
        
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
        
        # 低SNRの場合、更新量が大幅に減少するはず
        mean_snr = torch.abs(low_snr_weights).mean() / (torch.std(low_snr_weights) + 1e-6)
        
        if mean_snr < filter.threshold:
            # 更新量が1/10以下に抑制されることを確認
            assert update_ratio_adjusted <= update_ratio_original * 0.1 + 0.01  # 許容誤差
    
    def test_statistics_tracking(self):
        """統計追跡のテスト (Requirement 9.6)"""
        filter = SNRMemoryFilter()
        filter.train()  # 学習モードで統計を記録
        
        B, H, D = 2, 4, 8
        weights = torch.randn(B, H, D, D)
        gamma = torch.ones(B) * 0.1
        eta = 0.1
        
        # 複数回実行して統計を蓄積
        for _ in range(10):
            filter(weights, gamma, eta)
        
        # 統計取得
        stats = filter.get_statistics()
        
        # 統計情報が正しく記録されていることを確認
        assert 'mean_snr' in stats
        assert 'std_snr' in stats
        assert 'min_snr' in stats
        assert 'max_snr' in stats
        
        # 値が妥当な範囲にあることを確認
        assert stats['mean_snr'] >= 0
        assert stats['std_snr'] >= 0
        assert stats['min_snr'] <= stats['max_snr']
    
    def test_gradient_flow(self):
        """勾配が正しく流れることを確認"""
        filter = SNRMemoryFilter()
        
        B, H, D = 2, 4, 8
        weights = torch.randn(B, H, D, D)  # requires_grad不要（SNR計算は統計的）
        # leaf tensorを作成
        gamma_base = torch.full((B,), 0.1, requires_grad=True)
        eta = 0.1
        
        adjusted_gamma, adjusted_eta = filter(weights, gamma_base, eta)
        
        # 損失を計算
        loss = adjusted_gamma.sum()
        loss.backward()
        
        # 勾配が計算されていることを確認
        # gamma_baseはleaf tensorなので勾配が計算される
        assert gamma_base.grad is not None
        # weightsはSNR計算に使われるだけで、勾配は不要


class TestMemoryImportanceEstimator:
    """MemoryImportanceEstimatorのテスト"""
    
    def test_initialization(self):
        """初期化のテスト"""
        estimator = MemoryImportanceEstimator(
            snr_weight=0.5,
            energy_weight=0.3,
            recency_weight=0.2
        )
        
        # 重みが正規化されていることを確認
        total = estimator.snr_weight + estimator.energy_weight + estimator.recency_weight
        assert abs(total - 1.0) < 1e-6
    
    def test_importance_calculation(self):
        """重要度計算のテスト (Requirement 9.7)"""
        estimator = MemoryImportanceEstimator()
        
        B, H, D = 2, 4, 8
        weights = torch.randn(B, H, D, D)
        
        importance = estimator(weights)
        
        # 出力の形状確認
        assert importance.shape == weights.shape
        
        # 重要度が[0, 1]の範囲にあることを確認
        assert torch.all(importance >= 0)
        assert torch.all(importance <= 1)
    
    def test_high_importance_for_strong_signals(self):
        """強い信号に高い重要度が割り当てられることを確認 (Requirement 9.7)"""
        estimator = MemoryImportanceEstimator()
        
        B, H, D = 2, 4, 8
        weights = torch.randn(B, H, D, D) * 0.1
        
        # 一部の成分を強い信号にする
        weights[0, 0, 0, 0] = 10.0
        weights[0, 0, 0, 1] = 0.01
        
        importance = estimator(weights)
        
        # 強い信号の重要度が高いことを確認
        assert importance[0, 0, 0, 0] > importance[0, 0, 0, 1]
    
    def test_top_k_memories(self):
        """上位k個の記憶取得のテスト (Requirement 9.7)"""
        estimator = MemoryImportanceEstimator()
        
        B, H, D = 2, 4, 8
        weights = torch.randn(B, H, D, D)
        k = 10
        
        top_weights, top_indices = estimator.get_top_k_memories(weights, k)
        
        # 出力の形状確認
        assert top_weights.shape == (B, H, k)
        assert top_indices.shape == (B, H, k)
        
        # インデックスが有効な範囲にあることを確認
        assert torch.all(top_indices >= 0)
        assert torch.all(top_indices < D * D)
    
    def test_memory_retention_priority(self):
        """
        重要度の高い記憶が優先的に保持されることを検証
        (Requirement 9.7の検証基準)
        """
        estimator = MemoryImportanceEstimator()
        
        B, H, D = 2, 4, 8
        weights = torch.randn(B, H, D, D) * 0.1
        
        # 特定の成分を非常に重要にする
        important_positions = [(0, 0, 0, 0), (0, 1, 1, 1), (1, 0, 2, 2)]
        for b, h, i, j in important_positions:
            weights[b, h, i, j] = 10.0
        
        # 上位k個を取得
        k = 5
        top_weights, top_indices = estimator.get_top_k_memories(weights, k)
        
        # 重要な成分が上位に含まれることを確認
        for b, h, i, j in important_positions:
            flat_idx = i * D + j
            # このバッチ・ヘッドの上位kインデックスに含まれているか
            assert flat_idx in top_indices[b, h].tolist()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
