"""
Unit tests for HamiltonianNeuralODE with Automatic Fallback

このテストスイートは、HamiltonianNeuralODEの3段階フォールバック機構を検証します。

テスト項目:
1. 基本的なForward/Backward pass
2. Symplectic Adjointモードの動作
3. Checkpointingモードの動作
4. Full Backpropモードの動作
5. フォールバック機構の動作
6. モード切り替えの動作

Requirements: 2.17
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from src.models.phase3.hamiltonian_ode import HamiltonianNeuralODE
from src.models.phase3.symplectic_adjoint import ReconstructionError


class TestHamiltonianNeuralODEBasic:
    """基本的な動作テスト"""
    
    def test_initialization(self):
        """初期化のテスト"""
        ode = HamiltonianNeuralODE(d_model=64, potential_type='mlp')
        
        assert ode.d_model == 64
        assert ode.dt == 0.1
        assert ode.recon_threshold == 1e-5
        assert ode.mode == 'symplectic_adjoint'
        assert ode.fallback_count == 0
    
    def test_forward_pass_shape(self):
        """Forward passの出力形状テスト"""
        ode = HamiltonianNeuralODE(d_model=64, potential_type='mlp')
        
        # 初期状態: (B, N, 2D) = [q, p]
        B, N, D = 2, 10, 64
        x0 = torch.randn(B, N, 2 * D)
        
        # Forward pass
        x_final = ode(x0, t_span=(0, 1))
        
        # 形状チェック
        assert x_final.shape == (B, N, 2 * D)
    
    def test_backward_pass(self):
        """Backward passの動作テスト"""
        ode = HamiltonianNeuralODE(d_model=64, potential_type='mlp')
        
        B, N, D = 2, 10, 64
        x0 = torch.randn(B, N, 2 * D, requires_grad=True)
        
        # Forward pass
        x_final = ode(x0, t_span=(0, 1))
        
        # Loss計算
        loss = x_final.sum()
        
        # Backward pass
        loss.backward()
        
        # 勾配チェック
        assert x0.grad is not None
        assert not torch.isnan(x0.grad).any()
        assert not torch.isinf(x0.grad).any()
        
        # パラメータ勾配チェック
        for name, param in ode.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN in gradient of {name}"
                assert not torch.isinf(param.grad).any(), f"Inf in gradient of {name}"


class TestSymplecticAdjointMode:
    """Symplectic Adjointモードのテスト"""
    
    def test_symplectic_adjoint_mode(self):
        """Symplectic Adjointモードの動作テスト"""
        ode = HamiltonianNeuralODE(d_model=32, potential_type='mlp')
        ode.set_mode('symplectic_adjoint')
        
        B, N, D = 2, 8, 32
        x0 = torch.randn(B, N, 2 * D, requires_grad=True)
        
        # Forward pass
        x_final = ode(x0, t_span=(0, 0.5))  # 短い時間で安定性確保
        
        # Backward pass
        loss = x_final.sum()
        loss.backward()
        
        # モードが変わっていないことを確認
        assert ode.mode == 'symplectic_adjoint'
        assert ode.fallback_count == 0
    
    def test_symplectic_adjoint_memory_efficiency(self):
        """Symplectic AdjointのO(1)メモリ効率テスト"""
        ode = HamiltonianNeuralODE(d_model=32, potential_type='mlp')
        ode.set_mode('symplectic_adjoint')
        
        B, N, D = 2, 8, 32
        x0 = torch.randn(B, N, 2 * D, requires_grad=True)
        
        # メモリ使用量測定
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            x0 = x0.cuda()
            ode = ode.cuda()
            
            x_final = ode(x0, t_span=(0, 1))
            loss = x_final.sum()
            loss.backward()
            
            memory_mb = torch.cuda.max_memory_allocated() / 1024**2
            print(f"Symplectic Adjoint memory: {memory_mb:.2f} MB")
            
            # メモリ使用量が合理的な範囲内であることを確認
            # (具体的な閾値は環境依存)
            assert memory_mb < 1000, f"Memory usage too high: {memory_mb:.2f} MB"


class TestCheckpointingMode:
    """Checkpointingモードのテスト"""
    
    def test_checkpointing_mode(self):
        """Checkpointingモードの動作テスト"""
        ode = HamiltonianNeuralODE(d_model=32, potential_type='mlp', checkpoint_interval=5)
        ode.set_mode('checkpointing')
        
        B, N, D = 2, 8, 32
        x0 = torch.randn(B, N, 2 * D, requires_grad=True)
        
        # Forward pass
        x_final = ode(x0, t_span=(0, 1))
        
        # Backward pass
        loss = x_final.sum()
        loss.backward()
        
        # モードが変わっていないことを確認
        assert ode.mode == 'checkpointing'
        
        # 勾配チェック
        assert x0.grad is not None
        assert not torch.isnan(x0.grad).any()


class TestFullBackpropMode:
    """Full Backpropモードのテスト"""
    
    def test_full_backprop_mode(self):
        """Full Backpropモードの動作テスト"""
        ode = HamiltonianNeuralODE(d_model=32, potential_type='mlp')
        ode.set_mode('full_backprop')
        
        B, N, D = 2, 8, 32
        x0 = torch.randn(B, N, 2 * D, requires_grad=True)
        
        # Forward pass（警告が出ることを確認）
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            x_final = ode(x0, t_span=(0, 1))
            
            # 警告が出ていることを確認
            assert len(w) > 0
            assert "full backprop" in str(w[0].message).lower()
        
        # Backward pass
        loss = x_final.sum()
        loss.backward()
        
        # モードが変わっていないことを確認
        assert ode.mode == 'full_backprop'
        
        # 勾配チェック
        assert x0.grad is not None
        assert not torch.isnan(x0.grad).any()


class TestFallbackMechanism:
    """フォールバック機構のテスト"""
    
    def test_fallback_to_checkpointing(self):
        """Symplectic Adjoint → Checkpointingへのフォールバックテスト"""
        # 非常に小さい閾値を設定して、意図的にフォールバックを発生させる
        ode = HamiltonianNeuralODE(
            d_model=32,
            potential_type='mlp',
            recon_threshold=1e-10  # 非常に厳しい閾値
        )
        
        B, N, D = 2, 8, 32
        x0 = torch.randn(B, N, 2 * D, requires_grad=True)
        
        # Forward pass
        x_final = ode(x0, t_span=(0, 1))
        
        # Backward pass（ReconstructionErrorが発生する可能性）
        loss = x_final.sum()
        
        try:
            loss.backward()
            # Backward成功した場合、勾配チェック
            assert x0.grad is not None
            assert not torch.isnan(x0.grad).any()
        except ReconstructionError as e:
            # ReconstructionErrorが発生した場合、それは期待通り
            print(f"ReconstructionError caught: {e}")
            print(f"Error: {e.error:.2e}, Threshold: {e.threshold:.2e}, Step: {e.step}")
            
            # 手動でCheckpointingモードに切り替えて再試行
            ode.set_mode('checkpointing')
            x0.grad = None  # 勾配をリセット
            
            # 再度Forward/Backward
            x_final = ode(x0, t_span=(0, 1))
            loss = x_final.sum()
            loss.backward()
            
            # 勾配チェック
            assert x0.grad is not None
            assert not torch.isnan(x0.grad).any()
            
            print("Successfully switched to checkpointing mode")
    
    def test_reset_to_symplectic(self):
        """Symplectic Adjointモードへのリセットテスト"""
        ode = HamiltonianNeuralODE(d_model=32, potential_type='mlp')
        
        # Checkpointingモードに切り替え
        ode.set_mode('checkpointing')
        ode.fallback_count = 5
        
        # リセット
        ode.reset_to_symplectic()
        
        # モードとカウンターがリセットされていることを確認
        assert ode.mode == 'symplectic_adjoint'
        assert ode.fallback_count == 0


class TestModeSwitching:
    """モード切り替えのテスト"""
    
    def test_manual_mode_switching(self):
        """手動モード切り替えのテスト"""
        ode = HamiltonianNeuralODE(d_model=32, potential_type='mlp')
        
        # 各モードに切り替えてテスト
        modes = ['symplectic_adjoint', 'checkpointing', 'full_backprop']
        
        for mode in modes:
            ode.set_mode(mode)
            assert ode.mode == mode
            
            # Forward passが動作することを確認
            B, N, D = 2, 8, 32
            x0 = torch.randn(B, N, 2 * D)
            x_final = ode(x0, t_span=(0, 0.5))
            
            assert x_final.shape == x0.shape
    
    def test_invalid_mode(self):
        """無効なモード設定のテスト"""
        ode = HamiltonianNeuralODE(d_model=32, potential_type='mlp')
        
        with pytest.raises(ValueError):
            ode.set_mode('invalid_mode')


class TestDiagnostics:
    """診断情報のテスト"""
    
    def test_get_diagnostics(self):
        """診断情報取得のテスト"""
        ode = HamiltonianNeuralODE(d_model=32, potential_type='mlp')
        
        # 初期状態の診断情報
        diag = ode.get_diagnostics()
        
        assert 'mode' in diag
        assert 'fallback_count' in diag
        assert 'recon_error_history' in diag
        
        assert diag['mode'] == 'symplectic_adjoint'
        assert diag['fallback_count'] == 0
        assert len(diag['recon_error_history']) == 0


class TestIntegrationWithBKCore:
    """BK-Coreとの統合テスト"""
    
    def test_bk_core_potential(self):
        """BK-Coreポテンシャルのテスト"""
        try:
            ode = HamiltonianNeuralODE(d_model=64, potential_type='bk_core')
            
            B, N, D = 2, 8, 64
            x0 = torch.randn(B, N, 2 * D, requires_grad=True)
            
            # Forward pass
            x_final = ode(x0, t_span=(0, 0.5))
            
            # Backward pass
            loss = x_final.sum()
            loss.backward()
            
            # 勾配チェック
            assert x0.grad is not None
            assert not torch.isnan(x0.grad).any()
            
        except ImportError:
            pytest.skip("BK-Core not available")


class TestNumericalStability:
    """数値安定性のテスト"""
    
    def test_no_nan_inf_in_output(self):
        """出力にNaN/Infが含まれないことを確認"""
        ode = HamiltonianNeuralODE(d_model=32, potential_type='mlp')
        
        B, N, D = 2, 8, 32
        x0 = torch.randn(B, N, 2 * D)
        
        # Forward pass
        x_final = ode(x0, t_span=(0, 1))
        
        # NaN/Infチェック
        assert not torch.isnan(x_final).any(), "NaN detected in output"
        assert not torch.isinf(x_final).any(), "Inf detected in output"
    
    def test_gradient_stability(self):
        """勾配の安定性テスト"""
        ode = HamiltonianNeuralODE(d_model=32, potential_type='mlp')
        
        B, N, D = 2, 8, 32
        x0 = torch.randn(B, N, 2 * D, requires_grad=True)
        
        # Forward pass
        x_final = ode(x0, t_span=(0, 1))
        
        # Backward pass
        loss = x_final.sum()
        loss.backward()
        
        # 勾配の大きさをチェック
        grad_norm = x0.grad.norm().item()
        assert grad_norm > 1e-6, f"Gradient too small: {grad_norm}"
        assert grad_norm < 1e3, f"Gradient too large: {grad_norm}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
