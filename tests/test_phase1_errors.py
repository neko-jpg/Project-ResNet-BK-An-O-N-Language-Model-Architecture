"""
Unit tests for Phase 1 custom exception classes.

Requirements: 5.3, 10.4
"""

import pytest
import torch
from src.models.phase1.errors import (
    VRAMExhaustedError,
    NumericalInstabilityError,
    InvalidConfigError,
    HardwareCompatibilityError,
    raise_vram_exhausted,
    raise_numerical_instability,
    check_cuda_available,
)


class TestVRAMExhaustedError:
    """VRAMExhaustedErrorのテスト。"""
    
    def test_error_creation(self):
        """エラーが正しく作成されることを確認。"""
        error = VRAMExhaustedError(
            current_mb=9000.0,
            limit_mb=8000.0,
            suggestions=["Reduce batch size", "Enable checkpointing"]
        )
        
        assert error.current_mb == 9000.0
        assert error.limit_mb == 8000.0
        assert len(error.suggestions) == 2
    
    def test_error_message_format(self):
        """エラーメッセージが適切にフォーマットされることを確認。"""
        error = VRAMExhaustedError(
            current_mb=9000.0,
            limit_mb=8000.0,
            suggestions=["Reduce batch size"]
        )
        
        message = str(error)
        assert "9000.0MB" in message
        assert "8000.0MB" in message
        assert "Reduce batch size" in message
        assert "Suggestions" in message
    
    def test_raise_vram_exhausted_with_context(self):
        """コンテキストに応じた提案が生成されることを確認。"""
        with pytest.raises(VRAMExhaustedError) as exc_info:
            raise_vram_exhausted(
                current_mb=9000.0,
                limit_mb=8000.0,
                batch_size=8,
                seq_len=2048,
                ar_ssm_enabled=True,
                lns_enabled=True,
                gradient_checkpointing=False
            )
        
        error = exc_info.value
        suggestions_text = " ".join(error.suggestions)
        
        # バッチサイズ削減の提案を確認
        assert any("batch_size" in s.lower() for s in error.suggestions)
        
        # 勾配チェックポイントの提案を確認
        assert any("checkpointing" in s.lower() for s in error.suggestions)
        
        # LNS無効化の提案を確認
        assert any("lns" in s.lower() for s in error.suggestions)


class TestNumericalInstabilityError:
    """NumericalInstabilityErrorのテスト。"""
    
    def test_error_creation(self):
        """エラーが正しく作成されることを確認。"""
        diagnostics = {
            'has_nan': True,
            'max_value': 1e10,
            'gradient_norm': 100.0
        }
        error = NumericalInstabilityError(
            component="AR-SSM",
            diagnostics=diagnostics
        )
        
        assert error.component == "AR-SSM"
        assert error.diagnostics['has_nan'] is True
        assert error.diagnostics['max_value'] == 1e10
    
    def test_error_message_format(self):
        """エラーメッセージが適切にフォーマットされることを確認。"""
        diagnostics = {
            'has_nan': True,
            'gradient_norm': 100.0
        }
        error = NumericalInstabilityError(
            component="BK-Core",
            diagnostics=diagnostics
        )
        
        message = str(error)
        assert "BK-Core" in message
        assert "has_nan" in message
        assert "gradient_norm" in message
        assert "Recommended actions" in message
    
    def test_raise_numerical_instability(self):
        """raise_numerical_instability関数が正しく動作することを確認。"""
        with pytest.raises(NumericalInstabilityError) as exc_info:
            raise_numerical_instability(
                component="HTT-Embedding",
                has_nan=True,
                max_value=1e20,
                gradient_norm=50.0,
                custom_info="Additional diagnostic"
            )
        
        error = exc_info.value
        assert error.component == "HTT-Embedding"
        assert error.diagnostics['has_nan'] is True
        assert error.diagnostics['max_value'] == 1e20
        assert error.diagnostics['gradient_norm'] == 50.0
        assert error.diagnostics['custom_info'] == "Additional diagnostic"


class TestInvalidConfigError:
    """InvalidConfigErrorのテスト。"""
    
    def test_error_creation(self):
        """エラーが正しく作成されることを確認。"""
        error = InvalidConfigError(
            param_name="ar_ssm_max_rank",
            param_value=-1,
            reason="Must be positive"
        )
        
        assert error.param_name == "ar_ssm_max_rank"
        assert error.param_value == -1
        assert error.reason == "Must be positive"
    
    def test_error_message_format(self):
        """エラーメッセージが適切にフォーマットされることを確認。"""
        error = InvalidConfigError(
            param_name="htt_rank",
            param_value=0,
            reason="Must be >= 1"
        )
        
        message = str(error)
        assert "htt_rank" in message
        assert "0" in message
        assert "Must be >= 1" in message
        assert "Invalid configuration" in message


class TestHardwareCompatibilityError:
    """HardwareCompatibilityErrorのテスト。"""
    
    def test_error_creation_with_fallback(self):
        """フォールバック付きエラーが正しく作成されることを確認。"""
        error = HardwareCompatibilityError(
            required="CUDA 11.8+",
            available="CUDA 11.0",
            fallback="Upgrade CUDA toolkit"
        )
        
        assert error.required == "CUDA 11.8+"
        assert error.available == "CUDA 11.0"
        assert error.fallback == "Upgrade CUDA toolkit"
    
    def test_error_creation_without_fallback(self):
        """フォールバックなしエラーが正しく作成されることを確認。"""
        error = HardwareCompatibilityError(
            required="GPU with 24GB VRAM",
            available="GPU with 8GB VRAM",
            fallback=None
        )
        
        assert error.fallback is None
        message = str(error)
        assert "No fallback available" in message
    
    def test_error_message_format(self):
        """エラーメッセージが適切にフォーマットされることを確認。"""
        error = HardwareCompatibilityError(
            required="Triton library",
            available="Not installed",
            fallback="pip install triton"
        )
        
        message = str(error)
        assert "Triton library" in message
        assert "Not installed" in message
        assert "pip install triton" in message


class TestCheckCudaAvailable:
    """check_cuda_available関数のテスト。"""
    
    def test_cuda_check_when_available(self):
        """CUDAが利用可能な場合、エラーが発生しないことを確認。"""
        if torch.cuda.is_available():
            # エラーが発生しないことを確認
            check_cuda_available(require_triton=False)
        else:
            pytest.skip("CUDA not available")
    
    def test_cuda_check_when_not_available(self):
        """CUDAが利用できない場合、エラーが発生することを確認。"""
        if not torch.cuda.is_available():
            with pytest.raises(HardwareCompatibilityError) as exc_info:
                check_cuda_available(require_triton=False)
            
            error = exc_info.value
            assert "CUDA" in error.required
            assert "CPU" in error.available
        else:
            pytest.skip("CUDA is available")
    
    def test_triton_check(self):
        """Tritonチェックが正しく動作することを確認。"""
        try:
            import triton
            # Tritonが利用可能な場合、エラーが発生しないことを確認
            if torch.cuda.is_available():
                check_cuda_available(require_triton=True)
        except ImportError:
            # Tritonが利用できない場合、エラーが発生することを確認
            if torch.cuda.is_available():
                with pytest.raises(HardwareCompatibilityError) as exc_info:
                    check_cuda_available(require_triton=True)
                
                error = exc_info.value
                assert "Triton" in error.required


class TestErrorIntegration:
    """エラークラスの統合テスト。"""
    
    def test_all_errors_are_exceptions(self):
        """すべてのカスタムエラーがExceptionを継承していることを確認。"""
        assert issubclass(VRAMExhaustedError, Exception)
        assert issubclass(NumericalInstabilityError, Exception)
        assert issubclass(InvalidConfigError, Exception)
        assert issubclass(HardwareCompatibilityError, Exception)
    
    def test_error_messages_are_informative(self):
        """すべてのエラーメッセージが情報を含むことを確認。"""
        errors = [
            VRAMExhaustedError(9000, 8000, ["suggestion"]),
            NumericalInstabilityError("component", {"key": "value"}),
            InvalidConfigError("param", "value", "reason"),
            HardwareCompatibilityError("req", "avail", "fallback"),
        ]
        
        for error in errors:
            message = str(error)
            # メッセージが空でないことを確認
            assert len(message) > 0
            # メッセージが複数行であることを確認（詳細情報を含む）
            assert "\n" in message
