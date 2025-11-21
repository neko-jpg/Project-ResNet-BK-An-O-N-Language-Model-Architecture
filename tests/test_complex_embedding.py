"""
Unit Tests for Complex Embedding Layer

このテストスイートは、ComplexEmbeddingの機能を検証します。

Test Coverage:
    - 基本的な動作（forward pass）
    - ComplexTensor形式の出力
    - メモリ使用量の測定
    - Phase 2互換性
    - 統計情報の取得
    - 干渉効果の分析

Requirements:
    - Requirement 1.14: 出力がComplexTensor形式であることを確認
    - Requirement 1.14: メモリ使用量を測定
"""

import torch
import torch.nn as nn
import pytest
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.phase3.complex_embedding import (
    ComplexEmbedding,
    convert_phase2_embedding_to_complex,
    analyze_complex_embedding_interference
)
from src.models.phase3.complex_tensor import ComplexTensor


class TestComplexEmbeddingBasic:
    """ComplexEmbeddingの基本機能テスト"""
    
    def test_initialization(self):
        """初期化のテスト"""
        embedding = ComplexEmbedding(
            vocab_size=1000,
            d_model=128,
            max_seq_len=512
        )
        
        assert embedding.vocab_size == 1000
        assert embedding.d_model == 128
        assert embedding.max_seq_len == 512
        assert embedding.use_complex32 is True
    
    def test_forward_pass_complex32(self):
        """Forward pass（complex32モード）のテスト"""
        embedding = ComplexEmbedding(
            vocab_size=1000,
            d_model=128,
            use_complex32=True
        )
        
        # 入力
        input_ids = torch.randint(0, 1000, (4, 64))
        
        # Forward
        z = embedding(input_ids)
        
        # 出力検証
        assert isinstance(z, ComplexTensor), "Output should be ComplexTensor"
        assert z.shape == (4, 64, 128), f"Expected shape (4, 64, 128), got {z.shape}"
        assert z.dtype == torch.float16, f"Expected dtype float16, got {z.dtype}"
        
        # NaN/Infチェック
        assert not torch.isnan(z.real).any(), "Real part contains NaN"
        assert not torch.isnan(z.imag).any(), "Imag part contains NaN"
        assert not torch.isinf(z.real).any(), "Real part contains Inf"
        assert not torch.isinf(z.imag).any(), "Imag part contains Inf"
    
    def test_forward_pass_complex64(self):
        """Forward pass（complex64モード）のテスト"""
        embedding = ComplexEmbedding(
            vocab_size=1000,
            d_model=128,
            use_complex32=False
        )
        
        # 入力
        input_ids = torch.randint(0, 1000, (4, 64))
        
        # Forward
        z = embedding(input_ids)
        
        # 出力検証
        assert z.dtype == torch.complex64, f"Expected dtype complex64, got {z.dtype}"
        assert z.shape == (4, 64, 128), f"Expected shape (4, 64, 128), got {z.shape}"
        
        # NaN/Infチェック
        assert not torch.isnan(z).any(), "Output contains NaN"
        assert not torch.isinf(z).any(), "Output contains Inf"
    
    def test_custom_positions(self):
        """カスタム位置インデックスのテスト"""
        embedding = ComplexEmbedding(
            vocab_size=1000,
            d_model=128
        )
        
        # 入力
        input_ids = torch.randint(0, 1000, (4, 64))
        positions = torch.arange(64).unsqueeze(0).expand(4, -1)
        
        # Forward
        z = embedding(input_ids, positions=positions)
        
        # 出力検証
        assert isinstance(z, ComplexTensor)
        assert z.shape == (4, 64, 128)
    
    def test_sequence_length_validation(self):
        """シーケンス長の検証テスト"""
        embedding = ComplexEmbedding(
            vocab_size=1000,
            d_model=128,
            max_seq_len=512
        )
        
        # 正常なシーケンス長
        input_ids = torch.randint(0, 1000, (4, 512))
        z = embedding(input_ids)
        assert z.shape == (4, 512, 128)
        
        # 超過したシーケンス長
        input_ids_long = torch.randint(0, 1000, (4, 600))
        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            embedding(input_ids_long)
    
    def test_input_shape_validation(self):
        """入力形状の検証テスト"""
        embedding = ComplexEmbedding(
            vocab_size=1000,
            d_model=128
        )
        
        # 不正な形状（1D）
        input_ids_1d = torch.randint(0, 1000, (64,))
        with pytest.raises(ValueError, match="must be 2D"):
            embedding(input_ids_1d)
        
        # 不正な形状（3D）
        input_ids_3d = torch.randint(0, 1000, (4, 64, 10))
        with pytest.raises(ValueError, match="must be 2D"):
            embedding(input_ids_3d)


class TestComplexEmbeddingMemory:
    """メモリ使用量のテスト（Requirement 1.14）"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_usage_complex32_vs_complex64(self):
        """complex32とcomplex64のメモリ使用量比較"""
        vocab_size = 50000
        d_model = 512
        batch_size = 4
        seq_len = 1024
        
        # complex32モード
        torch.cuda.reset_peak_memory_stats()
        embedding_32 = ComplexEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            use_complex32=True
        ).cuda()
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
        z_32 = embedding_32(input_ids)
        
        memory_32 = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        # complex64モード
        torch.cuda.reset_peak_memory_stats()
        embedding_64 = ComplexEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            use_complex32=False
        ).cuda()
        
        z_64 = embedding_64(input_ids)
        
        memory_64 = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        # メモリ削減率の検証
        reduction_ratio = memory_32 / memory_64
        
        print(f"\nMemory Usage:")
        print(f"  complex32: {memory_32:.2f} MB")
        print(f"  complex64: {memory_64:.2f} MB")
        print(f"  Reduction: {(1 - reduction_ratio) * 100:.1f}%")
        
        # complex32はcomplex64の約50%のメモリを使用するはず
        # 実際には他のオーバーヘッドがあるため、60%以下を許容
        assert reduction_ratio < 0.6, \
            f"complex32 should use <60% memory of complex64, got {reduction_ratio:.2%}"
    
    def test_parameter_count(self):
        """パラメータ数のテスト"""
        vocab_size = 1000
        d_model = 128
        
        embedding = ComplexEmbedding(
            vocab_size=vocab_size,
            d_model=d_model
        )
        
        # パラメータ数を計算
        total_params = sum(p.numel() for p in embedding.parameters())
        
        # 期待値: Token Embedding (Real + Imag) + Position Embedding
        expected_params = (
            vocab_size * d_model +  # Token Embedding Real
            vocab_size * d_model +  # Token Embedding Imag
            embedding.max_seq_len * d_model  # Position Embedding
        )
        
        assert total_params == expected_params, \
            f"Expected {expected_params} parameters, got {total_params}"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_efficiency_report(self):
        """メモリ効率レポートの生成"""
        configs = [
            {'vocab_size': 10000, 'd_model': 256, 'seq_len': 512},
            {'vocab_size': 50000, 'd_model': 512, 'seq_len': 1024},
            {'vocab_size': 50000, 'd_model': 768, 'seq_len': 2048},
        ]
        
        print("\n" + "="*60)
        print("Complex Embedding Memory Efficiency Report")
        print("="*60)
        
        for config in configs:
            vocab_size = config['vocab_size']
            d_model = config['d_model']
            seq_len = config['seq_len']
            
            # complex32
            torch.cuda.reset_peak_memory_stats()
            embedding_32 = ComplexEmbedding(
                vocab_size=vocab_size,
                d_model=d_model,
                use_complex32=True
            ).cuda()
            
            input_ids = torch.randint(0, vocab_size, (2, seq_len)).cuda()
            _ = embedding_32(input_ids)
            memory_32 = torch.cuda.max_memory_allocated() / 1024**2
            
            # complex64
            torch.cuda.reset_peak_memory_stats()
            embedding_64 = ComplexEmbedding(
                vocab_size=vocab_size,
                d_model=d_model,
                use_complex32=False
            ).cuda()
            
            _ = embedding_64(input_ids)
            memory_64 = torch.cuda.max_memory_allocated() / 1024**2
            
            reduction = (1 - memory_32 / memory_64) * 100
            
            print(f"\nConfig: vocab={vocab_size}, d_model={d_model}, seq_len={seq_len}")
            print(f"  complex32: {memory_32:.2f} MB")
            print(f"  complex64: {memory_64:.2f} MB")
            print(f"  Reduction: {reduction:.1f}%")
            
            # クリーンアップ
            del embedding_32, embedding_64
            torch.cuda.empty_cache()


class TestComplexEmbeddingStatistics:
    """統計情報のテスト"""
    
    def test_get_statistics(self):
        """統計情報の取得テスト"""
        embedding = ComplexEmbedding(
            vocab_size=1000,
            d_model=128
        )
        
        # Forward pass
        input_ids = torch.randint(0, 1000, (4, 64))
        _ = embedding(input_ids)
        
        # 統計情報の取得
        stats = embedding.get_statistics()
        
        # 検証
        assert 'call_count' in stats
        assert 'real_norm' in stats
        assert 'imag_norm' in stats
        assert 'real_mean' in stats
        assert 'imag_mean' in stats
        assert 'real_std' in stats
        assert 'imag_std' in stats
        
        assert stats['call_count'] == 1
        assert stats['real_norm'] > 0
        assert stats['imag_norm'] > 0
    
    def test_get_embedding_weight(self):
        """埋め込み重みの取得テスト"""
        embedding = ComplexEmbedding(
            vocab_size=1000,
            d_model=128
        )
        
        # 実部の重み
        real_weight = embedding.get_embedding_weight('real')
        assert real_weight.shape == (1000, 128)
        
        # 虚部の重み
        imag_weight = embedding.get_embedding_weight('imag')
        assert imag_weight.shape == (1000, 128)
        
        # 不正な引数
        with pytest.raises(ValueError):
            embedding.get_embedding_weight('invalid')


class TestComplexEmbeddingPhase2Compatibility:
    """Phase 2互換性のテスト"""
    
    def test_convert_phase2_to_complex(self):
        """Phase 2 EmbeddingからComplexEmbeddingへの変換テスト"""
        # Phase 2のEmbedding
        phase2_emb = nn.Embedding(1000, 128)
        
        # Phase 3のComplexEmbeddingに変換
        phase3_emb = convert_phase2_embedding_to_complex(phase2_emb)
        
        # 検証
        assert phase3_emb.vocab_size == 1000
        assert phase3_emb.d_model == 128
        
        # 実部の重みがコピーされていることを確認
        assert torch.allclose(
            phase3_emb.token_embedding_real.weight.data,
            phase2_emb.weight.data,
            atol=1e-5
        )
        
        # 虚部がゼロで初期化されていることを確認
        assert torch.allclose(
            phase3_emb.token_embedding_imag.weight.data,
            torch.zeros_like(phase3_emb.token_embedding_imag.weight.data),
            atol=1e-8
        )


class TestComplexEmbeddingInterference:
    """干渉効果の分析テスト"""
    
    def test_analyze_interference(self):
        """干渉効果の分析テスト"""
        embedding = ComplexEmbedding(
            vocab_size=1000,
            d_model=128
        )
        
        # トークンID
        token_ids = torch.tensor([10, 20, 30])
        
        # 干渉効果の分析
        analysis = analyze_complex_embedding_interference(embedding, token_ids)
        
        # 検証
        assert 'magnitude' in analysis
        assert 'phase' in analysis
        assert 'interference' in analysis
        
        assert analysis['magnitude'].shape == (3,)
        assert analysis['phase'].shape == (3,)
        assert analysis['interference'].shape == (3, 3)
        
        # 干渉行列は対称行列
        assert torch.allclose(
            analysis['interference'],
            analysis['interference'].T,
            atol=1e-5
        )
        
        # 対角成分はゼロ
        assert torch.allclose(
            torch.diag(analysis['interference']),
            torch.zeros(3),
            atol=1e-5
        )


class TestComplexEmbeddingGradient:
    """勾配計算のテスト"""
    
    def test_backward_pass(self):
        """Backward passのテスト"""
        embedding = ComplexEmbedding(
            vocab_size=1000,
            d_model=128,
            use_complex32=False  # complex64で勾配計算
        )
        
        # 入力
        input_ids = torch.randint(0, 1000, (4, 64))
        
        # Forward
        z = embedding(input_ids)
        
        # 損失（ダミー）
        loss = z.abs().sum()
        
        # Backward
        loss.backward()
        
        # 勾配の検証
        assert embedding.token_embedding_real.weight.grad is not None
        assert embedding.token_embedding_imag.weight.grad is not None
        
        # NaN/Infチェック
        assert not torch.isnan(embedding.token_embedding_real.weight.grad).any()
        assert not torch.isnan(embedding.token_embedding_imag.weight.grad).any()
        assert not torch.isinf(embedding.token_embedding_real.weight.grad).any()
        assert not torch.isinf(embedding.token_embedding_imag.weight.grad).any()
    
    def test_gradient_flow_complex32(self):
        """complex32モードでの勾配伝播テスト"""
        embedding = ComplexEmbedding(
            vocab_size=1000,
            d_model=128,
            use_complex32=True
        )
        
        # 入力
        input_ids = torch.randint(0, 1000, (4, 64))
        
        # Forward
        z = embedding(input_ids)
        
        # ComplexTensorから実数損失を計算
        loss = z.abs().sum()
        
        # Backward
        loss.backward()
        
        # 勾配の検証
        assert embedding.token_embedding_real.weight.grad is not None
        assert embedding.token_embedding_imag.weight.grad is not None
        
        # 勾配のノルムが正の値
        real_grad_norm = embedding.token_embedding_real.weight.grad.norm().item()
        imag_grad_norm = embedding.token_embedding_imag.weight.grad.norm().item()
        
        assert real_grad_norm > 0, "Real gradient should be non-zero"
        assert imag_grad_norm > 0, "Imag gradient should be non-zero"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestComplexEmbeddingCUDA:
    """CUDA環境でのテスト"""
    
    def test_cuda_forward(self):
        """CUDA環境でのForward passテスト"""
        embedding = ComplexEmbedding(
            vocab_size=1000,
            d_model=128
        ).cuda()
        
        # 入力
        input_ids = torch.randint(0, 1000, (4, 64)).cuda()
        
        # Forward
        z = embedding(input_ids)
        
        # デバイスの検証
        assert z.device.type == 'cuda'
        assert z.real.device.type == 'cuda'
        assert z.imag.device.type == 'cuda'
    
    def test_cuda_backward(self):
        """CUDA環境でのBackward passテスト"""
        embedding = ComplexEmbedding(
            vocab_size=1000,
            d_model=128,
            use_complex32=False
        ).cuda()
        
        # 入力
        input_ids = torch.randint(0, 1000, (4, 64)).cuda()
        
        # Forward
        z = embedding(input_ids)
        
        # 損失
        loss = z.abs().sum()
        
        # Backward
        loss.backward()
        
        # 勾配の検証
        assert embedding.token_embedding_real.weight.grad.device.type == 'cuda'
        assert embedding.token_embedding_imag.weight.grad.device.type == 'cuda'


if __name__ == '__main__':
    # テストの実行
    pytest.main([__file__, '-v', '--tb=short'])
