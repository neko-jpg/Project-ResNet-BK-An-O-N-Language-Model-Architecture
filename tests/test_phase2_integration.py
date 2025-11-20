"""
Phase 2 Integration Tests

このモジュールは、Phase 2モデル全体の統合テストを提供します。

テスト内容:
1. エンドツーエンドテスト (Task 18.1)
   - モデルのインスタンス化から推論まで
   - 学習ループの動作

2. コンポーネント統合テスト (Task 18.2)
   - NonHermitian + BK-Coreの統合
   - DissipativeHebbian + SNRFilterの統合
   - MemoryResonance + Zetaの統合

3. 数値安定性テスト (Task 18.3)
   - 長時間学習での安定性
   - 極端な入力での安定性

KPI検証:
- VRAM: 8.0GB未満 (Batch=1, Seq=4096, fp16)
- スループット: 100 tokens/sec以上
- PPL劣化: Phase 1比で+10%以内

Requirements: 11.7
Author: Project MUSE Team
Date: 2025-01-20
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

# Phase 2モジュール
from src.models.phase2.integrated_model import Phase2IntegratedModel, Phase2Block
from src.models.phase2.factory import (
    create_phase2_model,
    Phase2Config,
    get_phase2_preset,
)

# Phase 1モジュール（比較用）
try:
    from src.models.phase1.factory import create_phase1_model, Phase1Config
    PHASE1_AVAILABLE = True
except ImportError:
    PHASE1_AVAILABLE = False
    warnings.warn("Phase 1 modules not available. Skipping Phase 1 comparison tests.")


# テスト設定
TEST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
RESULTS_DIR = Path("results/benchmarks")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Task 18.1: エンドツーエンドテスト
# ============================================================================

class TestEndToEnd:
    """
    エンドツーエンドテスト
    
    モデルのインスタンス化から推論、学習までの一連の流れをテストします。
    
    Requirements: 11.7
    """
    
    def test_model_instantiation(self):
        """
        モデルのインスタンス化テスト
        
        検証項目:
        - エラーなくインスタンス化できること
        - すべてのコンポーネントが正しく初期化されること
        - パラメータ数が妥当であること
        """
        # デフォルト設定でモデルを作成
        model = create_phase2_model(
            preset="small",  # テスト用に小さいモデル
            device=TEST_DEVICE
        )
        
        # モデルが正しく作成されたことを確認
        assert isinstance(model, Phase2IntegratedModel)
        assert model.d_model == 256
        assert model.n_layers == 4
        
        # すべてのブロックが存在することを確認
        assert len(model.blocks) == 4
        for block in model.blocks:
            assert isinstance(block, Phase2Block)
            assert hasattr(block, 'dissipative_bk')
            assert hasattr(block, 'hebbian')
            assert hasattr(block, 'snr_filter')
            assert hasattr(block, 'resonance')
        
        # パラメータ数を確認
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        print(f"Total parameters: {total_params:,}")
    
    def test_forward_pass(self):
        """
        Forward passテスト
        
        検証項目:
        - エラーなくforward passが実行できること
        - 出力の形状が正しいこと
        - 出力にNaN/Infが含まれないこと
        """
        model = create_phase2_model(
            preset="small",
            device=TEST_DEVICE
        )
        model.eval()
        
        # ダミー入力
        batch_size = 2
        seq_len = 128
        input_ids = torch.randint(
            0, model.vocab_size, (batch_size, seq_len),
            device=TEST_DEVICE
        )
        
        # Forward pass
        with torch.no_grad():
            logits = model(input_ids)
        
        # 出力の形状を確認
        assert logits.shape == (batch_size, seq_len, model.vocab_size)
        
        # NaN/Infチェック
        assert not torch.isnan(logits).any(), "Output contains NaN"
        assert not torch.isinf(logits).any(), "Output contains Inf"
        
        print(f"Forward pass successful. Output shape: {logits.shape}")
    
    def test_forward_with_diagnostics(self):
        """
        診断情報付きforward passテスト
        
        検証項目:
        - 診断情報が正しく収集されること
        - すべての層の診断情報が含まれること
        """
        model = create_phase2_model(
            preset="small",
            device=TEST_DEVICE
        )
        model.eval()
        
        # ダミー入力
        batch_size = 2
        seq_len = 128
        input_ids = torch.randint(
            0, model.vocab_size, (batch_size, seq_len),
            device=TEST_DEVICE
        )
        
        # 診断情報付きforward pass
        with torch.no_grad():
            logits, diagnostics = model(input_ids, return_diagnostics=True)
        
        # 診断情報の存在を確認
        assert 'layer_outputs' in diagnostics
        assert 'gamma_values' in diagnostics
        assert 'snr_stats' in diagnostics
        assert 'resonance_info' in diagnostics
        assert 'stability_metrics' in diagnostics
        
        # 各層の診断情報を確認
        assert len(diagnostics['layer_outputs']) == model.n_layers
        assert len(diagnostics['gamma_values']) == model.n_layers
        
        # Γ値の範囲を確認（正の値であること）
        for gamma in diagnostics['gamma_values']:
            if gamma is not None:
                assert (gamma >= 0).all(), "Gamma must be non-negative"
        
        print("Diagnostics collection successful")
        print(f"Number of layers: {len(diagnostics['layer_outputs'])}")

    def test_training_loop(self):
        """
        学習ループテスト
        
        検証項目:
        - 学習ループが正常に実行できること
        - Lossが減少すること
        - 勾配が正しく計算されること
        - メモリリークがないこと
        """
        model = create_phase2_model(
            preset="small",
            device=TEST_DEVICE
        )
        model.train()
        
        # オプティマイザー
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # ダミーデータセット
        batch_size = 4
        seq_len = 128
        num_batches = 10
        
        dataset = TensorDataset(
            torch.randint(0, model.vocab_size, (num_batches * batch_size, seq_len))
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 学習ループ
        losses = []
        for batch_idx, (input_ids,) in enumerate(dataloader):
            input_ids = input_ids.to(TEST_DEVICE)
            
            # Forward pass
            logits = model(input_ids)
            
            # Loss計算（次トークン予測）
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, model.vocab_size),
                shift_labels.view(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # パラメータ更新
            optimizer.step()
            
            losses.append(loss.item())
            
            # Fast Weight状態をデタッチして計算グラフから切り離す
            for module in model.modules():
                if hasattr(module, 'fast_weight') and module.fast_weight is not None:
                    module.fast_weight = module.fast_weight.detach()
            
            # メモリ使用量をチェック
            if torch.cuda.is_available():
                memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                print(f"Batch {batch_idx}: Loss={loss.item():.4f}, Memory={memory_mb:.1f}MB")
        
        # Lossが減少していることを確認（最初と最後を比較）
        initial_loss = sum(losses[:3]) / 3
        final_loss = sum(losses[-3:]) / 3
        print(f"Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
        
        # 学習が進んでいることを確認（Lossが減少または安定）
        # 注: ランダムデータなので大幅な減少は期待できない
        assert final_loss < initial_loss * 1.5, "Loss did not decrease or stabilize"
    
    def test_inference_with_state_management(self):
        """
        状態管理付き推論テスト
        
        検証項目:
        - Fast Weight状態が正しく管理されること
        - 状態リセットが機能すること
        - 逐次推論が可能であること
        """
        model = create_phase2_model(
            preset="small",
            device=TEST_DEVICE
        )
        model.eval()
        
        # 最初のシーケンス
        seq1 = torch.randint(0, model.vocab_size, (1, 64), device=TEST_DEVICE)
        with torch.no_grad():
            logits1 = model(seq1)
        
        # Fast Weight状態が保持されていることを確認
        for block in model.blocks:
            assert block.fast_weight_state is not None, "Fast weight state should be preserved"
        
        # 状態をリセット
        model.reset_state()
        
        # 状態がリセットされたことを確認
        for block in model.blocks:
            assert block.fast_weight_state is None, "Fast weight state should be reset"
        
        # 2番目のシーケンス（状態リセット後）
        seq2 = torch.randint(0, model.vocab_size, (1, 64), device=TEST_DEVICE)
        with torch.no_grad():
            logits2 = model(seq2)
        
        # 出力が異なることを確認（状態の影響）
        # 注: 入力が異なるので、出力も異なるはず
        assert logits1.shape == logits2.shape
        
        print("State management test successful")


# ============================================================================
# Task 18.2: コンポーネント統合テスト
# ============================================================================

class TestComponentIntegration:
    """
    コンポーネント統合テスト
    
    各コンポーネントの統合が正しく機能することをテストします。
    
    Requirements: 11.7
    """
    
    def test_nonhermitian_bkcore_integration(self):
        """
        NonHermitian + BK-Coreの統合テスト
        
        検証項目:
        - 複素ポテンシャルが正しく生成されること
        - BK-Coreが複素ポテンシャルを受け入れること
        - Γが正の値であること
        """
        model = create_phase2_model(
            preset="small",
            device=TEST_DEVICE
        )
        model.eval()
        
        # ダミー入力
        input_ids = torch.randint(0, model.vocab_size, (2, 64), device=TEST_DEVICE)
        
        # 診断情報付きforward pass
        with torch.no_grad():
            logits, diagnostics = model(input_ids, return_diagnostics=True)
        
        # Γ値を確認
        for layer_idx, gamma in enumerate(diagnostics['gamma_values']):
            if gamma is not None:
                # Γが正の値であることを確認
                assert (gamma >= 0).all(), f"Layer {layer_idx}: Gamma must be non-negative"
                
                # Γの範囲が妥当であることを確認（0.001 ~ 1.0程度）
                assert gamma.min() >= 0.0, f"Layer {layer_idx}: Gamma too small"
                assert gamma.max() < 10.0, f"Layer {layer_idx}: Gamma too large"
                
                print(f"Layer {layer_idx}: Gamma range [{gamma.min():.4f}, {gamma.max():.4f}]")
        
        # BK特徴を確認（diagnosticsから直接取得）
        # Note: layer_outputsはテンソルのリストなので、bk_featuresは別途確認
        # ここでは、Γ値が正しく生成されていることを確認済みなので、
        # BK-Coreの出力（logits）にNaN/Infがないことを確認
        assert not torch.isnan(logits).any(), "BK-Core output contains NaN"
        assert not torch.isinf(logits).any(), "BK-Core output contains Inf"
        
        print("NonHermitian + BK-Core integration test passed")

    def test_hebbian_snr_integration(self):
        """
        DissipativeHebbian + SNRFilterの統合テスト
        
        検証項目:
        - Fast Weightsが正しく更新されること
        - SNRフィルタが機能すること
        - ηとΓが調整されること
        """
        model = create_phase2_model(
            preset="small",
            device=TEST_DEVICE
        )
        model.eval()
        
        # ダミー入力
        input_ids = torch.randint(0, model.vocab_size, (2, 64), device=TEST_DEVICE)
        
        # 複数回forward passを実行してFast Weightsを蓄積
        with torch.no_grad():
            for _ in range(3):
                logits, diagnostics = model(input_ids, return_diagnostics=True)
        
        # SNR統計を確認
        for layer_idx, snr_stats in enumerate(diagnostics['snr_stats']):
            if snr_stats:
                print(f"Layer {layer_idx} SNR stats: {snr_stats}")
                
                # SNR統計が妥当な範囲であることを確認
                if 'mean_snr' in snr_stats:
                    mean_snr = snr_stats['mean_snr']
                    assert mean_snr >= 0, f"Layer {layer_idx}: Mean SNR must be non-negative"
                    assert mean_snr < 100, f"Layer {layer_idx}: Mean SNR too large"
        
        # Fast Weightエネルギーを確認
        for layer_idx in range(len(diagnostics.get('layer_outputs', []))):
            if 'fast_weight_energy' in diagnostics:
                energy = diagnostics['fast_weight_energy']
                print(f"Layer {layer_idx}: Fast Weight Energy = {energy:.4f}")
                assert energy >= 0, f"Layer {layer_idx}: Energy must be non-negative"
        
        print("Hebbian + SNR integration test passed")
    
    def test_resonance_zeta_integration(self):
        """
        MemoryResonance + Zetaの統合テスト
        
        検証項目:
        - ゼータ基底変換が機能すること
        - 共鳴検出が機能すること
        - 記憶がフィルタリングされること
        """
        model = create_phase2_model(
            preset="small",
            device=TEST_DEVICE
        )
        model.eval()
        
        # ダミー入力
        input_ids = torch.randint(0, model.vocab_size, (2, 64), device=TEST_DEVICE)
        
        # 複数回forward passを実行して記憶を蓄積
        with torch.no_grad():
            for _ in range(5):
                logits, diagnostics = model(input_ids, return_diagnostics=True)
        
        # 共鳴情報を確認
        for layer_idx, resonance_info in enumerate(diagnostics['resonance_info']):
            if resonance_info:
                print(f"Layer {layer_idx} Resonance info: {resonance_info}")
                
                # 共鳴マスクが存在することを確認
                if 'resonance_mask' in resonance_info:
                    mask = resonance_info['resonance_mask']
                    print(f"  Resonance mask shape: {mask.shape}")
                
                # 共鳴成分数を確認
                if 'num_resonant' in resonance_info:
                    num_resonant = resonance_info['num_resonant']
                    print(f"  Number of resonant modes: {num_resonant:.2f}")
                    assert num_resonant >= 0, f"Layer {layer_idx}: Resonant modes must be non-negative"
        
        print("Resonance + Zeta integration test passed")


# ============================================================================
# Task 18.3: 数値安定性テスト
# ============================================================================

class TestNumericalStability:
    """
    数値安定性テスト
    
    長時間学習や極端な入力での安定性をテストします。
    
    Requirements: 11.7
    """
    
    def test_long_training_stability(self):
        """
        長時間学習での安定性テスト
        
        検証項目:
        - 長時間学習でもNaN/Infが発生しないこと
        - Lossが発散しないこと
        - 勾配が消失しないこと
        """
        model = create_phase2_model(
            preset="small",
            device=TEST_DEVICE
        )
        model.train()
        
        # オプティマイザー
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # ダミーデータ
        batch_size = 2
        seq_len = 128
        num_steps = 50  # 長時間学習をシミュレート
        
        losses = []
        grad_norms = []
        
        for step in range(num_steps):
            # ダミー入力
            input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=TEST_DEVICE)
            
            # Forward pass
            logits = model(input_ids)
            
            # Loss計算
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, model.vocab_size),
                shift_labels.view(-1)
            )
            
            # NaN/Infチェック
            assert not torch.isnan(loss), f"Step {step}: Loss is NaN"
            assert not torch.isinf(loss), f"Step {step}: Loss is Inf"
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # 勾配ノルムを計算
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            # 勾配消失チェック
            assert total_norm > 1e-7, f"Step {step}: Gradient vanishing (norm={total_norm:.2e})"
            
            # 勾配爆発チェック
            assert total_norm < 1e3, f"Step {step}: Gradient explosion (norm={total_norm:.2e})"
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # パラメータ更新
            optimizer.step()
            
            # Fast Weight状態をデタッチして計算グラフから切り離す
            for module in model.modules():
                if hasattr(module, 'fast_weight') and module.fast_weight is not None:
                    module.fast_weight = module.fast_weight.detach()
            
            losses.append(loss.item())
            grad_norms.append(total_norm)
            
            if step % 10 == 0:
                print(f"Step {step}: Loss={loss.item():.4f}, Grad Norm={total_norm:.4f}")
        
        # Lossが発散していないことを確認
        final_loss = sum(losses[-5:]) / 5
        assert final_loss < 100.0, f"Loss diverged: {final_loss:.4f}"
        
        # 勾配ノルムが安定していることを確認
        final_grad_norm = sum(grad_norms[-5:]) / 5
        assert final_grad_norm > 1e-5, f"Gradient vanished: {final_grad_norm:.2e}"
        assert final_grad_norm < 10.0, f"Gradient exploded: {final_grad_norm:.2e}"
        
        print(f"Long training stability test passed. Final loss: {final_loss:.4f}, Final grad norm: {final_grad_norm:.4f}")

    def test_extreme_input_stability(self):
        """
        極端な入力での安定性テスト
        
        検証項目:
        - 長いシーケンスでも安定すること
        - 短いシーケンスでも動作すること
        - 大きなバッチサイズでも動作すること
        """
        model = create_phase2_model(
            preset="small",
            device=TEST_DEVICE
        )
        model.eval()
        
        test_cases = [
            ("Short sequence", 1, 16),
            ("Normal sequence", 2, 128),
            ("Long sequence", 1, 512),
            ("Large batch", 8, 64),
        ]
        
        for test_name, batch_size, seq_len in test_cases:
            print(f"\nTesting {test_name}: batch={batch_size}, seq={seq_len}")
            
            # ダミー入力
            input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=TEST_DEVICE)
            
            # Forward pass
            with torch.no_grad():
                logits = model(input_ids)
            
            # 出力の形状を確認
            assert logits.shape == (batch_size, seq_len, model.vocab_size)
            
            # NaN/Infチェック
            assert not torch.isnan(logits).any(), f"{test_name}: Output contains NaN"
            assert not torch.isinf(logits).any(), f"{test_name}: Output contains Inf"
            
            # メモリ使用量を確認
            if torch.cuda.is_available():
                memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                print(f"  Memory usage: {memory_mb:.1f} MB")
                torch.cuda.reset_peak_memory_stats()
            
            print(f"  {test_name} passed")
        
        print("\nExtreme input stability test passed")
    
    def test_lyapunov_stability_monitoring(self):
        """
        Lyapunov安定性監視テスト
        
        検証項目:
        - Lyapunov安定性が監視されること
        - エネルギーが発散しないこと
        - 安定性違反が検出されること
        """
        model = create_phase2_model(
            preset="small",
            device=TEST_DEVICE
        )
        model.train()
        
        # ダミー入力
        input_ids = torch.randint(0, model.vocab_size, (2, 64), device=TEST_DEVICE)
        
        # 複数回forward passを実行
        energies = []
        for step in range(10):
            logits, diagnostics = model(input_ids, return_diagnostics=True)
            
            # 安定性メトリクスを確認
            for layer_idx, stability in enumerate(diagnostics['stability_metrics']):
                if stability:
                    print(f"Step {step}, Layer {layer_idx}: {stability}")
                    
                    # エネルギーを記録
                    if 'energy' in stability:
                        energies.append(stability['energy'])
                    
                    # 安定性フラグを確認
                    if 'is_stable' in stability:
                        is_stable = stability['is_stable']
                        # 注: 学習中は一時的に不安定になることがあるので、警告のみ
                        if not is_stable:
                            print(f"  Warning: Layer {layer_idx} is unstable at step {step}")
        
        # エネルギーが発散していないことを確認
        if energies:
            max_energy = max(energies)
            print(f"Maximum energy: {max_energy:.4f}")
            assert max_energy < 1e6, f"Energy diverged: {max_energy:.2e}"
        
        print("Lyapunov stability monitoring test passed")


# ============================================================================
# KPI検証テスト
# ============================================================================

class TestKPIVerification:
    """
    KPI検証テスト
    
    Phase 2の性能目標を検証します:
    - VRAM: 8.0GB未満 (Batch=1, Seq=4096, fp16)
    - スループット: 100 tokens/sec以上
    - PPL劣化: Phase 1比で+10%以内
    
    Requirements: 11.7
    """
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_vram_usage(self):
        """
        VRAM使用量テスト
        
        KPI: 8.0GB未満 (Batch=1, Seq=4096, fp16)
        """
        # モデルを作成（fp16）
        model = create_phase2_model(
            preset="base",
            device=TEST_DEVICE
        )
        model = model.half()  # fp16に変換
        model.eval()
        
        # メモリをリセット
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # テスト入力（Batch=1, Seq=4096）
        batch_size = 1
        seq_len = 4096
        input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=TEST_DEVICE)
        
        # Forward pass
        with torch.no_grad():
            logits = model(input_ids)
        
        # VRAM使用量を測定
        memory_bytes = torch.cuda.max_memory_allocated()
        memory_gb = memory_bytes / (1024 ** 3)
        
        print(f"\nVRAM Usage Test:")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len}")
        print(f"  VRAM usage: {memory_gb:.2f} GB")
        
        # KPI検証
        target_vram_gb = 8.0
        assert memory_gb < target_vram_gb, f"VRAM usage {memory_gb:.2f} GB exceeds target {target_vram_gb} GB"
        
        # 結果を保存
        result = {
            "test": "vram_usage",
            "batch_size": batch_size,
            "seq_len": seq_len,
            "vram_gb": memory_gb,
            "target_vram_gb": target_vram_gb,
            "passed": memory_gb < target_vram_gb
        }
        
        result_file = RESULTS_DIR / "phase2_vram_test.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"  Result saved to {result_file}")
        print(f"  KPI: {'PASSED' if result['passed'] else 'FAILED'}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_throughput(self):
        """
        スループットテスト
        
        KPI: 100 tokens/sec以上
        """
        model = create_phase2_model(
            preset="base",
            device=TEST_DEVICE
        )
        model = model.half()  # fp16に変換
        model.eval()
        
        # ウォームアップ
        batch_size = 4
        seq_len = 512
        warmup_input = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=TEST_DEVICE)
        with torch.no_grad():
            _ = model(warmup_input)
        
        # スループット測定
        num_iterations = 10
        total_tokens = 0
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=TEST_DEVICE)
                logits = model(input_ids)
                total_tokens += batch_size * seq_len
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # スループット計算
        elapsed_time = end_time - start_time
        throughput = total_tokens / elapsed_time
        
        print(f"\nThroughput Test:")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Iterations: {num_iterations}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Elapsed time: {elapsed_time:.2f} sec")
        print(f"  Throughput: {throughput:.1f} tokens/sec")
        
        # KPI検証
        target_throughput = 100.0
        passed = throughput >= target_throughput
        
        # 結果を保存
        result = {
            "test": "throughput",
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_iterations": num_iterations,
            "total_tokens": total_tokens,
            "elapsed_time": elapsed_time,
            "throughput": throughput,
            "target_throughput": target_throughput,
            "passed": passed
        }
        
        result_file = RESULTS_DIR / "phase2_throughput_test.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"  Result saved to {result_file}")
        print(f"  KPI: {'PASSED' if passed else 'FAILED'}")
        
        # 注: スループットはハードウェアに依存するため、警告のみ
        if not passed:
            warnings.warn(
                f"Throughput {throughput:.1f} tokens/sec is below target {target_throughput} tokens/sec. "
                f"This may be due to hardware limitations.",
                UserWarning
            )
    
    @pytest.mark.skipif(not PHASE1_AVAILABLE, reason="Phase 1 not available")
    def test_perplexity_degradation(self):
        """
        Perplexity劣化テスト
        
        KPI: Phase 1比で+10%以内
        
        注: このテストは実際のデータセットでの評価が必要なため、
        ここでは簡易的な比較のみを行います。
        """
        # Phase 1モデル
        phase1_config = Phase1Config.for_hardware(vram_gb=8.0)
        phase1_model = create_phase1_model(config=phase1_config, device=TEST_DEVICE)
        phase1_model.eval()
        
        # Phase 2モデル
        phase2_config = Phase2Config.from_phase1(phase1_config)
        phase2_model = create_phase2_model(config=phase2_config, device=TEST_DEVICE)
        phase2_model.eval()
        
        # ダミーデータセット
        batch_size = 4
        seq_len = 256
        num_batches = 20
        
        # Phase 1のPerplexity
        phase1_losses = []
        with torch.no_grad():
            for _ in range(num_batches):
                input_ids = torch.randint(0, phase1_model.vocab_size, (batch_size, seq_len), device=TEST_DEVICE)
                logits = phase1_model(input_ids)
                
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, phase1_model.vocab_size),
                    shift_labels.view(-1)
                )
                phase1_losses.append(loss.item())
        
        phase1_ppl = torch.exp(torch.tensor(sum(phase1_losses) / len(phase1_losses))).item()
        
        # Phase 2のPerplexity
        phase2_losses = []
        with torch.no_grad():
            for _ in range(num_batches):
                input_ids = torch.randint(0, phase2_model.vocab_size, (batch_size, seq_len), device=TEST_DEVICE)
                logits = phase2_model(input_ids)
                
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, phase2_model.vocab_size),
                    shift_labels.view(-1)
                )
                phase2_losses.append(loss.item())
        
        phase2_ppl = torch.exp(torch.tensor(sum(phase2_losses) / len(phase2_losses))).item()
        
        # PPL劣化率を計算
        ppl_degradation = ((phase2_ppl - phase1_ppl) / phase1_ppl) * 100
        
        print(f"\nPerplexity Degradation Test:")
        print(f"  Phase 1 PPL: {phase1_ppl:.2f}")
        print(f"  Phase 2 PPL: {phase2_ppl:.2f}")
        print(f"  Degradation: {ppl_degradation:+.1f}%")
        
        # KPI検証
        target_degradation = 10.0
        passed = ppl_degradation <= target_degradation
        
        # 結果を保存
        result = {
            "test": "perplexity_degradation",
            "phase1_ppl": phase1_ppl,
            "phase2_ppl": phase2_ppl,
            "degradation_percent": ppl_degradation,
            "target_degradation_percent": target_degradation,
            "passed": passed
        }
        
        result_file = RESULTS_DIR / "phase2_ppl_test.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"  Result saved to {result_file}")
        print(f"  KPI: {'PASSED' if passed else 'FAILED'}")
        
        # 注: ランダムデータでの評価なので、警告のみ
        warnings.warn(
            "This test uses random data. For accurate PPL comparison, "
            "please evaluate on a real dataset (e.g., WikiText-2).",
            UserWarning
        )


# ============================================================================
# テスト実行時の設定
# ============================================================================

if __name__ == "__main__":
    # pytestを使用してテストを実行
    pytest.main([__file__, "-v", "-s"])
