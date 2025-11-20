# Phase 2 統合テスト クイックリファレンス

## 概要

Phase 2統合テストの使用方法とKPI検証手順のクイックリファレンスです。

## テスト実行コマンド

### 全テスト実行

```bash
# すべての統合テストを実行
pytest tests/test_phase2_integration.py -v

# 詳細な出力付き
pytest tests/test_phase2_integration.py -v -s

# 失敗時に即座に停止
pytest tests/test_phase2_integration.py -v -x
```

### カテゴリ別実行

```bash
# エンドツーエンドテスト
pytest tests/test_phase2_integration.py::TestEndToEnd -v

# コンポーネント統合テスト
pytest tests/test_phase2_integration.py::TestComponentIntegration -v

# 数値安定性テスト
pytest tests/test_phase2_integration.py::TestNumericalStability -v

# KPI検証テスト（CUDA必須）
pytest tests/test_phase2_integration.py::TestKPIVerification -v
```

### 個別テスト実行

```bash
# モデルインスタンス化
pytest tests/test_phase2_integration.py::TestEndToEnd::test_model_instantiation -v

# Forward pass
pytest tests/test_phase2_integration.py::TestEndToEnd::test_forward_pass -v

# 学習ループ
pytest tests/test_phase2_integration.py::TestEndToEnd::test_training_loop -v

# VRAM使用量（CUDA必須）
pytest tests/test_phase2_integration.py::TestKPIVerification::test_vram_usage -v

# スループット（CUDA必須）
pytest tests/test_phase2_integration.py::TestKPIVerification::test_throughput -v
```

## KPI検証手順

### 1. VRAM使用量テスト

**目標**: 8.0GB未満 (Batch=1, Seq=4096, fp16)

```bash
# CUDA環境で実行
pytest tests/test_phase2_integration.py::TestKPIVerification::test_vram_usage -v -s

# 結果確認
cat results/benchmarks/phase2_vram_test.json
```

**期待される出力**:
```json
{
  "test": "vram_usage",
  "batch_size": 1,
  "seq_len": 4096,
  "vram_gb": 7.5,
  "target_vram_gb": 8.0,
  "passed": true
}
```

### 2. スループットテスト

**目標**: 100 tokens/sec以上

```bash
# CUDA環境で実行
pytest tests/test_phase2_integration.py::TestKPIVerification::test_throughput -v -s

# 結果確認
cat results/benchmarks/phase2_throughput_test.json
```

**期待される出力**:
```json
{
  "test": "throughput",
  "batch_size": 4,
  "seq_len": 512,
  "throughput": 150.5,
  "target_throughput": 100.0,
  "passed": true
}
```

### 3. PPL劣化テスト

**目標**: Phase 1比で+10%以内

```bash
# Phase 1が利用可能な環境で実行
pytest tests/test_phase2_integration.py::TestKPIVerification::test_perplexity_degradation -v -s

# 結果確認
cat results/benchmarks/phase2_ppl_test.json
```

**期待される出力**:
```json
{
  "test": "perplexity_degradation",
  "phase1_ppl": 50.2,
  "phase2_ppl": 52.1,
  "degradation_percent": 3.8,
  "target_degradation_percent": 10.0,
  "passed": true
}
```

## テストケース一覧

### TestEndToEnd（エンドツーエンド）

| テスト名 | 検証内容 | 実行時間 |
|---------|---------|---------|
| `test_model_instantiation` | モデルのインスタンス化 | ~5秒 |
| `test_forward_pass` | Forward pass | ~7秒 |
| `test_forward_with_diagnostics` | 診断情報収集 | ~10秒 |
| `test_training_loop` | 学習ループ | ~30秒 |
| `test_inference_with_state_management` | 状態管理 | ~10秒 |

### TestComponentIntegration（コンポーネント統合）

| テスト名 | 検証内容 | 実行時間 |
|---------|---------|---------|
| `test_nonhermitian_bkcore_integration` | NonHermitian + BK-Core | ~10秒 |
| `test_hebbian_snr_integration` | Hebbian + SNR | ~15秒 |
| `test_resonance_zeta_integration` | Resonance + Zeta | ~15秒 |

### TestNumericalStability（数値安定性）

| テスト名 | 検証内容 | 実行時間 |
|---------|---------|---------|
| `test_long_training_stability` | 長時間学習 | ~60秒 |
| `test_extreme_input_stability` | 極端な入力 | ~20秒 |
| `test_lyapunov_stability_monitoring` | Lyapunov安定性 | ~15秒 |

### TestKPIVerification（KPI検証）

| テスト名 | 検証内容 | 要件 | 実行時間 |
|---------|---------|------|---------|
| `test_vram_usage` | VRAM使用量 | CUDA | ~30秒 |
| `test_throughput` | スループット | CUDA | ~20秒 |
| `test_perplexity_degradation` | PPL劣化 | Phase 1 | ~40秒 |

## トラブルシューティング

### Triton未インストール

**症状**:
```
UserWarning: Triton kernel failed: No module named 'triton'
```

**対応**:
- PyTorch実装へのフォールバックが自動的に行われます
- 性能テストを行う場合は、Tritonをインストールしてください:
  ```bash
  pip install triton
  ```

### CUDA未利用可能

**症状**:
```
SKIPPED [1] tests/test_phase2_integration.py:XXX: CUDA not available
```

**対応**:
- KPI検証テストはCUDA環境で実行してください
- CPU環境では基本テストのみ実行されます

### Phase 1未利用可能

**症状**:
```
SKIPPED [1] tests/test_phase2_integration.py:XXX: Phase 1 not available
```

**対応**:
- PPL劣化テストはPhase 1モジュールが必要です
- Phase 1を実装するか、このテストをスキップしてください

### Memory Resonance型エラー

**症状**:
```
UserWarning: Memory resonance computation failed: expected scalar type Float but found ComplexFloat
```

**対応**:
- 共鳴層がスキップされますが、他の機能は正常に動作します
- `src/models/phase2/memory_resonance.py`の型変換を修正してください

## CI/CD統合

### GitHub Actions設定例

```yaml
name: Phase 2 Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest
    
    - name: Run integration tests
      run: |
        pytest tests/test_phase2_integration.py -v
    
    - name: Upload test results
      if: always()
      uses: actions/upload-artifact@v2
      with:
        name: test-results
        path: results/benchmarks/
```

## ベストプラクティス

### 1. テスト前の準備

```bash
# 依存関係をインストール
pip install -r requirements.txt
pip install pytest

# 結果ディレクトリを作成
mkdir -p results/benchmarks
```

### 2. 定期的なテスト実行

```bash
# 毎日実行（cron）
0 0 * * * cd /path/to/project && pytest tests/test_phase2_integration.py -v
```

### 3. テスト結果の保存

```bash
# JSON形式で保存
pytest tests/test_phase2_integration.py --json-report --json-report-file=results/test_report.json

# HTML形式で保存
pytest tests/test_phase2_integration.py --html=results/test_report.html
```

## 参考資料

- [Phase 2実装ガイド](../PHASE2_IMPLEMENTATION_GUIDE.md)
- [Phase 2統合テスト報告書](../../results/benchmarks/PHASE2_INTEGRATION_TEST_REPORT.md)
- [Phase 2設計書](../../.kiro/specs/phase2-breath-of-life/design.md)

---

**最終更新**: 2025-01-20  
**バージョン**: 1.0.0
