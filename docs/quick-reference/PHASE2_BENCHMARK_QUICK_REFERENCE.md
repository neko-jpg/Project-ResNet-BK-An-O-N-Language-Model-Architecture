# Phase 2 Benchmark Quick Reference

Phase 2モデルのベンチマークテストの使い方を簡潔にまとめたクイックリファレンスです。

---

## 🚀 クイックスタート

### 全ベンチマークの実行:
```bash
pytest tests/test_phase2_benchmarks.py -v -s
```

### KPIテストのみ実行:
```bash
# BK-Core Triton高速化
pytest tests/test_phase2_benchmarks.py::TestBKCoreTritonBenchmark::test_bk_core_large_sequence -v -s

# VRAM使用量
pytest tests/test_phase2_benchmarks.py::TestMemoryBenchmark::test_memory_kpi -v -s

# スループット
pytest tests/test_phase2_benchmarks.py::TestThroughputBenchmark::test_throughput_kpi -v -s
```

### レポート生成:
```bash
pytest tests/test_phase2_benchmarks.py::TestBenchmarkReport::test_generate_comprehensive_report -v -s
```

---

## 📊 ベンチマーク種類

### 1. BK-Core Tritonベンチマーク

**目的**: Tritonカーネルの高速化を検証

**KPI**: 3.0倍以上の高速化、MSE < 1e-6

**テスト**:
- `test_bk_core_small_sequence()` - N=512
- `test_bk_core_medium_sequence()` - N=2048
- `test_bk_core_large_sequence()` - N=4096 (KPI)
- `test_bk_core_scaling()` - スケーリング特性

**出力**:
- `bk_core_triton_benchmark_kpi.json`
- `bk_core_triton_scaling.json`

---

### 2. メモリベンチマーク

**目的**: VRAM使用量を測定

**KPI**: < 8.0 GB (Batch=1, Seq=4096, fp16)

**テスト**:
- `test_memory_small_model()` - 小規模モデル
- `test_memory_base_model()` - 標準モデル
- `test_memory_kpi()` - KPI条件
- `test_memory_scaling()` - スケーリング
- `test_memory_dtype_comparison()` - fp16/fp32比較

**出力**:
- `phase2_memory_kpi.json`
- `phase2_memory_scaling.json`

---

### 3. スループットベンチマーク

**目的**: トークン処理速度を測定

**KPI**: >= 100 tokens/sec

**テスト**:
- `test_throughput_small_model()` - 小規模モデル
- `test_throughput_base_model()` - 標準モデル
- `test_throughput_kpi()` - KPI条件
- `test_throughput_with_backward()` - Forward+Backward
- `test_throughput_scaling()` - スケーリング

**出力**:
- `phase2_throughput_kpi.json`
- `phase2_throughput_scaling.json`

---

### 4. 総合レポート

**目的**: すべてのベンチマーク結果を集約

**テスト**:
- `test_generate_comprehensive_report()` - レポート生成

**出力**:
- `phase2_benchmark_comprehensive_report.json`
- `PHASE2_BENCHMARK_REPORT.md`

---

## 📁 出力ファイル

すべての結果は `results/benchmarks/` に保存されます:

```
results/benchmarks/
├── bk_core_triton_benchmark_kpi.json      # BK-Core KPI結果
├── bk_core_triton_scaling.json            # BK-Coreスケーリング
├── phase2_memory_kpi.json                 # メモリKPI結果
├── phase2_memory_scaling.json             # メモリスケーリング
├── phase2_throughput_kpi.json             # スループットKPI結果
├── phase2_throughput_scaling.json         # スループットスケーリング
├── phase2_benchmark_comprehensive_report.json  # 総合レポート(JSON)
└── PHASE2_BENCHMARK_REPORT.md             # 総合レポート(Markdown)
```

---

## 🎯 KPI一覧

| KPI | 目標値 | 測定条件 | テスト |
|-----|--------|----------|--------|
| BK-Core高速化 | >= 3.0x | Batch=16, Seq=4096 | `test_bk_core_large_sequence` |
| 数値精度 | MSE < 1e-6 | PyTorch実装との比較 | `test_bk_core_large_sequence` |
| VRAM使用量 | < 8.0 GB | Batch=1, Seq=4096, fp16 | `test_memory_kpi` |
| スループット | >= 100 tokens/sec | Batch=4, Seq=512, fp16 | `test_throughput_kpi` |

---

## 💻 プログラムからの利用

### BK-Coreベンチマーク:
```python
from tests.test_phase2_benchmarks import TestBKCoreTritonBenchmark

benchmark = TestBKCoreTritonBenchmark()
results = benchmark.benchmark_bk_core(
    batch_size=16,
    seq_len=4096,
    num_runs=100
)

print(f"Speedup: {results['speedup']:.2f}x")
print(f"MSE: {results['numerical_error']:.2e}")
```

### メモリ測定:
```python
from tests.test_phase2_benchmarks import TestMemoryBenchmark
from src.models.phase2.factory import create_phase2_model

benchmark = TestMemoryBenchmark()
model = create_phase2_model(preset="base", device="cuda")

results = benchmark.measure_memory(
    model,
    batch_size=1,
    seq_len=4096,
    dtype=torch.float16
)

print(f"Memory: {results['memory_gb']:.2f} GB")
```

### スループット測定:
```python
from tests.test_phase2_benchmarks import TestThroughputBenchmark
from src.models.phase2.factory import create_phase2_model

benchmark = TestThroughputBenchmark()
model = create_phase2_model(preset="base", device="cuda")
model = model.half()

results = benchmark.measure_throughput(
    model,
    batch_size=4,
    seq_len=512,
    num_iterations=20
)

print(f"Throughput: {results['forward_throughput']:.1f} tokens/sec")
```

---

## 🔍 結果の確認

### JSON結果の確認:
```bash
# BK-Core KPI
cat results/benchmarks/bk_core_triton_benchmark_kpi.json

# メモリ KPI
cat results/benchmarks/phase2_memory_kpi.json

# スループット KPI
cat results/benchmarks/phase2_throughput_kpi.json
```

### Markdownレポートの確認:
```bash
cat results/benchmarks/PHASE2_BENCHMARK_REPORT.md
```

---

## ⚙️ カスタマイズ

### ベンチマーク条件の変更:

```python
# カスタムバッチサイズとシーケンス長
results = benchmark.benchmark_bk_core(
    batch_size=32,      # デフォルト: 16
    seq_len=8192,       # デフォルト: 4096
    num_runs=200,       # デフォルト: 100
    warmup_runs=20      # デフォルト: 10
)
```

### 測定回数の調整:

```python
# 高精度測定（時間がかかる）
results = benchmark.benchmark_bk_core(
    batch_size=16,
    seq_len=4096,
    num_runs=1000,      # 多くの実行
    warmup_runs=50      # 十分なウォームアップ
)

# 高速測定（精度は低い）
results = benchmark.benchmark_bk_core(
    batch_size=16,
    seq_len=4096,
    num_runs=10,        # 少ない実行
    warmup_runs=2       # 最小限のウォームアップ
)
```

---

## 🐛 トラブルシューティング

### Tritonが利用できない:
```
SKIPPED [1] tests/test_phase2_benchmarks.py:XX: Triton not available
```
→ Tritonカーネルが利用できない環境です。CPUモードまたはPyTorch実装のみで動作します。

### CUDAが利用できない:
```
SKIPPED [1] tests/test_phase2_benchmarks.py:XX: CUDA not available
```
→ GPUが利用できない環境です。メモリとスループットのベンチマークはスキップされます。

### メモリ不足:
```
RuntimeError: CUDA out of memory
```
→ バッチサイズまたはシーケンス長を減らしてください。

### スループットが低い:
```
Warning: Throughput XX tokens/sec is below target 100 tokens/sec
```
→ ハードウェアの性能に依存します。警告のみで、テストは失敗しません。

---

## 📚 関連ドキュメント

- **実装ガイド**: `docs/PHASE2_IMPLEMENTATION_GUIDE.md`
- **BK-Core Triton**: `docs/implementation/BK_CORE_TRITON.md`
- **タスクリスト**: `.kiro/specs/phase2-breath-of-life/tasks.md`
- **完了報告**: `results/benchmarks/TASK19_COMPLETION_SUMMARY.md`

---

## 📞 サポート

問題が発生した場合:
1. 完了報告を確認: `results/benchmarks/TASK19_COMPLETION_SUMMARY.md`
2. テストログを確認: `pytest -v -s`
3. 生成されたJSONファイルを確認

---

**最終更新**: 2025-11-20  
**バージョン**: Phase 2 v1.0
