# Phase 1 Performance Validation and Benchmarking

このドキュメントでは、Phase 1 Efficiency Engineのパフォーマンス検証とベンチマークの実行方法を説明します。

## 概要

Phase 1のパフォーマンス検証には、以下の4つのスクリプトが用意されています：

1. **Memory Validation** - VRAM使用量の測定と8GB/10GBターゲットの検証
2. **Throughput Benchmark** - tokens/secondの測定とO(N)スケーリングの検証
3. **Perplexity Validation** - PPL測定とベースラインとの比較（5%劣化閾値）
4. **Comparison Tables** - 結果を統合した比較テーブルの生成

## 前提条件

```bash
# 必要なパッケージのインストール
pip install torch transformers datasets scikit-learn pandas tabulate matplotlib tqdm
```

## 1. メモリ使用量検証

### 基本的な使い方

```bash
# デフォルト設定で実行（batch_size=4, seq_len=2048）
python scripts/validate_phase1_memory.py

# カスタム設定で実行
python scripts/validate_phase1_memory.py \
    --batch-size 4 \
    --seq-length 2048 \
    --d-model 512 \
    --n-layers 8 \
    --ar-ssm-max-rank 32 \
    --htt-rank 16
```

### 出力

- `results/benchmarks/phase1_memory_validation.json` - 詳細な測定結果
- コンソール出力：ピークVRAM、forward/backward pass別のメモリ使用量、8GB/10GBターゲットの合否

### 検証項目

- ✅ Peak VRAM < 7.2GB (8GBターゲットの90%)
- ✅ Peak VRAM < 9.0GB (10GBターゲットの90%)
- ✅ ベースラインとの比較（メモリ削減率）

## 2. スループットベンチマーク

### 基本的な使い方

```bash
# デフォルト設定で実行（seq_lengths=[512, 1024, 2048, 4096]）
python scripts/benchmark_phase1_throughput.py

# カスタム設定で実行
python scripts/benchmark_phase1_throughput.py \
    --seq-lengths 512 1024 2048 4096 8192 \
    --batch-size 4 \
    --d-model 512 \
    --n-layers 8 \
    --warmup-steps 5 \
    --measure-steps 20 \
    --save-plots
```

### 出力

- `results/benchmarks/phase1_throughput_benchmark.json` - 詳細な測定結果
- `results/benchmarks/phase1_throughput_comparison.png` - スループットとスケーリングのプロット（--save-plotsオプション使用時）
- コンソール出力：tokens/second、計算量オーダー（O(N)、O(N log N)、O(N²)）、R²スコア

### 検証項目

- ✅ O(N)スケーリング（R² > 0.95）
- ✅ ベースラインとのスループット比較
- ✅ 異なるシーケンス長での線形スケーリング

## 3. Perplexity検証

### 基本的な使い方

```bash
# WikiText-103で実行（デフォルト）
python scripts/validate_phase1_perplexity.py

# C4データセットで実行
python scripts/validate_phase1_perplexity.py \
    --dataset-name c4 \
    --dataset-config en \
    --max-samples 1000

# 複数のHTTランクとAR-SSMランクをテスト
python scripts/validate_phase1_perplexity.py \
    --test-htt-ranks 8 16 32 \
    --test-ar-ssm-ranks 16 32 64 \
    --seq-length 512 \
    --batch-size 8
```

### 出力

- `results/benchmarks/phase1_perplexity_validation.json` - 詳細な測定結果
- コンソール出力：Perplexity、Bits per Byte、ベースラインとの劣化率、5%閾値の合否

### 検証項目

- ✅ PPL劣化 < 5% vs FP16ベースライン
- ✅ 異なるHTTランクとAR-SSMランクでの品質評価
- ✅ WikiText-103またはC4での言語モデリング性能

## 4. 比較テーブル生成

### 基本的な使い方

```bash
# すべてのベンチマーク結果から比較テーブルを生成
python scripts/generate_phase1_comparison_tables.py

# カスタム設定で実行
python scripts/generate_phase1_comparison_tables.py \
    --results-dir results/benchmarks \
    --out-dir results/benchmarks/tables \
    --hardware "RTX 3080" \
    --formats markdown csv latex
```

### 出力

`results/benchmarks/tables/` ディレクトリに以下のテーブルが生成されます：

1. **memory_comparison** - メモリ使用量の比較
2. **throughput_comparison** - スループットの比較
3. **scaling_comparison** - スケーリング特性の比較
4. **perplexity_comparison** - Perplexityの比較
5. **configuration_comparison** - 設定別の総合比較

各テーブルは以下のフォーマットで保存されます：
- `.md` - Markdown形式（ドキュメント用）
- `.csv` - CSV形式（データ分析用）
- `.tex` - LaTeX形式（論文用）

## 完全なベンチマークパイプライン

すべてのベンチマークを順番に実行する場合：

```bash
# 1. メモリ検証
python scripts/validate_phase1_memory.py \
    --batch-size 4 \
    --seq-length 2048 \
    --d-model 512 \
    --n-layers 8

# 2. スループットベンチマーク
python scripts/benchmark_phase1_throughput.py \
    --seq-lengths 512 1024 2048 4096 \
    --batch-size 4 \
    --d-model 512 \
    --n-layers 8 \
    --save-plots

# 3. Perplexity検証
python scripts/validate_phase1_perplexity.py \
    --dataset-name wikitext \
    --dataset-config wikitext-103-raw-v1 \
    --seq-length 512 \
    --batch-size 8 \
    --max-samples 1000 \
    --d-model 512 \
    --n-layers 8

# 4. 比較テーブル生成
python scripts/generate_phase1_comparison_tables.py \
    --results-dir results/benchmarks \
    --out-dir results/benchmarks/tables \
    --hardware "RTX 3080"
```

## ハードウェア別の推奨設定

### RTX 3080 (10GB VRAM)

```bash
# メモリ検証
python scripts/validate_phase1_memory.py \
    --batch-size 4 \
    --seq-length 2048

# スループットベンチマーク
python scripts/benchmark_phase1_throughput.py \
    --seq-lengths 512 1024 2048 4096 \
    --batch-size 4

# Perplexity検証
python scripts/validate_phase1_perplexity.py \
    --batch-size 8 \
    --seq-length 512
```

### RTX 3080 Mobile (8GB VRAM)

```bash
# メモリ検証
python scripts/validate_phase1_memory.py \
    --batch-size 2 \
    --seq-length 2048

# スループットベンチマーク
python scripts/benchmark_phase1_throughput.py \
    --seq-lengths 512 1024 2048 \
    --batch-size 2

# Perplexity検証
python scripts/validate_phase1_perplexity.py \
    --batch-size 4 \
    --seq-length 512
```

### RTX 4090 (24GB VRAM)

```bash
# メモリ検証
python scripts/validate_phase1_memory.py \
    --batch-size 8 \
    --seq-length 4096

# スループットベンチマーク
python scripts/benchmark_phase1_throughput.py \
    --seq-lengths 512 1024 2048 4096 8192 \
    --batch-size 8

# Perplexity検証
python scripts/validate_phase1_perplexity.py \
    --batch-size 16 \
    --seq-length 1024
```

## トラブルシューティング

### CUDA Out of Memory

メモリ不足エラーが発生した場合：

1. バッチサイズを減らす：`--batch-size 2` または `--batch-size 1`
2. シーケンス長を減らす：`--seq-length 1024` または `--seq-length 512`
3. AR-SSMランクを減らす：`--ar-ssm-max-rank 16`
4. グラディエントチェックポイントを有効化（デフォルトで有効）

### データセット読み込みエラー

WikiText-103やC4のダウンロードに失敗する場合：

```bash
# 事前にデータセットをダウンロード
python -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-103-raw-v1')"
```

### 依存パッケージのエラー

```bash
# すべての依存パッケージを再インストール
pip install --upgrade torch transformers datasets scikit-learn pandas tabulate matplotlib tqdm
```

## 結果の解釈

### メモリ検証

- **Peak VRAM < 7.2GB**: 8GB GPUで安全に動作
- **Memory Reduction > 50%**: ベースラインと比較して大幅なメモリ削減
- **Forward/Backward比率**: 通常、Backward passはForward passの1.5-2倍のメモリを使用

### スループットベンチマーク

- **O(N) Scaling (R² > 0.95)**: 線形スケーリングを達成
- **Tokens/second**: 高いほど良い（ハードウェアとモデルサイズに依存）
- **Scaling Coefficient**: 小さいほど効率的

### Perplexity検証

- **PPL Degradation < 5%**: 品質を維持しながら効率化を達成
- **Perplexity**: 低いほど良い（言語モデリング性能）
- **Bits per Byte**: 圧縮効率の指標

## 参考資料

- [Phase 1 Design Document](../specs/phase1-efficiency-engine/design.md)
- [Phase 1 Requirements](../specs/phase1-efficiency-engine/requirements.md)
- [Phase 1 Implementation Tasks](../specs/phase1-efficiency-engine/tasks.md)

## 問題報告

ベンチマーク実行中に問題が発生した場合は、以下の情報を含めてIssueを作成してください：

1. 使用したコマンド
2. エラーメッセージ（完全なスタックトレース）
3. ハードウェア情報（GPU、VRAM、CUDA version）
4. PyTorchとTransformersのバージョン
5. 生成された結果ファイル（あれば）
