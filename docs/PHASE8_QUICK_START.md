# Phase 8 Quick Start Guide
## Hyperbolic Transcendence - O(N) Language Model

Phase 8は、**Phase 7（ResNetBK）を継承**した双曲幾何学に基づくO(N)複雑度の言語モデルです。Phase 7のBK-Core（三重対角逆行列計算）を心臓部として、双曲幾何学、トポロジー、論理的含意などの高度な機能を追加しています。

## Phase 7からの継承

Phase 8は**Phase 7の拡張**であり、独立したモデルではありません：

- **BK-Core**: Phase 7のGreen関数対角成分（G_ii）を物理情報として活用
- **O(N)複雑度**: Phase 7のO(N)複雑度を完全に維持
- **8GB VRAM制約**: Phase 7と同様にRTX 3080 (10GB)で動作

## 主要な特徴

### 1. BK-Core Hyperbolic Integration（Phase 8新機能）
- **G_iiゲーティング**: Phase 7のGreen関数対角成分を活用
- **物理情報統合**: 散乱エネルギーによる注意重み調整
- **共鳴検出**: 物理的共鳴状態に基づく自動調整

### 2. AR-SSM Hyperbolic Fusion（Phase 8新機能）
- **双曲空間融合**: AR-SSMと双曲幾何学の統合
- **複雑度信号**: 双曲距離を複雑度の指標として活用
- **自動曲率調整**: 高ランク時の曲率増加

### 3. オプション機能
- **Entailment Cones**: 論理的含意関係の幾何学的検証
- **Persistent Homology**: トポロジカル解析と循環論理検出
- **Sheaf Attention**: マルチヘッド間の構造的整合性

### 4. Phase 7継承機能
- **BK-Core**: O(N)三重対角逆行列計算
- **Hybrid Hyperbolic Attention**: 双曲空間注意機構
- **AR-SSM**: 自己回帰状態空間モデル

## クイックスタート

### 1. 環境確認
```bash
make verify-phase7  # Phase 8もこれでチェック可能
```

### 2. テスト実行（ダミーデータ）
```bash
# 小規模テスト（数分で完了）
make train-phase8-small

# 最大設定テスト（ダミーデータ）
make train-phase8-test
```

### 3. 実際の訓練

#### 標準設定（512次元、12層）
```bash
# データセット設定
make recipe

# 訓練開始
make train-phase8
```

#### 最大設定（4096次元、32層、3Bパラメータ）
```bash
# データセット設定
make recipe

# 訓練開始（8GB VRAMで動作）
make train-phase8-max
```

### 4. 訓練の再開
```bash
make train-phase8-resume CHECKPOINT=checkpoints/phase8/epoch_5.pt
```

## 設定ファイル

### 標準設定: `configs/phase8_optimized.yaml`
- d_model: 512
- n_layers: 12
- Parameters: ~150M
- VRAM: ~2-3 GB

### 最大設定: `configs/phase8_max_push.yaml`
- d_model: 4096
- n_layers: 32
- Parameters: ~3.08B
- VRAM: ~5.81 GB

## Phase 7 vs Phase 8 比較

| 項目 | Phase 7 | Phase 8 |
|------|---------|---------|
| 基本アーキテクチャ | ResNetBK + Hyperbolic | Phase 7を継承 |
| BK-Core | ✅ あり | ✅ あり（継承） |
| G_ii活用 | 基本的 | 高度（ゲーティング） |
| AR-SSM | ✅ あり | ✅ 双曲融合版 |
| 複雑度 | O(N) | O(N)（維持） |
| メモリ使用量 | 5.81 GB | 5.81 GB（同等） |
| 追加機能 | なし | Entailment/Topology/Sheaf |

### ベンチマーク実行
```bash
# Phase 7 vs Phase 8比較（ResNetBKベース）
python scripts/benchmark_phase7_vs_phase8_resnetbk.py

# BK-Core統合効果の検証
python scripts/benchmark_bk_core_integration_effect.py

# O(N)複雑度の検証
python scripts/verify_phase8_complexity.py
```

## カスタマイズ

### パラメータ調整
```bash
# d_modelを変更
make train-phase8 D_MODEL=768

# レイヤー数を変更
make train-phase8 N_LAYERS=16

# バッチサイズを変更
make train-phase8 BATCH_SIZE=4
```

### 曲率の調整
低曲率（c < 0.1）: O(N)線形計算、高速
高曲率（c > 1.0）: O(N²)正確計算、遅い

デフォルト: c = 0.01（推奨）

### Hyperbolic SSMの有効化
```bash
# SSMを有効化（メモリ使用量増加）
make train-phase8-max-ssm
```

## トラブルシューティング

### OOM (Out of Memory)
1. バッチサイズを減らす: `BATCH_SIZE=1`
2. シーケンス長を減らす: `N_SEQ=256`
3. モデルサイズを減らす: `D_MODEL=2048 N_LAYERS=24`

### 訓練が遅い
1. Tritonカーネルを確認: `make verify-triton`
2. 混合精度を確認: FP16が有効か
3. Gradient Checkpointingを無効化（メモリに余裕がある場合）

### 精度が出ない
1. 学習率を調整: `--lr 1e-4`
2. Warmupステップを増やす: `--warmup-steps 2000`
3. データセットを確認: `make recipe`

## 次のステップ

### 1. 性能評価
```bash
# Perplexity測定
python scripts/evaluate_phase8.py --checkpoint checkpoints/phase8/final_model.pt

# 長文脈テスト
python scripts/test_long_context.py --phase 8
```

### 2. 推論
```bash
# チャット
make chat-ai CHECKPOINT=checkpoints/phase8/final_model.pt

# バッチ推論
python scripts/inference_phase8.py --input data/test.txt
```

### 3. 論文執筆
訓練結果を `paper/main.tex` に反映させる

## 参考資料

- **Phase 8 ResNetBK統合ガイド**: `docs/PHASE8_RESNETBK_INTEGRATION.md` ⭐ 必読
- Phase 8設計書: `.kiro/specs/phase8-hyperbolic-transcendence/design.md`
- Phase 7 vs Phase 8比較: `results/benchmarks/PHASE7_VS_PHASE8_FINAL_SUMMARY_JP.md`
- BK-Core Hyperbolic実装: `src/models/phase8/bk_core_hyperbolic.py`
- AR-SSM Fusion実装: `src/models/phase8/ar_ssm_fusion.py`
- Phase 8使用例: `examples/phase8_resnetbk_demo.py`

## よくある質問

### Q: Phase 8はPhase 7の置き換えですか？
A: いいえ、**Phase 8はPhase 7の拡張**です。Phase 7を継承し、その上に新機能を追加しています。

### Q: BK-Coreは使われていますか？
A: はい、Phase 7のBK-Coreを**完全に継承**しています。G_iiを物理情報として活用します。

### Q: O(N)複雑度は維持されていますか？
A: はい、Phase 7のO(N)複雑度を**完全に維持**しています。

### Q: メモリ使用量は同じですか？
A: はい、Phase 7と同等です（3Bパラメータで5.81 GB）。

### Q: Phase 7からの移行は簡単ですか？
A: はい、Phase 7の設定をそのまま継承できます。`examples/phase8_resnetbk_demo.py`を参照してください。

### Q: Phase 7とPhase 8、どちらを使うべき？
A: 
- **Phase 7**: 安定性重視、実用目的
- **Phase 8**: 研究目的、物理情報活用、高度な機能が必要な場合

## コマンド一覧

```bash
# テスト
make train-phase8-small      # 小規模テスト
make train-phase8-test       # ダミーデータテスト

# 訓練
make train-phase8            # 標準設定
make train-phase8-max        # 最大設定

# 再開
make train-phase8-resume CHECKPOINT=...

# ベンチマーク
make bench-phase8-vs-phase7  # Phase 7との比較

# 環境確認
make verify-phase7           # 環境チェック
make verify-triton           # Tritonカーネル確認
```

## サポート

問題が発生した場合:
1. `make doctor` で診断
2. `make verify-phase7` で環境確認
3. GitHubでIssueを作成
