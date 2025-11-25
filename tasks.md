# Phase 7 実装ロードマップ

## 1. リーマン多様体上の最適化 (Riemannian Optimization)

- [x] `RiemannianAdam` オプティマイザクラスの実装 (`src/optimizers/riemannian_adam.py`)
- [x] `HyperbolicInitializer` で `is_hyperbolic` 属性を付与 (`src/models/phase6/geometry/hyperbolic.py`)
- [x] `RiemannianAdam` のための検証スクリプトの作成 (`experiments/validate_riemannian_adam.py`)
- [x] 検証スクリプトによる動作確認

## 2. 双曲的注意機構 (Hyperbolic Attention)

- [x] `HyperbolicAttention` モジュールの作成
- [x] 既存モデルへの統合とテスト

## 3. 自由エネルギー原理に基づく損失関数 (FEP Loss)

- [x] `FreeEnergyLoss` クラスの実装
- [x] トレーニングループへの統合とテスト
