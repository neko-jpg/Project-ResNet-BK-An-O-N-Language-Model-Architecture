# Phase 7 - 1.5Bパラメータモデル訓練ガイド

## 🚀 クイックスタート

Phase 7の1.5Bパラメータモデルを、Triton必須ですべての最適化をONにして訓練する方法を説明します。

## 📋 必要な環境

### ハードウェア要件
- **GPU**: NVIDIA GPU (RTX 3080 10GB以上推奨)
- **VRAM**: 8-10GB以上
- **RAM**: 16GB以上推奨

### ソフトウェア要件
- **CUDA**: 11.8以上
- **Python**: 3.8以上
- **PyTorch**: 2.0以上 (CUDA対応版)
- **Triton**: 2.0以上 (必須)

## 🔧 環境セットアップ

### 1. 環境チェック

まず、Phase 7に必要な環境が整っているか確認します：

```bash
make check-phase7-env
```

このコマンドは以下をチェックします：
- ✓ PyTorchのインストール
- ✓ CUDAの利用可能性
- ✓ Tritonのインストール
- ✓ Tritonカーネルのロード
- ✓ GPU情報

### 2. Tritonのインストール（必要な場合）

Tritonがインストールされていない場合：

```bash
pip install triton
```

または特定のバージョン：

```bash
pip install triton==2.1.0
```

## 📊 データセット準備

### オプション1: データセットレシピの設定（推奨）

```bash
make recipe
```

対話的にデータセットの配合を設定できます。

### オプション2: テスト用データセット

```bash
make data-lite
```

小規模なテストデータセットをダウンロードします。

## 🎯 訓練開始

### 基本的な訓練コマンド

```bash
make train-phase7-1.5b
```

このコマンドは以下の設定で訓練を開始します：
- **パラメータ数**: ~1.5B (1,500,000,000)
- **d_model**: 2048
- **n_layers**: 24
- **n_seq**: 1024
- **batch_size**: 1
- **gradient_accumulation**: 16 (実効バッチサイズ=16)
- **Triton**: 必須 (全最適化ON)

### テスト実行（ダミーデータ）

データセットなしで動作確認：

```bash
make train-phase7-1.5b-test
```

### 訓練の再開

チェックポイントから訓練を再開：

```bash
make train-phase7-1.5b-resume CHECKPOINT=checkpoints/phase7_1.5b_triton/step_2000.pt
```

## 📈 訓練モニタリング

### ログの確認

訓練中、以下の情報がリアルタイムで表示されます：

```
Step   1000 | Loss: 3.2145 | PPL: 24.87 | LR: 2.50e-04 | Grad: 0.823 | Time: 0.45s
```

- **Loss**: 損失値
- **PPL**: パープレキシティ（低いほど良い）
- **LR**: 現在の学習率
- **Grad**: 勾配ノルム
- **Time**: ステップあたりの時間

### チェックポイント

チェックポイントは以下のタイミングで保存されます：
- 2000ステップごと: `phase7_step_XXXX.pt`
- 最良モデル: `phase7_best.pt`
- 最終モデル: `phase7_final.pt`

保存先: `checkpoints/phase7_1.5b_triton/`

## 🎮 訓練済みモデルの使用

### チャット推論

```bash
make chat-phase7-1.5b CHECKPOINT=checkpoints/phase7_1.5b_triton/phase7_best.pt
```

または自動検出：

```bash
make chat-phase7-1.5b
```

## 📊 ベンチマーク

### GPU性能測定

```bash
make bench-phase7-1.5b
```

GPUでの最大パラメータ数とメモリ使用量を測定します。

## ⚙️ 設定のカスタマイズ

### 設定ファイルの編集

`configs/phase7_1.5b_triton.yaml` を編集して、以下をカスタマイズできます：

```yaml
# モデルサイズの調整
d_model: 2048        # 埋め込み次元
n_layers: 24         # レイヤー数
n_seq: 1024          # シーケンス長

# 訓練設定
batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 0.0003
epochs: 10

# 最適化設定（すべてON）
use_mixed_precision: true
use_gradient_checkpointing: true
use_flash_attention: true
use_fused_optimizer: true
use_compile: true
```

### VRAMが不足する場合

8GB未満のGPUの場合、以下を調整：

```yaml
d_model: 1536        # 2048 → 1536
n_layers: 20         # 24 → 20
n_seq: 512           # 1024 → 512
```

これで約1.0Bパラメータになります。

## 🔍 トラブルシューティング

### CUDA Out of Memory

```bash
# 設定を小さくする
make train-phase7-small
```

または `configs/phase7_1.5b_triton.yaml` で：
- `batch_size` を減らす
- `n_seq` を減らす
- `d_model` を減らす

### Tritonカーネルエラー

```bash
# Tritonカーネルの動作確認
make triton-attn
```

### 環境の再確認

```bash
make verify-phase7
```

## 📚 設定の詳細

### 有効化されている最適化

1. **Tritonカーネル**: GPU最適化された高速演算
2. **混合精度訓練**: FP16で高速化
3. **勾配チェックポイント**: VRAM削減
4. **Flash Attention**: メモリ効率的アテンション
5. **融合オプティマイザ**: AdamWの高速化
6. **torch.compile**: PyTorch 2.0の最適化
7. **勾配累積**: 実効バッチサイズの増加
8. **HTT圧縮**: 埋め込み層のパラメータ圧縮
9. **AR-SSM**: 効率的な長距離依存性
10. **曲率スケジューラ**: 双曲空間の動的調整

### パラメータ数の計算

```
総パラメータ数 ≈ 1.5B
- 埋め込み層: ~100M (HTT圧縮)
- Transformerレイヤー: ~1.3B
- 出力層: ~100M
```

## 🎯 推奨ワークフロー

1. **環境チェック**: `make check-phase7-env`
2. **テスト実行**: `make train-phase7-1.5b-test`
3. **データ準備**: `make recipe` または `make data-lite`
4. **訓練開始**: `make train-phase7-1.5b`
5. **モニタリング**: ログとチェックポイントを確認
6. **推論テスト**: `make chat-phase7-1.5b`

## 💡 ヒント

- **初回実行**: Tritonカーネルのコンパイルに時間がかかります（数分）
- **学習率**: 大規模モデルでは低めの学習率（3e-4）が安定
- **ウォームアップ**: 最初の2000ステップで学習率を徐々に上げる
- **勾配累積**: VRAMが少ない場合は `gradient_accumulation_steps` を増やす
- **チェックポイント**: 定期的に保存されるので、いつでも再開可能

## 📞 サポート

問題が発生した場合：

1. `make check-phase7-env` で環境を確認
2. `make verify-phase7` で詳細診断
3. ログファイルを確認: `checkpoints/phase7_1.5b_triton/training_log.json`
4. GitHubのIssuesで報告

## 🚀 次のステップ

- Phase 8（O(N)線形アテンション）を試す: `make train-phase8`
- 性能比較: `make bench-phase8-vs-phase7`
- カスタムデータセットでファインチューニング

---

**Happy Training! 🎉**
