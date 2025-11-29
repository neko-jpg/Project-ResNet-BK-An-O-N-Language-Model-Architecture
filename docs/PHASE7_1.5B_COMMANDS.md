# Phase 7 - 1.5Bパラメータモデル コマンド一覧

## 🚀 クイックスタート

### 1. 環境チェック
```bash
make check-phase7-env
```

### 2. 訓練開始

#### 10GB以上のGPU (RTX 3080 10GB, RTX 3090, RTX 4080など)
```bash
make train-phase7-1.5b
```
- パラメータ数: ~1.4B
- d_model: 2048
- n_layers: 24
- VRAM: ~10-13GB

#### 8GBのGPU (RTX 3070, RTX 3080 8GB など)
```bash
make train-phase7-1.5b-8gb
```
- パラメータ数: ~1.2B
- d_model: 1792
- n_layers: 24
- VRAM: ~7-8GB
- 8bit AdamW使用

### 3. テスト実行（ダミーデータ）
```bash
make train-phase7-1.5b-test
```

### 4. 訓練再開
```bash
make train-phase7-1.5b-resume CHECKPOINT=checkpoints/phase7_1.5b_triton/step_2000.pt
```

### 5. チャット推論
```bash
make chat-phase7-1.5b CHECKPOINT=checkpoints/phase7_1.5b_triton/phase7_best.pt
```

### 6. GPUベンチマーク
```bash
make bench-phase7-1.5b
```

## 📋 全コマンド一覧

| コマンド | 説明 | VRAM要件 |
|---------|------|---------|
| `make check-phase7-env` | 環境チェック (CUDA+Triton) | - |
| `make train-phase7-1.5b` | 1.5B訓練 (フル設定) | 10GB+ |
| `make train-phase7-1.5b-8gb` | 1.2B訓練 (8GB最適化) | 8GB |
| `make train-phase7-1.5b-test` | テスト実行 | 2GB |
| `make train-phase7-1.5b-resume` | 訓練再開 | 10GB+ |
| `make bench-phase7-1.5b` | GPUベンチマーク | 可変 |
| `make chat-phase7-1.5b` | チャット推論 | 3GB |

## ⚙️ 設定ファイル

### 10GB+ GPU用
- **ファイル**: `configs/phase7_1.5b_triton.yaml`
- **パラメータ**: ~1.4B
- **d_model**: 2048
- **n_layers**: 24
- **n_seq**: 512

### 8GB GPU用
- **ファイル**: `configs/phase7_1.5b_triton_8gb.yaml`
- **パラメータ**: ~1.2B
- **d_model**: 1792
- **n_layers**: 24
- **n_seq**: 512
- **特徴**: 8bit AdamW

## 🔧 カスタマイズ

### パラメータ数を調整

設定ファイルを編集：
```yaml
d_model: 2048    # 埋め込み次元
n_layers: 24     # レイヤー数
n_seq: 512       # シーケンス長
```

パラメータ数の目安：
- d_model=1536, n_layers=20 → ~0.8B
- d_model=1792, n_layers=24 → ~1.2B
- d_model=2048, n_layers=24 → ~1.4B
- d_model=2304, n_layers=28 → ~1.8B

### バッチサイズ調整

```yaml
batch_size: 1
gradient_accumulation_steps: 16  # 実効バッチサイズ=16
```

VRAMが足りない場合：
- `batch_size: 1` のまま
- `gradient_accumulation_steps` を増やす（32, 64など）

## 📊 最適化設定

すべての最適化がONになっています：

```yaml
# Triton必須
use_triton_kernel: true
triton_kernel_version: 'fast'

# メモリ最適化
use_mixed_precision: true
use_gradient_checkpointing: true
use_flash_attention: true

# 速度最適化
use_fused_optimizer: true
use_fused_kernels: true
use_compile: true

# メモリ効率
use_memory_efficient_attention: true
```

## 🎯 推奨ワークフロー

1. **環境確認**
   ```bash
   make check-phase7-env
   ```

2. **テスト実行**
   ```bash
   make train-phase7-1.5b-test
   ```

3. **データ準備**
   ```bash
   make recipe
   ```

4. **訓練開始**
   ```bash
   # 10GB+ GPU
   make train-phase7-1.5b
   
   # 8GB GPU
   make train-phase7-1.5b-8gb
   ```

5. **モニタリング**
   - ログを確認
   - チェックポイントを確認

6. **推論テスト**
   ```bash
   make chat-phase7-1.5b
   ```

## 💡 トラブルシューティング

### CUDA Out of Memory

8GB版を使用：
```bash
make train-phase7-1.5b-8gb
```

または設定を調整：
```yaml
d_model: 1536
n_layers: 20
n_seq: 512
```

### Tritonエラー

カーネル確認：
```bash
make triton-attn
```

### 訓練が遅い

- `use_compile: true` を確認
- 初回はTritonコンパイルで遅い（数分）
- 2回目以降は高速化

## 📚 詳細ドキュメント

- [クイックスタートガイド](PHASE7_1.5B_QUICKSTART_JP.md)
- [Phase 7実装ガイド](PHASE7_IMPLEMENTATION_GUIDE.md)
- [Tritonカーネル詳細](../src/kernels/README.md)

---

**Happy Training! 🎉**
