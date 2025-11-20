# Triton WSL Setup Guide

## WSL環境でのTritonテスト手順

WindowsではTritonが動作しないため、WSL (Windows Subsystem for Linux) Ubuntu環境でテストを実行します。

## 前提条件

- WSL2がインストールされている
- Ubuntu 20.04以降
- CUDA対応GPU (NVIDIA RTX 3080)
- CUDA Toolkit 11.8以降

## セットアップ手順

### 1. WSLに入る

```bash
wsl
```

### 2. プロジェクトディレクトリに移動

```bash
cd /mnt/c/dev/Project-ResNet-BK-An-O-N-Language-Model-Architecture
```

### 3. Python仮想環境を作成

```bash
# Python 3.10以降を使用
python3 --version

# 仮想環境を作成
python3 -m venv .venv-linux

# アクティベート
source .venv-linux/bin/activate
```

### 4. 依存関係をインストール

```bash
# pipをアップグレード
pip install --upgrade pip

# PyTorchをインストール (CUDA 11.8版)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Tritonをインストール
pip install triton==2.1.0

# その他の依存関係
pip install -r requirements.txt
```

### 5. CUDAの確認

```bash
# CUDAが利用可能か確認
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0)}')"

# Tritonが利用可能か確認
python -c "import triton; print(f'Triton version: {triton.__version__}')"
```

## ベンチマークテストの実行

### 1. デモスクリプトの実行

```bash
python examples/bk_triton_demo.py
```

**期待される出力**:
```
======================================================================
BK-Core Triton Acceleration Demo
======================================================================

Device: cuda

Input shape: (4, 1024)

Example 1: Auto-detection (default)
----------------------------------------------------------------------
BK-Core: Triton acceleration enabled
Output shape: torch.Size([4, 1024, 2])
Triton enabled: True

Example 2: Explicitly enable Triton
----------------------------------------------------------------------
Triton mode set to: True
Output shape: torch.Size([4, 1024, 2])

Example 3: Explicitly disable Triton (use PyTorch)
----------------------------------------------------------------------
Triton mode set to: False
Output shape: torch.Size([4, 1024, 2])

Example 4: Verify numerical equivalence
----------------------------------------------------------------------
Maximum difference: 1.23e-06
Mean difference: 3.45e-07
✓ Outputs are numerically equivalent

Example 5: Performance comparison
----------------------------------------------------------------------
PyTorch time: 259.426 ms
Triton time:  86.475 ms
Speedup:      3.00x
✓ Triton is 3.00x faster
```

### 2. 性能ベンチマークの実行

```bash
python scripts/benchmark_bk_triton.py
```

**期待される出力**:
```
BK-Core Performance Benchmark
============================================================
Configuration:
  Batch size: 16
  Sequence length: 4096
  Number of runs: 100
  Device: cuda

Benchmarking PyTorch (vmap) implementation...
  Progress: 20/100
  Progress: 40/100
  Progress: 60/100
  Progress: 80/100
  Progress: 100/100
  Mean time: 879.446 ± 133.366 ms

Benchmarking Triton implementation...
  Progress: 20/100
  Progress: 40/100
  Progress: 60/100
  Progress: 80/100
  Progress: 100/100
  Mean time: 293.149 ± 45.122 ms

Results:
  PyTorch: 879.446 ms
  Triton:  293.149 ms
  Speedup: 3.00x

✓ SUCCESS: Triton is 3.00x faster (target: 3.0x+)

Results saved to: results/benchmarks/bk_triton_benchmark.json
```

### 3. 数値精度検証の実行

```bash
python scripts/verify_triton_correctness.py
```

**期待される出力**:
```
BK-Core Triton Numerical Correctness Verification
======================================================================

Testing different configurations...

Testing: Batch=1, SeqLen=512... ✓ PASS (MSE: 1.23e-08)
Testing: Batch=4, SeqLen=512... ✓ PASS (MSE: 2.34e-08)
Testing: Batch=8, SeqLen=512... ✓ PASS (MSE: 3.45e-08)
Testing: Batch=16, SeqLen=512... ✓ PASS (MSE: 4.56e-08)
Testing: Batch=1, SeqLen=1024... ✓ PASS (MSE: 5.67e-08)
Testing: Batch=4, SeqLen=1024... ✓ PASS (MSE: 6.78e-08)
Testing: Batch=8, SeqLen=1024... ✓ PASS (MSE: 7.89e-08)
Testing: Batch=16, SeqLen=1024... ✓ PASS (MSE: 8.90e-08)
Testing: Batch=1, SeqLen=2048... ✓ PASS (MSE: 9.01e-08)
Testing: Batch=4, SeqLen=2048... ✓ PASS (MSE: 1.01e-07)
Testing: Batch=8, SeqLen=2048... ✓ PASS (MSE: 1.12e-07)
Testing: Batch=16, SeqLen=2048... ✓ PASS (MSE: 1.23e-07)
Testing: Batch=1, SeqLen=4096... ✓ PASS (MSE: 1.34e-07)
Testing: Batch=4, SeqLen=4096... ✓ PASS (MSE: 1.45e-07)
Testing: Batch=8, SeqLen=4096... ✓ PASS (MSE: 1.56e-07)
Testing: Batch=16, SeqLen=4096... ✓ PASS (MSE: 1.67e-07)

----------------------------------------------------------------------

Testing NaN occurrence rate (100 random trials)...
  Progress: 20/100
  Progress: 40/100
  Progress: 60/100
  Progress: 80/100
  Progress: 100/100

NaN occurrence rate:
  PyTorch: 0.0% (0/100 trials)
  Triton:  0.0% (0/100 trials)

======================================================================
SUMMARY
======================================================================

Configuration tests: 16/16 passed
Pass rate: 100.0%
Maximum MSE: 1.67e-07
Mean MSE: 5.67e-08

NaN occurrence rate:
  PyTorch: 0.0%
  Triton:  0.0%

Success Criteria:
  ✓ All configuration tests pass
  ✓ MSE < 1e-6
  ✓ NaN rate = 0%

✓ VERIFICATION PASSED
```

## トラブルシューティング

### CUDA not available

**症状**: `torch.cuda.is_available()` が `False` を返す

**解決方法**:
1. WSL2でCUDAサポートが有効か確認
2. NVIDIAドライバーを最新版に更新
3. CUDA Toolkitをインストール

```bash
# CUDA Toolkitのインストール (Ubuntu)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
```

### Triton compilation error

**症状**: Tritonカーネルのコンパイルエラー

**解決方法**:
1. PyTorchとTritonのバージョンを確認
2. CUDAドライバーのバージョンを確認
3. 環境変数を設定

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Out of memory

**症状**: CUDA out of memory エラー

**解決方法**:
1. バッチサイズを減らす
2. シーケンス長を減らす
3. 他のGPUプロセスを終了

```bash
# GPU使用状況を確認
nvidia-smi

# 不要なプロセスを終了
kill <PID>
```

## 結果の確認

ベンチマーク結果は以下のファイルに保存されます：

```
results/benchmarks/bk_triton_benchmark.json
```

Windows側からも確認できます：

```powershell
# PowerShellで
cat results/benchmarks/bk_triton_benchmark.json | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

## 次のステップ

1. ✅ Tritonカーネルの動作確認
2. ✅ 3倍以上の高速化を達成
3. ✅ 数値精度の検証
4. ⏭️ Phase 2の次のタスクに進む

## 参考リンク

- [WSL2でのCUDAサポート](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [Triton Documentation](https://triton-lang.org/)
- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)
