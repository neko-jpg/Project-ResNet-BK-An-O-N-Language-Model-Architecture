#!/usr/bin/env python3
"""
訓練前包括的検証スクリプト

以下を検証:
1. 因果マスク（カンニング防止）
2. データパイプライン
3. モデルアーキテクチャ
4. 初期化・勾配
5. 設定ファイル
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only for quick testing

import random
import numpy as np
import yaml

print("=" * 70)
print("訓練前包括的検証")
print("=" * 70)

results = {"passed": 0, "failed": 0, "warnings": 0}

def check(name, condition, critical=True):
    """Check a condition and report result."""
    if condition:
        print(f"✅ {name}")
        results["passed"] += 1
        return True
    elif critical:
        print(f"❌ {name}")
        results["failed"] += 1
        return False
    else:
        print(f"⚠️  {name}")
        results["warnings"] += 1
        return False

# =============================================================================
# 1. 因果マスク検証
# =============================================================================
print("\n--- 1. 因果マスク（カンニング防止）検証 ---")

# Check hybrid_attention.py for torch.tril
hybrid_attention_path = Path("src/models/phase7/hybrid_attention.py")
if hybrid_attention_path.exists():
    content = hybrid_attention_path.read_text()
    check("HybridHyperbolicAttention: torch.trilが存在", "torch.tril" in content)
    check("HybridHyperbolicAttention: causalマスク関連コメントが存在", "causal" in content.lower())
else:
    check("hybrid_attention.pyが存在", False)

# Check hyperbolic_attention.py for mask application
hyperbolic_attention_path = Path("src/models/phase7/hyperbolic_attention.py")
if hyperbolic_attention_path.exists():
    content = hyperbolic_attention_path.read_text()
    check("HyperbolicAttention: masked_fillが存在", "masked_fill" in content)
    check("HyperbolicAttention: mask引数が存在", "mask=" in content or "mask:" in content)
else:
    check("hyperbolic_attention.pyが存在", False)

# =============================================================================
# 2. データパイプライン検証
# =============================================================================
print("\n--- 2. データパイプライン検証 ---")

from src.utils.data_utils import BinaryIndexedDataset

# Test short document concatenation logic
try:
    ds = BinaryIndexedDataset("data/japanese_instruct", split="train")
    rng = random.Random(42)
    
    # Test sampling with seq_len larger than some documents
    result = ds.sample_sequence(512, rng)
    check("sample_sequence: 512トークンのサンプリングが成功", result is not None)
    
    if result:
        x, y = result
        check("sample_sequence: x.shape == (512,)", len(x) == 512)
        check("sample_sequence: y.shape == (512,)", len(y) == 512)
        check("sample_sequence: トークン値が正の整数", x.min() >= 0 and y.min() >= 0)
except Exception as e:
    check(f"sample_sequenceテスト: エラー - {e}", False)

# Check all 4 datasets
datasets = ["japanese_instruct", "dolly_ja", "wiki_ja", "mc4_ja"]
for ds_name in datasets:
    ds_path = Path(f"data/{ds_name}/train.idx")
    if ds_path.exists():
        try:
            ds = BinaryIndexedDataset(f"data/{ds_name}", split="train")
            check(f"データセット {ds_name}: ロード成功 ({ds.num_docs} docs)", ds.num_docs > 0)
        except Exception as e:
            check(f"データセット {ds_name}: エラー - {e}", False)
    else:
        check(f"データセット {ds_name}: train.idxが存在", False, critical=False)

# =============================================================================
# 3. 設定ファイル検証
# =============================================================================
print("\n--- 3. 設定ファイル検証 ---")

# Check dataset config
dataset_config_path = Path("configs/dataset_japanese_chat_optimized.yaml")
if dataset_config_path.exists():
    with open(dataset_config_path) as f:
        ds_config = yaml.safe_load(f)
    
    datasets_in_config = ds_config.get("datasets", {})
    total_weight = sum(d.get("weight", 0) for d in datasets_in_config.values())
    check(f"データセット重み合計: {total_weight:.2f} (期待値: 1.0)", abs(total_weight - 1.0) < 0.01)
    
    for name, cfg in datasets_in_config.items():
        path = Path(cfg.get("path", ""))
        check(f"データセット {name}: パスが存在", (path / "train.bin").exists() or (Path("data") / name / "train.bin").exists(), critical=False)
else:
    check("dataset_japanese_chat_optimized.yamlが存在", False)

# Check model config
model_config_path = Path("configs/phase8_300m_japanese_chat.yaml")
if model_config_path.exists():
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)
    
    n_seq = model_config.get("n_seq", 0)
    check(f"n_seq: {n_seq} (期待値: 256-2048)", 256 <= n_seq <= 2048)
    
    lr = model_config.get("learning_rate", model_config.get("lr", 0))
    check(f"学習率: {lr} (期待値: 1e-5 ~ 1e-3)", 1e-6 <= lr <= 1e-2)
    
    batch_size = model_config.get("batch_size", 0)
    check(f"バッチサイズ: {batch_size} (期待値: 1-64)", 1 <= batch_size <= 64)
else:
    check("phase8_300m_japanese_chat.yamlが存在", False)

# =============================================================================
# 4. モデルアーキテクチャ検証
# =============================================================================
print("\n--- 4. モデルアーキテクチャ検証 ---")

try:
    import torch
    from src.models.configurable_resnet_bk import ResNetBKConfig, ConfigurableResNetBK
    
    # Create small test model
    test_config = ResNetBKConfig(
        d_model=64,
        n_layers=2,
        n_seq=32,
        num_heads=4,
        vocab_size=1000,
        model_type="resnet_bk",
    )
    
    model = ConfigurableResNetBK(test_config)
    check("モデル: インスタンス化成功", model is not None)
    
    # Test forward pass
    x = torch.randint(0, 1000, (2, 32))  # batch=2, seq=32
    
    with torch.no_grad():
        output = model(x)
    
    if isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output
    
    check("モデル: forward成功", logits is not None)
    check(f"モデル: 出力形状 {logits.shape} (期待: [2, 32, 1000])", logits.shape == (2, 32, 1000))
    check("モデル: NaNなし", not torch.isnan(logits).any().item())
    check("モデル: Infなし", not torch.isinf(logits).any().item())
    
    # Test gradient flow
    model.train()
    x = torch.randint(0, 1000, (2, 32))
    y = torch.randint(0, 1000, (2, 32))
    
    output = model(x)
    if isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output
    
    loss = torch.nn.functional.cross_entropy(logits.view(-1, 1000), y.view(-1))
    loss.backward()
    
    # Check gradient norms
    total_grad_norm = 0.0
    param_count = 0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item() ** 2
            param_count += 1
    total_grad_norm = total_grad_norm ** 0.5
    
    check(f"勾配: ノルム={total_grad_norm:.4f} (期待: > 0)", total_grad_norm > 0.001)
    check(f"勾配: パラメータ数={param_count} (期待: > 0)", param_count > 0)
    
    # Check initial loss
    expected_random_loss = np.log(1000)  # ~6.9 for vocab_size=1000
    check(f"初期損失: {loss.item():.2f} (ランダム期待値: ~{expected_random_loss:.1f})", 
          abs(loss.item() - expected_random_loss) < 3.0)
    
except Exception as e:
    import traceback
    print(f"❌ モデルテスト失敗: {e}")
    traceback.print_exc()
    results["failed"] += 1

# =============================================================================
# 5. BK-Core/AR-SSM接続検証
# =============================================================================
print("\n--- 5. BK-Core/AR-SSM接続検証 ---")

integrated_model_path = Path("src/models/phase8/integrated_model.py")
if integrated_model_path.exists():
    content = integrated_model_path.read_text()
    check("IntegratedModel: BK-Core参照が存在", "bk_core" in content.lower() or "bk-core" in content.lower())
    check("IntegratedModel: SSM参照が存在", "ssm" in content.lower())
    check("IntegratedModel: 残差接続が存在", "residual" in content.lower() or "+=" in content or "+ x" in content or "x +" in content)

# =============================================================================
# 最終結果
# =============================================================================
print("\n" + "=" * 70)
print("検証結果サマリー")
print("=" * 70)
print(f"✅ 合格: {results['passed']}")
print(f"❌ 失敗: {results['failed']}")
print(f"⚠️  警告: {results['warnings']}")

if results["failed"] == 0:
    print("\n🎉 すべての検証に合格しました！訓練を開始できます。")
    sys.exit(0)
else:
    print(f"\n⚠️  {results['failed']}件の問題が検出されました。修正が必要です。")
    sys.exit(1)
