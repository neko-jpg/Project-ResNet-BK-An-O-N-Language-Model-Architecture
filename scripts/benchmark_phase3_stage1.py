"""
Phase 3 Stage 1 Benchmark Script

このスクリプトは、Phase 3 Stage 1（Complex Dynamics Only）のベンチマークを実行します。

測定項目:
1. Perplexity (WikiText-2)
2. VRAM使用量
3. Throughput (tokens/sec)

完了条件:
- Perplexity: Phase 2比 +3%以内
- VRAM: Phase 2比 52%以下
- 数値安定性: NaN発生率 0%

Requirements: 1.18, 1.19, 1.20
Author: Project MUSE Team
Date: 2025-11-21
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Phase 3 imports
try:
    from src.models.phase3.stage1_model import Phase3Stage1Model, Phase3Stage1Config
except ImportError:
    warnings.warn("Phase 3 Stage 1 model not found. Please implement it first.")
    Phase3Stage1Model = None
    Phase3Stage1Config = None

# Phase 2 imports for baseline
try:
    from src.models.phase2.factory import create_phase2_model, Phase2Config
except ImportError:
    warnings.warn("Phase 2 model not found. Baseline comparison will be skipped.")
    create_phase2_model = None
    Phase2Config = None


def set_seed(seed: int = 42):
    """乱数シードを固定"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_tokenizer(name: str = "gpt2"):
    """トークナイザーを取得"""
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_wikitext2_loader(
    tokenizer,
    seq_length: int = 1024,
    batch_size: int = 4,
    split: str = "test",
    max_samples: Optional[int] = None
) -> DataLoader:
    """
    WikiText-2データローダーを準備
    
    Args:
        tokenizer: トークナイザー
        seq_length: シーケンス長
        batch_size: バッチサイズ
        split: データ分割 ("train", "validation", "test")
        max_samples: 最大サンプル数（Noneの場合は全データ）
    
    Returns:
        DataLoader
    """
    # WikiText-2をロード
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    
    # トークナイズ
    def tokenize_function(examples):
        return tokenizer(examples["text"], add_special_tokens=False)
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing"
    )
    
    # シーケンスにグループ化
    def group_texts(examples):
        # 全トークンを連結
        concatenated = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)
        
        # seq_length + 1 のチャンクに分割
        total_length = len(concatenated)
        chunk_size = seq_length + 1
        total_length = (total_length // chunk_size) * chunk_size
        
        result = {
            "input_ids": [
                concatenated[i : i + chunk_size]
                for i in range(0, total_length, chunk_size)
            ]
        }
        
        return result
    
    grouped = tokenized.map(
        group_texts,
        batched=True,
        remove_columns=tokenized.column_names,
        desc="Grouping texts"
    )
    
    # サンプル数を制限
    if max_samples is not None and len(grouped) > max_samples:
        grouped = grouped.select(range(max_samples))
    
    # PyTorch形式に変換
    grouped.set_format(type="torch", columns=["input_ids"])
    
    # DataLoaderを作成
    def collate_fn(batch):
        input_ids = torch.stack([b["input_ids"][:-1] for b in batch])
        labels = torch.stack([b["input_ids"][1:] for b in batch])
        return {"input_ids": input_ids, "labels": labels}
    
    loader = DataLoader(
        grouped,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    return loader


def measure_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    model_name: str = "model"
) -> Dict[str, float]:
    """
    Perplexityを測定
    
    測定条件（タスク7.1要件）:
    - Batch=4, Seq=1024, fp16, 同一シード
    - WikiText-2データセット
    
    目標:
    - Phase 2比 +3%以内（Phase 2が30.0なら30.9以下）
    
    Args:
        model: 評価するモデル
        dataloader: データローダー
        device: デバイス
        max_batches: 最大バッチ数（Noneの場合は全データ）
        model_name: モデル名（ログ用）
    
    Returns:
        {"ppl": float, "loss": float, "num_tokens": int, "nan_count": int, "inf_count": int}
    
    Requirements: 1.18
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    nan_count = 0
    inf_count = 0
    batch_count = 0
    
    print(f"\n  Measuring Perplexity for {model_name}...")
    print(f"  - Max batches: {max_batches if max_batches else 'All'}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            try:
                logits = model(input_ids)
                
                # NaN/Infチェック（詳細）
                has_nan = torch.isnan(logits).any().item()
                has_inf = torch.isinf(logits).any().item()
                
                if has_nan:
                    nan_count += 1
                    warnings.warn(f"NaN detected in batch {batch_idx} for {model_name}")
                    continue
                
                if has_inf:
                    inf_count += 1
                    warnings.warn(f"Inf detected in batch {batch_idx} for {model_name}")
                    continue
                
                # Loss計算
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction="sum"
                )
                
                # Loss自体のNaN/Infチェック
                if torch.isnan(loss).item() or torch.isinf(loss).item():
                    nan_count += 1
                    warnings.warn(f"NaN/Inf in loss for batch {batch_idx} for {model_name}")
                    continue
                
                total_loss += loss.item()
                total_tokens += labels.numel()
                batch_count += 1
                
                # 進捗表示（10バッチごと）
                if (batch_idx + 1) % 10 == 0:
                    current_ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
                    print(f"  - Batch {batch_idx + 1}: PPL={current_ppl:.2f}, Tokens={total_tokens:,}")
                
            except RuntimeError as e:
                warnings.warn(f"Runtime error in batch {batch_idx} for {model_name}: {e}")
                nan_count += 1
                continue
    
    # Perplexity計算
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        ppl = torch.exp(torch.tensor(avg_loss)).item()
    else:
        avg_loss = float('inf')
        ppl = float('inf')
    
    print(f"  - Final PPL: {ppl:.2f}")
    print(f"  - Total tokens: {total_tokens:,}")
    print(f"  - Valid batches: {batch_count}")
    print(f"  - NaN batches: {nan_count}")
    print(f"  - Inf batches: {inf_count}")
    
    return {
        "ppl": ppl,
        "loss": avg_loss,
        "num_tokens": total_tokens,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "valid_batches": batch_count
    }


def measure_vram(
    model: nn.Module,
    seq_length: int = 2048,
    batch_size: int = 2,
    device: torch.device = torch.device("cuda"),
    vocab_size: int = 50257,
    model_name: str = "model",
    use_gradient_checkpointing: bool = True
) -> Dict[str, float]:
    """
    VRAM使用量を測定
    
    測定条件（タスク7.2要件）:
    - Batch=2, Seq=2048
    - Forward + Backward pass
    - Gradient Checkpointing有効
    
    目標:
    - Phase 2比 52%以下（Phase 2が6.0GBなら3.12GB以下）
    
    Args:
        model: 評価するモデル
        seq_length: シーケンス長
        batch_size: バッチサイズ
        device: デバイス
        vocab_size: 語彙サイズ
        model_name: モデル名（ログ用）
        use_gradient_checkpointing: Gradient Checkpointingを使用するか
    
    Returns:
        {"vram_gb": float, "vram_mb": float, "peak_vram_gb": float, "forward_vram_gb": float}
    
    Requirements: 1.19
    """
    if not torch.cuda.is_available():
        print(f"  WARNING: CUDA not available. Skipping VRAM measurement for {model_name}.")
        return {
            "vram_gb": 0.0,
            "vram_mb": 0.0,
            "peak_vram_gb": 0.0,
            "forward_vram_gb": 0.0
        }
    
    print(f"\n  Measuring VRAM for {model_name}...")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Sequence length: {seq_length}")
    print(f"  - Gradient Checkpointing: {use_gradient_checkpointing}")
    
    model.train()  # Gradient Checkpointing有効化のため
    
    # Gradient Checkpointingを有効化（モデルがサポートしている場合）
    if use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print(f"  - Gradient Checkpointing enabled")
    
    # VRAMをリセット
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    
    # ダミー入力
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    
    # Forward + Backward pass
    try:
        # Forward pass
        torch.cuda.reset_peak_memory_stats(device)
        logits = model(input_ids)
        torch.cuda.synchronize()
        forward_memory = torch.cuda.max_memory_allocated(device)
        forward_vram_gb = forward_memory / (1024 ** 3)
        
        print(f"  - Forward VRAM: {forward_vram_gb:.2f} GB")
        
        # Loss計算
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # Backward pass
        loss.backward()
        torch.cuda.synchronize()
        
        # VRAM測定
        peak_memory = torch.cuda.max_memory_allocated(device)
        current_memory = torch.cuda.memory_allocated(device)
        
        vram_gb = current_memory / (1024 ** 3)
        vram_mb = current_memory / (1024 ** 2)
        peak_vram_gb = peak_memory / (1024 ** 3)
        
        print(f"  - Current VRAM: {vram_gb:.2f} GB")
        print(f"  - Peak VRAM: {peak_vram_gb:.2f} GB")
        
    except RuntimeError as e:
        warnings.warn(f"VRAM measurement failed for {model_name}: {e}")
        vram_gb = float('inf')
        vram_mb = float('inf')
        peak_vram_gb = float('inf')
        forward_vram_gb = float('inf')
    
    # クリーンアップ
    if use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    
    torch.cuda.empty_cache()
    
    return {
        "vram_gb": vram_gb,
        "vram_mb": vram_mb,
        "peak_vram_gb": peak_vram_gb,
        "forward_vram_gb": forward_vram_gb
    }


def measure_throughput(
    model: nn.Module,
    seq_length: int = 1024,
    batch_size: int = 4,
    device: torch.device = torch.device("cuda"),
    vocab_size: int = 50257,
    num_iterations: int = 10
) -> Dict[str, float]:
    """
    Throughputを測定
    
    Args:
        model: 評価するモデル
        seq_length: シーケンス長
        batch_size: バッチサイズ
        device: デバイス
        vocab_size: 語彙サイズ
        num_iterations: 測定回数
    
    Returns:
        {"tokens_per_sec": float, "ms_per_token": float}
    """
    model.eval()
    
    # ウォームアップ
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    with torch.no_grad():
        _ = model(input_ids)
    
    # 測定
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_ids)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    # 計算
    total_time = end_time - start_time
    total_tokens = batch_size * seq_length * num_iterations
    tokens_per_sec = total_tokens / total_time
    ms_per_token = (total_time * 1000) / total_tokens
    
    return {
        "tokens_per_sec": tokens_per_sec,
        "ms_per_token": ms_per_token
    }


def benchmark_phase3_stage1(
    phase3_model: nn.Module,
    phase2_model: Optional[nn.Module] = None,
    device: torch.device = torch.device("cuda"),
    seed: int = 42,
    ppl_batch_size: int = 4,
    ppl_seq_length: int = 1024,
    vram_batch_size: int = 2,
    vram_seq_length: int = 2048,
    max_ppl_batches: Optional[int] = None
) -> Dict[str, Any]:
    """
    Phase 3 Stage 1の完全ベンチマーク
    
    Args:
        phase3_model: Phase 3 Stage 1モデル
        phase2_model: Phase 2モデル（ベースライン）
        device: デバイス
        seed: 乱数シード
        ppl_batch_size: Perplexity測定のバッチサイズ
        ppl_seq_length: Perplexity測定のシーケンス長
        vram_batch_size: VRAM測定のバッチサイズ
        vram_seq_length: VRAM測定のシーケンス長
        max_ppl_batches: Perplexity測定の最大バッチ数
    
    Returns:
        ベンチマーク結果の辞書
    
    Requirements: 1.18, 1.19, 1.20
    """
    set_seed(seed)
    results = {
        "device": str(device),
        "seed": seed,
        "ppl_batch_size": ppl_batch_size,
        "ppl_seq_length": ppl_seq_length,
        "vram_batch_size": vram_batch_size,
        "vram_seq_length": vram_seq_length
    }
    
    # トークナイザー準備
    print("Preparing tokenizer...")
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    
    # データローダー準備
    print(f"Preparing WikiText-2 dataloader (batch={ppl_batch_size}, seq={ppl_seq_length})...")
    dataloader = prepare_wikitext2_loader(
        tokenizer,
        seq_length=ppl_seq_length,
        batch_size=ppl_batch_size,
        split="test"
    )
    
    # ===== Phase 3 Stage 1 測定 =====
    print("\n" + "=" * 60)
    print("Phase 3 Stage 1 Benchmark")
    print("=" * 60)
    
    # Perplexity測定
    print("\n[1/3] Measuring Perplexity...")
    phase3_ppl_results = measure_perplexity(
        phase3_model,
        dataloader,
        device,
        max_batches=max_ppl_batches,
        model_name="Phase 3 Stage 1"
    )
    
    results["phase3_ppl"] = phase3_ppl_results["ppl"]
    results["phase3_loss"] = phase3_ppl_results["loss"]
    results["phase3_nan_count"] = phase3_ppl_results["nan_count"]
    results["phase3_inf_count"] = phase3_ppl_results.get("inf_count", 0)
    results["phase3_valid_batches"] = phase3_ppl_results.get("valid_batches", 0)
    
    # VRAM測定
    print(f"\n[2/3] Measuring VRAM (batch={vram_batch_size}, seq={vram_seq_length})...")
    phase3_vram_results = measure_vram(
        phase3_model,
        seq_length=vram_seq_length,
        batch_size=vram_batch_size,
        device=device,
        vocab_size=vocab_size,
        model_name="Phase 3 Stage 1",
        use_gradient_checkpointing=True
    )
    
    results["phase3_vram_gb"] = phase3_vram_results["vram_gb"]
    results["phase3_peak_vram_gb"] = phase3_vram_results["peak_vram_gb"]
    results["phase3_forward_vram_gb"] = phase3_vram_results.get("forward_vram_gb", 0.0)
    
    # Throughput測定
    print("\n[3/3] Measuring Throughput...")
    phase3_throughput_results = measure_throughput(
        phase3_model,
        seq_length=ppl_seq_length,
        batch_size=ppl_batch_size,
        device=device,
        vocab_size=vocab_size
    )
    print(f"  - Throughput: {phase3_throughput_results['tokens_per_sec']:.1f} tokens/sec")
    print(f"  - Latency: {phase3_throughput_results['ms_per_token']:.3f} ms/token")
    
    results["phase3_throughput"] = phase3_throughput_results["tokens_per_sec"]
    results["phase3_latency"] = phase3_throughput_results["ms_per_token"]
    
    # ===== Phase 2 Baseline測定 =====
    if phase2_model is not None:
        print("\n" + "=" * 60)
        print("Phase 2 Baseline Benchmark")
        print("=" * 60)
        
        # Perplexity測定
        print("\n[1/3] Measuring Perplexity...")
        phase2_ppl_results = measure_perplexity(
            phase2_model,
            dataloader,
            device,
            max_batches=max_ppl_batches,
            model_name="Phase 2 Baseline"
        )
        
        results["phase2_ppl"] = phase2_ppl_results["ppl"]
        results["phase2_loss"] = phase2_ppl_results["loss"]
        
        # VRAM測定
        print(f"\n[2/3] Measuring VRAM (batch={vram_batch_size}, seq={vram_seq_length})...")
        phase2_vram_results = measure_vram(
            phase2_model,
            seq_length=vram_seq_length,
            batch_size=vram_batch_size,
            device=device,
            vocab_size=vocab_size,
            model_name="Phase 2 Baseline",
            use_gradient_checkpointing=True
        )
        
        results["phase2_vram_gb"] = phase2_vram_results["vram_gb"]
        results["phase2_peak_vram_gb"] = phase2_vram_results["peak_vram_gb"]
        results["phase2_forward_vram_gb"] = phase2_vram_results.get("forward_vram_gb", 0.0)
        
        # Throughput測定
        print("\n[3/3] Measuring Throughput...")
        phase2_throughput_results = measure_throughput(
            phase2_model,
            seq_length=ppl_seq_length,
            batch_size=ppl_batch_size,
            device=device,
            vocab_size=vocab_size
        )
        print(f"  - Throughput: {phase2_throughput_results['tokens_per_sec']:.1f} tokens/sec")
        
        results["phase2_throughput"] = phase2_throughput_results["tokens_per_sec"]
        
        # ===== 比較 =====
        print("\n" + "=" * 60)
        print("Comparison: Phase 3 Stage 1 vs Phase 2")
        print("=" * 60)
        
        # Perplexity比較
        ppl_ratio = results["phase3_ppl"] / results["phase2_ppl"]
        ppl_diff_pct = (ppl_ratio - 1.0) * 100
        ppl_target = 1.03  # +3%以内
        ppl_pass = ppl_ratio <= ppl_target
        
        print(f"\n[1/4] Perplexity:")
        print(f"  - Phase 3: {results['phase3_ppl']:.2f}")
        print(f"  - Phase 2: {results['phase2_ppl']:.2f}")
        print(f"  - Ratio: {ppl_ratio:.4f} ({ppl_diff_pct:+.2f}%)")
        print(f"  - Target: ≤ {ppl_target:.2f} (Phase 2 + 3%)")
        print(f"  - Status: {'✓ PASS' if ppl_pass else '✗ FAIL'}")
        
        results["ppl_ratio"] = ppl_ratio
        results["ppl_diff_pct"] = ppl_diff_pct
        results["ppl_target"] = ppl_target
        results["ppl_pass"] = ppl_pass
        
        # VRAM比較
        vram_ratio = results["phase3_vram_gb"] / results["phase2_vram_gb"]
        vram_reduction_pct = (1.0 - vram_ratio) * 100
        vram_target = 0.52  # 52%以下
        vram_pass = vram_ratio <= vram_target
        
        print(f"\n[2/4] VRAM:")
        print(f"  - Phase 3: {results['phase3_vram_gb']:.2f} GB")
        print(f"  - Phase 2: {results['phase2_vram_gb']:.2f} GB")
        print(f"  - Ratio: {vram_ratio:.4f} ({vram_reduction_pct:+.2f}% reduction)")
        print(f"  - Target: ≤ {vram_target:.2f} (52% of Phase 2)")
        print(f"  - Status: {'✓ PASS' if vram_pass else '✗ FAIL'}")
        
        results["vram_ratio"] = vram_ratio
        results["vram_reduction_pct"] = vram_reduction_pct
        results["vram_target"] = vram_target
        results["vram_pass"] = vram_pass
        
        # Throughput比較
        throughput_ratio = results["phase3_throughput"] / results["phase2_throughput"]
        throughput_diff_pct = (throughput_ratio - 1.0) * 100
        
        print(f"\n[3/4] Throughput:")
        print(f"  - Phase 3: {results['phase3_throughput']:.1f} tokens/sec")
        print(f"  - Phase 2: {results['phase2_throughput']:.1f} tokens/sec")
        print(f"  - Ratio: {throughput_ratio:.4f} ({throughput_diff_pct:+.2f}%)")
        print(f"  - Note: Throughput is informational (no pass/fail criteria)")
        
        results["throughput_ratio"] = throughput_ratio
        results["throughput_diff_pct"] = throughput_diff_pct
        
        # 数値安定性チェック
        nan_pass = results["phase3_nan_count"] == 0
        inf_pass = results.get("phase3_inf_count", 0) == 0
        stability_pass = nan_pass and inf_pass
        
        print(f"\n[4/4] Numerical Stability:")
        print(f"  - NaN Count: {results['phase3_nan_count']}")
        print(f"  - Inf Count: {results.get('phase3_inf_count', 0)}")
        print(f"  - Target: 0 (100% stability)")
        print(f"  - Status: {'✓ PASS' if stability_pass else '✗ FAIL'}")
        
        results["nan_pass"] = nan_pass
        results["inf_pass"] = inf_pass
        results["stability_pass"] = stability_pass
        
        # 総合判定
        all_pass = ppl_pass and vram_pass and stability_pass
        
        print(f"\n{'=' * 60}")
        print(f"Overall Status: {'✓ ALL PASS' if all_pass else '✗ SOME FAILED'}")
        print(f"{'=' * 60}")
        
        if all_pass:
            print("\n🎉 Phase 3 Stage 1 has achieved all numerical targets!")
            print("   - Perplexity: Within +3% of Phase 2")
            print("   - VRAM: 52% or less of Phase 2")
            print("   - Stability: 100% (no NaN/Inf)")
        else:
            print("\n⚠️  Some targets were not met:")
            if not ppl_pass:
                print(f"   - Perplexity: {ppl_ratio:.4f} > {ppl_target:.2f} (target)")
            if not vram_pass:
                print(f"   - VRAM: {vram_ratio:.4f} > {vram_target:.2f} (target)")
            if not stability_pass:
                print(f"   - Stability: NaN={results['phase3_nan_count']}, Inf={results.get('phase3_inf_count', 0)}")
        
        results["all_pass"] = all_pass
    
    else:
        # Phase 2がない場合
        results["phase2_ppl"] = None
        results["phase2_vram_gb"] = None
        results["phase2_throughput"] = None
        results["ppl_ratio"] = None
        results["vram_ratio"] = None
        results["throughput_ratio"] = None
        results["all_pass"] = results["phase3_nan_count"] == 0
    
    return results


def save_results(results: Dict[str, Any], output_path: Path):
    """
    結果をJSON形式で保存
    
    フォーマット（タスク7.3要件）:
    {
      "ppl": 30.5,
      "ppl_phase2": 30.0,
      "ppl_ratio": 1.017,
      "vram_gb": 3.1,
      "vram_phase2_gb": 6.0,
      "vram_ratio": 0.517,
      "pass": true
    }
    
    Args:
        results: ベンチマーク結果
        output_path: 出力パス
    
    Requirements: 1.20
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # メタデータを追加
    results["benchmark_name"] = "Phase 3 Stage 1 Benchmark"
    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results["device"] = str(results.get("device", "unknown"))
    
    # JSON形式で保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 60}")
    
    # サマリーを表示
    print(f"\nJSON Summary:")
    print(f"  - ppl: {results.get('phase3_ppl', 'N/A')}")
    print(f"  - ppl_phase2: {results.get('phase2_ppl', 'N/A')}")
    print(f"  - ppl_ratio: {results.get('ppl_ratio', 'N/A')}")
    print(f"  - vram_gb: {results.get('phase3_vram_gb', 'N/A')}")
    print(f"  - vram_phase2_gb: {results.get('phase2_vram_gb', 'N/A')}")
    print(f"  - vram_ratio: {results.get('vram_ratio', 'N/A')}")
    print(f"  - pass: {results.get('all_pass', False)}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Stage 1 Benchmark")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ppl-batch-size", type=int, default=4, help="Batch size for perplexity")
    parser.add_argument("--ppl-seq-length", type=int, default=1024, help="Sequence length for perplexity")
    parser.add_argument("--vram-batch-size", type=int, default=2, help="Batch size for VRAM measurement")
    parser.add_argument("--vram-seq-length", type=int, default=1024, help="Sequence length for VRAM measurement")
    parser.add_argument("--max-ppl-batches", type=int, default=None, help="Max batches for perplexity (for quick test)")
    parser.add_argument("--skip-phase2", action="store_true", help="Skip Phase 2 baseline")
    parser.add_argument("--output", type=str, default="results/benchmarks/phase3_stage1_comparison.json", help="Output JSON path")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Phase 3 Stage 1モデルを作成
    print("\nCreating Phase 3 Stage 1 model...")
    if Phase3Stage1Model is None or Phase3Stage1Config is None:
        print("ERROR: Phase 3 Stage 1 model not implemented yet.")
        print("Please implement src/models/phase3/stage1_model.py first.")
        sys.exit(1)
    
    phase3_config = Phase3Stage1Config(
        vocab_size=50257,
        d_model=512,
        n_layers=6,
        n_seq=1024,
        use_complex32=True
    )
    phase3_model = Phase3Stage1Model(phase3_config).to(device)
    
    # Phase 2モデルを作成（ベースライン）
    phase2_model = None
    if not args.skip_phase2 and create_phase2_model is not None:
        print("\nCreating Phase 2 baseline model...")
        phase2_config = Phase2Config(
            vocab_size=50257,
            d_model=512,
            n_layers=6,
            n_seq=1024
        )
        phase2_model = create_phase2_model(config=phase2_config).to(device)
    
    # ベンチマーク実行
    print("\nStarting benchmark...")
    results = benchmark_phase3_stage1(
        phase3_model=phase3_model,
        phase2_model=phase2_model,
        device=device,
        seed=args.seed,
        ppl_batch_size=args.ppl_batch_size,
        ppl_seq_length=args.ppl_seq_length,
        vram_batch_size=args.vram_batch_size,
        vram_seq_length=args.vram_seq_length,
        max_ppl_batches=args.max_ppl_batches
    )
    
    # 結果を保存
    output_path = Path(args.output)
    save_results(results, output_path)
    
    print("\nBenchmark completed!")
    
    # 終了コード
    if results.get("all_pass", False):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
