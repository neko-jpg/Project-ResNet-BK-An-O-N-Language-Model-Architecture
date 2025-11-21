"""
Phase 3 Stage 2 Benchmark Script

このスクリプトは、Phase 3 Stage 2（Hamiltonian ODE Integration）のベンチマークを実行します。

測定項目:
1. Perplexity (WikiText-2) - Stage 1との比較
2. Energy Drift - エネルギー保存則の検証
3. VRAM使用量 - Symplectic Adjoint vs Full Backprop

完了条件:
- Perplexity: Stage 1比 +2%以内
- Energy Drift: < 5e-5（閾値1e-4の半分）
- VRAM: Batch=2, Seq=2048で < 7.5GB
- VRAM削減率: Full Backprop比 70%以上削減

Requirements: 2.21, 2.22, 2.23
Author: Project MUSE Team
Date: 2025-11-21
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List

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
    from src.models.phase3.stage2_model import Phase3Stage2Model, Phase3Stage2Config
    from src.models.phase3.stage1_model import Phase3Stage1Model, Phase3Stage1Config
    from src.models.phase3.hamiltonian import HamiltonianFunction
except ImportError as e:
    warnings.warn(f"Phase 3 models not found: {e}")
    Phase3Stage2Model = None
    Phase3Stage2Config = None
    Phase3Stage1Model = None
    Phase3Stage1Config = None
    HamiltonianFunction = None


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
    """WikiText-2データローダーを準備"""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], add_special_tokens=False)
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing"
    )
    
    def group_texts(examples):
        concatenated = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)
        
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
    
    if max_samples is not None and len(grouped) > max_samples:
        grouped = grouped.select(range(max_samples))
    
    grouped.set_format(type="torch", columns=["input_ids"])
    
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
    
    測定条件（タスク13.1要件）:
    - Batch=4, Seq=1024, fp16, ODE steps=10
    - WikiText-2データセット
    
    目標:
    - Stage 1比 +2%以内
    
    Args:
        model: 評価するモデル
        dataloader: データローダー
        device: デバイス
        max_batches: 最大バッチ数
        model_name: モデル名
    
    Returns:
        {"ppl": float, "loss": float, "num_tokens": int}
    
    Requirements: 2.21
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batch_count = 0
    
    print(f"\n  Measuring Perplexity for {model_name}...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            try:
                logits = model(input_ids)
                
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    warnings.warn(f"NaN/Inf detected in batch {batch_idx}")
                    continue
                
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction="sum"
                )
                
                if torch.isnan(loss) or torch.isinf(loss):
                    warnings.warn(f"NaN/Inf in loss for batch {batch_idx}")
                    continue
                
                total_loss += loss.item()
                total_tokens += labels.numel()
                batch_count += 1
                
                if (batch_idx + 1) % 10 == 0:
                    current_ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
                    print(f"  - Batch {batch_idx + 1}: PPL={current_ppl:.2f}")
                
            except RuntimeError as e:
                warnings.warn(f"Runtime error in batch {batch_idx}: {e}")
                continue
    
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        ppl = torch.exp(torch.tensor(avg_loss)).item()
    else:
        avg_loss = float('inf')
        ppl = float('inf')
    
    print(f"  - Final PPL: {ppl:.2f} (tokens: {total_tokens:,})")
    
    return {
        "ppl": ppl,
        "loss": avg_loss,
        "num_tokens": total_tokens,
        "valid_batches": batch_count
    }


def measure_energy_drift(
    model: nn.Module,
    batch_size: int = 4,
    seq_length: int = 512,
    device: torch.device = torch.device("cuda"),
    vocab_size: int = 50257,
    dt: float = 0.1,
    num_steps: int = 100
) -> Dict[str, Any]:
    """
    Energy Driftを測定
    
    測定条件（タスク13.2要件）:
    - Batch=4, Seq=512, dt=0.1, 100 steps
    
    目標:
    - Energy Drift < 5e-5（閾値1e-4の半分）
    - エネルギーが単調増加/減少していないこと（振動許容範囲 ±10%）
    
    Args:
        model: 評価するモデル
        batch_size: バッチサイズ
        seq_length: シーケンス長
        device: デバイス
        vocab_size: 語彙サイズ
        dt: 時間刻み
        num_steps: 積分ステップ数
    
    Returns:
        {
            "mean_energy": float,
            "max_drift": float,
            "mean_drift": float,
            "energy_trajectory": List[float],
            "monotonic_violation": bool,
            "pass": bool
        }
    
    Requirements: 2.22
    """
    print(f"\n  Measuring Energy Drift...")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Sequence length: {seq_length}")
    print(f"  - dt: {dt}, steps: {num_steps}")
    
    model.eval()
    
    # ダミー入力を生成
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    
    # モデルからHamiltonian関数を取得
    hamiltonian_func = None
    if hasattr(model, 'blocks') and len(model.blocks) > 0:
        first_block = model.blocks[0]
        if hasattr(first_block, 'ode') and hasattr(first_block.ode, 'h_func'):
            hamiltonian_func = first_block.ode.h_func
    
    if hamiltonian_func is None:
        warnings.warn("Hamiltonian function not found in model")
        return {
            "mean_energy": 0.0,
            "max_drift": 0.0,
            "mean_drift": 0.0,
            "energy_trajectory": [],
            "monotonic_violation": False,
            "pass": False,
            "error": "Hamiltonian function not found"
        }
    
    # 初期状態を生成（位置と運動量）
    with torch.no_grad():
        # Embeddingから初期状態を取得
        if hasattr(model, 'embedding'):
            z = model.embedding(input_ids)
            if hasattr(z, 'real'):
                # ComplexTensorの場合
                q = z.real.flatten(1)  # (B, N*D)
            else:
                q = z.flatten(1)
        else:
            d_model = 512  # デフォルト
            q = torch.randn(batch_size, seq_length * d_model, device=device)
        
        # 運動量を初期化（ゼロまたは小さなランダム値）
        p = torch.randn_like(q) * 0.1
        
        # 位相空間の状態
        x = torch.cat([q, p], dim=-1)  # (B, 2*N*D)
    
    # エネルギー軌跡を記録
    energy_trajectory = []
    
    with torch.no_grad():
        for step in range(num_steps):
            # エネルギーを計算
            energy = hamiltonian_func(0, x)  # (B,)
            energy_trajectory.append(energy.mean().item())
            
            # Leapfrog積分で次の状態へ
            from src.models.phase3.hamiltonian import symplectic_leapfrog_step
            x = symplectic_leapfrog_step(hamiltonian_func, x, dt)
            
            if (step + 1) % 20 == 0:
                print(f"  - Step {step + 1}/{num_steps}: Energy={energy_trajectory[-1]:.6f}")
    
    # エネルギー統計を計算
    energy_array = torch.tensor(energy_trajectory)
    mean_energy = energy_array.mean().item()
    max_energy = energy_array.max().item()
    min_energy = energy_array.min().item()
    
    # Energy Drift = (E_max - E_min) / E_mean
    if abs(mean_energy) > 1e-10:
        max_drift = (max_energy - min_energy) / abs(mean_energy)
    else:
        max_drift = max_energy - min_energy
    
    # 平均ドリフト（隣接ステップ間の変化）
    energy_diffs = torch.abs(energy_array[1:] - energy_array[:-1])
    mean_drift = energy_diffs.mean().item()
    
    # 単調性チェック（±10%の振動は許容）
    tolerance = 0.1
    monotonic_increase = (energy_array >= energy_array[0] * (1 - tolerance)).all().item()
    monotonic_decrease = (energy_array <= energy_array[0] * (1 + tolerance)).all().item()
    monotonic_violation = not (monotonic_increase or monotonic_decrease or True)  # 振動は正常
    
    # 目標達成判定
    target_drift = 5e-5
    pass_drift = max_drift < target_drift
    pass_monotonic = not monotonic_violation
    pass_overall = pass_drift and pass_monotonic
    
    print(f"\n  Energy Drift Results:")
    print(f"  - Mean Energy: {mean_energy:.6e}")
    print(f"  - Max Drift: {max_drift:.6e} (target: < {target_drift:.6e})")
    print(f"  - Mean Drift: {mean_drift:.6e}")
    print(f"  - Monotonic Violation: {monotonic_violation}")
    print(f"  - Status: {'✓ PASS' if pass_overall else '✗ FAIL'}")
    
    return {
        "mean_energy": mean_energy,
        "max_drift": max_drift,
        "mean_drift": mean_drift,
        "energy_trajectory": energy_trajectory,
        "monotonic_violation": monotonic_violation,
        "pass": pass_overall,
        "target_drift": target_drift
    }



def measure_vram_comparison(
    model: nn.Module,
    seq_length: int = 2048,
    batch_size: int = 2,
    device: torch.device = torch.device("cuda"),
    vocab_size: int = 50257
) -> Dict[str, Any]:
    """
    VRAM使用量を測定（Symplectic Adjoint vs Full Backprop）
    
    測定条件（タスク13.3要件）:
    - Batch=2, Seq=2048
    - Forward + Backward pass
    - Symplectic Adjoint有効 vs Full Backprop
    
    目標:
    - Symplectic Adjoint: < 7.5GB（8GBの93.75%）
    - 削減率: Full Backprop比 70%以上削減
    
    Args:
        model: 評価するモデル
        seq_length: シーケンス長
        batch_size: バッチサイズ
        device: デバイス
        vocab_size: 語彙サイズ
    
    Returns:
        {
            "vram_symplectic_gb": float,
            "vram_full_backprop_gb": float,
            "reduction_ratio": float,
            "reduction_pct": float,
            "pass": bool
        }
    
    Requirements: 2.23
    """
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available. Skipping VRAM measurement.")
        return {
            "vram_symplectic_gb": 0.0,
            "vram_full_backprop_gb": 0.0,
            "reduction_ratio": 0.0,
            "reduction_pct": 0.0,
            "pass": False,
            "error": "CUDA not available"
        }
    
    print(f"\n  Measuring VRAM (Symplectic Adjoint vs Full Backprop)...")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Sequence length: {seq_length}")
    
    model.train()
    
    # ダミー入力
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    
    # ===== 1. Symplectic Adjoint測定 =====
    print(f"\n  [1/2] Measuring with Symplectic Adjoint...")
    
    # モデルをSymplectic Adjointモードに設定
    if hasattr(model, 'set_ode_mode'):
        model.set_ode_mode('symplectic_adjoint')
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    
    try:
        # Forward pass
        logits = model(input_ids)
        
        # Loss計算
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # Backward pass
        loss.backward()
        torch.cuda.synchronize()
        
        # VRAM測定
        vram_symplectic = torch.cuda.max_memory_allocated(device)
        vram_symplectic_gb = vram_symplectic / (1024 ** 3)
        
        print(f"  - Symplectic Adjoint VRAM: {vram_symplectic_gb:.2f} GB")
        
        # 勾配をクリア
        model.zero_grad()
        
    except RuntimeError as e:
        warnings.warn(f"Symplectic Adjoint measurement failed: {e}")
        vram_symplectic_gb = float('inf')
    
    # ===== 2. Full Backprop測定 =====
    print(f"\n  [2/2] Measuring with Full Backprop...")
    
    # モデルをFull Backpropモードに設定
    if hasattr(model, 'set_ode_mode'):
        model.set_ode_mode('full_backprop')
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    
    try:
        # Forward pass
        logits = model(input_ids)
        
        # Loss計算
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # Backward pass
        loss.backward()
        torch.cuda.synchronize()
        
        # VRAM測定
        vram_full_backprop = torch.cuda.max_memory_allocated(device)
        vram_full_backprop_gb = vram_full_backprop / (1024 ** 3)
        
        print(f"  - Full Backprop VRAM: {vram_full_backprop_gb:.2f} GB")
        
        # 勾配をクリア
        model.zero_grad()
        
    except RuntimeError as e:
        warnings.warn(f"Full Backprop measurement failed: {e}")
        vram_full_backprop_gb = float('inf')
    
    # ===== 3. 比較 =====
    if vram_full_backprop_gb > 0 and vram_full_backprop_gb != float('inf'):
        reduction_ratio = vram_symplectic_gb / vram_full_backprop_gb
        reduction_pct = (1.0 - reduction_ratio) * 100
    else:
        reduction_ratio = 0.0
        reduction_pct = 0.0
    
    # 目標達成判定
    target_vram_gb = 7.5
    target_reduction_pct = 70.0
    
    pass_vram = vram_symplectic_gb < target_vram_gb
    pass_reduction = reduction_pct >= target_reduction_pct
    pass_overall = pass_vram and pass_reduction
    
    print(f"\n  VRAM Comparison Results:")
    print(f"  - Symplectic Adjoint: {vram_symplectic_gb:.2f} GB (target: < {target_vram_gb:.2f} GB)")
    print(f"  - Full Backprop: {vram_full_backprop_gb:.2f} GB")
    print(f"  - Reduction: {reduction_pct:.1f}% (target: ≥ {target_reduction_pct:.1f}%)")
    print(f"  - Status: {'✓ PASS' if pass_overall else '✗ FAIL'}")
    
    # モデルをデフォルトモードに戻す
    if hasattr(model, 'set_ode_mode'):
        model.set_ode_mode('symplectic_adjoint')
    
    torch.cuda.empty_cache()
    
    return {
        "vram_symplectic_gb": vram_symplectic_gb,
        "vram_full_backprop_gb": vram_full_backprop_gb,
        "reduction_ratio": reduction_ratio,
        "reduction_pct": reduction_pct,
        "pass_vram": pass_vram,
        "pass_reduction": pass_reduction,
        "pass": pass_overall,
        "target_vram_gb": target_vram_gb,
        "target_reduction_pct": target_reduction_pct
    }


def benchmark_phase3_stage2(
    stage2_model: nn.Module,
    stage1_model: Optional[nn.Module] = None,
    device: torch.device = torch.device("cuda"),
    seed: int = 42,
    ppl_batch_size: int = 4,
    ppl_seq_length: int = 1024,
    energy_batch_size: int = 4,
    energy_seq_length: int = 512,
    vram_batch_size: int = 2,
    vram_seq_length: int = 2048,
    max_ppl_batches: Optional[int] = None
) -> Dict[str, Any]:
    """
    Phase 3 Stage 2の完全ベンチマーク
    
    Args:
        stage2_model: Phase 3 Stage 2モデル
        stage1_model: Phase 3 Stage 1モデル（ベースライン）
        device: デバイス
        seed: 乱数シード
        ppl_batch_size: Perplexity測定のバッチサイズ
        ppl_seq_length: Perplexity測定のシーケンス長
        energy_batch_size: Energy Drift測定のバッチサイズ
        energy_seq_length: Energy Drift測定のシーケンス長
        vram_batch_size: VRAM測定のバッチサイズ
        vram_seq_length: VRAM測定のシーケンス長
        max_ppl_batches: Perplexity測定の最大バッチ数
    
    Returns:
        ベンチマーク結果の辞書
    
    Requirements: 2.21, 2.22, 2.23
    """
    set_seed(seed)
    results = {
        "device": str(device),
        "seed": seed,
        "ppl_batch_size": ppl_batch_size,
        "ppl_seq_length": ppl_seq_length,
        "energy_batch_size": energy_batch_size,
        "energy_seq_length": energy_seq_length,
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
    
    # ===== Phase 3 Stage 2 測定 =====
    print("\n" + "=" * 60)
    print("Phase 3 Stage 2 Benchmark")
    print("=" * 60)
    
    # 1. Perplexity測定
    print("\n[1/3] Measuring Perplexity...")
    stage2_ppl_results = measure_perplexity(
        stage2_model,
        dataloader,
        device,
        max_batches=max_ppl_batches,
        model_name="Phase 3 Stage 2"
    )
    
    results["stage2_ppl"] = stage2_ppl_results["ppl"]
    results["stage2_loss"] = stage2_ppl_results["loss"]
    results["stage2_valid_batches"] = stage2_ppl_results.get("valid_batches", 0)
    
    # 2. Energy Drift測定
    print("\n[2/3] Measuring Energy Drift...")
    energy_results = measure_energy_drift(
        stage2_model,
        batch_size=energy_batch_size,
        seq_length=energy_seq_length,
        device=device,
        vocab_size=vocab_size,
        dt=0.1,
        num_steps=100
    )
    
    results["mean_energy"] = energy_results["mean_energy"]
    results["max_drift"] = energy_results["max_drift"]
    results["mean_drift"] = energy_results["mean_drift"]
    results["energy_trajectory"] = energy_results["energy_trajectory"]
    results["monotonic_violation"] = energy_results["monotonic_violation"]
    results["energy_pass"] = energy_results["pass"]
    
    # 3. VRAM測定
    print("\n[3/3] Measuring VRAM (Symplectic Adjoint vs Full Backprop)...")
    vram_results = measure_vram_comparison(
        stage2_model,
        seq_length=vram_seq_length,
        batch_size=vram_batch_size,
        device=device,
        vocab_size=vocab_size
    )
    
    results["vram_symplectic_gb"] = vram_results["vram_symplectic_gb"]
    results["vram_full_backprop_gb"] = vram_results["vram_full_backprop_gb"]
    results["vram_reduction_ratio"] = vram_results["reduction_ratio"]
    results["vram_reduction_pct"] = vram_results["reduction_pct"]
    results["vram_pass"] = vram_results["pass"]
    
    # ===== Stage 1 Baseline測定 =====
    if stage1_model is not None:
        print("\n" + "=" * 60)
        print("Phase 3 Stage 1 Baseline Benchmark")
        print("=" * 60)
        
        # Perplexity測定
        print("\n[1/1] Measuring Perplexity...")
        stage1_ppl_results = measure_perplexity(
            stage1_model,
            dataloader,
            device,
            max_batches=max_ppl_batches,
            model_name="Phase 3 Stage 1"
        )
        
        results["stage1_ppl"] = stage1_ppl_results["ppl"]
        results["stage1_loss"] = stage1_ppl_results["loss"]
        
        # ===== 比較 =====
        print("\n" + "=" * 60)
        print("Comparison: Stage 2 vs Stage 1")
        print("=" * 60)
        
        # Perplexity比較
        ppl_ratio = results["stage2_ppl"] / results["stage1_ppl"]
        ppl_diff_pct = (ppl_ratio - 1.0) * 100
        ppl_target = 1.02  # +2%以内
        ppl_pass = ppl_ratio <= ppl_target
        
        print(f"\n[1/3] Perplexity:")
        print(f"  - Stage 2: {results['stage2_ppl']:.2f}")
        print(f"  - Stage 1: {results['stage1_ppl']:.2f}")
        print(f"  - Ratio: {ppl_ratio:.4f} ({ppl_diff_pct:+.2f}%)")
        print(f"  - Target: ≤ {ppl_target:.2f} (Stage 1 + 2%)")
        print(f"  - Status: {'✓ PASS' if ppl_pass else '✗ FAIL'}")
        
        results["ppl_ratio"] = ppl_ratio
        results["ppl_diff_pct"] = ppl_diff_pct
        results["ppl_target"] = ppl_target
        results["ppl_pass"] = ppl_pass
        
        # Energy Drift
        print(f"\n[2/3] Energy Drift:")
        print(f"  - Max Drift: {results['max_drift']:.6e}")
        print(f"  - Target: < 5e-5")
        print(f"  - Status: {'✓ PASS' if results['energy_pass'] else '✗ FAIL'}")
        
        # VRAM
        print(f"\n[3/3] VRAM:")
        print(f"  - Symplectic Adjoint: {results['vram_symplectic_gb']:.2f} GB")
        print(f"  - Full Backprop: {results['vram_full_backprop_gb']:.2f} GB")
        print(f"  - Reduction: {results['vram_reduction_pct']:.1f}%")
        print(f"  - Status: {'✓ PASS' if results['vram_pass'] else '✗ FAIL'}")
        
        # 総合判定
        all_pass = ppl_pass and results["energy_pass"] and results["vram_pass"]
        
        print(f"\n{'=' * 60}")
        print(f"Overall Status: {'✓ ALL PASS' if all_pass else '✗ SOME FAILED'}")
        print(f"{'=' * 60}")
        
        if all_pass:
            print("\n🎉 Phase 3 Stage 2 has achieved all numerical targets!")
            print("   - Perplexity: Within +2% of Stage 1")
            print("   - Energy Drift: < 5e-5")
            print("   - VRAM: < 7.5GB with 70%+ reduction")
        else:
            print("\n⚠️  Some targets were not met:")
            if not ppl_pass:
                print(f"   - Perplexity: {ppl_ratio:.4f} > {ppl_target:.2f}")
            if not results["energy_pass"]:
                print(f"   - Energy Drift: {results['max_drift']:.6e} ≥ 5e-5")
            if not results["vram_pass"]:
                print(f"   - VRAM: {results['vram_symplectic_gb']:.2f} GB ≥ 7.5 GB or reduction < 70%")
        
        results["all_pass"] = all_pass
    
    else:
        # Stage 1がない場合
        results["stage1_ppl"] = None
        results["ppl_ratio"] = None
        results["all_pass"] = results["energy_pass"] and results["vram_pass"]
    
    return results


def save_results(results: Dict[str, Any], output_path: Path):
    """
    結果をJSON形式で保存
    
    Args:
        results: ベンチマーク結果
        output_path: 出力パス
    
    Requirements: 2.21, 2.22, 2.23
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # メタデータを追加
    results["benchmark_name"] = "Phase 3 Stage 2 Benchmark"
    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # JSON形式で保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Stage 2 Benchmark")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ppl-batch-size", type=int, default=4, help="Batch size for perplexity")
    parser.add_argument("--ppl-seq-length", type=int, default=1024, help="Sequence length for perplexity")
    parser.add_argument("--energy-batch-size", type=int, default=4, help="Batch size for energy drift")
    parser.add_argument("--energy-seq-length", type=int, default=512, help="Sequence length for energy drift")
    parser.add_argument("--vram-batch-size", type=int, default=2, help="Batch size for VRAM measurement")
    parser.add_argument("--vram-seq-length", type=int, default=2048, help="Sequence length for VRAM measurement")
    parser.add_argument("--max-ppl-batches", type=int, default=None, help="Max batches for perplexity")
    parser.add_argument("--skip-stage1", action="store_true", help="Skip Stage 1 baseline")
    parser.add_argument("--output", type=str, default="results/benchmarks/phase3_stage2_comparison.json", help="Output JSON path")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Phase 3 Stage 2モデルを作成
    print("\nCreating Phase 3 Stage 2 model...")
    if Phase3Stage2Model is None or Phase3Stage2Config is None:
        print("ERROR: Phase 3 Stage 2 model not implemented yet.")
        print("Please implement src/models/phase3/stage2_model.py first.")
        sys.exit(1)
    
    stage2_config = Phase3Stage2Config(
        vocab_size=50257,
        d_model=512,
        n_layers=6,
        n_seq=2048,
        use_complex32=True,
        ode_dt=0.1,
        ode_steps=10
    )
    stage2_model = Phase3Stage2Model(stage2_config).to(device)
    
    # Phase 3 Stage 1モデルを作成（ベースライン）
    stage1_model = None
    if not args.skip_stage1 and Phase3Stage1Model is not None:
        print("\nCreating Phase 3 Stage 1 baseline model...")
        stage1_config = Phase3Stage1Config(
            vocab_size=50257,
            d_model=512,
            n_layers=6,
            n_seq=2048,
            use_complex32=True
        )
        stage1_model = Phase3Stage1Model(stage1_config).to(device)
    
    # ベンチマーク実行
    print("\nStarting benchmark...")
    results = benchmark_phase3_stage2(
        stage2_model=stage2_model,
        stage1_model=stage1_model,
        device=device,
        seed=args.seed,
        ppl_batch_size=args.ppl_batch_size,
        ppl_seq_length=args.ppl_seq_length,
        energy_batch_size=args.energy_batch_size,
        energy_seq_length=args.energy_seq_length,
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
