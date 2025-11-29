#!/usr/bin/env python3
"""
Phase7 vs Phase8 ResNetBK比較ベンチマーク

同じResNetBKベースでの公平な比較を実施:
- スループット (tokens/sec)
- メモリ使用量 (GB)
- 精度 (Perplexity)
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import time
import json
from typing import Dict, Any

# Phase7とPhase8のモデルをインポート
from src.models.phase7.integrated_model import Phase7IntegratedModel, Phase7Config
from src.models.phase8.integrated_model import Phase8IntegratedModel
from src.models.phase8.config import Phase8Config


def measure_throughput(model: torch.nn.Module, batch_size: int, seq_len: int, 
                       vocab_size: int, device: str, num_iterations: int = 10) -> float:
    """スループット測定 (tokens/sec)"""
    model.eval()
    
    # ダミー入力（トークンID）
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # ウォームアップ
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input)
    
    # 測定
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    
    total_tokens = batch_size * seq_len * num_iterations
    throughput = total_tokens / elapsed_time
    
    return throughput


def measure_memory(model: torch.nn.Module, batch_size: int, seq_len: int,
                   vocab_size: int, device: str) -> Dict[str, float]:
    """メモリ使用量測定 (GB)"""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    model.eval()
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    with torch.no_grad():
        _ = model(dummy_input)
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
    current_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
    
    return {
        "peak_memory_gb": peak_memory,
        "current_memory_gb": current_memory
    }


def measure_perplexity(model: torch.nn.Module, batch_size: int, seq_len: int,
                       vocab_size: int, device: str) -> float:
    """Perplexity測定（簡易版）"""
    model.eval()
    
    # ダミーデータ生成
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    dummy_targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    with torch.no_grad():
        logits = model(dummy_input)
        
        # Cross entropy loss
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size),
            dummy_targets.reshape(-1)
        )
        
        perplexity = torch.exp(loss).item()
    
    return perplexity


def benchmark_phase7_vs_phase8(
    batch_size: int = 2,
    seq_len: int = 1024,
    d_model: int = 512,
    n_heads: int = 8,
    n_layers: int = 6,
    vocab_size: int = 32000,
    device: str = "cuda"
) -> Dict[str, Any]:
    """Phase7 vs Phase8の比較ベンチマーク"""
    
    print(f"=== Phase7 vs Phase8 ResNetBK比較ベンチマーク ===")
    print(f"Batch Size: {batch_size}, Seq Len: {seq_len}, D Model: {d_model}")
    print(f"N Heads: {n_heads}, N Layers: {n_layers}")
    print()
    
    results = {
        "config": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "vocab_size": vocab_size,
            "device": device
        },
        "phase7": {},
        "phase8": {},
        "comparison": {}
    }
    
    # Phase7モデル
    print("Phase7モデルを構築中...")
    phase7_config = Phase7Config(
        d_model=d_model,
        num_heads=n_heads,
        n_layers=n_layers,
        vocab_size=vocab_size,
        n_seq=seq_len,
        use_hybrid_attention=True,
        use_triton_kernel=False  # Triton利用不可のためPyTorchフォールバック
    )
    phase7_model = Phase7IntegratedModel(phase7_config).to(device)
    
    print("Phase7 スループット測定中...")
    phase7_throughput = measure_throughput(phase7_model, batch_size, seq_len, vocab_size, device)
    results["phase7"]["throughput_tokens_per_sec"] = phase7_throughput
    print(f"  Throughput: {phase7_throughput:.2f} tokens/sec")
    
    print("Phase7 メモリ測定中...")
    phase7_memory = measure_memory(phase7_model, batch_size, seq_len, vocab_size, device)
    results["phase7"]["memory"] = phase7_memory
    print(f"  Peak Memory: {phase7_memory['peak_memory_gb']:.3f} GB")
    
    print("Phase7 Perplexity測定中...")
    phase7_ppl = measure_perplexity(phase7_model, batch_size, seq_len, vocab_size, device)
    results["phase7"]["perplexity"] = phase7_ppl
    print(f"  Perplexity: {phase7_ppl:.2f}")
    print()
    
    # メモリクリア
    del phase7_model
    torch.cuda.empty_cache()
    
    # Phase8モデル
    print("Phase8モデルを構築中...")
    phase8_config = Phase8Config(
        d_model=d_model,
        num_heads=n_heads,
        n_layers=n_layers,
        vocab_size=vocab_size,
        n_seq=seq_len,
        use_hybrid_attention=True,
        use_triton_kernel=False,  # Triton利用不可のためPyTorchフォールバック
        # Phase8固有の機能
        use_bk_hyperbolic=True,
        use_ar_ssm_fusion=True,
        enable_entailment_cones=False,  # オプション機能はオフ
        enable_persistent_homology=False,
        enable_sheaf_attention=False
    )
    phase8_model = Phase8IntegratedModel(phase8_config).to(device)
    
    print("Phase8 スループット測定中...")
    phase8_throughput = measure_throughput(phase8_model, batch_size, seq_len, vocab_size, device)
    results["phase8"]["throughput_tokens_per_sec"] = phase8_throughput
    print(f"  Throughput: {phase8_throughput:.2f} tokens/sec")
    
    print("Phase8 メモリ測定中...")
    phase8_memory = measure_memory(phase8_model, batch_size, seq_len, vocab_size, device)
    results["phase8"]["memory"] = phase8_memory
    print(f"  Peak Memory: {phase8_memory['peak_memory_gb']:.3f} GB")
    
    print("Phase8 Perplexity測定中...")
    phase8_ppl = measure_perplexity(phase8_model, batch_size, seq_len, vocab_size, device)
    results["phase8"]["perplexity"] = phase8_ppl
    print(f"  Perplexity: {phase8_ppl:.2f}")
    print()
    
    # 比較
    print("=== 比較結果 ===")
    throughput_improvement = (phase8_throughput / phase7_throughput - 1) * 100
    memory_reduction = (1 - phase8_memory["peak_memory_gb"] / phase7_memory["peak_memory_gb"]) * 100
    ppl_change = (phase8_ppl / phase7_ppl - 1) * 100
    
    results["comparison"] = {
        "throughput_improvement_percent": throughput_improvement,
        "memory_reduction_percent": memory_reduction,
        "perplexity_change_percent": ppl_change
    }
    
    print(f"スループット改善: {throughput_improvement:+.2f}%")
    print(f"メモリ削減: {memory_reduction:+.2f}%")
    print(f"Perplexity変化: {ppl_change:+.2f}%")
    
    return results


def main():
    """メイン実行"""
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping GPU benchmark.")
        return
    
    # 出力ディレクトリ
    output_dir = Path("results/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ベンチマーク実行
    results = benchmark_phase7_vs_phase8(
        batch_size=2,
        seq_len=1024,
        d_model=512,
        n_heads=8,
        n_layers=6,
        vocab_size=32000,
        device="cuda"
    )
    
    # 結果保存
    output_file = output_dir / "phase7_vs_phase8_resnetbk_comparison.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n結果を保存しました: {output_file}")


if __name__ == "__main__":
    main()
