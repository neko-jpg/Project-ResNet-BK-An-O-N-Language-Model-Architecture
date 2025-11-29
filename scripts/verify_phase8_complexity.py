#!/usr/bin/env python3
"""
Phase8 O(N)複雑度の検証

シーケンス長を変えて計算時間を測定し、O(N)スケーリングを確認
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import time
import json
from typing import Dict, Any, List
import numpy as np
from scipy import stats

from src.models.phase8.integrated_model import Phase8IntegratedModel
from src.models.phase8.config import Phase8Config


def measure_forward_time(
    model: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    d_model: int,
    device: str,
    num_iterations: int = 10
) -> float:
    """Forward pass時間を測定 (秒)"""
    
    model.eval()
    
    # ウォームアップ
    with torch.no_grad():
        dummy_input = torch.randn(batch_size, seq_len, d_model, device=device)
        for _ in range(3):
            _ = model(dummy_input)
    
    # 測定
    times = []
    for _ in range(num_iterations):
        dummy_input = torch.randn(batch_size, seq_len, d_model, device=device)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
    
    # 平均時間
    return np.mean(times)


def verify_complexity_scaling(
    seq_lengths: List[int],
    batch_size: int = 2,
    d_model: int = 512,
    n_heads: int = 8,
    n_layers: int = 6,
    vocab_size: int = 32000,
    device: str = "cuda"
) -> Dict[str, Any]:
    """複雑度スケーリングの検証"""
    
    print(f"=== Phase8 O(N)複雑度の検証 ===")
    print(f"Sequence Lengths: {seq_lengths}")
    print(f"Batch Size: {batch_size}, D Model: {d_model}")
    print()
    
    results = {
        "config": {
            "seq_lengths": seq_lengths,
            "batch_size": batch_size,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "vocab_size": vocab_size,
            "device": device
        },
        "measurements": [],
        "complexity_analysis": {}
    }
    
    # Phase8モデル
    print("Phase8モデルを構築中...")
    config = Phase8Config(
        d_model=d_model,
        num_heads=n_heads,
        num_layers=n_layers,
        vocab_size=vocab_size,
        max_seq_len=max(seq_lengths) * 2,
        use_bk_core=True,
        use_hybrid_attention=True,
        use_ar_ssm=True,
        use_bk_hyperbolic=True,
        use_ar_ssm_fusion=True
    )
    model = Phase8IntegratedModel(config).to(device)
    
    # 各シーケンス長で測定
    for seq_len in seq_lengths:
        print(f"Seq Len {seq_len} を測定中...")
        
        forward_time = measure_forward_time(model, batch_size, seq_len, d_model, device)
        time_per_token = forward_time / (batch_size * seq_len)
        
        measurement = {
            "seq_len": seq_len,
            "forward_time_sec": forward_time,
            "time_per_token_ms": time_per_token * 1000
        }
        results["measurements"].append(measurement)
        
        print(f"  Forward Time: {forward_time:.4f} sec")
        print(f"  Time per Token: {time_per_token * 1000:.4f} ms")
    
    # 線形回帰で複雑度を推定
    seq_lens = np.array([m["seq_len"] for m in results["measurements"]])
    times = np.array([m["forward_time_sec"] for m in results["measurements"]])
    
    # O(N)仮説: time = a * N + b
    slope_linear, intercept_linear, r_value_linear, _, _ = stats.linregress(seq_lens, times)
    
    # O(N^2)仮説: time = a * N^2 + b
    seq_lens_squared = seq_lens ** 2
    slope_quadratic, intercept_quadratic, r_value_quadratic, _, _ = stats.linregress(seq_lens_squared, times)
    
    results["complexity_analysis"] = {
        "linear_fit": {
            "slope": float(slope_linear),
            "intercept": float(intercept_linear),
            "r_squared": float(r_value_linear ** 2)
        },
        "quadratic_fit": {
            "slope": float(slope_quadratic),
            "intercept": float(intercept_quadratic),
            "r_squared": float(r_value_quadratic ** 2)
        },
        "is_linear": bool(r_value_linear ** 2 > r_value_quadratic ** 2),
        "complexity_verdict": "O(N)" if r_value_linear ** 2 > r_value_quadratic ** 2 else "O(N^2)"
    }
    
    print()
    print("=== 複雑度解析 ===")
    print(f"Linear Fit R²: {r_value_linear ** 2:.4f}")
    print(f"Quadratic Fit R²: {r_value_quadratic ** 2:.4f}")
    print(f"Complexity Verdict: {results['complexity_analysis']['complexity_verdict']}")
    
    if results["complexity_analysis"]["is_linear"]:
        print("✅ O(N)複雑度が確認されました！")
    else:
        print("⚠️ O(N^2)の可能性があります。要確認。")
    
    return results


def main():
    """メイン実行"""
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping GPU benchmark.")
        return
    
    # 出力ディレクトリ
    output_dir = Path("results/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # シーケンス長のリスト
    seq_lengths = [256, 512, 1024, 2048, 4096]
    
    # ベンチマーク実行
    results = verify_complexity_scaling(
        seq_lengths=seq_lengths,
        batch_size=2,
        d_model=512,
        n_heads=8,
        n_layers=6,
        vocab_size=32000,
        device="cuda"
    )
    
    # 結果保存
    output_file = output_dir / "phase8_complexity_verification.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n結果を保存しました: {output_file}")


if __name__ == "__main__":
    main()
