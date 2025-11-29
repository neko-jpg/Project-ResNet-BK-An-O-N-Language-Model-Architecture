#!/usr/bin/env python3
"""
BK-Core統合効果の検証

G_iiゲーティングの効果測定と物理情報の活用度を評価
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import json
from typing import Dict, Any
import numpy as np

from src.models.phase8.integrated_model import Phase8IntegratedModel
from src.models.phase8.config import Phase8Config


def measure_bk_core_gating_effect(
    model: Phase8IntegratedModel,
    batch_size: int,
    seq_len: int,
    d_model: int,
    device: str
) -> Dict[str, Any]:
    """BK-Core G_iiゲーティングの効果測定"""
    
    model.eval()
    
    # ダミー入力
    dummy_input = torch.randn(batch_size, seq_len, d_model, device=device)
    
    results = {
        "g_ii_statistics": {},
        "gating_effect": {},
        "physics_utilization": {}
    }
    
    with torch.no_grad():
        output = model(dummy_input)
        
        # 診断情報を取得
        if isinstance(output, dict) and "diagnostics" in output:
            diagnostics = output["diagnostics"]
            
            # BK-Core診断情報
            if "bk_core" in diagnostics:
                bk_diag = diagnostics["bk_core"]
                
                # G_ii統計
                if "g_ii_real_mean" in bk_diag:
                    results["g_ii_statistics"] = {
                        "g_ii_real_mean": float(bk_diag["g_ii_real_mean"]),
                        "g_ii_real_std": float(bk_diag.get("g_ii_real_std", 0.0)),
                        "g_ii_imag_mean": float(bk_diag.get("g_ii_imag_mean", 0.0)),
                        "g_ii_imag_std": float(bk_diag.get("g_ii_imag_std", 0.0))
                    }
                
                # 共鳴検出
                if "resonance_detected" in bk_diag:
                    results["gating_effect"]["resonance_detected"] = bool(bk_diag["resonance_detected"])
                    results["gating_effect"]["resonance_ratio"] = float(bk_diag.get("resonance_ratio", 0.0))
            
            # BK-Hyperbolic統合診断情報
            if "bk_hyperbolic" in diagnostics:
                bk_hyp_diag = diagnostics["bk_hyperbolic"]
                
                # ゲーティング統計
                if "gate_mean" in bk_hyp_diag:
                    results["gating_effect"]["gate_mean"] = float(bk_hyp_diag["gate_mean"])
                    results["gating_effect"]["gate_std"] = float(bk_hyp_diag.get("gate_std", 0.0))
                    results["gating_effect"]["gate_min"] = float(bk_hyp_diag.get("gate_min", 0.0))
                    results["gating_effect"]["gate_max"] = float(bk_hyp_diag.get("gate_max", 1.0))
                
                # 曲率調整
                if "curvature_adjusted" in bk_hyp_diag:
                    results["physics_utilization"]["curvature_adjusted"] = bool(bk_hyp_diag["curvature_adjusted"])
                    results["physics_utilization"]["curvature_adjustment_count"] = int(bk_hyp_diag.get("curvature_adjustment_count", 0))
    
    return results


def measure_physics_information_utilization(
    model: Phase8IntegratedModel,
    batch_size: int,
    seq_len: int,
    d_model: int,
    device: str,
    num_samples: int = 10
) -> Dict[str, Any]:
    """物理情報の活用度を評価"""
    
    model.eval()
    
    g_ii_values = []
    gate_values = []
    resonance_counts = 0
    curvature_adjustment_counts = 0
    
    for _ in range(num_samples):
        dummy_input = torch.randn(batch_size, seq_len, d_model, device=device)
        
        with torch.no_grad():
            output = model(dummy_input)
            
            if isinstance(output, dict) and "diagnostics" in output:
                diagnostics = output["diagnostics"]
                
                # G_ii値を収集
                if "bk_core" in diagnostics and "g_ii_real_mean" in diagnostics["bk_core"]:
                    g_ii_values.append(float(diagnostics["bk_core"]["g_ii_real_mean"]))
                
                # ゲート値を収集
                if "bk_hyperbolic" in diagnostics and "gate_mean" in diagnostics["bk_hyperbolic"]:
                    gate_values.append(float(diagnostics["bk_hyperbolic"]["gate_mean"]))
                
                # 共鳴検出カウント
                if "bk_core" in diagnostics and diagnostics["bk_core"].get("resonance_detected", False):
                    resonance_counts += 1
                
                # 曲率調整カウント
                if "bk_hyperbolic" in diagnostics and diagnostics["bk_hyperbolic"].get("curvature_adjusted", False):
                    curvature_adjustment_counts += 1
    
    results = {
        "g_ii_distribution": {
            "mean": float(np.mean(g_ii_values)) if g_ii_values else 0.0,
            "std": float(np.std(g_ii_values)) if g_ii_values else 0.0,
            "min": float(np.min(g_ii_values)) if g_ii_values else 0.0,
            "max": float(np.max(g_ii_values)) if g_ii_values else 0.0
        },
        "gate_distribution": {
            "mean": float(np.mean(gate_values)) if gate_values else 0.0,
            "std": float(np.std(gate_values)) if gate_values else 0.0,
            "min": float(np.min(gate_values)) if gate_values else 0.0,
            "max": float(np.max(gate_values)) if gate_values else 0.0
        },
        "resonance_frequency": resonance_counts / num_samples,
        "curvature_adjustment_frequency": curvature_adjustment_counts / num_samples,
        "physics_utilization_score": (resonance_counts + curvature_adjustment_counts) / (2 * num_samples)
    }
    
    return results


def benchmark_bk_core_integration(
    batch_size: int = 2,
    seq_len: int = 1024,
    d_model: int = 512,
    n_heads: int = 8,
    n_layers: int = 6,
    vocab_size: int = 32000,
    device: str = "cuda"
) -> Dict[str, Any]:
    """BK-Core統合効果のベンチマーク"""
    
    print(f"=== BK-Core統合効果の検証 ===")
    print(f"Batch Size: {batch_size}, Seq Len: {seq_len}, D Model: {d_model}")
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
        }
    }
    
    # Phase8モデル（BK-Core統合あり）
    print("Phase8モデル（BK-Core統合あり）を構築中...")
    config = Phase8Config(
        d_model=d_model,
        num_heads=n_heads,
        num_layers=n_layers,
        vocab_size=vocab_size,
        max_seq_len=seq_len * 2,
        use_bk_core=True,
        use_hybrid_attention=True,
        use_ar_ssm=True,
        use_bk_hyperbolic=True,
        use_ar_ssm_fusion=True
    )
    model = Phase8IntegratedModel(config).to(device)
    
    print("BK-Core G_iiゲーティング効果を測定中...")
    gating_results = measure_bk_core_gating_effect(model, batch_size, seq_len, d_model, device)
    results["gating_effect"] = gating_results
    
    print(f"  G_ii Real Mean: {gating_results['g_ii_statistics'].get('g_ii_real_mean', 'N/A')}")
    print(f"  Gate Mean: {gating_results['gating_effect'].get('gate_mean', 'N/A')}")
    print(f"  Resonance Detected: {gating_results['gating_effect'].get('resonance_detected', 'N/A')}")
    print()
    
    print("物理情報の活用度を評価中...")
    utilization_results = measure_physics_information_utilization(
        model, batch_size, seq_len, d_model, device, num_samples=10
    )
    results["physics_utilization"] = utilization_results
    
    print(f"  G_ii Distribution Mean: {utilization_results['g_ii_distribution']['mean']:.4f}")
    print(f"  Gate Distribution Mean: {utilization_results['gate_distribution']['mean']:.4f}")
    print(f"  Resonance Frequency: {utilization_results['resonance_frequency']:.2%}")
    print(f"  Curvature Adjustment Frequency: {utilization_results['curvature_adjustment_frequency']:.2%}")
    print(f"  Physics Utilization Score: {utilization_results['physics_utilization_score']:.2%}")
    
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
    results = benchmark_bk_core_integration(
        batch_size=2,
        seq_len=1024,
        d_model=512,
        n_heads=8,
        n_layers=6,
        vocab_size=32000,
        device="cuda"
    )
    
    # 結果保存
    output_file = output_dir / "phase8_bk_core_integration_effect.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n結果を保存しました: {output_file}")


if __name__ == "__main__":
    main()
