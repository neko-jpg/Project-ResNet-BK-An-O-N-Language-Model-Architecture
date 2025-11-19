"""
Phase 1 Performance Comparison Table Generator

既存のベンチマーク結果から比較テーブルを生成します。
VRAM、スループット、PPLを異なる設定で比較します。

Requirements: 12.4, 12.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_benchmark_results(results_dir: Path) -> Dict[str, Dict]:
    """
    ベンチマーク結果を読み込む
    
    Args:
        results_dir: 結果ディレクトリ
    
    Returns:
        Dict[str, Dict]: ベンチマーク結果の辞書
    """
    results = {}
    
    # Memory validation results
    memory_path = results_dir / "phase1_memory_validation.json"
    if memory_path.exists():
        with open(memory_path) as f:
            results["memory"] = json.load(f)
        print(f"✅ Loaded memory validation results")
    else:
        print(f"⚠️  Memory validation results not found: {memory_path}")
    
    # Throughput benchmark results
    throughput_path = results_dir / "phase1_throughput_benchmark.json"
    if throughput_path.exists():
        with open(throughput_path) as f:
            results["throughput"] = json.load(f)
        print(f"✅ Loaded throughput benchmark results")
    else:
        print(f"⚠️  Throughput benchmark results not found: {throughput_path}")
    
    # Perplexity validation results
    perplexity_path = results_dir / "phase1_perplexity_validation.json"
    if perplexity_path.exists():
        with open(perplexity_path) as f:
            results["perplexity"] = json.load(f)
        print(f"✅ Loaded perplexity validation results")
    else:
        print(f"⚠️  Perplexity validation results not found: {perplexity_path}")
    
    return results


def generate_memory_comparison_table(
    memory_results: Dict,
    hardware_name: str = "RTX 3080",
) -> pd.DataFrame:
    """
    メモリ使用量比較テーブルを生成
    
    Args:
        memory_results: メモリ検証結果
        hardware_name: ハードウェア名
    
    Returns:
        pd.DataFrame: 比較テーブル
    """
    rows = []
    
    for model_name, result in memory_results.items():
        row = {
            "Model": result["model_name"],
            "Hardware": hardware_name,
            "Peak VRAM (MB)": f"{result['peak_total_mb']:.2f}",
            "Forward (MB)": f"{result['peak_forward_mb']:.2f}",
            "Backward (MB)": f"{result['peak_backward_mb']:.2f}",
            "8GB Target": "✅" if result["passes_8gb_target"] else "❌",
            "10GB Target": "✅" if result["passes_10gb_target"] else "❌",
        }
        
        if result.get("memory_reduction_percent") is not None:
            row["Reduction vs Baseline"] = f"{result['memory_reduction_percent']:.1f}%"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def generate_throughput_comparison_table(
    throughput_results: Dict,
    hardware_name: str = "RTX 3080",
) -> pd.DataFrame:
    """
    スループット比較テーブルを生成
    
    Args:
        throughput_results: スループットベンチマーク結果
        hardware_name: ハードウェア名
    
    Returns:
        pd.DataFrame: 比較テーブル
    """
    rows = []
    
    for model_name, result in throughput_results.items():
        measurements = result["measurements"]
        
        for measurement in measurements:
            row = {
                "Model": measurement["model_name"],
                "Hardware": hardware_name,
                "Seq Length": measurement["seq_length"],
                "Batch Size": measurement["batch_size"],
                "Throughput (tokens/s)": f"{measurement['tokens_per_second']:.2f}",
                "Forward Time (ms)": f"{measurement['forward_time_ms']:.2f}",
                "Backward Time (ms)": f"{measurement['backward_time_ms']:.2f}",
                "Total Time (ms)": f"{measurement['total_time_ms']:.2f}",
                "Peak VRAM (MB)": f"{measurement['peak_memory_mb']:.2f}",
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def generate_scaling_comparison_table(
    throughput_results: Dict,
) -> pd.DataFrame:
    """
    スケーリング特性比較テーブルを生成
    
    Args:
        throughput_results: スループットベンチマーク結果
    
    Returns:
        pd.DataFrame: 比較テーブル
    """
    rows = []
    
    for model_name, result in throughput_results.items():
        row = {
            "Model": result["model_name"],
            "Complexity": result["complexity_order"],
            "R² Score": f"{result['r_squared']:.4f}",
            "Scaling Coefficient": f"{result['scaling_coefficient']:.6e}",
        }
        
        # Calculate average throughput
        measurements = result["measurements"]
        avg_throughput = sum(m["tokens_per_second"] for m in measurements) / len(measurements)
        row["Avg Throughput (tokens/s)"] = f"{avg_throughput:.2f}"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def generate_perplexity_comparison_table(
    perplexity_results: Dict,
) -> pd.DataFrame:
    """
    Perplexity比較テーブルを生成
    
    Args:
        perplexity_results: Perplexity検証結果
    
    Returns:
        pd.DataFrame: 比較テーブル
    """
    rows = []
    
    for model_name, result in perplexity_results.items():
        row = {
            "Model": result["model_name"],
            "Dataset": f"{result['dataset_name']} ({result['dataset_config']})",
            "Perplexity": f"{result['perplexity']:.4f}",
            "Bits per Byte": f"{result['bits_per_byte']:.4f}",
            "Avg Loss": f"{result['avg_loss']:.4f}",
            "Samples": result["num_samples"],
        }
        
        if result.get("ppl_degradation_percent") is not None:
            row["Degradation vs Baseline"] = f"{result['ppl_degradation_percent']:.2f}%"
            row["< 5% Threshold"] = "✅" if result["passes_5_percent_threshold"] else "❌"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def generate_configuration_comparison_table(
    memory_results: Dict,
    throughput_results: Dict,
    perplexity_results: Dict,
) -> pd.DataFrame:
    """
    設定別の総合比較テーブルを生成
    
    Args:
        memory_results: メモリ検証結果
        throughput_results: スループットベンチマーク結果
        perplexity_results: Perplexity検証結果
    
    Returns:
        pd.DataFrame: 比較テーブル
    """
    rows = []
    
    # Baseline
    if "baseline" in memory_results and "baseline" in perplexity_results:
        baseline_memory = memory_results["baseline"]
        baseline_ppl = perplexity_results["baseline"]
        
        # Get baseline throughput (average across all seq lengths)
        baseline_throughput = None
        if "baseline" in throughput_results:
            measurements = throughput_results["baseline"]["measurements"]
            baseline_throughput = sum(m["tokens_per_second"] for m in measurements) / len(measurements)
        
        row = {
            "Configuration": "Baseline (ResNet-BK)",
            "AR-SSM": "❌",
            "HTT": "❌",
            "LNS": "❌",
            "Peak VRAM (MB)": f"{baseline_memory['peak_total_mb']:.2f}",
            "Perplexity": f"{baseline_ppl['perplexity']:.4f}",
        }
        
        if baseline_throughput is not None:
            row["Avg Throughput (tokens/s)"] = f"{baseline_throughput:.2f}"
        
        rows.append(row)
    
    # Phase 1 configurations
    phase1_configs = [k for k in memory_results.keys() if k.startswith("phase1")]
    
    for config_name in phase1_configs:
        if config_name in memory_results and config_name in perplexity_results:
            memory = memory_results[config_name]
            ppl = perplexity_results[config_name]
            
            # Extract configuration details
            config = memory.get("config", {})
            phase1_config = config.get("phase1_config", {})
            
            ar_ssm_enabled = phase1_config.get("ar_ssm_enabled", False)
            htt_enabled = phase1_config.get("htt_enabled", False)
            lns_enabled = phase1_config.get("lns_enabled", False)
            
            ar_ssm_rank = phase1_config.get("ar_ssm_max_rank", "N/A")
            htt_rank = phase1_config.get("htt_rank", "N/A")
            
            # Get throughput
            throughput = None
            if config_name in throughput_results:
                measurements = throughput_results[config_name]["measurements"]
                throughput = sum(m["tokens_per_second"] for m in measurements) / len(measurements)
            
            row = {
                "Configuration": memory["model_name"],
                "AR-SSM": f"✅ (r={ar_ssm_rank})" if ar_ssm_enabled else "❌",
                "HTT": f"✅ (r={htt_rank})" if htt_enabled else "❌",
                "LNS": "✅" if lns_enabled else "❌",
                "Peak VRAM (MB)": f"{memory['peak_total_mb']:.2f}",
                "Perplexity": f"{ppl['perplexity']:.4f}",
            }
            
            if throughput is not None:
                row["Avg Throughput (tokens/s)"] = f"{throughput:.2f}"
            
            if memory.get("memory_reduction_percent") is not None:
                row["VRAM Reduction"] = f"{memory['memory_reduction_percent']:.1f}%"
            
            if ppl.get("ppl_degradation_percent") is not None:
                row["PPL Degradation"] = f"{ppl['ppl_degradation_percent']:.2f}%"
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def save_tables(
    tables: Dict[str, pd.DataFrame],
    out_dir: Path,
    formats: List[str] = ["markdown", "csv", "latex"],
):
    """
    テーブルを複数のフォーマットで保存
    
    Args:
        tables: テーブルの辞書
        out_dir: 出力ディレクトリ
        formats: 保存フォーマットのリスト
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for table_name, df in tables.items():
        print(f"\n📊 {table_name}:")
        print(df.to_string(index=False))
        print()
        
        # Save in requested formats
        for fmt in formats:
            if fmt == "markdown":
                out_path = out_dir / f"{table_name}.md"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(f"# {table_name.replace('_', ' ').title()}\n\n")
                    f.write(df.to_markdown(index=False))
                print(f"   Saved: {out_path}")
            
            elif fmt == "csv":
                out_path = out_dir / f"{table_name}.csv"
                df.to_csv(out_path, index=False)
                print(f"   Saved: {out_path}")
            
            elif fmt == "latex":
                out_path = out_dir / f"{table_name}.tex"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(df.to_latex(index=False, escape=False))
                print(f"   Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Phase 1 Performance Comparison Tables"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/benchmarks",
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/benchmarks/tables",
        help="Output directory for tables",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="RTX 3080",
        help="Hardware name for tables",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["markdown", "csv", "latex"],
        help="Output formats (markdown, csv, latex)",
    )
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    
    print("=" * 70)
    print("PHASE 1 PERFORMANCE COMPARISON TABLE GENERATOR")
    print("=" * 70)
    print(f"Results Directory: {results_dir}")
    print(f"Output Directory: {out_dir}")
    print(f"Hardware: {args.hardware}")
    print()
    
    # Load benchmark results
    results = load_benchmark_results(results_dir)
    
    if not results:
        print("\n❌ No benchmark results found. Please run benchmarks first:")
        print("   1. python scripts/validate_phase1_memory.py")
        print("   2. python scripts/benchmark_phase1_throughput.py")
        print("   3. python scripts/validate_phase1_perplexity.py")
        return
    
    print()
    
    # Generate tables
    tables = {}
    
    if "memory" in results:
        print("📊 Generating memory comparison table...")
        tables["memory_comparison"] = generate_memory_comparison_table(
            results["memory"],
            hardware_name=args.hardware,
        )
    
    if "throughput" in results:
        print("📊 Generating throughput comparison table...")
        tables["throughput_comparison"] = generate_throughput_comparison_table(
            results["throughput"],
            hardware_name=args.hardware,
        )
        
        print("📊 Generating scaling comparison table...")
        tables["scaling_comparison"] = generate_scaling_comparison_table(
            results["throughput"],
        )
    
    if "perplexity" in results:
        print("📊 Generating perplexity comparison table...")
        tables["perplexity_comparison"] = generate_perplexity_comparison_table(
            results["perplexity"],
        )
    
    # Generate comprehensive configuration comparison
    if "memory" in results and "perplexity" in results:
        print("📊 Generating configuration comparison table...")
        tables["configuration_comparison"] = generate_configuration_comparison_table(
            results.get("memory", {}),
            results.get("throughput", {}),
            results.get("perplexity", {}),
        )
    
    # Save tables
    if tables:
        print("\n" + "=" * 70)
        print("SAVING TABLES")
        print("=" * 70)
        save_tables(tables, out_dir, formats=args.formats)
        
        print("\n" + "=" * 70)
        print("✅ All tables generated successfully!")
        print("=" * 70)
        print(f"Output directory: {out_dir}")
        print(f"Generated {len(tables)} tables in {len(args.formats)} formats")
    else:
        print("\n❌ No tables generated. Check benchmark results.")


if __name__ == "__main__":
    # Install required packages if not available
    try:
        import pandas
        import tabulate  # Required for to_markdown()
    except ImportError:
        print("⚠️  Required packages not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "tabulate"])
    
    main()
