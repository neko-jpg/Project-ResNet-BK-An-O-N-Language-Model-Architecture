"""
Phase 1 Throughput Benchmark Script

tokens/secondを測定し、異なる設定でのスループットを比較します。
O(N)スケーリングを実証的に検証します。

Requirements: 6.3, 6.4, 9.6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.phase1.config import Phase1Config
from src.models.phase1.factory import create_phase1_model
from src.models.resnet_bk import LanguageModel as BaselineModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ThroughputMeasurement:
    """スループット測定結果"""
    
    model_name: str
    seq_length: int
    batch_size: int
    
    # Timing measurements
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float
    
    # Throughput metrics
    tokens_per_second: float
    samples_per_second: float
    
    # Memory usage
    peak_memory_mb: float
    
    # Configuration
    config: Dict


@dataclass
class ScalingAnalysis:
    """スケーリング解析結果"""
    
    model_name: str
    measurements: List[ThroughputMeasurement]
    
    # Scaling characteristics
    complexity_order: str  # "O(N)", "O(N log N)", "O(N^2)"
    scaling_coefficient: float  # 線形回帰の傾き
    r_squared: float  # 決定係数
    
    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "measurements": [asdict(m) for m in self.measurements],
            "complexity_order": self.complexity_order,
            "scaling_coefficient": self.scaling_coefficient,
            "r_squared": self.r_squared,
        }


class ThroughputBenchmark:
    """スループットベンチマーク実行クラス"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        model_name: str,
        vocab_size: int,
        config_dict: Dict,
        warmup_steps: int = 5,
        measure_steps: int = 20,
    ):
        self.model = model.to(DEVICE)
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.config_dict = config_dict
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps
    
    def measure_throughput(
        self,
        batch_size: int,
        seq_length: int,
    ) -> ThroughputMeasurement:
        """
        指定されたバッチサイズとシーケンス長でスループットを測定
        
        Args:
            batch_size: バッチサイズ
            seq_length: シーケンス長
        
        Returns:
            ThroughputMeasurement: 測定結果
        """
        self.model.train()
        
        # Warmup
        for _ in range(self.warmup_steps):
            input_ids = torch.randint(
                0, self.vocab_size, (batch_size, seq_length), device=DEVICE
            )
            outputs = self.model(input_ids)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = logits.sum()
            loss.backward()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        # Measure forward pass
        forward_times = []
        for _ in range(self.measure_steps):
            input_ids = torch.randint(
                0, self.vocab_size, (batch_size, seq_length), device=DEVICE
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            
            with torch.no_grad():
                outputs = self.model(input_ids)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            forward_times.append((end - start) * 1000)  # ms
        
        avg_forward_ms = np.mean(forward_times)
        
        # Measure backward pass
        backward_times = []
        for _ in range(self.measure_steps):
            input_ids = torch.randint(
                0, self.vocab_size, (batch_size, seq_length), device=DEVICE
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            
            outputs = self.model(input_ids)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = logits.sum()
            loss.backward()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            backward_times.append((end - start) * 1000)  # ms
        
        avg_backward_ms = np.mean(backward_times)
        avg_total_ms = avg_forward_ms + avg_backward_ms
        
        # Calculate throughput
        total_tokens = batch_size * seq_length
        tokens_per_second = (total_tokens / avg_total_ms) * 1000
        samples_per_second = (batch_size / avg_total_ms) * 1000
        
        # Get peak memory
        peak_memory_mb = 0.0
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        return ThroughputMeasurement(
            model_name=self.model_name,
            seq_length=seq_length,
            batch_size=batch_size,
            forward_time_ms=avg_forward_ms,
            backward_time_ms=avg_backward_ms,
            total_time_ms=avg_total_ms,
            tokens_per_second=tokens_per_second,
            samples_per_second=samples_per_second,
            peak_memory_mb=peak_memory_mb,
            config=self.config_dict,
        )


def analyze_scaling(
    measurements: List[ThroughputMeasurement],
    model_name: str,
) -> ScalingAnalysis:
    """
    スケーリング特性を解析
    
    Args:
        measurements: 測定結果のリスト
        model_name: モデル名
    
    Returns:
        ScalingAnalysis: スケーリング解析結果
    """
    # Extract data
    seq_lengths = np.array([m.seq_length for m in measurements])
    times = np.array([m.total_time_ms for m in measurements])
    
    # Test different complexity orders
    # O(N): time = a * N
    # O(N log N): time = a * N * log(N)
    # O(N^2): time = a * N^2
    
    # Linear regression for O(N)
    X_linear = seq_lengths.reshape(-1, 1)
    y = times
    
    # Calculate linear fit
    from sklearn.linear_model import LinearRegression
    
    model_linear = LinearRegression()
    model_linear.fit(X_linear, y)
    r2_linear = model_linear.score(X_linear, y)
    
    # Calculate O(N log N) fit
    X_nlogn = (seq_lengths * np.log(seq_lengths)).reshape(-1, 1)
    model_nlogn = LinearRegression()
    model_nlogn.fit(X_nlogn, y)
    r2_nlogn = model_nlogn.score(X_nlogn, y)
    
    # Calculate O(N^2) fit
    X_n2 = (seq_lengths ** 2).reshape(-1, 1)
    model_n2 = LinearRegression()
    model_n2.fit(X_n2, y)
    r2_n2 = model_n2.score(X_n2, y)
    
    # Determine best fit
    best_r2 = max(r2_linear, r2_nlogn, r2_n2)
    
    if best_r2 == r2_linear:
        complexity_order = "O(N)"
        scaling_coefficient = model_linear.coef_[0]
        r_squared = r2_linear
    elif best_r2 == r2_nlogn:
        complexity_order = "O(N log N)"
        scaling_coefficient = model_nlogn.coef_[0]
        r_squared = r2_nlogn
    else:
        complexity_order = "O(N^2)"
        scaling_coefficient = model_n2.coef_[0]
        r_squared = r2_n2
    
    return ScalingAnalysis(
        model_name=model_name,
        measurements=measurements,
        complexity_order=complexity_order,
        scaling_coefficient=scaling_coefficient,
        r_squared=r_squared,
    )


def plot_throughput_comparison(
    analyses: List[ScalingAnalysis],
    out_dir: Path,
):
    """
    スループット比較プロットを生成
    
    Args:
        analyses: スケーリング解析結果のリスト
        out_dir: 出力ディレクトリ
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Throughput vs Sequence Length
    ax1 = axes[0]
    for analysis in analyses:
        seq_lengths = [m.seq_length for m in analysis.measurements]
        throughputs = [m.tokens_per_second for m in analysis.measurements]
        ax1.plot(seq_lengths, throughputs, marker='o', label=analysis.model_name)
    
    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Throughput (tokens/sec)")
    ax1.set_title("Throughput vs Sequence Length")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # Plot 2: Time vs Sequence Length (log-log for scaling analysis)
    ax2 = axes[1]
    for analysis in analyses:
        seq_lengths = [m.seq_length for m in analysis.measurements]
        times = [m.total_time_ms for m in analysis.measurements]
        label = f"{analysis.model_name} ({analysis.complexity_order}, R²={analysis.r_squared:.3f})"
        ax2.plot(seq_lengths, times, marker='o', label=label)
    
    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Time (ms)")
    ax2.set_title("Scaling Analysis (log-log)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    out_path = out_dir / "phase1_throughput_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"📊 Plot saved to: {out_path}")
    plt.close()


def benchmark_phase1_throughput(
    seq_lengths: List[int] = [512, 1024, 2048, 4096],
    batch_size: int = 4,
    vocab_size: int = 50257,
    d_model: int = 512,
    n_layers: int = 8,
    compare_baseline: bool = True,
    phase1_config: Optional[Phase1Config] = None,
    warmup_steps: int = 5,
    measure_steps: int = 20,
) -> Dict[str, ScalingAnalysis]:
    """
    Phase 1モデルのスループットをベンチマーク
    
    Args:
        seq_lengths: テストするシーケンス長のリスト
        batch_size: バッチサイズ
        vocab_size: 語彙サイズ
        d_model: モデル次元
        n_layers: レイヤー数
        compare_baseline: ベースラインと比較するか
        phase1_config: Phase 1設定
        warmup_steps: ウォームアップステップ数
        measure_steps: 測定ステップ数
    
    Returns:
        Dict[str, ScalingAnalysis]: モデル名をキーとするスケーリング解析結果
    """
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Throughput benchmark requires GPU.")
        return {}
    
    print(f"🚀 Phase 1 Throughput Benchmark")
    print(f"   Sequence Lengths: {seq_lengths}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Vocab Size: {vocab_size}")
    print(f"   Model Dim: {d_model}")
    print(f"   Layers: {n_layers}")
    print()
    
    analyses = {}
    
    # Baseline model
    if compare_baseline:
        print("📊 Benchmarking baseline model...")
        baseline_config = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_layers": n_layers,
            "num_experts": 4,
            "top_k": 1,
            "dropout_p": 0.1,
        }
        
        baseline_measurements = []
        for seq_len in seq_lengths:
            print(f"   Testing seq_length={seq_len}...")
            
            baseline_config["n_seq"] = seq_len
            baseline_model = BaselineModel(**baseline_config)
            
            benchmark = ThroughputBenchmark(
                model=baseline_model,
                model_name="Baseline (ResNet-BK)",
                vocab_size=vocab_size,
                config_dict=baseline_config,
                warmup_steps=warmup_steps,
                measure_steps=measure_steps,
            )
            
            measurement = benchmark.measure_throughput(batch_size, seq_len)
            baseline_measurements.append(measurement)
            
            print(f"      Throughput: {measurement.tokens_per_second:.2f} tokens/sec")
            print(f"      Time: {measurement.total_time_ms:.2f} ms")
            
            del baseline_model
            torch.cuda.empty_cache()
        
        baseline_analysis = analyze_scaling(baseline_measurements, "Baseline (ResNet-BK)")
        analyses["baseline"] = baseline_analysis
        
        print(f"   Scaling: {baseline_analysis.complexity_order} (R²={baseline_analysis.r_squared:.3f})")
        print()
    
    # Phase 1 model
    print("📊 Benchmarking Phase 1 model...")
    
    if phase1_config is None:
        phase1_config = Phase1Config(
            ar_ssm_enabled=True,
            ar_ssm_max_rank=32,
            ar_ssm_min_rank=4,
            ar_ssm_use_fused_scan=True,
            htt_enabled=True,
            htt_rank=16,
            htt_num_cores=2,
            lns_enabled=False,
            use_gradient_checkpointing=True,
        )
    
    phase1_measurements = []
    for seq_len in seq_lengths:
        print(f"   Testing seq_length={seq_len}...")
        
        phase1_config_dict = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_seq": seq_len,
            "phase1_config": phase1_config,
        }
        
        phase1_model = create_phase1_model(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_seq=seq_len,
            config=phase1_config,
        )
        
        benchmark = ThroughputBenchmark(
            model=phase1_model,
            model_name="Phase 1 (AR-SSM + HTT)",
            vocab_size=vocab_size,
            config_dict=phase1_config_dict,
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
        )
        
        measurement = benchmark.measure_throughput(batch_size, seq_len)
        phase1_measurements.append(measurement)
        
        print(f"      Throughput: {measurement.tokens_per_second:.2f} tokens/sec")
        print(f"      Time: {measurement.total_time_ms:.2f} ms")
        
        del phase1_model
        torch.cuda.empty_cache()
    
    phase1_analysis = analyze_scaling(phase1_measurements, "Phase 1 (AR-SSM + HTT)")
    analyses["phase1"] = phase1_analysis
    
    print(f"   Scaling: {phase1_analysis.complexity_order} (R²={phase1_analysis.r_squared:.3f})")
    print()
    
    return analyses


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 Throughput Benchmark Script"
    )
    parser.add_argument(
        "--seq-lengths",
        nargs="+",
        type=int,
        default=[512, 1024, 2048, 4096],
        help="Sequence lengths to test",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for benchmark"
    )
    parser.add_argument(
        "--vocab-size", type=int, default=50257, help="Vocabulary size"
    )
    parser.add_argument(
        "--d-model", type=int, default=512, help="Model dimension"
    )
    parser.add_argument(
        "--n-layers", type=int, default=8, help="Number of layers"
    )
    parser.add_argument(
        "--no-baseline", action="store_true", help="Skip baseline comparison"
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=5, help="Warmup steps"
    )
    parser.add_argument(
        "--measure-steps", type=int, default=20, help="Measurement steps"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/benchmarks",
        help="Output directory",
    )
    parser.add_argument(
        "--save-plots", action="store_true", help="Save comparison plots"
    )
    args = parser.parse_args()
    
    # Create Phase 1 config
    phase1_config = Phase1Config(
        ar_ssm_enabled=True,
        ar_ssm_max_rank=32,
        ar_ssm_min_rank=4,
        ar_ssm_use_fused_scan=True,
        htt_enabled=True,
        htt_rank=16,
        htt_num_cores=2,
        lns_enabled=False,
        use_gradient_checkpointing=True,
    )
    
    # Run benchmark
    analyses = benchmark_phase1_throughput(
        seq_lengths=args.seq_lengths,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        compare_baseline=not args.no_baseline,
        phase1_config=phase1_config,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
    )
    
    # Save results
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results_dict = {
        name: analysis.to_dict() for name, analysis in analyses.items()
    }
    
    out_path = out_dir / "phase1_throughput_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"💾 Results saved to: {out_path}")
    
    # Generate plots
    if args.save_plots and analyses:
        plot_throughput_comparison(list(analyses.values()), out_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("THROUGHPUT BENCHMARK SUMMARY")
    print("=" * 70)
    
    for name, analysis in analyses.items():
        print(f"\n{analysis.model_name}:")
        print(f"  Complexity: {analysis.complexity_order}")
        print(f"  R² Score: {analysis.r_squared:.4f}")
        print(f"  Scaling Coefficient: {analysis.scaling_coefficient:.6f}")
        
        if analysis.measurements:
            avg_throughput = np.mean([m.tokens_per_second for m in analysis.measurements])
            print(f"  Avg Throughput: {avg_throughput:.2f} tokens/sec")
    
    # Verify O(N) scaling for Phase 1
    if "phase1" in analyses:
        phase1_analysis = analyses["phase1"]
        if phase1_analysis.complexity_order == "O(N)" and phase1_analysis.r_squared > 0.95:
            print("\n✅ Phase 1 model demonstrates O(N) scaling!")
        else:
            print(f"\n⚠️  Phase 1 scaling: {phase1_analysis.complexity_order} (R²={phase1_analysis.r_squared:.3f})")
    
    print("=" * 70)


if __name__ == "__main__":
    # Install sklearn if not available
    try:
        import sklearn
    except ImportError:
        print("⚠️  scikit-learn not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    
    main()
