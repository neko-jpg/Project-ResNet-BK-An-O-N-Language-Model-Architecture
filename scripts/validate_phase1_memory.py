"""
Phase 1 Memory Usage Validation Script

VRAMトラッキングを実装し、forward/backward passでのメモリ使用量を測定します。
8GB VRAMターゲットでの動作を検証し、詳細なメモリレポートを生成します。

Requirements: 5.3, 5.4, 9.3, 9.4
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.phase1.config import Phase1Config
from src.models.phase1.factory import create_phase1_model
from src.models.resnet_bk import LanguageModel as BaselineModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class MemorySnapshot:
    """単一のメモリスナップショット"""
    
    stage: str  # "initial", "after_forward", "after_backward", "peak"
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    timestamp_ms: float


@dataclass
class MemoryReport:
    """メモリ使用量の完全なレポート"""
    
    model_name: str
    config: Dict
    batch_size: int
    seq_length: int
    vocab_size: int
    d_model: int
    n_layers: int
    
    # Memory snapshots
    snapshots: List[MemorySnapshot]
    
    # Peak memory usage
    peak_forward_mb: float
    peak_backward_mb: float
    peak_total_mb: float
    
    # Memory efficiency metrics
    baseline_peak_mb: Optional[float] = None
    memory_reduction_percent: Optional[float] = None
    
    # Validation results
    passes_8gb_target: bool = False
    passes_10gb_target: bool = False
    
    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return asdict(self)
    
    def print_summary(self):
        """サマリーを出力"""
        print("\n" + "=" * 70)
        print(f"MEMORY VALIDATION REPORT: {self.model_name}")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Sequence Length: {self.seq_length}")
        print(f"  Vocab Size: {self.vocab_size}")
        print(f"  Model Dim: {self.d_model}")
        print(f"  Layers: {self.n_layers}")
        print()
        print(f"Peak Memory Usage:")
        print(f"  Forward Pass:  {self.peak_forward_mb:>8.2f} MB")
        print(f"  Backward Pass: {self.peak_backward_mb:>8.2f} MB")
        print(f"  Total Peak:    {self.peak_total_mb:>8.2f} MB")
        print()
        
        if self.baseline_peak_mb is not None:
            print(f"Comparison to Baseline:")
            print(f"  Baseline Peak: {self.baseline_peak_mb:>8.2f} MB")
            print(f"  Reduction:     {self.memory_reduction_percent:>8.2f}%")
            print()
        
        print(f"Target Validation:")
        target_8gb = "✅ PASS" if self.passes_8gb_target else "❌ FAIL"
        target_10gb = "✅ PASS" if self.passes_10gb_target else "❌ FAIL"
        print(f"  8GB Target (< 7.2GB):  {target_8gb}")
        print(f"  10GB Target (< 9.0GB): {target_10gb}")
        print("=" * 70)


class MemoryTracker:
    """VRAM使用量を追跡するユーティリティ"""
    
    def __init__(self):
        self.snapshots: List[MemorySnapshot] = []
        self.start_time = None
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    
    def snapshot(self, stage: str):
        """現在のメモリ状態をスナップショット"""
        if not torch.cuda.is_available():
            return
        
        if self.start_time is None:
            import time
            self.start_time = time.time()
        
        import time
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        timestamp = (time.time() - self.start_time) * 1000
        
        snapshot = MemorySnapshot(
            stage=stage,
            allocated_mb=allocated,
            reserved_mb=reserved,
            max_allocated_mb=max_allocated,
            timestamp_ms=timestamp,
        )
        self.snapshots.append(snapshot)
        
        return snapshot
    
    def get_peak_memory(self) -> float:
        """ピークメモリ使用量を取得 (MB)"""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    def reset(self):
        """メモリ統計をリセット"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()


def measure_model_memory(
    model: torch.nn.Module,
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    model_name: str,
    config_dict: Dict,
) -> MemoryReport:
    """
    モデルのメモリ使用量を測定
    
    Args:
        model: 測定対象のモデル
        batch_size: バッチサイズ
        seq_length: シーケンス長
        vocab_size: 語彙サイズ
        model_name: モデル名
        config_dict: 設定辞書
    
    Returns:
        MemoryReport: メモリレポート
    """
    tracker = MemoryTracker()
    model = model.to(DEVICE)
    model.train()
    
    # Initial snapshot
    tracker.snapshot("initial")
    
    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=DEVICE)
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=DEVICE)
    
    # Forward pass
    tracker.reset()
    tracker.snapshot("before_forward")
    
    outputs = model(input_ids)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
    
    tracker.snapshot("after_forward")
    peak_forward = tracker.get_peak_memory()
    
    # Backward pass
    tracker.reset()
    tracker.snapshot("before_backward")
    
    loss.backward()
    
    tracker.snapshot("after_backward")
    peak_backward = tracker.get_peak_memory()
    
    # Final snapshot
    tracker.snapshot("final")
    
    # Calculate peak total (forward + backward)
    tracker.reset()
    tracker.snapshot("before_full_pass")
    
    outputs = model(input_ids)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
    loss.backward()
    
    tracker.snapshot("after_full_pass")
    peak_total = tracker.get_peak_memory()
    
    # Create report
    report = MemoryReport(
        model_name=model_name,
        config=config_dict,
        batch_size=batch_size,
        seq_length=seq_length,
        vocab_size=vocab_size,
        d_model=config_dict.get("d_model", 256),
        n_layers=config_dict.get("n_layers", 6),
        snapshots=tracker.snapshots,
        peak_forward_mb=peak_forward,
        peak_backward_mb=peak_backward,
        peak_total_mb=peak_total,
        passes_8gb_target=(peak_total < 7200),  # 90% of 8GB
        passes_10gb_target=(peak_total < 9000),  # 90% of 10GB
    )
    
    return report


def validate_phase1_memory(
    batch_size: int = 4,
    seq_length: int = 2048,
    vocab_size: int = 50257,
    d_model: int = 512,
    n_layers: int = 8,
    compare_baseline: bool = True,
    phase1_config: Optional[Phase1Config] = None,
) -> Dict[str, MemoryReport]:
    """
    Phase 1モデルのメモリ使用量を検証
    
    Args:
        batch_size: バッチサイズ
        seq_length: シーケンス長
        vocab_size: 語彙サイズ
        d_model: モデル次元
        n_layers: レイヤー数
        compare_baseline: ベースラインと比較するか
        phase1_config: Phase 1設定（Noneの場合はデフォルト）
    
    Returns:
        Dict[str, MemoryReport]: モデル名をキーとするメモリレポート
    """
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Memory validation requires GPU.")
        return {}
    
    print(f"🔍 Phase 1 Memory Validation")
    print(f"   Batch Size: {batch_size}")
    print(f"   Sequence Length: {seq_length}")
    print(f"   Vocab Size: {vocab_size}")
    print(f"   Model Dim: {d_model}")
    print(f"   Layers: {n_layers}")
    print()
    
    reports = {}
    
    # Baseline model (if requested)
    if compare_baseline:
        print("📊 Measuring baseline model...")
        baseline_config = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_seq": seq_length,
            "num_experts": 4,
            "top_k": 1,
            "dropout_p": 0.1,
        }
        
        baseline_model = BaselineModel(**baseline_config)
        baseline_report = measure_model_memory(
            model=baseline_model,
            batch_size=batch_size,
            seq_length=seq_length,
            vocab_size=vocab_size,
            model_name="Baseline (ResNet-BK)",
            config_dict=baseline_config,
        )
        reports["baseline"] = baseline_report
        baseline_report.print_summary()
        
        # Clean up
        del baseline_model
        torch.cuda.empty_cache()
    
    # Phase 1 model
    print("\n📊 Measuring Phase 1 model...")
    
    if phase1_config is None:
        phase1_config = Phase1Config(
            ar_ssm_enabled=True,
            ar_ssm_max_rank=32,
            ar_ssm_min_rank=4,
            htt_enabled=True,
            htt_rank=16,
            htt_num_cores=2,
            lns_enabled=False,  # LNS is experimental
            use_gradient_checkpointing=True,
            checkpoint_ar_ssm=True,
        )
    
    phase1_config_dict = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_seq": seq_length,
        "phase1_config": phase1_config,
    }
    
    phase1_model = create_phase1_model(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_seq=seq_length,
        config=phase1_config,
    )
    
    phase1_report = measure_model_memory(
        model=phase1_model,
        batch_size=batch_size,
        seq_length=seq_length,
        vocab_size=vocab_size,
        model_name="Phase 1 (AR-SSM + HTT)",
        config_dict=phase1_config_dict,
    )
    reports["phase1"] = phase1_report
    
    # Calculate reduction if baseline available
    if compare_baseline and "baseline" in reports:
        baseline_peak = reports["baseline"].peak_total_mb
        phase1_report.baseline_peak_mb = baseline_peak
        phase1_report.memory_reduction_percent = (
            (baseline_peak - phase1_report.peak_total_mb) / baseline_peak * 100
        )
    
    phase1_report.print_summary()
    
    # Clean up
    del phase1_model
    torch.cuda.empty_cache()
    
    return reports


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 Memory Usage Validation Script"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for validation"
    )
    parser.add_argument(
        "--seq-length", type=int, default=2048, help="Sequence length for validation"
    )
    parser.add_argument(
        "--vocab-size", type=int, default=50257, help="Vocabulary size (GPT-2 default)"
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
        "--ar-ssm-max-rank", type=int, default=32, help="AR-SSM maximum rank"
    )
    parser.add_argument(
        "--htt-rank", type=int, default=16, help="HTT rank"
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/benchmarks",
        help="Output directory for results",
    )
    args = parser.parse_args()
    
    # Create Phase 1 config
    phase1_config = Phase1Config(
        ar_ssm_enabled=True,
        ar_ssm_max_rank=args.ar_ssm_max_rank,
        ar_ssm_min_rank=4,
        htt_enabled=True,
        htt_rank=args.htt_rank,
        htt_num_cores=2,
        lns_enabled=False,
        use_gradient_checkpointing=not args.no_gradient_checkpointing,
        checkpoint_ar_ssm=True,
    )
    
    # Run validation
    reports = validate_phase1_memory(
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        compare_baseline=not args.no_baseline,
        phase1_config=phase1_config,
    )
    
    # Save results
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results_dict = {
        name: report.to_dict() for name, report in reports.items()
    }
    
    out_path = out_dir / "phase1_memory_validation.json"
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n💾 Results saved to: {out_path}")
    
    # Print final summary
    if "phase1" in reports:
        phase1_report = reports["phase1"]
        print("\n" + "=" * 70)
        print("FINAL VALIDATION RESULT")
        print("=" * 70)
        
        if phase1_report.passes_8gb_target:
            print("✅ Phase 1 model PASSES 8GB VRAM target!")
        else:
            print("❌ Phase 1 model FAILS 8GB VRAM target.")
            print(f"   Peak: {phase1_report.peak_total_mb:.2f} MB (target: < 7200 MB)")
            print("   Suggestions:")
            print("   - Reduce batch size")
            print("   - Reduce sequence length")
            print("   - Reduce ar_ssm_max_rank")
            print("   - Enable gradient checkpointing")
        
        print("=" * 70)


if __name__ == "__main__":
    main()
