"""
HTT Embedding Runtime VRAM Verification Script

Phase 1最終関門: HTT Embeddingの実行時VRAMが削減されていることを証明します。

このスクリプトは以下を測定します:
1. 標準nn.Embeddingの実行時VRAM
2. HTT Embeddingの実行時VRAM
3. Forward/Backward pass中のピークVRAM
4. 削減率の計算

目標: 90%以上のVRAM削減を実証

Author: Project MUSE Team
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.phase1.htt_embedding import HolographicTTEmbedding
from src.models.phase1.config import Phase1Config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EmbeddingMemoryReport:
    """Embedding層のメモリレポート"""
    
    embedding_type: str  # "Standard" or "HTT"
    vocab_size: int
    d_model: int
    batch_size: int
    seq_length: int
    
    # Parameter memory (disk/storage)
    param_count: int
    param_memory_mb: float
    
    # Runtime VRAM (execution)
    initial_vram_mb: float
    after_forward_vram_mb: float
    after_backward_vram_mb: float
    peak_vram_mb: float
    
    # Comparison
    baseline_peak_vram_mb: Optional[float] = None
    vram_reduction_percent: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print(f"EMBEDDING MEMORY REPORT: {self.embedding_type}")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  Vocab Size: {self.vocab_size:,}")
        print(f"  Model Dim: {self.d_model}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Sequence Length: {self.seq_length}")
        print()
        print(f"Parameter Memory (Storage):")
        print(f"  Parameter Count: {self.param_count:,}")
        print(f"  Memory: {self.param_memory_mb:.2f} MB")
        print()
        print(f"Runtime VRAM (Execution):")
        print(f"  Initial: {self.initial_vram_mb:.2f} MB")
        print(f"  After Forward: {self.after_forward_vram_mb:.2f} MB")
        print(f"  After Backward: {self.after_backward_vram_mb:.2f} MB")
        print(f"  Peak: {self.peak_vram_mb:.2f} MB")
        print()
        
        if self.baseline_peak_vram_mb is not None:
            print(f"Comparison to Baseline:")
            print(f"  Baseline Peak: {self.baseline_peak_vram_mb:.2f} MB")
            print(f"  VRAM Reduction: {self.vram_reduction_percent:.2f}%")
            
            if self.vram_reduction_percent >= 90:
                print(f"  Status: ✅ PASS (≥90% reduction)")
            elif self.vram_reduction_percent >= 50:
                print(f"  Status: ⚠️  PARTIAL (50-90% reduction)")
            else:
                print(f"  Status: ❌ FAIL (<50% reduction)")
        
        print("=" * 70)


def measure_embedding_vram(
    embedding: nn.Module,
    vocab_size: int,
    d_model: int,
    batch_size: int,
    seq_length: int,
    embedding_type: str,
) -> EmbeddingMemoryReport:
    """
    Embedding層の実行時VRAMを測定
    
    Args:
        embedding: 測定対象のEmbedding層
        vocab_size: 語彙サイズ
        d_model: 出力次元
        batch_size: バッチサイズ
        seq_length: シーケンス長
        embedding_type: "Standard" or "HTT"
    
    Returns:
        EmbeddingMemoryReport
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for VRAM measurement")
    
    # Move to GPU
    embedding = embedding.to(DEVICE)
    embedding.train()
    
    # Reset VRAM stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Count parameters
    param_count = sum(p.numel() for p in embedding.parameters())
    param_memory_mb = sum(
        p.numel() * p.element_size() for p in embedding.parameters()
    ) / (1024 ** 2)
    
    # Initial VRAM
    initial_vram = torch.cuda.memory_allocated() / (1024 ** 2)
    
    # Create input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=DEVICE)
    
    # Forward pass
    torch.cuda.reset_peak_memory_stats()
    output = embedding(input_ids)
    after_forward_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    # Backward pass
    torch.cuda.reset_peak_memory_stats()
    loss = output.sum()
    loss.backward()
    after_backward_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    # Peak VRAM (full forward + backward)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=DEVICE)
    output = embedding(input_ids)
    loss = output.sum()
    loss.backward()
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    return EmbeddingMemoryReport(
        embedding_type=embedding_type,
        vocab_size=vocab_size,
        d_model=d_model,
        batch_size=batch_size,
        seq_length=seq_length,
        param_count=param_count,
        param_memory_mb=param_memory_mb,
        initial_vram_mb=initial_vram,
        after_forward_vram_mb=after_forward_vram,
        after_backward_vram_mb=after_backward_vram,
        peak_vram_mb=peak_vram,
    )


def verify_htt_runtime_memory(
    vocab_size: int = 50257,
    d_model: int = 1024,
    batch_size: int = 4,
    seq_length: int = 2048,
    htt_rank: int = 16,
) -> Dict[str, EmbeddingMemoryReport]:
    """
    HTT Embeddingの実行時VRAMを検証
    
    Args:
        vocab_size: 語彙サイズ
        d_model: 出力次元
        batch_size: バッチサイズ
        seq_length: シーケンス長
        htt_rank: HTTランク
    
    Returns:
        Dict[str, EmbeddingMemoryReport]: "standard" と "htt" のレポート
    """
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. This test requires GPU.")
        return {}
    
    print(f"🔍 HTT Embedding Runtime VRAM Verification")
    print(f"   Vocab Size: {vocab_size:,}")
    print(f"   Model Dim: {d_model}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Sequence Length: {seq_length}")
    print(f"   HTT Rank: {htt_rank}")
    print()
    
    reports = {}
    
    # 1. Measure Standard Embedding
    print("📊 Measuring Standard nn.Embedding...")
    standard_embedding = nn.Embedding(vocab_size, d_model)
    standard_report = measure_embedding_vram(
        embedding=standard_embedding,
        vocab_size=vocab_size,
        d_model=d_model,
        batch_size=batch_size,
        seq_length=seq_length,
        embedding_type="Standard nn.Embedding",
    )
    reports["standard"] = standard_report
    standard_report.print_summary()
    
    # Clean up
    del standard_embedding
    torch.cuda.empty_cache()
    
    # 2. Measure HTT Embedding
    print("\n📊 Measuring HTT Embedding...")
    htt_embedding = HolographicTTEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        rank=htt_rank,
        num_cores=2,
        phase_encoding=True,
    )
    htt_report = measure_embedding_vram(
        embedding=htt_embedding,
        vocab_size=vocab_size,
        d_model=d_model,
        batch_size=batch_size,
        seq_length=seq_length,
        embedding_type="HTT Embedding",
    )
    reports["htt"] = htt_report
    
    # Calculate reduction
    htt_report.baseline_peak_vram_mb = standard_report.peak_vram_mb
    htt_report.vram_reduction_percent = (
        (standard_report.peak_vram_mb - htt_report.peak_vram_mb) 
        / standard_report.peak_vram_mb * 100
    )
    
    htt_report.print_summary()
    
    # Clean up
    del htt_embedding
    torch.cuda.empty_cache()
    
    return reports


def main():
    parser = argparse.ArgumentParser(
        description="HTT Embedding Runtime VRAM Verification"
    )
    parser.add_argument(
        "--vocab-size", type=int, default=50257, help="Vocabulary size"
    )
    parser.add_argument(
        "--d-model", type=int, default=1024, help="Model dimension"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size"
    )
    parser.add_argument(
        "--seq-length", type=int, default=2048, help="Sequence length"
    )
    parser.add_argument(
        "--htt-rank", type=int, default=16, help="HTT rank"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/benchmarks",
        help="Output directory",
    )
    args = parser.parse_args()
    
    # Run verification
    reports = verify_htt_runtime_memory(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        htt_rank=args.htt_rank,
    )
    
    # Save results
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results_dict = {
        name: report.to_dict() for name, report in reports.items()
    }
    
    out_path = out_dir / "htt_runtime_vram_verification.json"
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n💾 Results saved to: {out_path}")
    
    # Print final verdict
    if "htt" in reports:
        htt_report = reports["htt"]
        print("\n" + "=" * 70)
        print("FINAL VERDICT: HTT Embedding Runtime VRAM")
        print("=" * 70)
        
        if htt_report.vram_reduction_percent >= 90:
            print("✅ SUCCESS: HTT Embedding achieves ≥90% runtime VRAM reduction!")
            print(f"   Reduction: {htt_report.vram_reduction_percent:.2f}%")
            print("   Phase 1 Goal: ACHIEVED")
        elif htt_report.vram_reduction_percent >= 50:
            print("⚠️  PARTIAL: HTT Embedding achieves 50-90% runtime VRAM reduction.")
            print(f"   Reduction: {htt_report.vram_reduction_percent:.2f}%")
            print("   Phase 1 Goal: PARTIALLY ACHIEVED")
            print("   Recommendation: Optimize gather operations or use Triton kernel")
        else:
            print("❌ FAIL: HTT Embedding does not achieve sufficient VRAM reduction.")
            print(f"   Reduction: {htt_report.vram_reduction_percent:.2f}%")
            print("   Phase 1 Goal: NOT ACHIEVED")
            print("   Action Required: Implement memory-efficient TT contraction kernel")
        
        print("=" * 70)


if __name__ == "__main__":
    main()
