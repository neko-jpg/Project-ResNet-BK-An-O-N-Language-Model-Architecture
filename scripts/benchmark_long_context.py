#!/usr/bin/env python3
"""
Phase 8 Long Context Benchmark Script

タスク30.6: ロングコンテキストベンチマーク
- 2048, 4096, 8192, 16384トークンでのPerplexity測定
- 各長さでのメモリ使用量
- 目標: RTX 3080 (10GB)で16384トークンをサポート

Requirements: 30.5
"""

import json
import gc
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Phase 8モジュールのインポート
try:
    from src.models.phase8 import (
        BlockWiseDistanceComputation,
        HyperbolicSSM,
        TangentSpaceLinearAttention,
    )
    PHASE8_AVAILABLE = True
except ImportError:
    PHASE8_AVAILABLE = False

# Phase 7モジュールのインポート
try:
    from src.models.phase7 import HyperbolicAttention
    PHASE7_AVAILABLE = True
except ImportError:
    PHASE7_AVAILABLE = False


def get_device() -> torch.device:
    """利用可能なデバイスを取得"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_gpu_memory_info() -> Dict[str, float]:
    """GPU メモリ情報を取得"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    return {
        "total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
        "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        "free_gb": (torch.cuda.get_device_properties(0).total_memory - 
                   torch.cuda.memory_allocated()) / (1024**3),
    }


def clear_memory(device: torch.device):
    """メモリをクリア"""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


class SimpleLM(nn.Module):
    """シンプルな言語モデル（ベンチマーク用）"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        attention_module: nn.Module,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            attention_module for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
            if isinstance(x, tuple):
                x = x[0]
        x = self.ln(x)
        return self.lm_head(x)


def compute_perplexity(
    model: nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
) -> float:
    """Perplexityを計算"""
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        # シフトしてロス計算
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean'
        )
        
        perplexity = torch.exp(loss).item()
    
    return perplexity


def measure_long_context(
    model: nn.Module,
    seq_len: int,
    vocab_size: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    ロングコンテキストでのパフォーマンスを測定
    
    Returns:
        Dict with perplexity, memory_mb, success
    """
    model = model.to(device)
    clear_memory(device)
    
    # ランダムな入力を生成
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    try:
        # メモリ測定開始
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        
        # Perplexity計算
        perplexity = compute_perplexity(model, input_ids, device)
        
        # メモリ使用量
        peak_memory_mb = 0.0
        if device.type == "cuda":
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        return {
            "success": True,
            "perplexity": perplexity,
            "peak_memory_mb": peak_memory_mb,
            "peak_memory_gb": peak_memory_mb / 1024,
            "seq_len": seq_len,
        }
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {
                "success": False,
                "error": "OOM",
                "message": str(e),
                "seq_len": seq_len,
            }
        raise


class LongContextBenchmark:
    """ロングコンテキストベンチマーク"""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device or get_device()
        self.results: Dict[str, Any] = {}
    
    def create_phase8_attention(self) -> nn.Module:
        """Phase 8アテンションモジュールを作成"""
        if not PHASE8_AVAILABLE:
            return nn.MultiheadAttention(
                self.d_model, self.num_heads, batch_first=True
            )
        
        try:
            return BlockWiseDistanceComputation(
                d_model=self.d_model,
                num_heads=self.num_heads,
                block_size=128,
                curvature=1.0,
            )
        except Exception:
            return nn.MultiheadAttention(
                self.d_model, self.num_heads, batch_first=True
            )
    
    def create_phase7_attention(self) -> nn.Module:
        """Phase 7アテンションモジュールを作成"""
        if not PHASE7_AVAILABLE:
            return nn.MultiheadAttention(
                self.d_model, self.num_heads, batch_first=True
            )
        
        try:
            return HyperbolicAttention(
                d_model=self.d_model,
                num_heads=self.num_heads,
                curvature=1.0,
            )
        except Exception:
            return nn.MultiheadAttention(
                self.d_model, self.num_heads, batch_first=True
            )
    
    def run_benchmark(
        self,
        seq_lengths: List[int] = [2048, 4096, 8192, 16384],
    ) -> Dict[str, Any]:
        """ベンチマークを実行"""
        print(f"\n{'='*60}")
        print("Phase 8 Long Context Benchmark")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Vocab size: {self.vocab_size}")
        print(f"Model dim: {self.d_model}")
        print(f"Num heads: {self.num_heads}")
        print(f"Num layers: {self.num_layers}")
        print(f"Batch size: {self.batch_size}")
        
        gpu_info = get_gpu_memory_info()
        print(f"GPU Memory: {gpu_info.get('total_gb', 'N/A'):.1f} GB total")
        print(f"{'='*60}\n")
        
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "device": str(self.device),
                "gpu_info": gpu_info,
                "vocab_size": self.vocab_size,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "batch_size": self.batch_size,
            },
            "phase8": {},
            "phase7_baseline": {},
            "comparison": {},
        }
        
        # RTX 3080 (10GB) での目標
        target_max_seq = 16384
        target_memory_gb = 10.0
        
        for seq_len in seq_lengths:
            print(f"\nSequence length: {seq_len}")
            print("-" * 40)
            
            # Phase 8テスト
            try:
                phase8_attn = self.create_phase8_attention()
                phase8_model = SimpleLM(
                    self.vocab_size, self.d_model, self.num_heads,
                    self.num_layers, phase8_attn
                )
                
                phase8_result = measure_long_context(
                    phase8_model, seq_len, self.vocab_size,
                    self.batch_size, self.device
                )
                results["phase8"][str(seq_len)] = phase8_result
                
                if phase8_result["success"]:
                    print(f"  Phase 8: PPL={phase8_result['perplexity']:.2f}, "
                          f"Memory={phase8_result['peak_memory_gb']:.2f} GB")
                else:
                    print(f"  Phase 8: {phase8_result['error']}")
                
                del phase8_model, phase8_attn
            except Exception as e:
                print(f"  Phase 8: Error - {e}")
                results["phase8"][str(seq_len)] = {"error": str(e)}
            
            clear_memory(self.device)
            
            # Phase 7テスト
            try:
                phase7_attn = self.create_phase7_attention()
                phase7_model = SimpleLM(
                    self.vocab_size, self.d_model, self.num_heads,
                    self.num_layers, phase7_attn
                )
                
                phase7_result = measure_long_context(
                    phase7_model, seq_len, self.vocab_size,
                    self.batch_size, self.device
                )
                results["phase7_baseline"][str(seq_len)] = phase7_result
                
                if phase7_result["success"]:
                    print(f"  Phase 7: PPL={phase7_result['perplexity']:.2f}, "
                          f"Memory={phase7_result['peak_memory_gb']:.2f} GB")
                else:
                    print(f"  Phase 7: {phase7_result['error']}")
                
                del phase7_model, phase7_attn
            except Exception as e:
                print(f"  Phase 7: Error - {e}")
                results["phase7_baseline"][str(seq_len)] = {"error": str(e)}
            
            clear_memory(self.device)
            
            # 比較
            p8 = results["phase8"].get(str(seq_len), {})
            p7 = results["phase7_baseline"].get(str(seq_len), {})
            
            if p8.get("success") and p7.get("success"):
                memory_reduction = (1 - p8["peak_memory_gb"] / p7["peak_memory_gb"]) * 100
                results["comparison"][str(seq_len)] = {
                    "phase8_memory_gb": p8["peak_memory_gb"],
                    "phase7_memory_gb": p7["peak_memory_gb"],
                    "memory_reduction_percent": memory_reduction,
                    "phase8_fits_10gb": p8["peak_memory_gb"] < target_memory_gb,
                }
                print(f"  Memory reduction: {memory_reduction:.1f}%")
        
        # サマリー
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        
        # 最大サポートシーケンス長
        max_supported_seq = 0
        for seq_len in seq_lengths:
            p8 = results["phase8"].get(str(seq_len), {})
            if p8.get("success") and p8.get("peak_memory_gb", float('inf')) < target_memory_gb:
                max_supported_seq = seq_len
        
        results["summary"] = {
            "max_supported_seq_10gb": max_supported_seq,
            "target_met": max_supported_seq >= target_max_seq,
            "target_seq": target_max_seq,
            "target_memory_gb": target_memory_gb,
        }
        
        print(f"Max supported seq (10GB): {max_supported_seq}")
        print(f"Target ({target_max_seq}): {'MET' if max_supported_seq >= target_max_seq else 'NOT MET'}")
        
        self.results = results
        return results
    
    def save_results(self, output_path: str):
        """結果をJSONファイルに保存"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 8 Long Context Benchmark"
    )
    parser.add_argument(
        "--vocab-size", type=int, default=32000,
        help="Vocabulary size (default: 32000)"
    )
    parser.add_argument(
        "--d-model", type=int, default=512,
        help="Model dimension (default: 512)"
    )
    parser.add_argument(
        "--num-heads", type=int, default=8,
        help="Number of attention heads (default: 8)"
    )
    parser.add_argument(
        "--num-layers", type=int, default=4,
        help="Number of layers (default: 4)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size (default: 1)"
    )
    parser.add_argument(
        "--seq-lengths", type=int, nargs="+",
        default=[2048, 4096, 8192, 16384],
        help="Sequence lengths to benchmark"
    )
    parser.add_argument(
        "--output", type=str,
        default="results/benchmarks/phase8_long_context_benchmark.json",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    benchmark = LongContextBenchmark(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
    )
    
    benchmark.run_benchmark(seq_lengths=args.seq_lengths)
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()
