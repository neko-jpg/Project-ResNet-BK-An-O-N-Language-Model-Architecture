#!/usr/bin/env python3
"""
Phase 8 Quantization Benchmark Script

タスク30.4: 量子化ベンチマーク
- FP16, INT8, INT4スループット比較
- 精度劣化の測定
- 目標: INT8で2xスピードアップ、<1% PPL劣化

Requirements: 36.5, 4.3, 4.4
"""

import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn

# Phase 8量子化モジュールのインポート
try:
    from src.models.phase8.quantization import (
        LogarithmicQuantizer,
        INT8QuantizedKernel,
        CalibrationPipeline,
        create_logarithmic_quantizer,
        create_int8_kernel,
    )
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False


def get_device() -> torch.device:
    """利用可能なデバイスを取得"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def measure_quantization_accuracy(
    original: torch.Tensor,
    quantized: torch.Tensor,
) -> Dict[str, float]:
    """
    量子化精度を測定
    
    Returns:
        Dict with mse, mae, max_error, correlation
    """
    diff = original.float() - quantized.float()
    
    mse = (diff ** 2).mean().item()
    mae = diff.abs().mean().item()
    max_error = diff.abs().max().item()
    
    # 相関係数
    orig_flat = original.float().flatten()
    quant_flat = quantized.float().flatten()
    
    orig_mean = orig_flat.mean()
    quant_mean = quant_flat.mean()
    
    numerator = ((orig_flat - orig_mean) * (quant_flat - quant_mean)).sum()
    denominator = torch.sqrt(
        ((orig_flat - orig_mean) ** 2).sum() * 
        ((quant_flat - quant_mean) ** 2).sum()
    )
    
    correlation = (numerator / (denominator + 1e-8)).item()
    
    return {
        "mse": mse,
        "mae": mae,
        "max_error": max_error,
        "correlation": correlation,
    }


def measure_throughput(
    fn,
    *args,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    関数のスループットを測定
    
    Returns:
        Dict with time_ms, throughput
    """
    device = device or get_device()
    
    # ウォームアップ
    for _ in range(warmup_iterations):
        _ = fn(*args)
        if device.type == "cuda":
            torch.cuda.synchronize()
    
    # 測定
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    for _ in range(num_iterations):
        _ = fn(*args)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    time_per_iter_ms = (total_time / num_iterations) * 1000
    
    return {
        "time_ms": time_per_iter_ms,
        "iterations_per_sec": num_iterations / total_time,
    }


class QuantizationBenchmark:
    """量子化ベンチマーク"""
    
    def __init__(
        self,
        batch_size: int = 4,
        d_model: int = 512,
        device: Optional[torch.device] = None,
    ):
        self.batch_size = batch_size
        self.d_model = d_model
        self.device = device or get_device()
        self.results: Dict[str, Any] = {}
    
    def run_benchmark(
        self,
        seq_lengths: List[int] = [1024, 2048, 4096],
        num_iterations: int = 100,
    ) -> Dict[str, Any]:
        """ベンチマークを実行"""
        print(f"\n{'='*60}")
        print("Phase 8 Quantization Benchmark")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Model dim: {self.d_model}")
        print(f"Quantization available: {QUANTIZATION_AVAILABLE}")
        print(f"{'='*60}\n")
        
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "device": str(self.device),
                "batch_size": self.batch_size,
                "d_model": self.d_model,
                "num_iterations": num_iterations,
            },
            "fp16": {},
            "int8": {},
            "int4": {},
            "accuracy": {},
            "comparison": {},
        }
        
        for seq_len in seq_lengths:
            print(f"\nSequence length: {seq_len}")
            print("-" * 40)
            
            # テストデータ生成
            x = torch.randn(
                self.batch_size, seq_len, self.d_model,
                device=self.device, dtype=torch.float16
            )
            
            # FP16ベースライン
            print("  FP16 baseline...")
            fp16_fn = lambda t: t * 1.0  # 単純なコピー操作
            fp16_perf = measure_throughput(
                fp16_fn, x,
                num_iterations=num_iterations,
                device=self.device,
            )
            results["fp16"][str(seq_len)] = fp16_perf
            print(f"    Time: {fp16_perf['time_ms']:.3f} ms")
            
            # INT8量子化
            if QUANTIZATION_AVAILABLE:
                print("  INT8 quantization...")
                try:
                    quantizer = create_logarithmic_quantizer(
                        bits=8,
                        boundary_scale=0.99,
                    )
                    
                    # 量子化スループット測定
                    def int8_quantize(t):
                        return quantizer.quantize(t)
                  