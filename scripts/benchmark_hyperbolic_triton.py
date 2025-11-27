"""
Hyperbolic Attention Triton Kernel Benchmark

各カーネルバージョンの性能を比較:
- fast: 最速版（近似双曲距離）
- v2: 最適化版（事前計算 + autotune）
- v1: 従来版
- pytorch: PyTorch参照実装

出力: JSON形式のベンチマーク結果
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn.functional as F


def benchmark_kernel(
    kernel_name: str,
    batch: int,
    seq_len: int,
    d_model: int,
    heads: int,
    device: torch.device,
    warmup_iters: int = 5,
    bench_iters: int = 20,
) -> Dict[str, Any]:
    """単一カーネルのベンチマーク"""
    d_head = d_model // heads
    
    # 入力生成
    torch.manual_seed(42)
    q = torch.randn(batch, heads, seq_len, d_head, device=device, dtype=torch.float32)
    k = torch.randn(batch, heads, seq_len, d_head, device=device, dtype=torch.float32)
    v = torch.randn(batch, heads, seq_len, d_head, device=device, dtype=torch.float32)
    c = torch.tensor(1.0, device=device)
    beta = torch.tensor(1.0, device=device)
    
    # カーネル選択
    if kernel_name == 'fast':
        from src.kernels.hyperbolic_attention_fast import fast_hyperbolic_attention
        kernel_fn = lambda: fast_hyperbolic_attention(q, k, v, c, beta, causal=True)
    elif kernel_name == 'v2':
        from src.kernels.hyperbolic_attention_triton_v2 import hyperbolic_attention_triton_v2
        kernel_fn = lambda: hyperbolic_attention_triton_v2(q, k, v, c, beta, causal=True)
    elif kernel_name == 'v1':
        from src.kernels.hyperbolic_attention_triton import hyperbolic_attention_triton
        kernel_fn = lambda: hyperbolic_attention_triton(q, k, v, c, beta, causal=True)
    elif kernel_name == 'pytorch':
        kernel_fn = lambda: pytorch_hyperbolic_attention(q, k, v, c, beta, causal=True)
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")
    
    # Warmup
    for _ in range(warmup_iters):
        try:
            _ = kernel_fn()
            if device.type == 'cuda':
                torch.cuda.synchronize()
        except Exception as e:
            return {
                'kernel': kernel_name,
                'status': 'error',
                'error': str(e),
            }
    
    # Benchmark
    times = []
    for _ in range(bench_iters):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = kernel_fn()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # Throughput計算
    tokens_per_sec = (batch * seq_len) / avg_time
    
    return {
        'kernel': kernel_name,
        'status': 'ok',
        'avg_time_ms': avg_time * 1000,
        'min_time_ms': min_time * 1000,
        'max_time_ms': max_time * 1000,
        'tokens_per_sec': tokens_per_sec,
    }


def pytorch_hyperbolic_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    c: torch.Tensor,
    beta: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """PyTorch参照実装"""
    B, H, N, D = q.shape
    device = q.device
    
    EPS = 1e-5
    sqrt_c = torch.sqrt(c.clamp(min=EPS))
    
    # exp_map
    def exp_map(x):
        norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
        tanh_arg = (sqrt_c * norm).clamp(max=15.0)
        return x * (torch.tanh(tanh_arg) / (sqrt_c * norm))
    
    q_hyp = exp_map(q)
    k_hyp = exp_map(k)
    
    # 距離計算
    q_norm_sq = (q_hyp * q_hyp).sum(dim=-1, keepdim=True)
    k_norm_sq = (k_hyp * k_hyp).sum(dim=-1, keepdim=True)
    
    qk_dot = torch.matmul(q_hyp, k_hyp.transpose(-2, -1))
    diff_sq = q_norm_sq - 2.0 * qk_dot + k_norm_sq.transpose(-2, -1)
    diff_sq = diff_sq.clamp(min=0.0)
    
    denom = (1.0 - c * q_norm_sq) * (1.0 - c * k_norm_sq.transpose(-2, -1))
    denom = denom.clamp(min=EPS)
    
    arg = 1.0 + 2.0 * c * diff_sq / denom
    arg = arg.clamp(min=1.0 + EPS)
    
    dist = (1.0 / sqrt_c) * torch.acosh(arg)
    scores = -beta * dist
    
    if causal:
        mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
    
    attn = F.softmax(scores, dim=-1)
    acc = torch.matmul(attn, v)
    out = exp_map(acc)
    
    return out


def run_benchmark(
    batch: int,
    seq_len: int,
    d_model: int,
    heads: int,
    device_str: str,
    json_path: str | None,
    kernels: List[str],
) -> Dict[str, Any]:
    """全カーネルのベンチマーク実行"""
    device = torch.device(device_str if device_str != 'cuda' or torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"Hyperbolic Attention Benchmark")
    print(f"{'='*60}")
    print(f"Config: batch={batch}, seq_len={seq_len}, d_model={d_model}, heads={heads}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    results = {
        'config': {
            'batch': batch,
            'seq_len': seq_len,
            'd_model': d_model,
            'heads': heads,
            'device': str(device),
        },
        'kernels': [],
    }
    
    for kernel in kernels:
        print(f"Benchmarking {kernel}...", end=' ', flush=True)
        result = benchmark_kernel(kernel, batch, seq_len, d_model, heads, device)
        results['kernels'].append(result)
        
        if result['status'] == 'ok':
            print(f"avg={result['avg_time_ms']:.2f}ms, {result['tokens_per_sec']:.0f} tok/s")
        else:
            print(f"ERROR: {result.get('error', 'unknown')}")
    
    # 最速カーネルを特定
    ok_results = [r for r in results['kernels'] if r['status'] == 'ok']
    if ok_results:
        fastest = min(ok_results, key=lambda x: x['avg_time_ms'])
        results['fastest_kernel'] = fastest['kernel']
        results['fastest_time_ms'] = fastest['avg_time_ms']
        
        # PyTorchとの比較
        pytorch_result = next((r for r in ok_results if r['kernel'] == 'pytorch'), None)
        if pytorch_result and fastest['kernel'] != 'pytorch':
            speedup = pytorch_result['avg_time_ms'] / fastest['avg_time_ms']
            results['speedup_vs_pytorch'] = speedup
            print(f"\n最速: {fastest['kernel']} ({speedup:.2f}x faster than PyTorch)")
    
    # JSON出力
    if json_path:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2))
        print(f"\n結果を保存: {json_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--json", type=str, default="results/benchmarks/hyperbolic_triton_benchmark.json")
    parser.add_argument(
        "--kernels",
        type=str,
        nargs='+',
        default=['fast', 'v2', 'v1', 'pytorch'],
        help="Kernels to benchmark",
    )
    args = parser.parse_args()
    
    run_benchmark(
        batch=args.batch,
        seq_len=args.seq_len,
        d_model=args.d_model,
        heads=args.heads,
        device_str=args.device,
        json_path=args.json,
        kernels=args.kernels,
    )


if __name__ == "__main__":
    main()
