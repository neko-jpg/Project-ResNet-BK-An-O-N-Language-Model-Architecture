#!/usr/bin/env python3
"""
95% VRAM削減検証スクリプト

このスクリプトは、メモリ最適化モデルが実際に95%のVRAM削減を
達成していることを厳密に検証します。

検証項目:
1. パラメータメモリ削減
2. Activationメモリ削減
3. ピークVRAM削減
4. 実行時メモリプロファイリング

Author: MUSE Kernel Architect
"""

import sys
import os
import gc
import torch
import torch.nn as nn

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.phase1.memory_optimizer import (
    create_memory_optimized_model,
    MemoryOptimizedModel,
)
from src.models.phase1.config import Phase1Config

# 色付き出力用
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'


def get_gpu_memory_mb():
    """現在のGPUメモリ使用量を取得（MB）"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def reset_memory():
    """メモリをリセット"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


class StandardTransformerModel(nn.Module):
    """比較用の標準Transformerモデル"""
    
    def __init__(self, vocab_size, d_model, n_layers, device=None):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model, device=device)
        
        # 標準的なTransformerブロック
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=4 * d_model,
            batch_first=True,
            device=device,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output head
        self.output = nn.Linear(d_model, vocab_size, device=device)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.output(x)
        return logits


def benchmark_model(model_factory, input_shape, description, device="cuda"):
    """モデルのメモリ使用量を測定"""
    reset_memory()
    
    if device == "cpu" or not torch.cuda.is_available():
        print(f"{Colors.YELLOW}Warning: CUDA not available. Skipping {description}{Colors.END}")
        return None
    
    try:
        # モデル作成
        model = model_factory().to(device)
        param_mem = get_gpu_memory_mb()
        
        # 入力データ作成
        input_ids = torch.randint(0, 1000, input_shape).to(device)
        
        # Forward pass
        reset_memory()
        base_mem = get_gpu_memory_mb()
        
        output = model(input_ids)
        forward_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        result = {
            'description': description,
            'param_mem_mb': param_mem,
            'forward_mem_mb': forward_mem,
            'peak_mem_mb': peak_mem,
            'activation_mem_mb': peak_mem - param_mem,
        }
        
        # クリーンアップ
        del model, input_ids, output, loss
        reset_memory()
        
        return result
    
    except Exception as e:
        print(f"{Colors.RED}Error in {description}: {e}{Colors.END}")
        return None


def print_result(result):
    """結果を表示"""
    if result is None:
        return
    
    print(f"\n{Colors.BOLD}{result['description']}{Colors.END}")
    print(f"  Parameter Memory:  {result['param_mem_mb']:>8.1f} MB")
    print(f"  Forward Memory:    {result['forward_mem_mb']:>8.1f} MB")
    print(f"  Peak Memory:       {result['peak_mem_mb']:>8.1f} MB")
    print(f"  Activation Memory: {result['activation_mem_mb']:>8.1f} MB")


def main():
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}")
    print(f"95% VRAM削減検証スクリプト")
    print(f"{'='*80}{Colors.END}\n")
    
    if not torch.cuda.is_available():
        print(f"{Colors.RED}❌ CUDA not available. This test requires a GPU.{Colors.END}")
        return
    
    # 設定（小さめのモデルでテスト）
    VOCAB_SIZE = 10000
    D_MODEL = 512
    N_LAYERS = 6
    BATCH_SIZE = 2
    SEQ_LEN = 512
    
    input_shape = (BATCH_SIZE, SEQ_LEN)
    
    print(f"Configuration:")
    print(f"  Vocab Size: {VOCAB_SIZE}")
    print(f"  Model Dim:  {D_MODEL}")
    print(f"  Layers:     {N_LAYERS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Seq Length: {SEQ_LEN}")
    print(f"  Input Shape: {input_shape}")
    print("-" * 80)
    
    # 1. Baseline (Standard Transformer)
    print(f"\n{Colors.BOLD}1. Baseline Model (Standard Transformer){Colors.END}")
    
    def create_baseline():
        return StandardTransformerModel(VOCAB_SIZE, D_MODEL, N_LAYERS)
    
    baseline_result = benchmark_model(
        create_baseline,
        input_shape,
        "Standard Transformer (Baseline)"
    )
    
    if baseline_result:
        print_result(baseline_result)
    
    # 2. Memory Optimized Model
    print(f"\n{Colors.BOLD}2. Memory Optimized Model (Phase 1){Colors.END}")
    
    config = Phase1Config()
    config.use_gradient_checkpointing = True
    config.htt_rank = 16
    config.ar_ssm_max_rank = 32
    
    def create_optimized():
        return create_memory_optimized_model(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            n_layers=N_LAYERS,
            config=config,
        )
    
    optimized_result = benchmark_model(
        create_optimized,
        input_shape,
        "Memory Optimized Model (Phase 1)"
    )
    
    if optimized_result:
        print_result(optimized_result)
    
    # 3. 理論値との比較
    print(f"\n{Colors.BOLD}3. Theoretical Memory Breakdown{Colors.END}")
    
    if optimized_result:
        model = create_optimized().to("cuda")
        mem_breakdown = model.get_memory_breakdown(BATCH_SIZE, SEQ_LEN)
        
        print(f"\n{Colors.CYAN}Optimized Model Breakdown:{Colors.END}")
        opt = mem_breakdown['optimized']
        print(f"  Embedding Params:  {opt['embedding_param_mb']:>8.1f} MB")
        print(f"  Embedding Acts:    {opt['embedding_act_mb']:>8.1f} MB")
        print(f"  Blocks Params:     {opt['blocks_param_mb']:>8.1f} MB")
        print(f"  Blocks Acts:       {opt['blocks_act_mb']:>8.1f} MB")
        print(f"  Output Params:     {opt['output_param_mb']:>8.1f} MB")
        print(f"  Output Acts:       {opt['output_act_mb']:>8.1f} MB")
        print(f"  {Colors.BOLD}Total:             {opt['total_mb']:>8.1f} MB{Colors.END}")
        
        print(f"\n{Colors.CYAN}Baseline Model Breakdown:{Colors.END}")
        base = mem_breakdown['baseline']
        print(f"  Embedding:         {base['embedding_mb']:>8.1f} MB")
        print(f"  Attention:         {base['attention_mb']:>8.1f} MB")
        print(f"  FFN:               {base['ffn_mb']:>8.1f} MB")
        print(f"  {Colors.BOLD}Total:             {base['total_mb']:>8.1f} MB{Colors.END}")
        
        del model
        reset_memory()
    
    # 4. 削減率の計算と判定
    print(f"\n{Colors.BOLD}{'='*80}")
    print(f"FINAL VERDICT")
    print(f"{'='*80}{Colors.END}\n")
    
    if baseline_result and optimized_result:
        # Parameter Memory Reduction
        param_reduction = (1 - optimized_result['param_mem_mb'] / baseline_result['param_mem_mb']) * 100
        print(f"Parameter Memory Reduction: {Colors.GREEN if param_reduction >= 95 else Colors.RED}"
              f"{param_reduction:>6.1f}%{Colors.END} "
              f"({baseline_result['param_mem_mb']:.1f} MB → {optimized_result['param_mem_mb']:.1f} MB)")
        
        # Peak Memory Reduction
        peak_reduction = (1 - optimized_result['peak_mem_mb'] / baseline_result['peak_mem_mb']) * 100
        print(f"Peak Memory Reduction:      {Colors.GREEN if peak_reduction >= 95 else Colors.RED}"
              f"{peak_reduction:>6.1f}%{Colors.END} "
              f"({baseline_result['peak_mem_mb']:.1f} MB → {optimized_result['peak_mem_mb']:.1f} MB)")
        
        # Activation Memory Reduction
        act_reduction = (1 - optimized_result['activation_mem_mb'] / baseline_result['activation_mem_mb']) * 100
        print(f"Activation Memory Reduction: {Colors.GREEN if act_reduction >= 90 else Colors.RED}"
              f"{act_reduction:>6.1f}%{Colors.END} "
              f"({baseline_result['activation_mem_mb']:.1f} MB → {optimized_result['activation_mem_mb']:.1f} MB)")
        
        # 理論値との比較
        if mem_breakdown:
            theoretical_reduction = mem_breakdown['reduction']['percentage']
            print(f"\nTheoretical Reduction:      {Colors.CYAN}{theoretical_reduction:>6.1f}%{Colors.END}")
            print(f"Meets 95% Target:           {Colors.GREEN if mem_breakdown['reduction']['meets_95_target'] else Colors.RED}"
                  f"{'YES' if mem_breakdown['reduction']['meets_95_target'] else 'NO'}{Colors.END}")
        
        # 最終判定
        print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
        
        if peak_reduction >= 95.0:
            print(f"{Colors.BOLD}{Colors.GREEN}🏆 SUCCESS: 95%削減目標を達成しました！{Colors.END}")
            print(f"{Colors.GREEN}Phase 1は完全に成功です。Phase 2に進んでください。{Colors.END}")
        elif peak_reduction >= 90.0:
            print(f"{Colors.BOLD}{Colors.YELLOW}⚠️  CLOSE: {peak_reduction:.1f}%削減を達成（目標95%）{Colors.END}")
            print(f"{Colors.YELLOW}あと少しです。Mixed Precision (FP16) を有効化してください。{Colors.END}")
        elif peak_reduction >= 80.0:
            print(f"{Colors.BOLD}{Colors.YELLOW}⚠️  PARTIAL: {peak_reduction:.1f}%削減を達成（目標95%）{Colors.END}")
            print(f"{Colors.YELLOW}Triton Kernelの統合とFP16の有効化が必要です。{Colors.END}")
        else:
            print(f"{Colors.BOLD}{Colors.RED}❌ FAILED: {peak_reduction:.1f}%削減のみ（目標95%）{Colors.END}")
            print(f"{Colors.RED}根本的な最適化が必要です。{Colors.END}")
        
        print(f"{Colors.BOLD}{'='*80}{Colors.END}\n")
    
    else:
        print(f"{Colors.RED}❌ Benchmark failed. Could not compute reduction.{Colors.END}")


if __name__ == "__main__":
    main()
