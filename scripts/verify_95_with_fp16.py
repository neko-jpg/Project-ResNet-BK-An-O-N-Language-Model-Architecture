#!/usr/bin/env python3
"""
95% VRAM削減検証スクリプト (FP16対応版)

Mixed Precision (FP16) を使用して、さらに50%のメモリ削減を実現します。

Author: MUSE Kernel Architect
"""

import sys
import os
import gc
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.phase1.memory_optimizer import create_memory_optimized_model
from src.models.phase1.ultra_optimizer import create_ultra_memory_optimized_model
from src.models.phase1.config import Phase1Config


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
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=4 * d_model,
            batch_first=True,
            device=device,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Linear(d_model, vocab_size, device=device)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.output(x)
        return logits


def benchmark_model(model_factory, input_shape, description, use_fp16=False, device="cuda"):
    """モデルのメモリ使用量を測定"""
    reset_memory()
    
    if device == "cpu" or not torch.cuda.is_available():
        print(f"Warning: CUDA not available. Skipping {description}")
        return None
    
    try:
        # モデル作成
        model = model_factory().to(device)
        
        # FP16に変換
        if use_fp16:
            model = model.half()
        
        param_mem = get_gpu_memory_mb()
        
        # 入力データ作成
        input_ids = torch.randint(0, 1000, input_shape).to(device)
        
        reset_memory()
        base_mem = get_gpu_memory_mb()
        
        # Forward pass
        if use_fp16:
            with autocast():
                output = model(input_ids)
        else:
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
            'use_fp16': use_fp16,
        }
        
        # クリーンアップ
        del model, input_ids, output, loss
        reset_memory()
        
        return result
    
    except Exception as e:
        print(f"Error in {description}: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_result(result):
    """結果を表示"""
    if result is None:
        return
    
    fp16_tag = " (FP16)" if result.get('use_fp16', False) else " (FP32)"
    print(f"\n{result['description']}{fp16_tag}")
    print(f"  Parameter Memory:  {result['param_mem_mb']:>8.1f} MB")
    print(f"  Forward Memory:    {result['forward_mem_mb']:>8.1f} MB")
    print(f"  Peak Memory:       {result['peak_mem_mb']:>8.1f} MB")
    print(f"  Activation Memory: {result['activation_mem_mb']:>8.1f} MB")


def main():
    print("\n" + "="*80)
    print("95% VRAM削減検証スクリプト (FP16対応版)")
    print("="*80 + "\n")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This test requires a GPU.")
        return
    
    # 設定
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
    print("-" * 80)
    
    results = {}
    
    # 1. Baseline (FP32)
    print(f"\n1. Baseline Model (Standard Transformer, FP32)")
    
    def create_baseline():
        return StandardTransformerModel(VOCAB_SIZE, D_MODEL, N_LAYERS)
    
    results['baseline_fp32'] = benchmark_model(
        create_baseline,
        input_shape,
        "Standard Transformer",
        use_fp16=False
    )
    
    if results['baseline_fp32']:
        print_result(results['baseline_fp32'])
    
    # 2. Baseline (FP16)
    print(f"\n2. Baseline Model (Standard Transformer, FP16)")
    
    results['baseline_fp16'] = benchmark_model(
        create_baseline,
        input_shape,
        "Standard Transformer",
        use_fp16=True
    )
    
    if results['baseline_fp16']:
        print_result(results['baseline_fp16'])
    
    # 3. Memory Optimized (FP32)
    print(f"\n3. Memory Optimized Model (Phase 1, FP32)")
    
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
    
    results['optimized_fp32'] = benchmark_model(
        create_optimized,
        input_shape,
        "Memory Optimized Model",
        use_fp16=False
    )
    
    if results['optimized_fp32']:
        print_result(results['optimized_fp32'])
    
    # 4. Memory Optimized (FP16) - 標準モード
    print(f"\n4. Memory Optimized Model (Phase 1, FP16) - Standard")
    
    results['optimized_fp16'] = benchmark_model(
        create_optimized,
        input_shape,
        "Memory Optimized Model (Standard)",
        use_fp16=True
    )
    
    if results['optimized_fp16']:
        print_result(results['optimized_fp16'])
    
    # 5. Memory Optimized (FP16) - 極端モード (95%+削減を目指す)
    print(f"\n5. Memory Optimized Model (Phase 1, FP16) - EXTREME MODE")
    
    def create_extreme():
        return create_memory_optimized_model(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            n_layers=N_LAYERS,
            config=config,
            extreme_mode=True,  # 極端な最適化
        )
    
    results['extreme_fp16'] = benchmark_model(
        create_extreme,
        input_shape,
        "Memory Optimized Model (EXTREME)",
        use_fp16=True
    )
    
    if results['extreme_fp16']:
        print_result(results['extreme_fp16'])
    
    # 6. Memory Optimized (FP16) - ULTRA MODE (95%+削減を目指す)
    print(f"\n6. Memory Optimized Model (Phase 1, FP16) - ULTRA MODE")
    
    def create_ultra():
        return create_ultra_memory_optimized_model(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            n_layers=N_LAYERS,
        )
    
    results['ultra_fp16'] = benchmark_model(
        create_ultra,
        input_shape,
        "Memory Optimized Model (ULTRA)",
        use_fp16=True
    )
    
    if results['ultra_fp16']:
        print_result(results['ultra_fp16'])
    
    # 7. 削減率の計算と判定
    print(f"\n" + "="*80)
    print(f"FINAL VERDICT")
    print(f"="*80 + "\n")
    
    # Ultraモードの結果を最優先
    if results['baseline_fp32'] and results.get('ultra_fp16'):
        baseline = results['baseline_fp32']
        optimized = results['ultra_fp16']
        mode_name = "ULTRA MODE"
    elif results['baseline_fp32'] and results.get('extreme_fp16'):
        baseline = results['baseline_fp32']
        optimized = results['extreme_fp16']
        mode_name = "EXTREME MODE"
    elif results['baseline_fp32'] and results['optimized_fp16']:
        baseline = results['baseline_fp32']
        optimized = results['optimized_fp16']
        mode_name = "Standard Mode"
    else:
        baseline = None
        optimized = None
        mode_name = "Unknown"
    
    if baseline and optimized:
        
        print(f"Mode: {mode_name}\n")
        
        # Parameter Memory Reduction
        param_reduction = (1 - optimized['param_mem_mb'] / baseline['param_mem_mb']) * 100
        print(f"Parameter Memory Reduction: {param_reduction:>6.1f}% "
              f"({baseline['param_mem_mb']:.1f} MB -> {optimized['param_mem_mb']:.1f} MB)")
        
        # Peak Memory Reduction
        peak_reduction = (1 - optimized['peak_mem_mb'] / baseline['peak_mem_mb']) * 100
        print(f"Peak Memory Reduction:      {peak_reduction:>6.1f}% "
              f"({baseline['peak_mem_mb']:.1f} MB -> {optimized['peak_mem_mb']:.1f} MB)")
        
        # Activation Memory Reduction
        act_reduction = (1 - optimized['activation_mem_mb'] / baseline['activation_mem_mb']) * 100
        print(f"Activation Memory Reduction: {act_reduction:>6.1f}% "
              f"({baseline['activation_mem_mb']:.1f} MB -> {optimized['activation_mem_mb']:.1f} MB)")
        
        print(f"\n" + "="*80)
        
        if peak_reduction >= 95.0:
            print(f"SUCCESS: 95%削減目標を達成しました！")
            print(f"Phase 1は完全に成功です。Phase 2に進んでください。")
        elif peak_reduction >= 90.0:
            print(f"CLOSE: {peak_reduction:.1f}%削減を達成（目標95%）")
            print(f"あと少しです。さらなる最適化を検討してください。")
        elif peak_reduction >= 80.0:
            print(f"GOOD: {peak_reduction:.1f}%削減を達成（目標95%）")
            print(f"良い結果ですが、さらなる改善の余地があります。")
        else:
            print(f"PARTIAL: {peak_reduction:.1f}%削減を達成（目標95%）")
            print(f"追加の最適化が必要です。")
        
        print(f"="*80 + "\n")
        
        # 内訳の比較
        print(f"Breakdown Comparison:")
        print(f"\nBaseline (FP32):")
        print(f"  Params:      {baseline['param_mem_mb']:>8.1f} MB")
        print(f"  Activations: {baseline['activation_mem_mb']:>8.1f} MB")
        print(f"  Peak:        {baseline['peak_mem_mb']:>8.1f} MB")
        
        if results['baseline_fp16']:
            fp16_base = results['baseline_fp16']
            print(f"\nBaseline (FP16):")
            print(f"  Params:      {fp16_base['param_mem_mb']:>8.1f} MB")
            print(f"  Activations: {fp16_base['activation_mem_mb']:>8.1f} MB")
            print(f"  Peak:        {fp16_base['peak_mem_mb']:>8.1f} MB")
        
        if results['optimized_fp32']:
            fp32_opt = results['optimized_fp32']
            print(f"\nOptimized (FP32):")
            print(f"  Params:      {fp32_opt['param_mem_mb']:>8.1f} MB")
            print(f"  Activations: {fp32_opt['activation_mem_mb']:>8.1f} MB")
            print(f"  Peak:        {fp32_opt['peak_mem_mb']:>8.1f} MB")
        
        print(f"\nOptimized (FP16) - Standard:")
        if results.get('optimized_fp16'):
            opt_std = results['optimized_fp16']
            print(f"  Params:      {opt_std['param_mem_mb']:>8.1f} MB")
            print(f"  Activations: {opt_std['activation_mem_mb']:>8.1f} MB")
            print(f"  Peak:        {opt_std['peak_mem_mb']:>8.1f} MB")
        
        if results.get('extreme_fp16'):
            ext = results['extreme_fp16']
            print(f"\nOptimized (FP16) - EXTREME:")
            print(f"  Params:      {ext['param_mem_mb']:>8.1f} MB")
            print(f"  Activations: {ext['activation_mem_mb']:>8.1f} MB")
            print(f"  Peak:        {ext['peak_mem_mb']:>8.1f} MB")
        
        if results.get('ultra_fp16'):
            print(f"\nOptimized (FP16) - ULTRA:")
            print(f"  Params:      {optimized['param_mem_mb']:>8.1f} MB")
            print(f"  Activations: {optimized['activation_mem_mb']:>8.1f} MB")
            print(f"  Peak:        {optimized['peak_mem_mb']:>8.1f} MB")
        
        # 各最適化の寄与度
        print(f"\n" + "="*80)
        print(f"Optimization Contributions:")
        print(f"="*80)
        
        if results['optimized_fp32']:
            fp32_opt = results['optimized_fp32']
            
            # 1. Architecture optimization (FP32 baseline -> FP32 optimized)
            arch_reduction = (1 - fp32_opt['peak_mem_mb'] / baseline['peak_mem_mb']) * 100
            print(f"1. Architecture Optimization (HTT + AR-SSM + Low-Rank FFN):")
            print(f"   {arch_reduction:>6.1f}% reduction")
            
            # 2. Mixed Precision (FP32 optimized -> FP16 optimized)
            if results.get('optimized_fp16'):
                fp16_std = results['optimized_fp16']
                fp16_reduction = (1 - fp16_std['peak_mem_mb'] / fp32_opt['peak_mem_mb']) * 100
                print(f"2. Mixed Precision (FP16):")
                print(f"   {fp16_reduction:>6.1f}% additional reduction")
            
            # 3. Extreme optimizations
            if results.get('extreme_fp16'):
                extreme_additional = (1 - optimized['peak_mem_mb'] / fp32_opt['peak_mem_mb']) * 100
                print(f"3. Extreme Optimizations (Weight Tying + Extreme Checkpointing):")
                print(f"   {extreme_additional:>6.1f}% additional reduction")
            
            # 4. Total
            print(f"4. Total Reduction ({mode_name}):")
            print(f"   {peak_reduction:>6.1f}% (combined)")
        
        # 95%達成の判定
        print(f"\n" + "="*80)
        if peak_reduction >= 95.0:
            print(f"🎉 SUCCESS: 95%削減目標を達成しました！")
            print(f"   Peak VRAM: {baseline['peak_mem_mb']:.1f} MB -> {optimized['peak_mem_mb']:.1f} MB")
            print(f"   削減率: {peak_reduction:.1f}%")
        elif peak_reduction >= 90.0:
            print(f"⚠️  CLOSE: {peak_reduction:.1f}%削減を達成（目標95%）")
            print(f"   あと {95.0 - peak_reduction:.1f}% で目標達成です。")
        else:
            print(f"📊 PROGRESS: {peak_reduction:.1f}%削減を達成（目標95%）")
            print(f"   さらなる最適化が必要です。")
        print(f"="*80)
    
    else:
        print("ERROR: Benchmark failed. Could not compute reduction.")


if __name__ == "__main__":
    main()
