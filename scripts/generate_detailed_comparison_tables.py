#!/usr/bin/env python3
"""
詳細な比較テーブル生成スクリプト

パラメータ圧縮とVRAM削減の詳細な内訳を生成します。

Author: MUSE Kernel Architect
"""

import sys
import os
import gc
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.phase1.ultra_optimizer import create_ultra_memory_optimized_model
from src.models.phase1.config import Phase1Config


def get_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def count_parameters(model):
    """モデルのパラメータ数を詳細にカウント"""
    total = 0
    details = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                details[name] = params
                total += params
    
    return total, details


class StandardTransformerModel(nn.Module):
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


def benchmark_detailed(model, input_ids, description):
    """詳細なベンチマーク"""
    reset_memory()
    
    if not torch.cuda.is_available():
        return None
    
    try:
        model = model.cuda().half()
        input_ids = input_ids.cuda()
        
        # パラメータカウント
        total_params, param_details = count_parameters(model)
        
        # メモリ測定
        param_mem = get_gpu_memory_mb()
        
        reset_memory()
        
        # Forward pass
        with autocast():
            output = model(input_ids)
        
        forward_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        result = {
            'description': description,
            'total_params': total_params,
            'param_details': param_details,
            'param_mem_mb': param_mem,
            'forward_mem_mb': forward_mem,
            'peak_mem_mb': peak_mem,
            'activation_mem_mb': peak_mem - param_mem,
        }
        
        del model, input_ids, output, loss
        reset_memory()
        
        return result
    
    except Exception as e:
        print(f"Error in {description}: {e}")
        import traceback
        traceback.print_exc()
        return None


def format_number(num):
    """数値を読みやすくフォーマット"""
    if num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num/1_000:.2f}K"
    else:
        return f"{num}"


def main():
    print("\n" + "="*80)
    print("詳細な比較テーブル生成")
    print("="*80 + "\n")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return
    
    VOCAB_SIZE = 10000
    D_MODEL = 512
    N_LAYERS = 6
    BATCH_SIZE = 2
    SEQ_LEN = 512
    
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    
    print(f"Configuration:")
    print(f"  Vocab Size: {VOCAB_SIZE}")
    print(f"  Model Dim:  {D_MODEL}")
    print(f"  Layers:     {N_LAYERS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Seq Length: {SEQ_LEN}")
    print("-" * 80)
    
    # 1. Baseline (FP32)
    print(f"\n[1/3] Benchmarking Baseline Model (FP32)...")
    baseline_model_fp32 = StandardTransformerModel(VOCAB_SIZE, D_MODEL, N_LAYERS)
    baseline_result_fp32 = benchmark_detailed(baseline_model_fp32, input_ids, "Baseline (FP32)")
    
    # 2. Baseline (FP16)
    print(f"\n[2/3] Benchmarking Baseline Model (FP16)...")
    baseline_model_fp16 = StandardTransformerModel(VOCAB_SIZE, D_MODEL, N_LAYERS)
    baseline_result_fp16 = benchmark_detailed(baseline_model_fp16, input_ids, "Baseline (FP16)")
    
    # 3. Ultra Optimized
    print(f"\n[3/3] Benchmarking Ultra Optimized Model...")
    ultra_model = create_ultra_memory_optimized_model(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
    )
    ultra_result = benchmark_detailed(ultra_model, input_ids, "Ultra Optimized")
    
    # FP32 baselineを使用
    baseline_result = baseline_result_fp32
    
    if not baseline_result_fp32 or not baseline_result_fp16 or not ultra_result:
        print("ERROR: Benchmark failed")
        return
    
    # 比較用にFP32 baselineを使用
    baseline_result = baseline_result_fp32
    
    # テーブル生成
    print("\n" + "="*80)
    print("DETAILED COMPARISON TABLES")
    print("="*80 + "\n")
    
    # Table 1: パラメータ数の比較
    print("Table 1: パラメータ数の比較")
    print("-" * 80)
    print(f"{'Component':<40} {'Baseline':<15} {'Ultra Opt.':<15} {'Reduction':<15}")
    print("-" * 80)
    
    baseline_params = baseline_result['total_params']
    ultra_params = ultra_result['total_params']
    total_reduction = (1 - ultra_params / baseline_params) * 100
    
    # コンポーネント別の比較
    components = {
        'Embedding': ('embedding', 'embedding'),
        'Transformer Layers': ('transformer', 'blocks'),
        'Output Head': ('output', 'output'),
    }
    
    for comp_name, (baseline_key, ultra_key) in components.items():
        baseline_comp = sum(v for k, v in baseline_result['param_details'].items() if baseline_key in k)
        ultra_comp = sum(v for k, v in ultra_result['param_details'].items() if ultra_key in k)
        
        if baseline_comp > 0:
            reduction = (1 - ultra_comp / baseline_comp) * 100
            print(f"{comp_name:<40} {format_number(baseline_comp):<15} {format_number(ultra_comp):<15} {reduction:>6.1f}%")
    
    print("-" * 80)
    print(f"{'Total':<40} {format_number(baseline_params):<15} {format_number(ultra_params):<15} {total_reduction:>6.1f}%")
    print("-" * 80)
    
    # Table 2: VRAM使用量の比較
    print("\n\nTable 2: VRAM使用量の比較（学習時）")
    print("-" * 80)
    print(f"{'Metric':<40} {'Baseline(FP32)':<15} {'Baseline(FP16)':<15} {'Ultra(FP16)':<15} {'Reduction':<15}")
    print("-" * 80)
    
    metrics = [
        ('Parameter Memory', 'param_mem_mb'),
        ('Forward Memory', 'forward_mem_mb'),
        ('Peak Memory (Training)', 'peak_mem_mb'),
        ('Activation Memory', 'activation_mem_mb'),
    ]
    
    for metric_name, key in metrics:
        baseline_fp32_val = baseline_result_fp32[key]
        baseline_fp16_val = baseline_result_fp16[key]
        ultra_val = ultra_result[key]
        reduction = (1 - ultra_val / baseline_fp32_val) * 100
        print(f"{metric_name:<40} {baseline_fp32_val:>7.1f} MB     {baseline_fp16_val:>7.1f} MB     {ultra_val:>7.1f} MB     {reduction:>6.1f}%")
    
    print("-" * 80)
    
    # Table 3: 内訳の詳細
    print("\n\nTable 3: Ultra Optimized Model の内訳")
    print("-" * 80)
    print(f"{'Component':<50} {'Parameters':<15} {'Memory (MB)':<15}")
    print("-" * 80)
    
    # パラメータの詳細
    param_categories = {}
    for name, params in ultra_result['param_details'].items():
        if 'embedding' in name:
            category = 'HTT Embedding'
        elif 'ar_ssm' in name:
            category = 'AR-SSM Layers'
        elif 'ffn' in name:
            category = 'Low-Rank FFN'
        elif 'norm' in name:
            category = 'Normalization'
        elif 'output' in name:
            category = 'Output Head'
        else:
            category = 'Other'
        
        if category not in param_categories:
            param_categories[category] = 0
        param_categories[category] += params
    
    for category, params in sorted(param_categories.items()):
        mem_mb = params * 2 / 1024 / 1024  # FP16 = 2 bytes
        print(f"{category:<50} {format_number(params):<15} {mem_mb:>8.2f} MB")
    
    print("-" * 80)
    print(f"{'Total Parameters':<50} {format_number(ultra_params):<15} {ultra_result['param_mem_mb']:>8.2f} MB")
    print(f"{'Activation Memory':<50} {'':<15} {ultra_result['activation_mem_mb']:>8.2f} MB")
    print(f"{'Peak Memory (Training)':<50} {'':<15} {ultra_result['peak_mem_mb']:>8.2f} MB")
    print("-" * 80)
    
    # Markdown形式で保存
    output_dir = "results/benchmarks/tables"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/detailed_comparison.md", "w", encoding="utf-8") as f:
        f.write("# 詳細な比較テーブル\n\n")
        f.write(f"**日付**: 2025-11-19\n")
        f.write(f"**構成**: vocab={VOCAB_SIZE}, d={D_MODEL}, layers={N_LAYERS}, batch={BATCH_SIZE}, seq={SEQ_LEN}\n\n")
        
        f.write("## Table 1: パラメータ数の比較\n\n")
        f.write("| Component | Baseline | Ultra Optimized | Reduction |\n")
        f.write("|-----------|----------|-----------------|----------|\n")
        
        for comp_name, (baseline_key, ultra_key) in components.items():
            baseline_comp = sum(v for k, v in baseline_result['param_details'].items() if baseline_key in k)
            ultra_comp = sum(v for k, v in ultra_result['param_details'].items() if ultra_key in k)
            if baseline_comp > 0:
                reduction = (1 - ultra_comp / baseline_comp) * 100
                f.write(f"| {comp_name} | {format_number(baseline_comp)} | {format_number(ultra_comp)} | {reduction:.1f}% |\n")
        
        f.write(f"| **Total** | **{format_number(baseline_params)}** | **{format_number(ultra_params)}** | **{total_reduction:.1f}%** |\n\n")
        
        f.write("## Table 2: VRAM使用量の比較（学習時）\n\n")
        f.write("| Metric | Baseline (FP32) | Baseline (FP16) | Ultra Optimized (FP16) | Reduction |\n")
        f.write("|--------|-----------------|-----------------|------------------------|----------|\n")
        
        for metric_name, key in metrics:
            baseline_fp32_val = baseline_result_fp32[key]
            baseline_fp16_val = baseline_result_fp16[key]
            ultra_val = ultra_result[key]
            reduction = (1 - ultra_val / baseline_fp32_val) * 100
            f.write(f"| {metric_name} | {baseline_fp32_val:.1f} MB | {baseline_fp16_val:.1f} MB | {ultra_val:.1f} MB | {reduction:.1f}% |\n")
        
        f.write("\n## Table 3: Ultra Optimized Model の内訳\n\n")
        f.write("| Component | Parameters | Memory (FP16) |\n")
        f.write("|-----------|------------|---------------|\n")
        
        for category, params in sorted(param_categories.items()):
            mem_mb = params * 2 / 1024 / 1024
            f.write(f"| {category} | {format_number(params)} | {mem_mb:.2f} MB |\n")
        
        f.write(f"| **Total Parameters** | **{format_number(ultra_params)}** | **{ultra_result['param_mem_mb']:.2f} MB** |\n")
        f.write(f"| **Activation Memory** | - | **{ultra_result['activation_mem_mb']:.2f} MB** |\n")
        f.write(f"| **Peak Memory (Training)** | - | **{ultra_result['peak_mem_mb']:.2f} MB** |\n")
    
    print(f"\n\nMarkdown table saved to: {output_dir}/detailed_comparison.md")
    
    # CSV形式で保存
    with open(f"{output_dir}/detailed_comparison.csv", "w", encoding="utf-8") as f:
        f.write("Category,Component,Baseline,Ultra_Optimized,Reduction_Percent\n")
        
        # パラメータ
        f.write("Parameters,Total,{},{},{:.1f}\n".format(
            baseline_params, ultra_params, total_reduction
        ))
        
        for comp_name, (baseline_key, ultra_key) in components.items():
            baseline_comp = sum(v for k, v in baseline_result['param_details'].items() if baseline_key in k)
            ultra_comp = sum(v for k, v in ultra_result['param_details'].items() if ultra_key in k)
            if baseline_comp > 0:
                reduction = (1 - ultra_comp / baseline_comp) * 100
                f.write(f"Parameters,{comp_name},{baseline_comp},{ultra_comp},{reduction:.1f}\n")
        
        # VRAM
        for metric_name, key in metrics:
            baseline_val = baseline_result[key]
            ultra_val = ultra_result[key]
            reduction = (1 - ultra_val / baseline_val) * 100
            f.write(f"VRAM,{metric_name},{baseline_val:.1f},{ultra_val:.1f},{reduction:.1f}\n")
    
    print(f"CSV table saved to: {output_dir}/detailed_comparison.csv")
    
    # LaTeX形式で保存
    with open(f"{output_dir}/detailed_comparison.tex", "w", encoding="utf-8") as f:
        f.write("% 詳細な比較テーブル (LaTeX)\n")
        f.write("% Generated: 2025-11-19\n\n")
        
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{パラメータ数の比較}\n")
        f.write("\\label{tab:param_comparison}\n")
        f.write("\\begin{tabular}{lrrr}\n")
        f.write("\\toprule\n")
        f.write("Component & Baseline & Ultra Optimized & Reduction \\\\\n")
        f.write("\\midrule\n")
        
        for comp_name, (baseline_key, ultra_key) in components.items():
            baseline_comp = sum(v for k, v in baseline_result['param_details'].items() if baseline_key in k)
            ultra_comp = sum(v for k, v in ultra_result['param_details'].items() if ultra_key in k)
            if baseline_comp > 0:
                reduction = (1 - ultra_comp / baseline_comp) * 100
                f.write(f"{comp_name} & {format_number(baseline_comp)} & {format_number(ultra_comp)} & {reduction:.1f}\\% \\\\\n")
        
        f.write("\\midrule\n")
        f.write(f"\\textbf{{Total}} & \\textbf{{{format_number(baseline_params)}}} & \\textbf{{{format_number(ultra_params)}}} & \\textbf{{{total_reduction:.1f}\\%}} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{VRAM使用量の比較（学習時、FP16）}\n")
        f.write("\\label{tab:vram_comparison}\n")
        f.write("\\begin{tabular}{lrrr}\n")
        f.write("\\toprule\n")
        f.write("Metric & Baseline & Ultra Optimized & Reduction \\\\\n")
        f.write("\\midrule\n")
        
        for metric_name, key in metrics:
            baseline_val = baseline_result[key]
            ultra_val = ultra_result[key]
            reduction = (1 - ultra_val / baseline_val) * 100
            f.write(f"{metric_name} & {baseline_val:.1f} MB & {ultra_val:.1f} MB & {reduction:.1f}\\% \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to: {output_dir}/detailed_comparison.tex")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nパラメータ削減: {total_reduction:.1f}%")
    print(f"  Baseline: {format_number(baseline_params)} parameters")
    print(f"  Ultra Optimized: {format_number(ultra_params)} parameters")
    print(f"\nVRAM削減（Peak Memory）: {(1 - ultra_result['peak_mem_mb'] / baseline_result['peak_mem_mb']) * 100:.1f}%")
    print(f"  Baseline: {baseline_result['peak_mem_mb']:.1f} MB")
    print(f"  Ultra Optimized: {ultra_result['peak_mem_mb']:.1f} MB")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
