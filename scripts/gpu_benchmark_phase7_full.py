#!/usr/bin/env python3
"""
Phase 7 完全最適化ベンチマーク (8GB VRAM用)

全最適化機能を有効化:
- HTT Embedding (99.7%圧縮)
- Ultra Low-Rank FFN (98%+圧縮)
- AR-SSM (O(N)計算)
- Triton Kernels
- FP16 Mixed Precision
- Gradient Checkpointing
- Ultra Memory Optimizer

物理的直観: 8GB VRAMでチャットAI用の最大パラメータ数を探索
"""
import gc
import json
import sys
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def get_gpu_info():
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_vram_gb": props.total_memory / 1024**3,
        "compute_capability": f"{props.major}.{props.minor}",
    }

def get_vram_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(0) / 1024**3, torch.cuda.memory_reserved(0) / 1024**3
    return 0, 0

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def test_ultra_optimized_model(d_model, n_layers, batch_size, seq_len, vocab_size=32000):
    """Ultra Memory Optimizerを使用したテスト"""
    clear_memory()
    try:
        from src.models.phase1.ultra_optimizer import create_ultra_memory_optimized_model
        
        model = create_ultra_memory_optimized_model(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            device='cuda',
            dtype=torch.float16,
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
        
        # Forward + Backward
        with torch.cuda.amp.autocast(enabled=True):
            output = model(input_ids)
            loss = output.mean()
            loss.backward()
        
        torch.cuda.synchronize()
        allocated, reserved = get_vram_usage()
        
        del model, output, loss, input_ids
        clear_memory()
        
        return {
            "success": True,
            "model_type": "UltraMemoryOptimized",
            "total_params": total_params,
            "params_millions": total_params / 1e6,
            "vram_allocated_gb": allocated,
            "vram_reserved_gb": reserved,
            "config": {"d_model": d_model, "n_layers": n_layers, "batch_size": batch_size, "seq_len": seq_len}
        }
    except Exception as e:
        clear_memory()
        return {"success": False, "error": str(e), "model_type": "UltraMemoryOptimized"}

def test_memory_optimized_model(d_model, n_layers, batch_size, seq_len, vocab_size=32000, extreme_mode=True):
    """Memory Optimizerを使用したテスト"""
    clear_memory()
    try:
        from src.models.phase1.memory_optimizer import create_memory_optimized_model
        from src.models.phase1.config import Phase1Config
        
        config = Phase1Config()
        config.use_gradient_checkpointing = True
        config.htt_rank = 8
        config.ar_ssm_max_rank = 16
        
        model = create_memory_optimized_model(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            config=config,
            device='cuda',
            dtype=torch.float16,
            extreme_mode=extreme_mode,
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
        
        with torch.cuda.amp.autocast(enabled=True):
            output = model(input_ids)
            loss = output.mean()
            loss.backward()
        
        torch.cuda.synchronize()
        allocated, reserved = get_vram_usage()
        
        del model, output, loss, input_ids
        clear_memory()
        
        return {
            "success": True,
            "model_type": "MemoryOptimized_Extreme" if extreme_mode else "MemoryOptimized",
            "total_params": total_params,
            "params_millions": total_params / 1e6,
            "vram_allocated_gb": allocated,
            "vram_reserved_gb": reserved,
            "config": {"d_model": d_model, "n_layers": n_layers, "batch_size": batch_size, "seq_len": seq_len}
        }
    except Exception as e:
        clear_memory()
        return {"success": False, "error": str(e), "model_type": "MemoryOptimized"}

def test_phase1_factory_model(d_model, n_layers, batch_size, seq_len, vocab_size=32000):
    """Phase1 Factoryを使用したテスト"""
    clear_memory()
    try:
        from src.models.phase1.factory import create_phase1_model
        from src.models.phase1.config import Phase1Config
        
        config = Phase1Config()
        config.vocab_size = vocab_size
        config.d_model = d_model
        config.n_layers = n_layers
        config.n_seq = seq_len
        config.use_gradient_checkpointing = True
        config.htt_rank = 8
        config.ar_ssm_max_rank = 16
        config.ar_ssm_use_fused_scan = True
        
        model = create_phase1_model(config)
        model = model.cuda().half()
        
        total_params = sum(p.numel() for p in model.parameters())
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
        
        with torch.cuda.amp.autocast(enabled=True):
            output = model(input_ids)
            loss = output.mean()
            loss.backward()
        
        torch.cuda.synchronize()
        allocated, reserved = get_vram_usage()
        
        del model, output, loss, input_ids
        clear_memory()
        
        return {
            "success": True,
            "model_type": "Phase1Factory",
            "total_params": total_params,
            "params_millions": total_params / 1e6,
            "vram_allocated_gb": allocated,
            "vram_reserved_gb": reserved,
            "config": {"d_model": d_model, "n_layers": n_layers, "batch_size": batch_size, "seq_len": seq_len}
        }
    except Exception as e:
        clear_memory()
        return {"success": False, "error": str(e), "model_type": "Phase1Factory"}

def binary_search_max_params(test_func, base_d=128, max_d=2048, step=64, **kwargs):
    """二分探索で最大d_modelを見つける"""
    low, high = base_d, max_d
    best_result = None
    
    while low <= high:
        mid = ((low + high) // 2 // step) * step
        if mid < base_d:
            mid = base_d
        print(f"  Testing d_model={mid}...", end=" ", flush=True)
        
        result = test_func(d_model=mid, **kwargs)
        
        if result["success"]:
            print(f"✓ {result['params_millions']:.1f}M, {result['vram_allocated_gb']:.2f}GB")
            best_result = result
            low = mid + step
        else:
            err = result.get('error', 'OOM')[:50]
            print(f"✗ {err}")
            high = mid - step
    
    return best_result

def run_full_benchmark():
    """完全ベンチマーク実行"""
    print("=" * 70)
    print("Phase 7 完全最適化ベンチマーク (8GB VRAM)")
    print("=" * 70)
    
    gpu_info = get_gpu_info()
    print(f"\nGPU: {gpu_info.get('name', 'Unknown')}")
    print(f"VRAM: {gpu_info.get('total_vram_gb', 0):.2f} GB")
    
    # Triton確認
    try:
        import triton
        print(f"Triton: v{triton.__version__}")
        triton_available = True
    except:
        print("Triton: Not Available")
        triton_available = False
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "gpu_info": gpu_info,
        "triton_available": triton_available,
        "benchmarks": []
    }
    
    # ========================================
    # Test 1: Ultra Memory Optimized Model
    # ========================================
    print("\n" + "=" * 70)
    print("[1] Ultra Memory Optimized Model (95%+ VRAM削減)")
    print("=" * 70)
    
    test_configs = [
        (6, 1, 512, "Chat AI (6層, batch=1, seq=512)"),
        (6, 2, 256, "Chat AI (6層, batch=2, seq=256)"),
        (8, 1, 512, "Larger (8層, batch=1, seq=512)"),
        (12, 1, 256, "Deep (12層, batch=1, seq=256)"),
    ]
    
    for n_layers, batch_size, seq_len, desc in test_configs:
        print(f"\n[TEST] {desc}")
        result = binary_search_max_params(
            test_ultra_optimized_model,
            base_d=128, max_d=2048, step=64,
            n_layers=n_layers, batch_size=batch_size, seq_len=seq_len
        )
        if result:
            results["benchmarks"].append({"description": f"Ultra: {desc}", "result": result})
            print(f"  → MAX: {result['params_millions']:.1f}M @ d={result['config']['d_model']}")
    
    # ========================================
    # Test 2: Memory Optimized Model (Extreme)
    # ========================================
    print("\n" + "=" * 70)
    print("[2] Memory Optimized Model (Extreme Mode)")
    print("=" * 70)
    
    for n_layers, batch_size, seq_len, desc in test_configs[:2]:
        print(f"\n[TEST] {desc}")
        result = binary_search_max_params(
            test_memory_optimized_model,
            base_d=128, max_d=2048, step=64,
            n_layers=n_layers, batch_size=batch_size, seq_len=seq_len, extreme_mode=True
        )
        if result:
            results["benchmarks"].append({"description": f"MemOpt: {desc}", "result": result})
            print(f"  → MAX: {result['params_millions']:.1f}M @ d={result['config']['d_model']}")
    
    # ========================================
    # Test 3: Phase1 Factory Model
    # ========================================
    print("\n" + "=" * 70)
    print("[3] Phase1 Factory Model (HTT + AR-SSM + Triton)")
    print("=" * 70)
    
    for n_layers, batch_size, seq_len, desc in test_configs[:2]:
        print(f"\n[TEST] {desc}")
        result = binary_search_max_params(
            test_phase1_factory_model,
            base_d=128, max_d=1536, step=64,
            n_layers=n_layers, batch_size=batch_size, seq_len=seq_len
        )
        if result:
            results["benchmarks"].append({"description": f"Phase1: {desc}", "result": result})
            print(f"  → MAX: {result['params_millions']:.1f}M @ d={result['config']['d_model']}")
    
    # ========================================
    # 推奨設定
    # ========================================
    print("\n" + "=" * 70)
    print("推奨設定 (8GB VRAM)")
    print("=" * 70)
    
    if results["benchmarks"]:
        best = max(results["benchmarks"], key=lambda x: x["result"]["params_millions"])
        results["recommended"] = best
        print(f"\n最大パラメータ構成:")
        print(f"  モデルタイプ: {best['result']['model_type']}")
        print(f"  d_model: {best['result']['config']['d_model']}")
        print(f"  n_layers: {best['result']['config']['n_layers']}")
        print(f"  パラメータ数: {best['result']['params_millions']:.1f}M")
        print(f"  VRAM使用量: {best['result']['vram_allocated_gb']:.2f}GB")
    
    # 保存
    output_path = Path("results/benchmarks/phase7_full_optimization_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n結果保存: {output_path}")
    return results

if __name__ == "__main__":
    run_full_benchmark()
