#!/usr/bin/env python3
"""
Phase 7 GPU Maximum Parameter Benchmark
RTX 3080 (8GB) でのPhase 7モデルの最大パラメータ数を測定

物理的直観:
- VRAMの限界を探りながら、チャットAI用の最適な設定を見つける
- Triton ON/OFF、Mixed Precision、Gradient Checkpointingの効果を測定
"""
import gc
import json
import sys
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def get_gpu_info():
    """GPU情報を取得"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_vram_gb": props.total_memory / 1024**3,
        "compute_capability": f"{props.major}.{props.minor}",
        "multiprocessors": props.multi_processor_count
    }

def get_vram_usage():
    """現在のVRAM使用量を取得"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        return allocated, reserved
    return 0, 0

def clear_memory():
    """メモリをクリア"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def test_phase7_config(d_model, n_layers, batch_size, seq_len, use_triton=True, use_fp16=True, use_checkpointing=True):
    """Phase 7設定でモデルをテスト"""
    clear_memory()
    
    try:
        from src.models.phase7.integrated_model import Phase7IntegratedModel, Phase7Config
        
        config = Phase7Config(
            d_model=d_model,
            n_layers=n_layers,
            vocab_size=50257,
            n_seq=seq_len,
            num_heads=max(8, d_model // 64),
            htt_rank=16,
            use_hybrid_attention=True,
            hyperbolic_window_size=64,
            use_triton_kernel=use_triton,
            triton_kernel_version='fast',
            use_gradient_checkpointing=use_checkpointing,
            use_mixed_precision=use_fp16
        )
        
        model = Phase7IntegratedModel(config)
        model = model.cuda()
        
        if use_fp16:
            model = model.half()
        
        # パラメータ数を計算
        total_params = sum(p.numel() for p in model.parameters())
        
        # Forward pass test
        input_ids = torch.randint(0, 50257, (batch_size, seq_len), device='cuda')
        
        with torch.cuda.amp.autocast(enabled=use_fp16):
            output = model(input_ids)
        
        # Backward pass test (training simulation)
        loss = output.mean()
        loss.backward()
        
        torch.cuda.synchronize()
        allocated, reserved = get_vram_usage()
        
        del model, output, loss, input_ids
        clear_memory()
        
        return {
            "success": True,
            "total_params": total_params,
            "params_millions": total_params / 1e6,
            "vram_allocated_gb": allocated,
            "vram_reserved_gb": reserved,
            "config": {
                "d_model": d_model,
                "n_layers": n_layers,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "use_triton": use_triton,
                "use_fp16": use_fp16,
                "use_checkpointing": use_checkpointing
            }
        }
    except Exception as e:
        clear_memory()
        return {
            "success": False,
            "error": str(e),
            "config": {
                "d_model": d_model,
                "n_layers": n_layers,
                "batch_size": batch_size,
                "seq_len": seq_len
            }
        }

def binary_search_max_params(base_d_model=512, max_d_model=2048, n_layers=6, batch_size=4, seq_len=512, **kwargs):
    """二分探索で最大パラメータを見つける"""
    low, high = base_d_model, max_d_model
    best_result = None
    
    while low <= high:
        mid = ((low + high) // 2 // 64) * 64  # 64の倍数に丸める
        print(f"  Testing d_model={mid}...", end=" ", flush=True)
        
        result = test_phase7_config(mid, n_layers, batch_size, seq_len, **kwargs)
        
        if result["success"]:
            print(f"✓ {result['params_millions']:.1f}M params, {result['vram_allocated_gb']:.2f}GB VRAM")
            best_result = result
            low = mid + 64
        else:
            print(f"✗ OOM")
            high = mid - 64
    
    return best_result

def run_comprehensive_benchmark():
    """包括的なベンチマークを実行"""
    print("=" * 70)
    print("Phase 7 GPU Maximum Parameter Benchmark")
    print("=" * 70)
    
    # GPU情報
    gpu_info = get_gpu_info()
    print(f"\nGPU: {gpu_info.get('name', 'Unknown')}")
    print(f"VRAM: {gpu_info.get('total_vram_gb', 0):.2f} GB")
    print(f"Compute Capability: {gpu_info.get('compute_capability', 'Unknown')}")
    
    # Triton確認
    try:
        import triton
        triton_available = True
        print(f"Triton: Available (v{triton.__version__})")
    except ImportError:
        triton_available = False
        print("Triton: Not Available")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "gpu_info": gpu_info,
        "triton_available": triton_available,
        "benchmarks": []
    }
    
    # テスト設定 (8GB VRAM用に最適化)
    test_configs = [
        # (n_layers, batch_size, seq_len, use_triton, use_fp16, use_checkpointing, description)
        (6, 2, 512, True, True, True, "Chat AI (Triton ON, FP16, Checkpointing)"),
        (6, 2, 512, False, True, True, "Chat AI (Triton OFF, FP16, Checkpointing)"),
        (8, 2, 512, True, True, True, "Larger Model (8 layers)"),
        (12, 1, 512, True, True, True, "Deep Model (12 layers, batch=1)"),
        (6, 4, 256, True, True, True, "High Throughput (batch=4, seq=256)"),
        (6, 1, 1024, True, True, True, "Long Context (seq=1024)"),
        (6, 1, 2048, True, True, True, "Very Long Context (seq=2048)"),
    ]
    
    print("\n" + "=" * 70)
    print("Running Maximum Parameter Search...")
    print("=" * 70)
    
    for n_layers, batch_size, seq_len, use_triton, use_fp16, use_checkpointing, desc in test_configs:
        if use_triton and not triton_available:
            print(f"\n[SKIP] {desc} - Triton not available")
            continue
            
        print(f"\n[TEST] {desc}")
        print(f"       n_layers={n_layers}, batch={batch_size}, seq={seq_len}")
        
        result = binary_search_max_params(
            base_d_model=256,
            max_d_model=2048,
            n_layers=n_layers,
            batch_size=batch_size,
            seq_len=seq_len,
            use_triton=use_triton,
            use_fp16=use_fp16,
            use_checkpointing=use_checkpointing
        )
        
        if result:
            results["benchmarks"].append({
                "description": desc,
                "result": result
            })
            print(f"  → MAX: {result['params_millions']:.1f}M params @ d_model={result['config']['d_model']}")
        else:
            print(f"  → Failed to find working configuration")
    
    # 推奨設定を決定
    print("\n" + "=" * 70)
    print("RECOMMENDED CONFIGURATIONS")
    print("=" * 70)
    
    # Chat AI用の最適設定を見つける
    chat_results = [r for r in results["benchmarks"] if "Chat AI" in r["description"] and r["result"]["success"]]
    if chat_results:
        best_chat = max(chat_results, key=lambda x: x["result"]["params_millions"])
        results["recommended_chat_config"] = best_chat["result"]["config"]
        print(f"\nChat AI Recommended:")
        print(f"  d_model: {best_chat['result']['config']['d_model']}")
        print(f"  n_layers: {best_chat['result']['config']['n_layers']}")
        print(f"  Parameters: {best_chat['result']['params_millions']:.1f}M")
        print(f"  VRAM Usage: {best_chat['result']['vram_allocated_gb']:.2f}GB")
    
    # 結果を保存
    output_path = Path("results/benchmarks/phase7_max_params_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")
    
    return results

if __name__ == "__main__":
    results = run_comprehensive_benchmark()
