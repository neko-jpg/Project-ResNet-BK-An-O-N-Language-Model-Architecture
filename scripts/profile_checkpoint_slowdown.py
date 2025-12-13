#!/usr/bin/env python3
"""
Checkpoint Slowdown Profiler
=============================
本番と同じ環境で訓練し、チェックポイント保存前後の速度を
詳細にプロファイリングして、どの処理が遅延の原因かを特定する。

Usage:
    python scripts/profile_checkpoint_slowdown.py

出力:
    - 各処理のタイミング（forward, backward, optimizer, state_dict等）
    - メモリ使用量の変化
    - torch.compile グラフの状態
"""

import os
import sys
import time
import gc
import tempfile
import torch
import torch.nn as nn
from datetime import datetime
from dataclasses import asdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DetailedTimer:
    """詳細なタイミング計測"""
    
    def __init__(self, name: str, sync_cuda: bool = True):
        self.name = name
        self.sync_cuda = sync_cuda
        self.start_time = None
        self.elapsed_ms = 0
        
    def __enter__(self):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000


def get_memory_stats():
    """GPU/CPUメモリ統計を取得"""
    stats = {}
    if torch.cuda.is_available():
        stats['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
        stats['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024**2
        stats['gpu_max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024**2
    return stats


def profile_step(model, optimizer, scaler, x, y, step_num: int, use_amp: bool = True):
    """1ステップの詳細プロファイリング"""
    
    result = {'step': step_num, 'timings': {}}
    
    # Forward
    with DetailedTimer('forward') as t:
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(x)
            # Handle various output types
            if isinstance(output, tuple):
                logits = output[0]  # First element is usually logits
            elif isinstance(output, dict):
                logits = output.get('logits', output.get('output', list(output.values())[0]))
            else:
                logits = output
            # Ensure tensor
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    result['timings']['forward_ms'] = t.elapsed_ms
    result['loss'] = loss.item()
    
    # Backward
    with DetailedTimer('backward') as t:
        optimizer.zero_grad()
        scaler.scale(loss).backward()
    result['timings']['backward_ms'] = t.elapsed_ms
    
    # Unscale
    with DetailedTimer('unscale') as t:
        scaler.unscale_(optimizer)
    result['timings']['unscale_ms'] = t.elapsed_ms
    
    # Grad clip
    with DetailedTimer('grad_clip') as t:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    result['timings']['grad_clip_ms'] = t.elapsed_ms
    
    # Optimizer step
    with DetailedTimer('optimizer_step') as t:
        scaler.step(optimizer)
    result['timings']['optimizer_step_ms'] = t.elapsed_ms
    
    # Scaler update
    with DetailedTimer('scaler_update') as t:
        scaler.update()
    result['timings']['scaler_update_ms'] = t.elapsed_ms
    
    # Total step time
    result['timings']['total_ms'] = sum(result['timings'].values())
    result['memory'] = get_memory_stats()
    
    return result


def profile_checkpoint_save(model, optimizer, scaler, scheduler, path: str, 
                           use_orig_mod: bool = True):
    """チェックポイント保存の詳細プロファイリング"""
    
    result = {'path': path, 'timings': {}}
    
    # _orig_modの使用を確認
    model_to_save = model
    has_orig_mod = hasattr(model, '_orig_mod')
    result['has_orig_mod'] = has_orig_mod
    
    if use_orig_mod and has_orig_mod:
        model_to_save = model._orig_mod
        result['used_orig_mod'] = True
    else:
        result['used_orig_mod'] = False
    
    # model.state_dict()
    with DetailedTimer('model_state_dict') as t:
        model_state = model_to_save.state_dict()
    result['timings']['model_state_dict_ms'] = t.elapsed_ms
    
    # optimizer.state_dict()
    with DetailedTimer('optimizer_state_dict') as t:
        optimizer_state = optimizer.state_dict()
    result['timings']['optimizer_state_dict_ms'] = t.elapsed_ms
    
    # scaler.state_dict()
    with DetailedTimer('scaler_state_dict') as t:
        scaler_state = scaler.state_dict()
    result['timings']['scaler_state_dict_ms'] = t.elapsed_ms
    
    # scheduler.state_dict()
    with DetailedTimer('scheduler_state_dict') as t:
        scheduler_state = scheduler.state_dict() if scheduler else {}
    result['timings']['scheduler_state_dict_ms'] = t.elapsed_ms
    
    # Build checkpoint dict
    with DetailedTimer('build_checkpoint') as t:
        checkpoint = {
            'step': 0,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'scaler_state_dict': scaler_state,
            'scheduler_state_dict': scheduler_state,
        }
    result['timings']['build_checkpoint_ms'] = t.elapsed_ms
    
    # torch.save()
    with DetailedTimer('torch_save') as t:
        torch.save(checkpoint, path)
    result['timings']['torch_save_ms'] = t.elapsed_ms
    
    # Cleanup
    with DetailedTimer('cleanup') as t:
        del checkpoint, model_state, optimizer_state, scaler_state, scheduler_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result['timings']['cleanup_ms'] = t.elapsed_ms
    
    result['timings']['total_ms'] = sum(result['timings'].values())
    result['memory_after'] = get_memory_stats()
    
    return result


def run_profiling():
    """メインプロファイリング実行"""
    
    print("=" * 70)
    print("🔬 Checkpoint Slowdown Profiler")
    print("=" * 70)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Import model (same as training)
    from src.models.phase8.integrated_model import Phase8IntegratedModel
    from src.models.phase8.config import Phase8Config
    from scripts.train_phase8 import Phase8TrainingConfig, CosineWarmupScheduler
    
    # Config (same as training)
    config = Phase8TrainingConfig()
    model_config = Phase8Config(
        vocab_size=50256,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_seq=config.n_seq,
        num_heads=config.num_heads,
        htt_rank=config.htt_rank,
        hyperbolic_window_size=config.hyperbolic_window_size,
        use_bk_hyperbolic=config.use_bk_hyperbolic,
        use_ar_ssm_fusion=config.use_ar_ssm_fusion,
        low_rank_ffn=config.low_rank_ffn,
        low_rank_attention=config.low_rank_attention,
        low_rank_rank=config.low_rank_rank,
        use_bitnet=config.use_bitnet,
    )
    
    print(f"Model: {config.d_model}d x {config.n_layers}L")
    print()
    
    # Create model
    print("Creating model...")
    model = Phase8IntegratedModel(model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    # Test with and without torch.compile
    for use_compile in [False, True]:
        print()
        print("=" * 70)
        print(f"🧪 Testing with torch.compile = {use_compile}")
        print("=" * 70)
        
        # Fresh model for each test
        test_model = Phase8IntegratedModel(model_config).to(device)
        
        if use_compile:
            try:
                print("Applying torch.compile()...")
                test_model = torch.compile(test_model, mode="reduce-overhead", fullgraph=False)
                print("  ✔ torch.compile() applied")
            except Exception as e:
                print(f"  ✘ torch.compile() failed: {e}")
                continue
        
        # Optimizer
        optimizer = torch.optim.AdamW(test_model.parameters(), lr=0.05)
        scaler = torch.cuda.amp.GradScaler()
        
        # Mock scheduler
        class MockScheduler:
            def state_dict(self):
                return {'step': 0}
        scheduler = MockScheduler()
        
        # Mock data
        x = torch.randint(0, 50256, (1, 512), device=device)
        y = torch.randint(0, 50256, (1, 512), device=device)
        
        # Warmup steps
        print()
        print("Phase 1: Warmup (3 steps)")
        print("-" * 50)
        for i in range(3):
            result = profile_step(test_model, optimizer, scaler, x, y, i)
            print(f"  Step {i}: {result['timings']['total_ms']:.1f}ms, loss={result['loss']:.4f}")
        
        # Pre-save steps
        print()
        print("Phase 2: Pre-save (3 steps)")
        print("-" * 50)
        pre_save_times = []
        for i in range(3, 6):
            result = profile_step(test_model, optimizer, scaler, x, y, i)
            pre_save_times.append(result['timings']['total_ms'])
            print(f"  Step {i}: {result['timings']['total_ms']:.1f}ms")
        avg_pre = sum(pre_save_times) / len(pre_save_times)
        print(f"  Average: {avg_pre:.1f}ms")
        
        # Checkpoint save
        print()
        print("Phase 3: Checkpoint Save")
        print("-" * 50)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        save_result = profile_checkpoint_save(
            test_model, optimizer, scaler, scheduler, temp_path,
            use_orig_mod=True
        )
        
        print(f"  has _orig_mod: {save_result['has_orig_mod']}")
        print(f"  used _orig_mod: {save_result['used_orig_mod']}")
        print()
        print("  Timing breakdown:")
        for name, ms in save_result['timings'].items():
            if name != 'total_ms':
                pct = ms / save_result['timings']['total_ms'] * 100
                bar = '█' * int(pct / 5)
                print(f"    {name:25s}: {ms:8.1f}ms ({pct:5.1f}%) {bar}")
        print(f"    {'TOTAL':25s}: {save_result['timings']['total_ms']:8.1f}ms")
        
        # Delete temp file
        try:
            os.unlink(temp_path)
        except:
            pass
        
        # Post-save steps
        print()
        print("Phase 4: Post-save (5 steps) - CRITICAL")
        print("-" * 50)
        post_save_times = []
        for i in range(6, 11):
            result = profile_step(test_model, optimizer, scaler, x, y, i)
            post_save_times.append(result['timings']['total_ms'])
            
            # Detail breakdown
            t = result['timings']
            print(f"  Step {i}: {t['total_ms']:8.1f}ms | "
                  f"fwd={t['forward_ms']:6.1f} bwd={t['backward_ms']:6.1f} "
                  f"opt={t['optimizer_step_ms']:6.1f}")
        
        avg_post = sum(post_save_times) / len(post_save_times)
        slowdown = avg_post / avg_pre
        
        print()
        print(f"  Average: {avg_post:.1f}ms")
        print(f"  Slowdown: {slowdown:.2f}x {'⚠️  PROBLEM!' if slowdown > 1.5 else '✅ OK'}")
        
        # Cleanup
        del test_model, optimizer, scaler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print()
    print("=" * 70)
    print("✅ Profiling Complete")
    print("=" * 70)


if __name__ == '__main__':
    run_profiling()
