#!/usr/bin/env python3
"""
Production Checkpoint Profiler
===============================
本番と同じコードフローでチェックポイント保存を行い、
保存前後の速度変化を詳細にログする。

本番で起きている現象:
- 500 step → pt保存 → 古いファイル削除 → メモリクリーンアップ
- 残り時間: 261時間 → 700時間 → 1000時間+ に増加

このプロファイラは:
1. 実際のモデル/オプティマイザを使用
2. 5 stepごとにptファイルを保存（短縮版）
3. 各処理の詳細なタイミングをログ
4. 保存後の5 stepで速度低下を検出

Usage:
    python scripts/profile_production_checkpoint.py
"""

import os
import sys
import time
import gc
import torch
import torch.nn as nn
from datetime import datetime
from dataclasses import asdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TimingLogger:
    """詳細なタイミングロガー"""
    
    def __init__(self):
        self.logs = []
        self.current_step = 0
        
    def log(self, event: str, duration_ms: float = None, extra: dict = None):
        """イベントをログ"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'step': self.current_step,
            'event': event,
        }
        if duration_ms is not None:
            entry['duration_ms'] = round(duration_ms, 2)
        if extra:
            entry.update(extra)
        
        self.logs.append(entry)
        
        # リアルタイム出力
        if duration_ms is not None:
            print(f"  [{self.current_step:3d}] {event:40s} {duration_ms:8.1f}ms")
        else:
            print(f"  [{self.current_step:3d}] {event}")
    
    def timed(self, event: str):
        """タイミング測定コンテキストマネージャ"""
        class Timer:
            def __init__(self, logger, event):
                self.logger = logger
                self.event = event
                self.start = None
                self.elapsed_ms = 0
                
            def __enter__(self):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.start = time.perf_counter()
                return self
                
            def __exit__(self, *args):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.elapsed_ms = (time.perf_counter() - self.start) * 1000
                self.logger.log(self.event, self.elapsed_ms)
                
        return Timer(self, event)


def cleanup_old_checkpoints(save_dir: str, max_keep: int, logger: TimingLogger):
    """古いチェックポイントを削除（本番と同じロジック）"""
    import glob
    
    with logger.timed("Listing checkpoint files"):
        pattern = os.path.join(save_dir, "step_*.pt")
        checkpoints = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    
    deleted = 0
    for old_ckpt in checkpoints[max_keep:]:
        with logger.timed(f"Deleting {os.path.basename(old_ckpt)}"):
            try:
                os.remove(old_ckpt)
                deleted += 1
            except Exception as e:
                logger.log(f"Delete failed: {e}")
    
    return deleted


def save_checkpoint_with_profiling(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    config: dict,
    logger: TimingLogger,
):
    """チェックポイント保存（各処理を詳細にプロファイル）"""
    
    logger.log("=== CHECKPOINT SAVE START ===")
    
    # ディレクトリ作成
    with logger.timed("mkdir"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # モデルの_orig_modチェック（torch.compile対応）
    model_to_save = model
    has_orig_mod = hasattr(model, '_orig_mod')
    if has_orig_mod:
        with logger.timed("Access _orig_mod"):
            model_to_save = model._orig_mod
        logger.log(f"Using _orig_mod: True")
    else:
        logger.log(f"Using _orig_mod: False (not compiled)")
    
    # model.state_dict()
    with logger.timed("model.state_dict()"):
        model_state = model_to_save.state_dict()
    
    # optimizer.state_dict()
    with logger.timed("optimizer.state_dict()"):
        optimizer_state = optimizer.state_dict()
    
    # scaler.state_dict()
    with logger.timed("scaler.state_dict()"):
        scaler_state = scaler.state_dict()
    
    # チェックポイント構築
    with logger.timed("Build checkpoint dict"):
        checkpoint = {
            'step': step,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'scaler_state_dict': scaler_state,
            'config': config,
        }
    
    # torch.save()
    with logger.timed("torch.save()"):
        torch.save(checkpoint, path)
    
    # 明示的にメモリ解放
    with logger.timed("Del checkpoint refs"):
        del checkpoint, model_state, optimizer_state, scaler_state
    
    logger.log("=== CHECKPOINT SAVE END ===")


def memory_cleanup_with_profiling(logger: TimingLogger):
    """メモリクリーンアップ（本番と同じロジック）"""
    
    logger.log("=== MEMORY CLEANUP START ===")
    
    # GC 3回
    for i in range(3):
        with logger.timed(f"gc.collect() #{i+1}"):
            gc.collect()
    
    # CUDA キャッシュクリア
    if torch.cuda.is_available():
        with logger.timed("torch.cuda.empty_cache()"):
            torch.cuda.empty_cache()
        
        with logger.timed("torch.cuda.synchronize()"):
            torch.cuda.synchronize()
        
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            with logger.timed("reset_peak_memory_stats()"):
                torch.cuda.reset_peak_memory_stats()
    
    logger.log("=== MEMORY CLEANUP END ===")


def run_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    x: torch.Tensor,
    y: torch.Tensor,
    logger: TimingLogger,
    use_amp: bool = True,
):
    """1ステップの訓練（本番と同じロジック）"""
    
    step_start = time.perf_counter()
    
    # Forward
    with logger.timed("Forward"):
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(x)
            if isinstance(output, tuple):
                logits = output[0]
            elif isinstance(output, dict):
                logits = output.get('logits', list(output.values())[0])
            else:
                logits = output
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    
    # Zero grad
    with logger.timed("zero_grad"):
        optimizer.zero_grad()
    
    # Backward
    with logger.timed("Backward"):
        scaler.scale(loss).backward()
    
    # Unscale
    with logger.timed("unscale_"):
        scaler.unscale_(optimizer)
    
    # Grad clip
    with logger.timed("clip_grad_norm_"):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Optimizer step
    with logger.timed("optimizer.step"):
        scaler.step(optimizer)
    
    # Scaler update
    with logger.timed("scaler.update"):
        scaler.update()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    step_time = (time.perf_counter() - step_start) * 1000
    return loss.item(), step_time


def run_production_profiler():
    """本番環境再現プロファイラ"""
    
    print("=" * 70)
    print("🔬 Production Checkpoint Profiler")
    print("=" * 70)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Import (same as production)
    from src.models.phase8.integrated_model import Phase8IntegratedModel
    from src.models.phase8.config import Phase8Config
    from scripts.train_phase8 import Phase8TrainingConfig
    
    # Config (same as production)
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
    
    # Create model
    print("Creating model...")
    model = Phase8IntegratedModel(model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    
    # Optimizer (same as production - AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    scaler = torch.cuda.amp.GradScaler()
    
    # Mock data
    x = torch.randint(0, 50256, (1, 512), device=device)
    y = torch.randint(0, 50256, (1, 512), device=device)
    
    # Temp checkpoint dir
    save_dir = "checkpoints/profiler_temp"
    os.makedirs(save_dir, exist_ok=True)
    
    # Logger
    logger = TimingLogger()
    
    # === Phase 1: Warmup (3 steps) ===
    print()
    print("=" * 70)
    print("Phase 1: Warmup (3 steps)")
    print("=" * 70)
    
    for i in range(3):
        logger.current_step = i
        loss, step_time = run_training_step(model, optimizer, scaler, x, y, logger)
        print(f"  >>> Step {i} total: {step_time:.1f}ms, loss: {loss:.4f}")
        print()
    
    # === Phase 2: Pre-save (3 steps) ===
    print()
    print("=" * 70)
    print("Phase 2: Pre-save baseline (3 steps)")
    print("=" * 70)
    
    pre_save_times = []
    for i in range(3, 6):
        logger.current_step = i
        loss, step_time = run_training_step(model, optimizer, scaler, x, y, logger)
        pre_save_times.append(step_time)
        print(f"  >>> Step {i} total: {step_time:.1f}ms")
        print()
    
    avg_pre = sum(pre_save_times) / len(pre_save_times)
    print(f"  Pre-save average: {avg_pre:.1f}ms")
    
    # === Phase 3: Checkpoint Save (exactly like production) ===
    print()
    print("=" * 70)
    print("Phase 3: CHECKPOINT SAVE (Production flow)")
    print("=" * 70)
    
    logger.current_step = 6
    
    # 3.1: Save checkpoint
    ckpt_path = os.path.join(save_dir, f"step_{6}.pt")
    save_checkpoint_with_profiling(
        ckpt_path, model, optimizer, scaler, 6,
        asdict(config), logger
    )
    
    # 3.2: Delete old checkpoints (keep 2)
    print()
    deleted = cleanup_old_checkpoints(save_dir, max_keep=2, logger=logger)
    if deleted > 0:
        logger.log(f"Deleted {deleted} old checkpoint(s)")
    
    # 3.3: Memory cleanup
    print()
    memory_cleanup_with_profiling(logger)
    
    print()
    print("  >>> Checkpoint save complete")
    
    # === Phase 4: Post-save (5 steps) - CRITICAL ===
    print()
    print("=" * 70)
    print("Phase 4: POST-SAVE (5 steps) - DETECTING SLOWDOWN")
    print("=" * 70)
    
    post_save_times = []
    for i in range(7, 12):
        logger.current_step = i
        loss, step_time = run_training_step(model, optimizer, scaler, x, y, logger)
        post_save_times.append(step_time)
        
        # 速度低下の検出
        slowdown = step_time / avg_pre
        indicator = "⚠️  SLOW!" if slowdown > 1.5 else "✅"
        print(f"  >>> Step {i} total: {step_time:.1f}ms ({slowdown:.2f}x) {indicator}")
        print()
    
    avg_post = sum(post_save_times) / len(post_save_times)
    overall_slowdown = avg_post / avg_pre
    
    # === Summary ===
    print()
    print("=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"  Pre-save average:  {avg_pre:.1f}ms")
    print(f"  Post-save average: {avg_post:.1f}ms")
    print(f"  Overall slowdown:  {overall_slowdown:.2f}x")
    print()
    
    if overall_slowdown > 1.5:
        print("  ⚠️  SIGNIFICANT SLOWDOWN DETECTED!")
        print("  Check the detailed logs above to identify the bottleneck.")
    else:
        print("  ✅ No significant slowdown in this test.")
        print("  The production slowdown may be caused by:")
        print("    - Accumulation over many checkpoints")
        print("    - DataLoader issues")
        print("    - Memory fragmentation over time")
    
    # Cleanup temp files
    print()
    print("Cleaning up temp files...")
    import shutil
    try:
        shutil.rmtree(save_dir)
    except:
        pass
    
    print()
    print("=" * 70)
    print("✅ Profiling Complete")
    print("=" * 70)


if __name__ == '__main__':
    run_production_profiler()
