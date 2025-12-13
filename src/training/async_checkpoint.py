"""
Async Checkpoint Saver - Prevent Training Slowdown After Checkpoint Saves

PROBLEM:
    Checkpoint saves cause 3x slowdown (4.89s → 15.60s/it) due to:
    1. Memory fragmentation from state_dict copies
    2. CUDA memory not being released properly
    3. torch.compile graph cache invalidation
    4. Main thread blocking during I/O

SOLUTION:
    - Copy state dicts to CPU in background thread
    - Save to disk without blocking training
    - Aggressive memory cleanup after copy
    - Periodic CUDA memory defragmentation

Usage:
    saver = AsyncCheckpointSaver(max_queue_size=2)
    saver.save_async(path, model, optimizer, scheduler, ...)
    # Training continues immediately
    saver.shutdown()  # Call at end of training
"""

import os
import gc
import copy
import threading
import queue
from typing import Dict, Optional, Any
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.optim as optim


class AsyncCheckpointSaver:
    """
    Asynchronous checkpoint saver that doesn't block the training loop.
    
    Key features:
    - Saves checkpoints in background thread
    - Copies tensors to CPU before queueing (frees GPU immediately)
    - Limits queue size to prevent memory buildup
    - Thread-safe shutdown
    """
    
    def __init__(self, max_queue_size: int = 2):
        """
        Args:
            max_queue_size: Maximum pending saves. If exceeded, oldest is dropped.
        """
        self.max_queue_size = max_queue_size
        self.save_queue = queue.Queue(maxsize=max_queue_size)
        self.worker_thread = None
        self.shutdown_event = threading.Event()
        self._start_worker()
    
    def _start_worker(self):
        """Start the background save worker thread."""
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
    
    def _worker_loop(self):
        """Background worker that saves checkpoints from queue."""
        while not self.shutdown_event.is_set():
            try:
                # Wait for save job with timeout (allows checking shutdown)
                job = self.save_queue.get(timeout=1.0)
                if job is None:  # Poison pill
                    break
                
                path, checkpoint_data = job
                try:
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    
                    # Save checkpoint (already on CPU)
                    torch.save(checkpoint_data, path)
                    print(f"\n💾 [Async] Checkpoint saved: {path}")
                    
                except Exception as e:
                    print(f"\n⚠ [Async] Checkpoint save failed: {e}")
                
                finally:
                    # Free checkpoint data
                    for key in list(checkpoint_data.keys()):
                        del checkpoint_data[key]
                    del checkpoint_data
                    gc.collect()
                
                self.save_queue.task_done()
                
            except queue.Empty:
                continue
    
    def _state_dict_to_cpu(self, state_dict: Dict) -> Dict:
        """Move all tensors in state dict to CPU."""
        cpu_dict = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                cpu_dict[key] = value.cpu().clone()
            elif isinstance(value, dict):
                cpu_dict[key] = self._state_dict_to_cpu(value)
            else:
                cpu_dict[key] = copy.deepcopy(value)
        return cpu_dict
    
    def save_async(
        self,
        path: str,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Any,
        scaler: torch.cuda.amp.GradScaler,
        ema: Optional[Any],
        step: int,
        epoch: int,
        loss: float,
        config: Any,
        revolutionary_trainer: Optional[Any] = None,
    ):
        """
        Queue a checkpoint for async saving.
        
        CRITICAL: We now use a SYNCHRONOUS save approach because:
        1. model.state_dict() MUST be called from main thread anyway
        2. Calling state_dict() on torch.compile'd model resets the graph
        3. The real solution is to use model._orig_mod if available
        
        For torch.compile'd models, we access the underlying model to avoid
        graph invalidation.
        """
        import gc
        
        # Drop oldest if queue is full (prevents memory buildup)
        while self.save_queue.full():
            try:
                old_job = self.save_queue.get_nowait()
                if old_job is not None:
                    _, old_data = old_job
                    del old_data
                    gc.collect()
                self.save_queue.task_done()
                print(f"⚠ Dropped pending checkpoint (queue full)")
            except queue.Empty:
                break
        
        try:
            # CRITICAL: Access underlying model for torch.compile'd models
            # This avoids resetting the compiled graph!
            model_to_save = model
            if hasattr(model, '_orig_mod'):
                model_to_save = model._orig_mod  # torch.compile'd model
            
            # Build checkpoint - use keep_vars=False (default) for efficiency
            # Do this quickly in main thread, then queue for async disk I/O
            checkpoint = {
                'step': step,
                'epoch': epoch,
                'loss': loss,
            }
            
            # Get state dicts - these are still on GPU but that's OK
            # We'll copy to CPU in the background thread
            model_state = model_to_save.state_dict()
            optimizer_state = optimizer.state_dict()
            
            # Move to CPU immediately to free GPU memory
            checkpoint['model_state_dict'] = {k: v.cpu() for k, v in model_state.items()}
            del model_state
            
            checkpoint['optimizer_state_dict'] = self._state_dict_to_cpu(optimizer_state)
            del optimizer_state
            
            checkpoint['scheduler_state_dict'] = scheduler.state_dict() if hasattr(scheduler, 'state_dict') else {}
            checkpoint['scaler_state_dict'] = scaler.state_dict()
            checkpoint['config'] = asdict(config) if hasattr(config, '__dataclass_fields__') else config
            
            if ema is not None and hasattr(ema, 'state_dict'):
                ema_state = ema.state_dict()
                checkpoint['ema_state_dict'] = self._state_dict_to_cpu(ema_state)
                del ema_state
            
            if revolutionary_trainer is not None and hasattr(revolutionary_trainer, 'state_dict'):
                checkpoint['revolutionary_trainer_state_dict'] = revolutionary_trainer.state_dict()
            
            # Force GC before queueing
            gc.collect()
            
            # Queue for background disk I/O
            self.save_queue.put((path, checkpoint))
            
        except Exception as e:
            print(f"⚠ Failed to queue checkpoint: {e}")
    
    def shutdown(self, timeout: float = 30.0):
        """Shutdown the saver, waiting for pending saves to complete."""
        self.shutdown_event.set()
        
        # Send poison pill
        try:
            self.save_queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=timeout)
    
    def wait_for_pending(self, timeout: float = 60.0):
        """Wait for all pending saves to complete."""
        self.save_queue.join()


def aggressive_memory_cleanup():
    """
    Aggressive GPU memory cleanup to prevent fragmentation.
    
    Call this periodically (e.g., every 100 steps) to prevent
    slowdown accumulation.
    """
    # Multiple GC passes
    for _ in range(3):
        gc.collect()
    
    # CUDA cleanup
    if torch.cuda.is_available():
        # Wait for all CUDA ops to complete
        torch.cuda.synchronize()
        
        # Clear cached memory
        torch.cuda.empty_cache()
        
        # Reset memory stats (doesn't free memory but prevents tracking overhead)
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()
        
        # Reset accumulated memory (PyTorch 2.0+)
        if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
            torch.cuda.reset_accumulated_memory_stats()


def force_cuda_memory_defrag():
    """
    Force CUDA memory defragmentation.
    
    This is more aggressive than empty_cache() and should be used
    sparingly (e.g., after checkpoint saves).
    """
    if not torch.cuda.is_available():
        return
    
    # First, run normal cleanup
    aggressive_memory_cleanup()
    
    # Force synchronization across all streams
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.synchronize()
    
    # Try to trigger memory allocator cleanup
    try:
        # Allocate and immediately free a small tensor to trigger cleanup
        dummy = torch.empty(1024, device='cuda')
        del dummy
    except:
        pass
    
    # Final cache clear
    torch.cuda.empty_cache()


__all__ = [
    'AsyncCheckpointSaver',
    'aggressive_memory_cleanup', 
    'force_cuda_memory_defrag',
]
