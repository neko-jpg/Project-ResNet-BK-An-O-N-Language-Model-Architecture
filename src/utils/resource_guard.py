import torch
import logging
import gc

logger = logging.getLogger(__name__)

class ResourceGuard:
    """
    Monitors GPU memory and safeguards against OOM by suggesting batch size adjustments.
    Constraints: 8GB (Mobile/Consumer), 10GB (RTX 3080).
    """
    def __init__(self, memory_limit_gb: float = 9.5, low_memory_threshold_gb: float = 1.0):
        self.memory_limit_bytes = memory_limit_gb * 1024**3
        self.low_memory_threshold_bytes = low_memory_threshold_gb * 1024**3
        self.oom_history = 0

    def check_memory(self) -> bool:
        """
        Check if memory is within safe limits.
        Returns True if safe, False if critical (need to reduce load).
        """
        if not torch.cuda.is_available():
            return True

        try:
            free, total = torch.cuda.mem_get_info()
            used = total - free

            # Check absolute limit if explicitly set (e.g. strictly < 8GB)
            # We assume the limit provided in init is the target max usage.
            if used > self.memory_limit_bytes:
                logger.warning(f"ResourceGuard: Memory usage {used/1024**3:.2f}GB exceeds limit {self.memory_limit_bytes/1024**3:.2f}GB")
                return False

            # Check headroom
            if free < self.low_memory_threshold_bytes:
                logger.warning(f"ResourceGuard: Low memory headroom {free/1024**3:.2f}GB < {self.low_memory_threshold_bytes/1024**3:.2f}GB")
                return False
        except Exception:
            # If mem_get_info fails, assume safe but log warning
            return True

        return True

    def handle_oom(self, current_batch_size: int) -> int:
        """
        Suggest new batch size after OOM or critical memory state.
        """
        self.oom_history += 1

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        new_bs = max(1, current_batch_size // 2)
        logger.warning(f"ResourceGuard: OOM detected or imminent. Reducing batch size: {current_batch_size} -> {new_bs}")
        return new_bs

    def check_and_adjust(self, current_batch_size: int) -> int:
        """
        Proactive check. If critical, suggest reduction.
        """
        if not self.check_memory():
            return self.handle_oom(current_batch_size)
        return current_batch_size
