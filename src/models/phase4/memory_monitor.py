import torch
import os
from typing import Tuple

class MemoryMonitor:
    """
    Monitors VRAM usage and provides safe fallbacks for CPU environments.
    Supports mocking for testing memory-dependent logic.
    """

    def __init__(self, mock_limit_gb: float = 7.5, mode: str = 'auto'):
        """
        Args:
            mock_limit_gb: Limit for mock mode in GB.
            mode: 'auto', 'cuda', or 'mock'.
        """
        self.mode = mode
        self.mock_limit_bytes = int(mock_limit_gb * 1024**3)
        self.current_mock_usage_bytes = 0

        if self.mode == 'auto':
            self.use_mock = not torch.cuda.is_available()
        elif self.mode == 'mock':
            self.use_mock = True
        else:
            self.use_mock = False

    def get_free_memory(self) -> int:
        """Returns free memory in bytes."""
        if self.use_mock:
            return max(0, self.mock_limit_bytes - self.current_mock_usage_bytes)
        else:
            try:
                free, total = torch.cuda.mem_get_info()
                return free
            except (AssertionError, RuntimeError):
                # Fallback if cuda is initialized but fails or not available unexpectedly
                return self.mock_limit_bytes

    def get_total_memory(self) -> int:
        """Returns total memory in bytes."""
        if self.use_mock:
            return self.mock_limit_bytes
        else:
            try:
                free, total = torch.cuda.mem_get_info()
                return total
            except (AssertionError, RuntimeError):
                return self.mock_limit_bytes

    def get_memory_stats(self) -> dict:
        """Returns dictionary with memory statistics."""
        free = self.get_free_memory()
        total = self.get_total_memory()
        used = total - free
        return {
            'free_mb': free / (1024**2),
            'total_mb': total / (1024**2),
            'used_mb': used / (1024**2),
            'percent_used': (used / total) * 100 if total > 0 else 0
        }

    def set_mock_usage(self, usage_gb: float):
        """Sets the simulated used memory (for testing)."""
        self.current_mock_usage_bytes = int(usage_gb * 1024**3)

    def reset_mock(self):
        """Resets mock usage to 0."""
        self.current_mock_usage_bytes = 0
