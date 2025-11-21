import torch
import threading
import time
import warnings
from typing import Optional, Dict, Any, Callable
from src.models.phase4.dream_core.inverse_diffusion import DreamCore

class PassivePipelineManager:
    """
    Manages the Passive Pipeline (Dream Core) for idle-time processing.

    Features:
    - Runs Dream Core in a separate execution context (simulated via thread).
    - Integrates new concepts into Topological Memory after Ethical Filtering.
    - Handles JIT compilation of the Dream Core for efficiency (optional).
    """

    def __init__(
        self,
        dream_core: DreamCore,
        topological_memory: Any, # SparseKnotRepresentation
        ethical_filter: Any, # EthicalFilter
        use_jit: bool = False
    ):
        self.dream_core = dream_core
        self.topological_memory = topological_memory
        self.ethical_filter = ethical_filter

        # Try JIT compilation
        if use_jit:
            try:
                # Note: DreamCore uses torch.utils.checkpoint which is hard to JIT.
                # We expect users to disable JIT if they use checkpointing features that aren't supported.
                self.dream_core_net = torch.jit.script(dream_core)
            except Exception as e:
                warnings.warn(f"DreamCore JIT compilation failed: {e}. Using standard eager mode.")
                self.dream_core_net = dream_core
        else:
            self.dream_core_net = dream_core

        self.is_running = False
        self.thread = None

    def start_passive_loop(
        self,
        memory_provider_func: Callable[[], Optional[torch.Tensor]],
        interval: float = 1.0
    ):
        """
        Start the passive loop in a background thread.

        Args:
            memory_provider_func: Function that returns (n_fragments, d_model) tensor.
            interval: Sleep interval between dreams (seconds).
        """
        if self.is_running:
            return

        self.is_running = True
        self.thread = threading.Thread(
            target=self._loop,
            args=(memory_provider_func, interval),
            daemon=True
        )
        self.thread.start()
        print("Passive Pipeline started (Dream Core active)")

    def stop_passive_loop(self):
        self.is_running = False
        if self.thread:
            self.thread.join()
        print("Passive Pipeline stopped")

    def _loop(self, memory_provider_func, interval):
        while self.is_running:
            try:
                fragments = memory_provider_func()
                if fragments is not None:
                    self.generate_dream(fragments)
            except Exception as e:
                print(f"Error in passive loop: {e}")

            time.sleep(interval)

    def generate_dream(
        self,
        memory_fragments: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Generate a dream and integrate it if ethical.

        Args:
            memory_fragments: (n, d)

        Returns:
            new_concept: (d,) or None if rejected
        """
        # 1. Generate Dream
        # Ensure we are in eval mode for inference
        was_training = self.dream_core.training
        self.dream_core.eval()

        try:
            with torch.no_grad():
                # Use the potentially JIT-ed model
                # Note: We pass None for initial_state explicitly if needed,
                # but JIT might require typed inputs. DreamCore.forward definition handles Optional.
                new_concept, _ = self.dream_core_net(memory_fragments)
        except Exception as e:
            print(f"Dream generation failed: {e}")
            self.dream_core.train(was_training)
            return None

        self.dream_core.train(was_training)

        # 2. Ethical Filter
        if self.ethical_filter.check(new_concept):
            # 3. Integrate into Memory
            metadata = {
                "source": "dream",
                "timestamp": time.time(),
                "type": "concept"
            }
            # add_knot might be async or sync. We call it directly.
            self.topological_memory.add_knot(new_concept, metadata)
            return new_concept
        else:
            # Rejected
            # print("Dream rejected by ethical filter")
            return None
