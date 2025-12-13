"""
Revolutionary Trainer - Unified Integration of All Revolutionary Algorithms

Integrates 7 passing revolutionary algorithms into the training pipeline:
1. Holographic Weight Synthesis (CUDA kernel)
2. BK-Core Closed-Form Solution
3. Topological Training Collapse
4. Retrocausal Learning (Delta Rule FWP)
5. Riemann Zeta Resonance
6. Sheaf Cohomology Compilation
7. Diffractive Weight Optics

Usage:
    from src.training.revolutionary_trainer import RevolutionaryTrainer
    
    trainer = RevolutionaryTrainer(model, config)
    for batch in dataloader:
        loss, metrics = trainer.train_step(batch)

Author: Project MUSE Team
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import time

# Import revolutionary algorithms
from src.training.holographic_training import HolographicWeightSynthesis
from src.training.closed_form_training import BKCoreClosedFormOptimizer
from src.training.topological_optimizer import TopologicalTrainingCollapse
from src.training.retrocausal_learning import RetrocausalLearning
from src.training.zeta_resonance import RiemannZetaResonance
from src.training.sheaf_compilation import SheafCohomologyCompilation
from src.training.diffractive_optics import DiffractiveWeightOptics


@dataclass
class RevolutionaryConfig:
    """Configuration for revolutionary training."""
    # Algorithm enables
    use_holographic: bool = True
    use_closed_form: bool = True
    use_topological: bool = True
    use_retrocausal: bool = True
    use_zeta: bool = True
    use_sheaf: bool = True
    use_diffractive: bool = True
    
    # Hyperparameters
    learning_rate: float = 0.001
    holographic_lr: float = 0.01
    topological_skip_threshold: float = 0.9
    retrocausal_beta: float = 0.1
    diffractive_steps: int = 10
    
    # Scheduling - FIXED: Use absolute steps instead of percentages
    # This ensures proper resume from checkpoints
    total_steps: int = 195312  # Total training steps (for phase calculation)
    warmup_steps: int = 100
    algorithm_cycle: int = 10  # Cycle through algorithms every N steps
    
    # Phase-based auto-scheduling thresholds (as fraction of total_steps)
    # - Warmup (0-10%): OFF (focus on stability)
    # - Early (10-30%): holographic, closed_form only
    # - Mid (30-70%): + topological, zeta
    # - Late (70-100%): ALL algorithms enabled
    phase_warmup_end: float = 0.10
    phase_early_end: float = 0.30
    phase_mid_end: float = 0.70
    
    # Logging
    log_interval: int = 10


class RevolutionaryTrainer:
    """
    Unified trainer that integrates all revolutionary algorithms.
    
    Algorithms are applied in a complementary way:
    - Holographic: Fast weight synthesis via FFT
    - Closed-form: Skip gradient descent when possible
    - Topological: Skip redundant computation regions
    - Retrocausal: Future gradient prediction
    - Zeta: Dimensional reduction via resonance
    - Sheaf: Coherent weight compilation
    - Diffractive: Optical-inspired optimization
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: RevolutionaryConfig = None,
        device: torch.device = None,
    ):
        self.model = model
        self.config = config or RevolutionaryConfig()
        self.device = device or next(model.parameters()).device
        
        # Standard optimizer as fallback
        self.base_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )
        
        # Initialize revolutionary algorithms
        self._init_algorithms()
        
        # Warmup mode - skip weight modifications during warmup for stability
        self._warmup_mode = False
        
        # Metrics
        self.step_count = 0
        self.metrics_history = []
    
    def _init_algorithms(self):
        """Initialize all revolutionary algorithm modules."""
        cfg = self.config
        
        # 1. Holographic
        if cfg.use_holographic:
            self.holographic = HolographicWeightSynthesis(
                self.model,
                learning_rate=cfg.holographic_lr,
            )
        else:
            self.holographic = None
        
        # 2. Closed-form (BK-Core)
        if cfg.use_closed_form:
            self.closed_form = BKCoreClosedFormOptimizer(
                self.model,
            )
        else:
            self.closed_form = None
        
        # 3. Topological
        if cfg.use_topological:
            self.topological = TopologicalTrainingCollapse(
                self.model,
            )
        else:
            self.topological = None
        
        # 4. Retrocausal
        if cfg.use_retrocausal:
            self.retrocausal = RetrocausalLearning(
                self.model,
            )
        else:
            self.retrocausal = None
        
        # 5. Zeta
        if cfg.use_zeta:
            self.zeta = RiemannZetaResonance(
                self.model,
            )
        else:
            self.zeta = None
        
        # 6. Sheaf
        if cfg.use_sheaf:
            self.sheaf = SheafCohomologyCompilation(
                self.model,
            )
        else:
            self.sheaf = None
        
        # 7. Diffractive
        if cfg.use_diffractive:
            self.diffractive = DiffractiveWeightOptics(
                self.model,
                learning_rate=cfg.learning_rate,
                optimization_steps=cfg.diffractive_steps,
            )
        else:
            self.diffractive = None
    
    def _get_training_phase(self) -> str:
        """
        Determine current training phase based on global step.
        
        Uses absolute step counts (not percentages) to ensure proper
        behavior when resuming from checkpoints.
        
        Returns:
            'warmup', 'early', 'mid', or 'late'
        """
        cfg = self.config
        total = cfg.total_steps
        step = self.step_count
        
        # Calculate thresholds as absolute steps
        warmup_end = int(total * cfg.phase_warmup_end)
        early_end = int(total * cfg.phase_early_end)
        mid_end = int(total * cfg.phase_mid_end)
        
        if step < warmup_end:
            return 'warmup'
        elif step < early_end:
            return 'early'
        elif step < mid_end:
            return 'mid'
        else:
            return 'late'
    
    def _get_enabled_algorithms_for_phase(self, phase: str) -> list:
        """
        Get list of enabled algorithms for the current phase.
        
        Phase schedule:
        - Warmup (0-10%): OFF (focus on stability)
        - Early (10-30%): holographic, closed_form only
        - Mid (30-70%): + topological, zeta
        - Late (70-100%): ALL algorithms enabled
        """
        if phase == 'warmup':
            return []  # No algorithms during warmup
        
        enabled = []
        
        # Early phase: basic algorithms
        if phase in ('early', 'mid', 'late'):
            if self.holographic:
                enabled.append(('holographic', self.holographic))
            if self.closed_form:
                enabled.append(('closed_form', self.closed_form))
        
        # Mid phase: add topological and zeta
        if phase in ('mid', 'late'):
            if self.topological:
                enabled.append(('topological', self.topological))
            if self.zeta:
                enabled.append(('zeta', self.zeta))
        
        # Late phase: all algorithms
        if phase == 'late':
            if self.retrocausal:
                enabled.append(('retrocausal', self.retrocausal))
            if self.sheaf:
                enabled.append(('sheaf', self.sheaf))
            if self.diffractive:
                enabled.append(('diffractive', self.diffractive))
        
        return enabled
    
    def _select_algorithm(self) -> str:
        """
        Select which algorithm to use based on training phase and step count.
        
        FIXED: Uses absolute global_step instead of percentages to ensure
        proper resume from checkpoints.
        """
        phase = self._get_training_phase()
        enabled = self._get_enabled_algorithms_for_phase(phase)
        
        if not enabled:
            return 'base'  # Fallback during warmup
        
        # Cycle through enabled algorithms
        cycle = self.config.algorithm_cycle
        idx = (self.step_count // cycle) % len(enabled)
        return enabled[idx][0]
    
    def set_warmup_mode(self, enabled: bool):
        """Enable/disable warmup mode. In warmup mode, all weight modifications are skipped."""
        self._warmup_mode = enabled
    
    def train_step(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: nn.Module = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Execute one training step with revolutionary algorithms.
        
        Args:
            data: Input batch
            targets: Target labels
            loss_fn: Loss function (default: CrossEntropyLoss)
        
        Returns:
            (loss, metrics_dict)
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        
        # Ensure data dtype matches model dtype (fixes float vs bfloat16 mismatch)
        model_dtype = next(self.model.parameters()).dtype
        if data.dtype != model_dtype and data.dtype in (torch.float32, torch.float16, torch.bfloat16):
            data = data.to(model_dtype)
        
        # Skip weight modifications during warmup for stability
        if self._warmup_mode:
            with torch.no_grad():
                outputs = self.model(data)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                if outputs.dim() == 3:
                    outputs = outputs.view(-1, outputs.size(-1))
                    targets = targets.view(-1)
                loss = loss_fn(outputs, targets)
            return loss, {'algorithm': 'warmup_skip', 'skipped': True, 'step': self.step_count}
        
        start_time = time.perf_counter()
        
        # Select algorithm
        algo_name = self._select_algorithm()
        
        metrics = {
            'step': self.step_count,
            'algorithm': algo_name,
        }
        
        # Execute selected algorithm
        if algo_name == 'holographic' and self.holographic:
            loss, algo_metrics = self.holographic.synthesize(data, targets, loss_fn)
            metrics.update(algo_metrics)
            
        elif algo_name == 'closed_form' and self.closed_form:
            loss, algo_metrics = self.closed_form.train_one_shot(data, targets, loss_fn)
            metrics.update(algo_metrics)
            
        elif algo_name == 'topological' and self.topological:
            loss, algo_metrics = self.topological.collapse_to_global(data, targets, loss_fn)
            metrics.update(algo_metrics)
            
        elif algo_name == 'retrocausal' and self.retrocausal:
            loss, algo_metrics = self.retrocausal.train_retrocausal(data, targets, loss_fn)
            metrics.update(algo_metrics)
            
        elif algo_name == 'zeta' and self.zeta:
            loss, algo_metrics = self.zeta.optimize_via_zeta(data, targets, loss_fn)
            metrics.update(algo_metrics)
            
        elif algo_name == 'sheaf' and self.sheaf:
            loss, algo_metrics = self.sheaf.compile_to_zero_cohomology(data, targets, loss_fn)
            metrics.update(algo_metrics)
            
        elif algo_name == 'diffractive' and self.diffractive:
            loss, algo_metrics = self.diffractive.synthesize_weights(data, targets, loss_fn)
            metrics.update(algo_metrics)
            
        else:
            # Fallback to standard training
            loss = self._standard_step(data, targets, loss_fn)
            metrics['fallback'] = True
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics['step_time_ms'] = elapsed_ms
        
        self.step_count += 1
        self.metrics_history.append(metrics)
        
        # Log periodically
        if self.step_count % self.config.log_interval == 0:
            self._log_metrics(metrics)
        
        return loss, metrics
    
    def _standard_step(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        """Standard gradient descent step."""
        self.base_optimizer.zero_grad()
        
        outputs = self.model(data)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        if outputs.dim() == 3:
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
        
        loss = loss_fn(outputs, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.base_optimizer.step()
        
        return loss
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log training metrics."""
        algo = metrics.get('algorithm', 'unknown')
        step_time = metrics.get('step_time_ms', 0)
        
        print(f"[Step {self.step_count}] {algo}: {step_time:.2f}ms")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        if not self.metrics_history:
            return {}
        
        algo_times = {}
        for m in self.metrics_history:
            algo = m.get('algorithm', 'unknown')
            time_ms = m.get('step_time_ms', 0)
            
            if algo not in algo_times:
                algo_times[algo] = []
            algo_times[algo].append(time_ms)
        
        summary = {
            'total_steps': self.step_count,
            'algorithms': {},
        }
        
        for algo, times in algo_times.items():
            summary['algorithms'][algo] = {
                'count': len(times),
                'avg_time_ms': sum(times) / len(times),
                'min_time_ms': min(times),
                'max_time_ms': max(times),
            }
        
        return summary
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Return trainer state for checkpointing.
        
        CRITICAL: This must be saved/loaded to ensure proper resume behavior.
        Without this, step_count resets to 0 and phase-based scheduling breaks.
        """
        return {
            'step_count': self.step_count,
            'warmup_mode': self._warmup_mode,
            # Save config for validation on load
            'total_steps': self.config.total_steps,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load trainer state from checkpoint.
        
        Restores step_count so phase-based scheduling works correctly
        when resuming training.
        """
        self.step_count = state_dict.get('step_count', 0)
        self._warmup_mode = state_dict.get('warmup_mode', False)
        
        # Log phase info on restore
        phase = self._get_training_phase()
        enabled = self._get_enabled_algorithms_for_phase(phase)
        algo_names = [name for name, _ in enabled] if enabled else ['base']
        print(f"✔ RevolutionaryTrainer restored: step={self.step_count}, phase={phase}, algorithms={algo_names}")


def create_revolutionary_trainer(
    model: nn.Module,
    enable_all: bool = True,
    **kwargs,
) -> RevolutionaryTrainer:
    """Factory function to create RevolutionaryTrainer with common configs."""
    config = RevolutionaryConfig(
        use_holographic=enable_all,
        use_closed_form=enable_all,
        use_topological=enable_all,
        use_retrocausal=enable_all,
        use_zeta=enable_all,
        use_sheaf=enable_all,
        use_diffractive=enable_all,
        **kwargs,
    )
    return RevolutionaryTrainer(model, config)


__all__ = [
    'RevolutionaryConfig',
    'RevolutionaryTrainer',
    'create_revolutionary_trainer',
]
