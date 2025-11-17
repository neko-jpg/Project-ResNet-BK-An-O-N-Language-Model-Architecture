# ResNet-BK API Reference

Complete API documentation for all ResNet-BK modules, classes, and functions.

---

## Table of Contents

1. [Core Models](#core-models)
2. [Training](#training)
3. [Benchmarks](#benchmarks)
4. [Utilities](#utilities)
5. [Configuration](#configuration)

---

## Core Models

### LanguageModel

Main language model class.

```python
class LanguageModel(nn.Module):
    """
    ResNet-BK language model with Birman-Schwinger core.
    
    Args:
        vocab_size (int): Vocabulary size
        d_model (int): Model dimension
        n_layers (int): Number of layers
        n_seq (int): Maximum sequence length
        epsilon (float): Regularization parameter (0.5-1.0)
        use_prime_bump (bool): Use Prime-Bump initialization
        num_experts (int): Number of MoE experts
        top_k (int): Number of experts to route to
        use_scattering_router (bool): Use scattering-based routing
        use_act (bool): Enable adaptive computation time
        
    Examples:
        >>> model = LanguageModel(vocab_size=30000, d_model=256, n_layers=6)
        >>> output = model(input_ids)
        >>> print(output.shape)  # [batch, seq_len, vocab_size]
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 256,
        n_layers: int = 6,
        n_seq: int = 512,
        epsilon: float = 1.0,
        use_prime_bump: bool = True,
        num_experts: int = 4,
        top_k: int = 2,
        use_scattering_router: bool = True,
        use_act: bool = False,
        **kwargs
    ):
        ...
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            return_dict: Return dictionary with additional outputs
            
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            or dict with keys: logits, loss, hidden_states, etc.
        """
        ...
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Prompt token IDs [batch, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            do_sample: Use sampling (vs greedy)
            
        Returns:
            generated_ids: Generated token IDs [batch, max_length]
            
        Examples:
            >>> prompt = torch.tensor([[1, 2, 3]])
            >>> output = model.generate(prompt, max_length=50)
        """
        ...
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "LanguageModel":
        """
        Load pre-trained model from Hugging Face Hub or local path.
        
        Args:
            model_name_or_path: Model identifier or path
            
        Returns:
            Loaded model
            
        Examples:
            >>> model = LanguageModel.from_pretrained("resnetbk/mamba-killer-1b")
        """
        ...
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> "LanguageModel":
        """
        Load model from checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint (.pt file)
            
        Returns:
            Loaded model
        """
        ...
    
    def save_pretrained(self, save_directory: str):
        """
        Save model to directory.
        
        Args:
            save_directory: Directory to save model
        """
        ...
```

---

### BirmanSchwingerCore

Core Birman-Schwinger operator implementation.

```python
class BirmanSchwingerCore(nn.Module):
    """
    Birman-Schwinger operator with LAP-based numerical stability.
    
    Implements: K_ε(z) = |V_ε|^{1/2} R_0(z) |V_ε|^{1/2}
    
    Args:
        n_seq (int): Sequence length
        epsilon (float): Regularization parameter
        use_mourre (bool): Enable Mourre estimate verification
        use_lap (bool): Enable Limiting Absorption Principle
        precision (str): Computation precision ('complex64' or 'complex128')
        
    Mathematical Guarantees:
        - Hilbert-Schmidt: ||K_ε||_S2 ≤ (1/2)(Im z)^{-1/2} ||V_ε||_L2
        - Trace-class: ||K_ε||_S1 ≤ (1/2)(Im z)^{-1} ||V_ε||_L1
        - Mourre estimate: [H_0, iA] = I
        
    Examples:
        >>> bk_core = BirmanSchwingerCore(n_seq=512, epsilon=1.0)
        >>> v = torch.randn(8, 512)  # Potential
        >>> G_ii = bk_core(v, z=1.0j)
        >>> print(G_ii.shape)  # [8, 512, 2] (real, imag)
    """
    
    def __init__(
        self,
        n_seq: int,
        epsilon: float = 1.0,
        use_mourre: bool = True,
        use_lap: bool = True,
        precision: str = 'complex128'
    ):
        ...
    
    def forward(
        self,
        v: torch.Tensor,
        z: complex = 1.0j
    ) -> torch.Tensor:
        """
        Compute diagonal resolvent G_ii = diag((H_ε - zI)^{-1}).
        
        Args:
            v: Potential [batch, n_seq]
            z: Complex shift (default: 1.0j)
            
        Returns:
            G_ii: Diagonal resolvent [batch, n_seq, 2] (real, imag)
        """
        ...
    
    def compute_schatten_norms(self) -> Tuple[float, float]:
        """
        Compute Schatten norms for monitoring.
        
        Returns:
            (||K||_S1, ||K||_S2): Trace-class and Hilbert-Schmidt norms
        """
        ...
    
    def verify_mourre_estimate(self) -> bool:
        """
        Verify Mourre estimate: [H_0, iA] = I.
        
        Returns:
            True if estimate holds (error < 1e-6)
        """
        ...
    
    def apply_spectral_clipping(self, threshold: float):
        """
        Clip eigenvalues exceeding trace-class bounds.
        
        Args:
            threshold: Clipping threshold
        """
        ...
```

---

### PrimeBumpPotential

Prime-Bump initialization module.

```python
class PrimeBumpPotential(nn.Module):
    """
    Prime-Bump potential with GUE eigenvalue statistics.
    
    Implements: V_ε(x) = Σ_p α_{p,k}(ε) ψ_ε(x - log p)
    
    Args:
        n_seq (int): Sequence length
        epsilon (float): Cutoff width
        k_max (int): Maximum prime power
        
    Examples:
        >>> prime_bump = PrimeBumpPotential(n_seq=512, epsilon=1.0)
        >>> x = torch.randn(8, 512, 256)
        >>> v = prime_bump(x)
        >>> print(v.shape)  # [8, 512]
    """
    
    def __init__(
        self,
        n_seq: int,
        epsilon: float = 1.0,
        k_max: int = 3
    ):
        ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute potential V_ε(x).
        
        Args:
            x: Input features [batch, n_seq, d_model]
            
        Returns:
            v: Potential [batch, n_seq]
        """
        ...
    
    def get_prime_indices(self) -> List[int]:
        """
        Get list of prime positions < n_seq.
        
        Returns:
            List of prime indices
        """
        ...
    
    def compute_alpha_coefficients(self, p: int, k: int) -> float:
        """
        Compute canonical coefficient α_{p,k}(ε).
        
        Args:
            p: Prime number
            k: Prime power
            
        Returns:
            α_{p,k}(ε) = (log p) / p^{k(1/2+ε)}
        """
        ...
    
    def verify_gue_statistics(self) -> Dict[str, float]:
        """
        Verify eigenvalue spacing follows GUE statistics.
        
        Returns:
            Dictionary with Wigner surmise fit quality
        """
        ...
```

---

### ScatteringRouter

Zero-parameter MoE routing using scattering phase.

```python
class ScatteringRouter(nn.Module):
    """
    Parameter-free MoE routing using scattering phase.
    
    Implements: δ_ε(λ) = arg(det_2(I + K_ε(λ + i0)))
    
    Args:
        num_experts (int): Number of experts
        use_clark_measure (bool): Use Clark measure for routing
        
    Examples:
        >>> router = ScatteringRouter(num_experts=8)
        >>> G_ii = torch.randn(8, 512, 2)  # From BK-Core
        >>> expert_indices, weights = router(G_ii)
    """
    
    def __init__(
        self,
        num_experts: int,
        use_clark_measure: bool = False
    ):
        ...
    
    def forward(
        self,
        G_ii: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens based on scattering phase.
        
        Args:
            G_ii: Complex resolvent diagonal [batch, n_seq, 2]
            
        Returns:
            expert_indices: Selected experts [batch, n_seq, top_k]
            routing_weights: Mixing weights [batch, n_seq, top_k]
        """
        ...
    
    def compute_scattering_phase(self, G_ii: torch.Tensor) -> torch.Tensor:
        """
        Compute scattering phase δ_ε(λ).
        
        Args:
            G_ii: Complex resolvent diagonal
            
        Returns:
            phase: Scattering phase [batch, n_seq]
        """
        ...
    
    def compute_spectral_shift(self, lambda_: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral shift function ξ(λ).
        
        Args:
            lambda_: Energy values
            
        Returns:
            xi: Spectral shift [batch, n_seq]
        """
        ...
    
    def detect_resonances(self, D_eps: torch.Tensor) -> torch.Tensor:
        """
        Identify resonances where |D_ε(λ)| is small.
        
        Args:
            D_eps: Regularized determinant
            
        Returns:
            is_resonance: Boolean mask [batch, n_seq]
        """
        ...
```

---

### SemiseparableMatrix

Semiseparable matrix structure for O(N log N) memory.

```python
class SemiseparableMatrix:
    """
    Semiseparable matrix: H = T + U·V^T.
    
    Args:
        n_seq (int): Sequence length
        rank (Optional[int]): Low-rank component rank (default: log(n_seq))
        
    Examples:
        >>> H = torch.randn(512, 512)
        >>> semi = SemiseparableMatrix(n_seq=512)
        >>> T, U, V = semi.factorize(H)
        >>> x = torch.randn(512)
        >>> y = semi.matvec(x)  # O(N) complexity
    """
    
    def __init__(self, n_seq: int, rank: Optional[int] = None):
        ...
    
    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """
        O(N) matrix-vector product: y = H·x.
        
        Args:
            x: Input vector [n_seq]
            
        Returns:
            y: Output vector [n_seq]
        """
        ...
    
    def factorize(
        self,
        H: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decompose H into tridiagonal + low-rank.
        
        Args:
            H: Dense matrix [n_seq, n_seq]
            
        Returns:
            T: Tridiagonal part [n_seq, n_seq]
            U: Left low-rank factor [n_seq, rank]
            V: Right low-rank factor [n_seq, rank]
        """
        ...
    
    def checkpoint_recompute(
        self,
        x: torch.Tensor,
        k: int = 4
    ) -> torch.Tensor:
        """
        Gradient checkpointing with semiseparable structure.
        
        Args:
            x: Input [batch, n_seq, d_model]
            k: Number of checkpoints
            
        Returns:
            output: Processed features
        """
        ...
```

---

## Training

### Trainer

Main training class.

```python
class Trainer:
    """
    Training orchestrator for ResNet-BK models.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        optimizer (Optimizer): Optimizer
        scheduler (Optional[LRScheduler]): Learning rate scheduler
        config (Dict): Training configuration
        
    Examples:
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     optimizer=optimizer,
        ...     config={'num_epochs': 3, 'gradient_clip_norm': 1.0}
        ... )
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler] = None,
        config: Optional[Dict] = None
    ):
        ...
    
    def train(self) -> Dict[str, List[float]]:
        """
        Run full training loop.
        
        Returns:
            history: Dictionary with training metrics
        """
        ...
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            avg_loss: Average loss for epoch
        """
        ...
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on validation set.
        
        Returns:
            metrics: Dictionary with evaluation metrics
        """
        ...
    
    def save_checkpoint(self, path: str):
        """
        Save training checkpoint.
        
        Args:
            path: Checkpoint file path
        """
        ...
    
    def load_checkpoint(self, path: str):
        """
        Load training checkpoint.
        
        Args:
            path: Checkpoint file path
        """
        ...
```

---

### StabilityMonitor

Real-time numerical stability monitoring.

```python
class StabilityMonitor:
    """
    Monitor numerical health during training.
    
    Examples:
        >>> monitor = StabilityMonitor()
        >>> health = monitor.check_tensors({'loss': loss, 'grads': grads})
        >>> if not health['all_finite']:
        ...     action = monitor.suggest_recovery('nan_detected')
    """
    
    def check_tensors(
        self,
        tensors: Dict[str, torch.Tensor]
    ) -> Dict[str, bool]:
        """
        Check for NaN/Inf in tensors.
        
        Args:
            tensors: Dictionary of tensors to check
            
        Returns:
            health: Dictionary with health status
        """
        ...
    
    def check_condition_number(self, H: torch.Tensor) -> float:
        """
        Compute condition number κ(H) = σ_max / σ_min.
        
        Args:
            H: Matrix to check
            
        Returns:
            condition_number: κ(H)
        """
        ...
    
    def check_schatten_bounds(
        self,
        K: torch.Tensor,
        epsilon: float,
        z: complex
    ) -> bool:
        """
        Verify Schatten norm bounds.
        
        Args:
            K: Birman-Schwinger kernel
            epsilon: Regularization parameter
            z: Complex shift
            
        Returns:
            bounds_satisfied: True if bounds hold
        """
        ...
    
    def suggest_recovery(self, failure_type: str) -> str:
        """
        Suggest recovery action based on failure mode.
        
        Args:
            failure_type: Type of failure detected
            
        Returns:
            action: Suggested recovery action
        """
        ...
```

---

### AutoRecovery

Automatic failure detection and recovery.

```python
class AutoRecovery:
    """
    Automatic failure detection and recovery system.
    
    Args:
        checkpoint_dir (str): Directory for checkpoints
        max_retries (int): Maximum recovery attempts
        
    Examples:
        >>> recovery = AutoRecovery(checkpoint_dir="checkpoints/")
        >>> trainer.add_callback(recovery)
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_retries: int = 3
    ):
        ...
    
    def detect_failure(
        self,
        state: Dict
    ) -> Optional[str]:
        """
        Detect failure mode from training state.
        
        Args:
            state: Training state dictionary
            
        Returns:
            failure_type: Type of failure or None
        """
        ...
    
    def recover(
        self,
        failure_type: str,
        model: nn.Module,
        optimizer: Optimizer
    ) -> bool:
        """
        Attempt recovery from failure.
        
        Args:
            failure_type: Type of failure
            model: Model to recover
            optimizer: Optimizer to recover
            
        Returns:
            success: True if recovery successful
        """
        ...
```

---

## Benchmarks

### WikiText2Benchmark

WikiText-2 evaluation benchmark.

```python
class WikiText2Benchmark:
    """
    WikiText-2 benchmark for language modeling.
    
    Examples:
        >>> benchmark = WikiText2Benchmark()
        >>> results = benchmark.evaluate(model)
        >>> print(f"Perplexity: {results['perplexity']:.2f}")
    """
    
    def evaluate(
        self,
        model: nn.Module,
        batch_size: int = 8
    ) -> Dict[str, float]:
        """
        Evaluate model on WikiText-2 test set.
        
        Args:
            model: Model to evaluate
            batch_size: Batch size for evaluation
            
        Returns:
            results: Dictionary with metrics (perplexity, loss, etc.)
        """
        ...
```

---

### FairComparison

Fair comparison framework for ResNet-BK vs Mamba.

```python
class FairComparison:
    """
    Fair comparison between ResNet-BK and Mamba.
    
    Args:
        model_a (str): First model name
        model_b (str): Second model name
        num_seeds (int): Number of random seeds
        
    Examples:
        >>> comparison = FairComparison("resnetbk", "mamba", num_seeds=5)
        >>> results = comparison.run_all_benchmarks()
    """
    
    def __init__(
        self,
        model_a: str,
        model_b: str,
        num_seeds: int = 5
    ):
        ...
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """
        Run all comparison benchmarks.
        
        Returns:
            results: Dictionary with all benchmark results
        """
        ...
    
    def compute_statistical_significance(
        self,
        results_a: List[float],
        results_b: List[float]
    ) -> float:
        """
        Compute p-value using paired t-test.
        
        Args:
            results_a: Results for model A
            results_b: Results for model B
            
        Returns:
            p_value: Statistical significance
        """
        ...
```

---

## Utilities

### ConfigLoader

Configuration file loader.

```python
class ConfigLoader:
    """
    Load and validate configuration files.
    
    Examples:
        >>> config = ConfigLoader.load("configs/base_config.yaml")
        >>> model = LanguageModel(**config['model'])
    """
    
    @staticmethod
    def load(config_path: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            config: Configuration dictionary
        """
        ...
    
    @staticmethod
    def validate(config: Dict) -> bool:
        """
        Validate configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            valid: True if configuration is valid
        """
        ...
```

---

### CheckpointManager

Checkpoint management utilities.

```python
class CheckpointManager:
    """
    Manage model checkpoints.
    
    Args:
        checkpoint_dir (str): Directory for checkpoints
        keep_last_n (int): Number of checkpoints to keep
        
    Examples:
        >>> manager = CheckpointManager("checkpoints/", keep_last_n=3)
        >>> manager.save(model, optimizer, epoch=5)
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 3
    ):
        ...
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        metrics: Optional[Dict] = None
    ):
        """
        Save checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            metrics: Optional metrics dictionary
        """
        ...
    
    def load_latest(self) -> Dict:
        """
        Load latest checkpoint.
        
        Returns:
            checkpoint: Checkpoint dictionary
        """
        ...
    
    def load_best(self, metric: str = 'perplexity') -> Dict:
        """
        Load best checkpoint based on metric.
        
        Args:
            metric: Metric to use for selection
            
        Returns:
            checkpoint: Best checkpoint dictionary
        """
        ...
```

---

## Configuration

### Model Configuration

```yaml
model:
  vocab_size: 30000
  d_model: 256
  n_layers: 6
  n_seq: 512
  
  # Birman-Schwinger parameters
  epsilon: 1.0
  use_prime_bump: true
  prime_bump_scale: 0.02
  k_max: 3
  
  # MoE parameters
  num_experts: 4
  top_k: 2
  use_scattering_router: true
  
  # Numerical stability
  use_mourre: true
  use_lap: true
  schatten_threshold: 100.0
  
  # Memory optimization
  use_semiseparable: true
  low_rank: null  # Auto: log(n_seq)
  
  # Adaptive computation
  use_act: false
  act_halt_threshold: 0.2
```

### Training Configuration

```yaml
training:
  batch_size: 8
  learning_rate: 1e-3
  num_epochs: 3
  warmup_steps: 100
  gradient_clip_norm: 1.0
  
  # Optimization
  gradient_checkpointing: true
  gradient_accumulation_steps: 1
  mixed_precision: true
  
  # Logging
  log_interval: 50
  eval_interval: 500
  save_interval: 1000
```

---

For more examples and tutorials, see [TUTORIAL.md](TUTORIAL.md).
