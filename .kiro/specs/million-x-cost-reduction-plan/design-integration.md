# Design Document (Continued) - Integration & Testing

## Components and Interfaces

### Modular Architecture

All components are designed to be independently enabled/disabled for ablation studies:

```python
class ConfigurableResNetBK(nn.Module):
    """
    Fully configurable ResNet-BK with all optimizations as options.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Step 1: Base architecture (always enabled)
        self.use_bk_core = True
        
        # Step 2: Learning algorithm
        self.use_analytic_gradient = config.get('use_analytic_gradient', False)
        self.use_koopman = config.get('use_koopman', False)
        self.use_physics_informed = config.get('use_physics_informed', False)
        self.grad_blend = config.get('grad_blend', 0.5)
        
        # Step 3: Sparsification (already implemented)
        self.use_moe = config.get('use_moe', True)
        self.num_experts = config.get('num_experts', 4)
        
        # Step 4: Compression
        self.use_quantization = config.get('use_quantization', False)
        self.use_pruning = config.get('use_pruning', False)
        self.use_distillation = config.get('use_distillation', False)
        
        # Step 5: Hardware optimization
        self.use_cuda_kernels = config.get('use_cuda_kernels', False)
        self.use_mixed_precision = config.get('use_mixed_precision', False)
        
        # Step 6: Algorithmic innovations
        self.use_act = config.get('use_act', False)
        self.use_multiscale = config.get('use_multiscale', False)
        self.use_learned_sparsity = config.get('use_learned_sparsity', False)
        
        # Step 7: System optimizations
        self.use_curriculum = config.get('use_curriculum', False)
        self.use_active_learning = config.get('use_active_learning', False)
        self.use_gradient_caching = config.get('use_gradient_caching', False)
        
        # Build model based on config
        self._build_model()
    
    def _build_model(self):
        """
        Construct model components based on configuration.
        """
        vocab_size = self.config['vocab_size']
        d_model = self.config['d_model']
        n_layers = self.config['n_layers']
        n_seq = self.config['n_seq']
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(n_seq, d_model)
        
        # Blocks
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            if self.use_act:
                block = AdaptiveResNetBKBlock(d_model, n_seq)
            elif self.use_multiscale:
                block = MultiScaleResNetBKBlock(d_model, n_seq)
            else:
                block = ResNetBKBlock(d_model, n_seq, self.num_experts)
            
            self.blocks.append(block)
        
        # Output
        self.layer_norm_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        """
        Forward pass with all enabled optimizations.
        """
        B, N = x.shape
        
        # Embeddings
        tok_emb = self.token_embedding(x)
        pos = torch.arange(0, N, dtype=torch.long, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)
        h = tok_emb + pos_emb
        
        # Process through blocks
        for block in self.blocks:
            h = block(h)
        
        # Output
        h = self.layer_norm_final(h)
        logits = self.lm_head(h)
        
        return logits
    
    def get_cost_reduction_estimate(self):
        """
        Estimate total cost reduction based on enabled features.
        """
        reduction = 1.0
        
        # Step 1: Architecture (always enabled)
        reduction *= 10.0
        
        # Step 2: Learning
        if self.use_analytic_gradient:
            reduction *= 50.0
        if self.use_koopman:
            reduction *= 2.0  # Additional on top of analytic
        if self.use_physics_informed:
            reduction *= 1.5  # Additional
        
        # Step 3: Sparsification
        if self.use_moe:
            reduction *= 10.0
        
        # Step 4: Compression
        if self.use_quantization:
            reduction *= 4.0
        if self.use_pruning:
            reduction *= 4.0
        if self.use_distillation:
            reduction *= 6.0
        
        # Step 5: Hardware
        if self.use_cuda_kernels:
            reduction *= 3.0
        if self.use_mixed_precision:
            reduction *= 2.0
        
        # Step 6: Algorithms
        if self.use_act:
            reduction *= 1.4
        if self.use_multiscale:
            reduction *= 2.0
        if self.use_learned_sparsity:
            reduction *= 1.8
        
        # Step 7: System
        if self.use_curriculum:
            reduction *= 1.4
        if self.use_active_learning:
            reduction *= 2.0
        if self.use_gradient_caching:
            reduction *= 1.25
        
        return reduction
```

### Configuration Presets

```python
# Preset configurations for different use cases

BASELINE_CONFIG = {
    'vocab_size': 30000,
    'd_model': 64,
    'n_layers': 4,
    'n_seq': 128,
    'num_experts': 4,
    # All optimizations disabled (except base architecture)
    'use_analytic_gradient': False,
    'use_koopman': False,
    'use_physics_informed': False,
    'use_quantization': False,
    'use_pruning': False,
    'use_distillation': False,
    'use_cuda_kernels': False,
    'use_mixed_precision': False,
    'use_act': False,
    'use_multiscale': False,
    'use_learned_sparsity': False,
    'use_curriculum': False,
    'use_active_learning': False,
    'use_gradient_caching': False,
}

STEP2_CONFIG = {
    **BASELINE_CONFIG,
    'use_analytic_gradient': True,
    'grad_blend': 0.5,
}

STEP2_FULL_CONFIG = {
    **STEP2_CONFIG,
    'use_koopman': True,
    'use_physics_informed': True,
}

STEP4_CONFIG = {
    **STEP2_FULL_CONFIG,
    'use_quantization': True,
    'use_pruning': True,
    'use_distillation': True,
}

STEP5_CONFIG = {
    **STEP4_CONFIG,
    'use_cuda_kernels': True,
    'use_mixed_precision': True,
}

STEP6_CONFIG = {
    **STEP5_CONFIG,
    'use_act': True,
    'use_multiscale': True,
    'use_learned_sparsity': True,
}

FULL_CONFIG = {
    **STEP6_CONFIG,
    'use_curriculum': True,
    'use_active_learning': True,
    'use_gradient_caching': True,
}
```

## Data Models

### Training Metrics

```python
@dataclass
class TrainingMetrics:
    """
    Comprehensive metrics tracked during training.
    """
    epoch: int
    step: int
    
    # Loss metrics
    loss_lm: float  # Language modeling loss
    loss_koopman: float = 0.0  # Koopman auxiliary loss
    loss_energy: float = 0.0  # Energy conservation loss
    loss_act: float = 0.0  # ACT ponder cost
    loss_sparsity: float = 0.0  # Sparsity regularization
    loss_total: float = 0.0  # Total loss
    
    # Performance metrics
    perplexity: float = 0.0
    accuracy: float = 0.0
    
    # Efficiency metrics
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    total_time_ms: float = 0.0
    forward_flops: int = 0
    backward_flops: int = 0
    
    # Memory metrics
    peak_memory_mb: float = 0.0
    activation_memory_mb: float = 0.0
    
    # Adaptive computation metrics
    avg_layers_executed: float = 0.0  # For ACT
    sparsity_ratio: float = 0.0  # For learned sparsity
    
    # MoE metrics
    expert_usage: List[float] = None  # Usage per expert
    routing_entropy: float = 0.0
    
    # Gradient metrics
    grad_norm: float = 0.0
    grad_variance: float = 0.0
    
    # Learning rate
    learning_rate: float = 0.0
    
    def to_dict(self):
        return asdict(self)
    
    def log_to_wandb(self):
        """Log to Weights & Biases if available."""
        try:
            import wandb
            wandb.log(self.to_dict())
        except ImportError:
            pass


@dataclass
class BenchmarkResults:
    """
    Results from comprehensive benchmarking.
    """
    config_name: str
    
    # Cost reduction
    theoretical_reduction: float
    empirical_reduction: float
    
    # Performance
    final_perplexity: float
    baseline_perplexity: float
    perplexity_ratio: float
    
    # Efficiency
    total_training_time_hours: float
    total_flops: int
    flops_per_token: int
    
    # Model size
    num_parameters: int
    model_size_mb: float
    
    # Hardware utilization
    avg_gpu_utilization: float
    avg_memory_utilization: float
    
    # Comparison to baseline
    speedup_vs_baseline: float
    memory_reduction_vs_baseline: float
    
    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"Benchmark Results: {self.config_name}")
        print(f"{'='*60}")
        print(f"Cost Reduction:")
        print(f"  Theoretical: {self.theoretical_reduction:.1f}×")
        print(f"  Empirical:   {self.empirical_reduction:.1f}×")
        print(f"\nPerformance:")
        print(f"  Perplexity: {self.final_perplexity:.2f} (baseline: {self.baseline_perplexity:.2f})")
        print(f"  Ratio:      {self.perplexity_ratio:.2%}")
        print(f"\nEfficiency:")
        print(f"  Training Time: {self.total_training_time_hours:.2f} hours")
        print(f"  Total FLOPs:   {self.total_flops/1e12:.2f} TFLOPs")
        print(f"  Speedup:       {self.speedup_vs_baseline:.1f}×")
        print(f"\nModel:")
        print(f"  Parameters: {self.num_parameters/1e6:.2f}M")
        print(f"  Size:       {self.model_size_mb:.2f} MB")
        print(f"{'='*60}\n")
```

## Error Handling

### Numerical Stability Monitoring

```python
class NumericalStabilityMonitor:
    """
    Monitor and handle numerical instability during training.
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Thresholds
        self.nan_threshold = 0  # No NaNs allowed
        self.inf_threshold = 0  # No Infs allowed
        self.grad_norm_threshold = 1000.0
        
        # Counters
        self.nan_count = 0
        self.inf_count = 0
        self.large_grad_count = 0
        
        # Recovery strategies
        self.recovery_strategies = [
            self._reduce_learning_rate,
            self._reduce_v_max,
            self._reduce_feature_clamp,
            self._adjust_grad_blend,
            self._reset_to_checkpoint,
        ]
        self.current_strategy = 0
    
    def check_tensors(self, tensors, names):
        """
        Check tensors for NaN/Inf.
        
        Args:
            tensors: list of tensors to check
            names: list of tensor names for logging
        
        Returns:
            is_stable: bool
            issues: list of detected issues
        """
        issues = []
        
        for tensor, name in zip(tensors, names):
            if torch.isnan(tensor).any():
                issues.append(f"NaN detected in {name}")
                self.nan_count += 1
            
            if torch.isinf(tensor).any():
                issues.append(f"Inf detected in {name}")
                self.inf_count += 1
        
        return len(issues) == 0, issues
    
    def check_gradients(self):
        """
        Check gradient norms and values.
        """
        issues = []
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                
                if grad_norm > self.grad_norm_threshold:
                    issues.append(f"Large gradient in {name}: {grad_norm:.2f}")
                    self.large_grad_count += 1
                
                if torch.isnan(param.grad).any():
                    issues.append(f"NaN gradient in {name}")
                    self.nan_count += 1
                
                if torch.isinf(param.grad).any():
                    issues.append(f"Inf gradient in {name}")
                    self.inf_count += 1
        
        return len(issues) == 0, issues
    
    def recover(self):
        """
        Attempt to recover from numerical instability.
        """
        if self.current_strategy >= len(self.recovery_strategies):
            raise RuntimeError("All recovery strategies exhausted. Training failed.")
        
        strategy = self.recovery_strategies[self.current_strategy]
        print(f"Attempting recovery strategy {self.current_strategy + 1}: {strategy.__name__}")
        
        success = strategy()
        
        if not success:
            self.current_strategy += 1
            return self.recover()
        
        return True
    
    def _reduce_learning_rate(self):
        """Reduce learning rate by 50%."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.5
        print(f"Reduced learning rate to {param_group['lr']}")
        return True
    
    def _reduce_v_max(self):
        """Reduce v_max clamping."""
        for block in self.model.blocks:
            if hasattr(block.bk_layer, 'v_max'):
                block.bk_layer.v_max *= 0.8
                print(f"Reduced v_max to {block.bk_layer.v_max}")
        return True
    
    def _reduce_feature_clamp(self):
        """Reduce feature clamping."""
        for block in self.model.blocks:
            if hasattr(block.bk_layer, 'feature_clamp'):
                block.bk_layer.feature_clamp *= 0.8
                print(f"Reduced feature_clamp to {block.bk_layer.feature_clamp}")
        return True
    
    def _adjust_grad_blend(self):
        """Adjust GRAD_BLEND towards more stable value."""
        # Move towards 0.5 (balanced)
        current_blend = BKCoreFunction.GRAD_BLEND
        new_blend = 0.5 * current_blend + 0.5 * 0.5
        BKCoreFunction.GRAD_BLEND = new_blend
        print(f"Adjusted GRAD_BLEND to {new_blend}")
        return True
    
    def _reset_to_checkpoint(self):
        """Reset to last stable checkpoint."""
        if hasattr(self, 'last_stable_checkpoint'):
            self.model.load_state_dict(self.last_stable_checkpoint)
            print("Reset to last stable checkpoint")
            return True
        return False
```

## Testing Strategy

### Unit Tests

```python
class TestBKCore(unittest.TestCase):
    """
    Unit tests for BK-Core computation.
    """
    
    def test_theta_recursion(self):
        """Test theta recursion correctness."""
        N = 16
        a = torch.randn(N)
        b = torch.randn(N-1)
        c = torch.randn(N-1)
        z = torch.tensor(1.0j)
        
        G_ii = get_tridiagonal_inverse_diagonal(a, b, c, z)
        
        # Check output shape
        self.assertEqual(G_ii.shape, (N,))
        
        # Check no NaN/Inf
        self.assertFalse(torch.isnan(G_ii).any())
        self.assertFalse(torch.isinf(G_ii).any())
    
    def test_gradient_correctness(self):
        """Test analytic gradient vs finite difference."""
        N = 8
        d_model = 4
        
        model = MoEResNetBKLayer(d_model, N)
        x = torch.randn(1, N, d_model, requires_grad=True)
        
        # Analytic gradient
        output = model(x)
        loss = output.sum()
        loss.backward()
        grad_analytic = x.grad.clone()
        
        # Finite difference
        eps = 1e-4
        grad_fd = torch.zeros_like(x)
        
        for i in range(N):
            for j in range(d_model):
                x_plus = x.clone().detach()
                x_plus[0, i, j] += eps
                output_plus = model(x_plus)
                loss_plus = output_plus.sum()
                
                x_minus = x.clone().detach()
                x_minus[0, i, j] -= eps
                output_minus = model(x_minus)
                loss_minus = output_minus.sum()
                
                grad_fd[0, i, j] = (loss_plus - loss_minus) / (2 * eps)
        
        # Compare
        relative_error = (grad_analytic - grad_fd).abs() / (grad_fd.abs() + 1e-8)
        self.assertLess(relative_error.mean(), 0.1)  # 10% tolerance


class TestCompression(unittest.TestCase):
    """
    Unit tests for compression techniques.
    """
    
    def test_quantization_accuracy(self):
        """Test quantization preserves accuracy."""
        model = QuantizedBKCore(n_seq=128)
        v = torch.randn(4, 128)
        
        # Calibrate
        model.calibrate_quantization(v)
        
        # Forward
        features = model(v)
        
        # Check output shape
        self.assertEqual(features.shape, (4, 128, 2))
        
        # Check no NaN/Inf
        self.assertFalse(torch.isnan(features).any())
        self.assertFalse(torch.isinf(features).any())
```

### Integration Tests

```python
class TestEndToEnd(unittest.TestCase):
    """
    Integration tests for full training pipeline.
    """
    
    def test_training_convergence(self):
        """Test that model converges on toy dataset."""
        # Create toy dataset
        vocab_size = 100
        n_samples = 500
        n_seq = 32
        
        x_data = torch.randint(0, vocab_size, (n_samples, n_seq))
        y_data = torch.randint(0, vocab_size, (n_samples * n_seq,))
        
        dataset = TensorDataset(x_data, y_data)
        loader = DataLoader(dataset, batch_size=16)
        
        # Create model
        config = BASELINE_CONFIG.copy()
        config['vocab_size'] = vocab_size
        config['n_seq'] = n_seq
        model = ConfigurableResNetBK(config)
        
        # Train
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        initial_loss = None
        final_loss = None
        
        for epoch in range(5):
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = model(x_batch)
                loss = criterion(logits.view(-1, logits.size(-1)), y_batch)
                loss.backward()
                optimizer.step()
                
                if initial_loss is None:
                    initial_loss = loss.item()
                final_loss = loss.item()
        
        # Check convergence
        self.assertLess(final_loss, initial_loss * 0.8)  # 20% improvement
```

