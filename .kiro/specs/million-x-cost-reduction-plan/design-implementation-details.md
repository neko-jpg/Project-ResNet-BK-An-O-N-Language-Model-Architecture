# Implementation Details and Optimization Strategies

## Memory Optimization for Google Colab

### Current Memory Usage Analysis

**Model Parameters** (d_model=64, n_layers=4, num_experts=4):
```
Token Embedding: vocab_size × d_model = 30000 × 64 = 1.92M params
Position Embedding: n_seq × d_model = 128 × 64 = 8.2K params

Per Layer:
  MoE (4 experts):
    Gating: d_model × num_experts = 64 × 4 = 256 params
    Expert 1: 64 × 128 + 128 × 64 = 16,384 params
    Expert 2-4: 3 × 16,384 = 49,152 params
    Total MoE: 49,408 params
  
  Output Projection: 2 × d_model = 128 params
  LayerNorm: 2 × d_model = 128 params
  
  Total per layer: 49,664 params

Total 4 layers: 4 × 49,664 = 198,656 params

LM Head: d_model × vocab_size = 64 × 30000 = 1.92M params
Final LayerNorm: 128 params

Total Model: 1.92M + 8.2K + 198K + 1.92M + 128 = 4.05M params
```

**Memory Footprint**:
```
FP32: 4.05M × 4 bytes = 16.2 MB
FP16: 4.05M × 2 bytes = 8.1 MB
INT8: 4.05M × 1 byte = 4.05 MB
```

**Activation Memory** (batch_size=20, n_seq=128):
```
Per Layer:
  Input: B × N × D = 20 × 128 × 64 = 163,840 values
  MoE intermediate: B × N × (2D) = 20 × 128 × 128 = 327,680 values
  BK-Core (complex): B × N = 20 × 128 = 2,560 complex values = 5,120 values
  Output: B × N × D = 163,840 values
  
  Total per layer: ~660K values

Total 4 layers: 2.64M values
FP32: 2.64M × 4 = 10.56 MB
```

**Optimizer State** (AdamW):
```
Momentum: 4.05M params
Variance: 4.05M params
Total: 8.1M params
FP32: 8.1M × 4 = 32.4 MB
```

**Total Memory** (training):
```
Model (FP32): 16.2 MB
Activations: 10.56 MB
Optimizer: 32.4 MB
Gradients: 16.2 MB
Total: ~75 MB
```

**Google Colab Free Tier**: 15 GB RAM, 12 GB GPU RAM
**Headroom**: 12 GB - 75 MB = 11.925 GB (plenty of room!)

### Scaling to Larger Models

**Target**: 100M parameters (25× larger)

**Memory Estimate**:
```
Model: 100M × 4 = 400 MB
Activations: 10.56 MB × 25 = 264 MB
Optimizer: 100M × 8 = 800 MB
Gradients: 400 MB
Total: ~1.86 GB
```

**Still fits in Colab!**

**Target**: 1B parameters (250× larger)

**Memory Estimate**:
```
Model: 1B × 4 = 4 GB
Activations: 10.56 MB × 250 = 2.64 GB
Optimizer: 1B × 8 = 8 GB
Gradients: 4 GB
Total: ~18.64 GB
```

**Exceeds Colab GPU RAM (12 GB)**

**Solutions**:
1. **Gradient Checkpointing**: Trade compute for memory
   - Recompute activations during backward pass
   - Reduces activation memory by ~80%
   - New total: 4 + 0.53 + 8 + 4 = 16.53 GB (still too much)

2. **CPU Offloading** (ZeRO-Offload):
   - Keep optimizer states on CPU
   - New total: 4 + 0.53 + 4 = 8.53 GB (fits!)

3. **Mixed Precision** (FP16):
   - Model: 1B × 2 = 2 GB
   - Activations: 0.53 GB
   - Optimizer (FP32): 8 GB (on CPU)
   - Gradients (FP16): 2 GB
   - Total GPU: 2 + 0.53 + 2 = 4.53 GB (fits easily!)

### Implementation: Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedResNetBKBlock(nn.Module):
    def __init__(self, d_model, n_seq):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.bk_layer = MoEResNetBKLayer(d_model, n_seq)
    
    def forward(self, x):
        # Use gradient checkpointing
        return checkpoint(self._forward, x, use_reentrant=False)
    
    def _forward(self, x):
        return x + self.bk_layer(self.layer_norm(x))
```

### Implementation: CPU Offloading

```python
class CPUOffloadedOptimizer:
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.lr = lr
        
        # Create optimizer on CPU
        self.optimizer = torch.optim.AdamW(
            [p.cpu() for p in model.parameters()],
            lr=lr
        )
        
        # Keep model on GPU
        self.model = model.cuda()
    
    def step(self):
        # Move gradients to CPU
        for param_gpu in self.model.parameters():
            if param_gpu.grad is not None:
                param_cpu = param_gpu.cpu()
                param_cpu.grad = param_gpu.grad.cpu()
        
        # Optimizer step on CPU
        self.optimizer.step()
        
        # Move updated parameters back to GPU
        for param_gpu, param_cpu in zip(self.model.parameters(), self.optimizer.param_groups[0]['params']):
            param_gpu.data = param_cpu.data.cuda()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
        self.model.zero_grad()
```

## Detailed Benchmarking Methodology

### FLOPs Counting

**Implementation**:

```python
class FLOPsCounter:
    """
    Accurate FLOPs counting for ResNet-BK.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.forward_flops = 0
        self.backward_flops = 0
    
    def count_bk_core_forward(self, n_seq):
        """
        BK-Core forward pass FLOPs.
        """
        # Theta recursion: N iterations × 6 FLOPs (complex multiply-add)
        theta_flops = n_seq * 6
        
        # Phi recursion: N iterations × 6 FLOPs
        phi_flops = n_seq * 6
        
        # Division: N × 8 FLOPs (complex division)
        div_flops = n_seq * 8
        
        # Real/Imag extraction: N × 2 FLOPs
        extract_flops = n_seq * 2
        
        total = theta_flops + phi_flops + div_flops + extract_flops
        self.forward_flops += total
        return total
    
    def count_moe_forward(self, batch_size, n_seq, d_model, num_experts):
        """
        MoE forward pass FLOPs.
        """
        # Gating network: (B×N) × d_model × num_experts
        gating_flops = batch_size * n_seq * d_model * num_experts
        
        # Gumbel-Softmax: (B×N) × num_experts × 10
        gumbel_flops = batch_size * n_seq * num_experts * 10
        
        # Expert computation (top-1): (B×N) × [d_model × (2×d_model) + (2×d_model) × d_model]
        expert_flops = batch_size * n_seq * (d_model * 2 * d_model + 2 * d_model * d_model)
        
        total = gating_flops + gumbel_flops + expert_flops
        self.forward_flops += total
        return total
    
    def count_linear_forward(self, batch_size, n_seq, in_features, out_features):
        """
        Linear layer forward pass FLOPs.
        """
        flops = batch_size * n_seq * in_features * out_features
        self.forward_flops += flops
        return flops
    
    def count_bk_core_backward(self, n_seq):
        """
        BK-Core analytic gradient FLOPs.
        """
        # G_ii² computation: N × 6 FLOPs
        g_sq_flops = n_seq * 6
        
        # 1/G_ii² computation: N × 8 FLOPs
        inv_g_sq_flops = n_seq * 8
        
        # Gradient blending: N × 4 FLOPs
        blend_flops = n_seq * 4
        
        total = g_sq_flops + inv_g_sq_flops + blend_flops
        self.backward_flops += total
        return total
    
    def count_autograd_backward(self, forward_flops):
        """
        Autograd backward pass: approximately 2× forward FLOPs.
        """
        backward_flops = 2 * forward_flops
        self.backward_flops += backward_flops
        return backward_flops
    
    def get_total_flops(self):
        return self.forward_flops + self.backward_flops
    
    def get_report(self):
        return {
            'forward_flops': self.forward_flops,
            'backward_flops': self.backward_flops,
            'total_flops': self.get_total_flops(),
            'forward_gflops': self.forward_flops / 1e9,
            'backward_gflops': self.backward_flops / 1e9,
            'total_gflops': self.get_total_flops() / 1e9,
        }


# Usage example
counter = FLOPsCounter()

# Count forward pass
counter.count_bk_core_forward(n_seq=128)
counter.count_moe_forward(batch_size=20, n_seq=128, d_model=64, num_experts=4)
counter.count_linear_forward(batch_size=20, n_seq=128, in_features=2, out_features=64)

# Count backward pass
counter.count_bk_core_backward(n_seq=128)
counter.count_autograd_backward(counter.forward_flops - counter.count_bk_core_forward(128))

print(counter.get_report())
```

### Wall-Clock Time Measurement

**Implementation**:

```python
import time
import torch

class TimingBenchmark:
    """
    Accurate wall-clock time measurement with GPU synchronization.
    """
    
    def __init__(self, device='cuda', warmup_iters=10, measure_iters=100):
        self.device = device
        self.warmup_iters = warmup_iters
        self.measure_iters = measure_iters
    
    def benchmark_forward(self, model, x):
        """
        Benchmark forward pass time.
        """
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_iters):
                _ = model(x)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(self.measure_iters):
                start = time.perf_counter()
                _ = model(x)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)
        
        return {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000,
        }
    
    def benchmark_backward(self, model, x, y, criterion):
        """
        Benchmark backward pass time.
        """
        model.train()
        
        # Warmup
        for _ in range(self.warmup_iters):
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y)
            loss.backward()
            model.zero_grad()
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # Measure
        times = []
        for _ in range(self.measure_iters):
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            loss.backward()
            if self.device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            times.append(end - start)
            model.zero_grad()
        
        return {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000,
        }
    
    def benchmark_full_step(self, model, x, y, optimizer, criterion):
        """
        Benchmark full training step (forward + backward + optimizer).
        """
        model.train()
        
        # Warmup
        for _ in range(self.warmup_iters):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y)
            loss.backward()
            optimizer.step()
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # Measure
        times = []
        for _ in range(self.measure_iters):
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y)
            loss.backward()
            optimizer.step()
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append(end - start)
        
        return {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000,
        }
```

### Memory Profiling

**Implementation**:

```python
import torch

class MemoryProfiler:
    """
    Profile GPU memory usage.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    
    def get_current_memory_mb(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0
    
    def get_peak_memory_mb(self):
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
        return 0
    
    def profile_forward(self, model, x):
        """
        Profile memory usage during forward pass.
        """
        self.reset()
        
        initial_memory = self.get_current_memory_mb()
        
        with torch.no_grad():
            _ = model(x)
        
        peak_memory = self.get_peak_memory_mb()
        final_memory = self.get_current_memory_mb()
        
        return {
            'initial_mb': initial_memory,
            'peak_mb': peak_memory,
            'final_mb': final_memory,
            'activation_mb': peak_memory - initial_memory,
        }
    
    def profile_backward(self, model, x, y, criterion):
        """
        Profile memory usage during backward pass.
        """
        self.reset()
        
        # Forward pass
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y)
        
        memory_after_forward = self.get_current_memory_mb()
        
        # Backward pass
        loss.backward()
        
        peak_memory = self.get_peak_memory_mb()
        final_memory = self.get_current_memory_mb()
        
        return {
            'after_forward_mb': memory_after_forward,
            'peak_mb': peak_memory,
            'final_mb': final_memory,
            'gradient_mb': final_memory - memory_after_forward,
        }
```

## Comprehensive Benchmark Suite

```python
class ComprehensiveBenchmark:
    """
    Run all benchmarks and generate report.
    """
    
    def __init__(self, model, baseline_model, config):
        self.model = model
        self.baseline_model = baseline_model
        self.config = config
        
        self.flops_counter = FLOPsCounter()
        self.timing_benchmark = TimingBenchmark()
        self.memory_profiler = MemoryProfiler()
    
    def run_all_benchmarks(self, train_loader, val_loader):
        """
        Run comprehensive benchmark suite.
        """
        results = {}
        
        # 1. FLOPs Analysis
        print("Running FLOPs analysis...")
        results['flops'] = self.benchmark_flops()
        
        # 2. Wall-Clock Time
        print("Running timing benchmarks...")
        results['timing'] = self.benchmark_timing(train_loader)
        
        # 3. Memory Usage
        print("Running memory profiling...")
        results['memory'] = self.benchmark_memory(train_loader)
        
        # 4. Training Convergence
        print("Running convergence test...")
        results['convergence'] = self.benchmark_convergence(train_loader, val_loader)
        
        # 5. Scaling Analysis
        print("Running scaling analysis...")
        results['scaling'] = self.benchmark_scaling()
        
        # 6. Numerical Stability
        print("Running stability test...")
        results['stability'] = self.benchmark_stability(train_loader)
        
        # Generate report
        self.generate_report(results)
        
        return results
    
    def benchmark_flops(self):
        """Benchmark FLOPs for model and baseline."""
        # Count model FLOPs
        self.flops_counter.reset()
        # ... (count all operations)
        model_flops = self.flops_counter.get_report()
        
        # Count baseline FLOPs
        self.flops_counter.reset()
        # ... (count baseline operations)
        baseline_flops = self.flops_counter.get_report()
        
        return {
            'model': model_flops,
            'baseline': baseline_flops,
            'speedup': baseline_flops['total_flops'] / model_flops['total_flops'],
        }
    
    def benchmark_timing(self, train_loader):
        """Benchmark wall-clock time."""
        x_batch, y_batch = next(iter(train_loader))
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
        
        # Model timing
        model_forward = self.timing_benchmark.benchmark_forward(self.model, x_batch)
        model_backward = self.timing_benchmark.benchmark_backward(
            self.model, x_batch, y_batch, nn.CrossEntropyLoss()
        )
        
        # Baseline timing
        baseline_forward = self.timing_benchmark.benchmark_forward(self.baseline_model, x_batch)
        baseline_backward = self.timing_benchmark.benchmark_backward(
            self.baseline_model, x_batch, y_batch, nn.CrossEntropyLoss()
        )
        
        return {
            'model_forward': model_forward,
            'model_backward': model_backward,
            'baseline_forward': baseline_forward,
            'baseline_backward': baseline_backward,
            'forward_speedup': baseline_forward['mean_ms'] / model_forward['mean_ms'],
            'backward_speedup': baseline_backward['mean_ms'] / model_backward['mean_ms'],
        }
    
    def benchmark_memory(self, train_loader):
        """Benchmark memory usage."""
        x_batch, y_batch = next(iter(train_loader))
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
        
        # Model memory
        model_forward_mem = self.memory_profiler.profile_forward(self.model, x_batch)
        model_backward_mem = self.memory_profiler.profile_backward(
            self.model, x_batch, y_batch, nn.CrossEntropyLoss()
        )
        
        # Baseline memory
        baseline_forward_mem = self.memory_profiler.profile_forward(self.baseline_model, x_batch)
        baseline_backward_mem = self.memory_profiler.profile_backward(
            self.baseline_model, x_batch, y_batch, nn.CrossEntropyLoss()
        )
        
        return {
            'model_forward': model_forward_mem,
            'model_backward': model_backward_mem,
            'baseline_forward': baseline_forward_mem,
            'baseline_backward': baseline_backward_mem,
            'memory_reduction': baseline_backward_mem['peak_mb'] / model_backward_mem['peak_mb'],
        }
    
    def generate_report(self, results):
        """Generate comprehensive PDF report."""
        # ... (generate plots, tables, summary)
        pass
```

