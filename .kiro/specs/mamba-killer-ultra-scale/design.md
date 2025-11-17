# Design Document: Mamba-Killer Ultra-Scale ResNet-BK

## Overview

This design document describes the architecture and implementation strategy for transforming ResNet-BK into a world-class O(N) language model that surpasses Mamba across three critical dimensions: long-context stability, quantization robustness, and dynamic compute efficiency.

### Design Philosophy

The design is grounded in rigorous mathematical foundations from the Birman-Schwinger operator theory and Riemann zeta function spectral analysis, as detailed in the research paper `改善案/論文/riemann_hypothesis_main.tex`. Rather than empirical tuning, we leverage proven mathematical properties:

1. **Trace-class operators** guarantee numerical stability
2. **Semiseparable matrix structure** enables O(N) complexity with logarithmic memory overhead
3. **Scattering phase theory** provides parameter-free routing
4. **Clark measure preservation** ensures lossless compression

### Mathematical Foundations (from Paper)

The implementation is based on the following rigorous mathematical results:

**Birman-Krein Formula (Proposition BK-formula):**
```
d/dz log D_ε(z) = -Tr((H_ε - z)^{-1} - (H_0 - z)^{-1})
```
This connects the regularized determinant D_ε to the resolvent difference, enabling the explicit formula matching Weil's distribution.

**Schatten Bounds (Propositions BS-HS, BS-trace):**
- Hilbert-Schmidt: ||K_ε(z)||_S2 ≤ (1/2)(Im z)^{-1/2} ||V_ε||_L2
- Trace-class (ε > 1/2): ||K_ε(z)||_S1 ≤ (1/2)(Im z)^{-1} ||V_ε||_L1

**Mourre Estimate (Theorem mourre-H0):**
```
[H_0, iA] = I  (optimal with c_I = 1)
```
This provides the fundamental positive commutator estimate for numerical stability.

**Limiting Absorption Principle (Theorem lap-H0, Corollary lap-Heps):**
The weighted resolvent ⟨x⟩^{-s}(H_ε - λ ∓ iη)^{-1}⟨x⟩^{-s} extends continuously to η = 0, ensuring uniform invertibility of the Birman-Schwinger operator as Im z → 0.

**Weil Explicit Formula (eq:explicit-formula):**
```
(1/2πi) ∫ φ(λ) d log D_ε(λ) = -Σ_p Σ_k (log p / p^{k(1/2+ε)}) φ̂(k log p) + W_∞(φ)
```
This matches the prime sums and archimedean terms, providing the theoretical foundation for Prime-Bump initialization.

### Current State Analysis

**Existing Implementation:**
- BK-Core with O(N) theta/phi recursions (complex128 precision)
- Hybrid analytic gradient (GRAD_BLEND=0.5, optimal at 0.0)
- Sparse MoE with learned gating network
- 4.15M parameter model achieving PPL 1122 on WikiText-2
- 6.7× speedup over attention at N=2048

**Gaps to Address:**
- No Birman-Schwinger kernel implementation
- No Prime-Bump initialization
- No scattering-based routing (only learned MLP gating)
- No trace-class stability guarantees
- No semiseparable structure exploitation
- Limited to N=512 sequences (not 128k-1M)
- No quantization-aware training
- No adaptive computation (ACT)

## Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Mamba-Killer ResNet-BK                      │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────┴────────────┐
         │                         │
    ┌────▼────┐              ┌────▼────┐
    │ Input   │              │ Config  │
    │ Tokens  │              │ System  │
    └────┬────┘              └────┬────┘
         │                         │
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │  Token + Position Emb   │
         │  (Prime-Bump Init)      │
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │   ResNet-BK Block ×L    │
         │                         │
         │  ┌──────────────────┐   │
         │  │ LayerNorm        │   │
         │  └────────┬─────────┘   │
         │           │             │
         │  ┌────────▼─────────┐   │
         │  │ Scattering-MoE   │   │
         │  │ (Physics Router) │   │
         │  └────────┬─────────┘   │
         │           │             │
         │  ┌────────▼─────────┐   │
         │  │ Potential v_i    │   │
         │  └────────┬─────────┘   │
         │           │             │
         │  ┌────────▼─────────┐   │
         │  │ Birman-Schwinger │   │
         │  │ BK-Core (LAP)    │   │
         │  └────────┬─────────┘   │
         │           │             │
         │  ┌────────▼─────────┐   │
         │  │ Residual Add     │   │
         │  └────────┬─────────┘   │
         │           │             │
         └───────────┼─────────────┘
                     │
         ┌───────────▼─────────────┐
         │  ACT Early Exit         │
         │  (Scattering Halting)   │
         └───────────┬─────────────┘
                     │
         ┌───────────▼─────────────┐
         │  Final LayerNorm        │
         └───────────┬─────────────┘
                     │
         ┌───────────▼─────────────┐
         │  LM Head                │
         └───────────┬─────────────┘
                     │
                ┌────▼────┐
                │ Logits  │
                └─────────┘
```

### Component Hierarchy


```
LanguageModel
├── TokenEmbedding (with Prime-Bump init)
├── PositionEmbedding (with Prime-Bump init)
├── ResNetBKBlock[] (n_layers)
│   ├── LayerNorm
│   └── MoEResNetBKLayer
│       ├── ScatteringMoELayer (replaces SparseMoELayer)
│       │   ├── ScatteringRouter (parameter-free)
│       │   └── Expert[] (FFN networks)
│       ├── PotentialProjection (v_proj)
│       ├── BirmanSchwingerCore (replaces BKCoreFunction)
│       │   ├── PrimeBumpPotential
│       │   ├── ResolventKernel (LAP-stable)
│       │   ├── SchattenNormMonitor
│       │   └── SpectralShiftFunction
│       └── OutputProjection
├── ACTModule (adaptive computation)
├── FinalLayerNorm
└── LMHead

Supporting Systems:
├── SemiseparableMatrix (memory optimization)
├── ClarkMeasure (compression)
├── NumericalStabilityMonitor
├── QuantizationAwareTraining
└── BenchmarkPipeline
```

## Theoretical Foundation Summary

This implementation is based on the rigorous mathematical framework established in `改善案/論文/riemann_hypothesis_main.tex`. The key theoretical results that guarantee correctness and stability are:

### Core Theorems and Propositions

| Theorem/Proposition | Statement | Implementation Impact |
|---------------------|-----------|----------------------|
| **Proposition BK-formula** | d/dz log D_ε(z) = -Tr((H_ε-z)^{-1}-(H_0-z)^{-1}) | Enables scattering phase computation |
| **Proposition BS-HS** | \\|K_ε(z)\\|_S2 ≤ (1/2)(Im z)^{-1/2} \\|V_ε\\|_L2 | Guarantees Hilbert-Schmidt property |
| **Proposition BS-trace** | \\|K_ε(z)\\|_S1 ≤ (1/2)(Im z)^{-1} \\|V_ε\\|_L1 (ε > 1/2) | Guarantees trace-class property |
| **Theorem mourre-H0** | [H_0, iA] = I (optimal Mourre estimate) | Ensures numerical stability |
| **Theorem lap-H0** | Weighted resolvent extends to η = 0 | Enables boundary computation |
| **Corollary lap-Heps** | LAP holds for perturbed H_ε uniformly in ε | Uniform stability across ε schedule |
| **Theorem uniform-resolvent-bounds** | \\|(H_ε - λ - iη)^{-1}\\| ≤ C uniformly as η → 0 | Justifies Im z → 0 limit |
| **Corollary BK-boundary** | Birman-Krein formula extends to boundary | Enables scattering-based routing |
| **eq:explicit-formula** | Weil explicit formula via D_ε | Validates Prime-Bump initialization |

### Implementation Verification Strategy

For each theorem/proposition, we implement corresponding verification tests:

1. **Schatten Norm Tests:** Verify ||K_ε||_S1 and ||K_ε||_S2 satisfy bounds
2. **Mourre Estimate Test:** Verify [H_0, iA] = I numerically
3. **LAP Test:** Verify resolvent remains bounded as η → 0
4. **Boundary Extension Test:** Verify continuity of Birman-Krein derivative
5. **Weil Formula Test:** Verify prime sum matches spectral trace

**Reference:** All theorem numbers refer to `改善案/論文/riemann_hypothesis_main.tex`

## Components and Interfaces

### 1. Birman-Schwinger Core Module

**Purpose:** Implement the mathematically rigorous Birman-Schwinger operator K_ε(z) = |V_ε|^{1/2} R_0(z) |V_ε|^{1/2} with guaranteed trace-class properties.

**Interface:**
```python
class BirmanSchwingerCore(nn.Module):
    """
    Birman-Schwinger operator with LAP-based numerical stability.
    
    Args:
        n_seq: sequence length
        epsilon: regularization parameter (ε ∈ [0.5, 1.0])
        use_mourre: enable Mourre estimate verification
        use_lap: enable Limiting Absorption Principle
    """
    def __init__(self, n_seq: int, epsilon: float = 1.0, 
                 use_mourre: bool = True, use_lap: bool = True)
    
    def forward(self, v: Tensor, z: complex) -> Tensor:
        """
        Compute G_ii = diag((H_ε - zI)^{-1}) with stability guarantees.
        
        Args:
            v: (B, N) potential from Prime-Bump initialization
            z: complex shift (default: 1.0j)
        
        Returns:
            features: (B, N, 2) [real(G_ii), imag(G_ii)]
        """

    def compute_schatten_norms(self) -> Tuple[float, float]:
        """Return (||K||_S1, ||K||_S2) for monitoring."""
    
    def verify_mourre_estimate(self) -> bool:
        """Verify [H_0, iA] = I where A = position operator."""
    
    def apply_spectral_clipping(self, threshold: float):
        """Clip eigenvalues exceeding trace-class bounds."""
```

**Key Design Decisions:**

1. **Precision Strategy:** Use complex128 for recursions, complex64 for output (matching current BKCore)
2. **Stability Enforcement:** Automatic precision upgrade when condition number κ > 10^6
3. **Schatten Norm Monitoring:** Real-time tracking with automatic clipping when ||K||_S2 > C·ε^{-1/2}
4. **LAP Integration:** Weighted resolvent ⟨x⟩^{-s}(H - λ - iη)^{-1}⟨x⟩^{-s} with s=1

**Mathematical Guarantees (from Paper):**
- Hilbert-Schmidt bound (Proposition BS-HS): ||K_ε(z)||_S2 ≤ (1/2)(Im z)^{-1/2} ||V_ε||_L2
- Trace-class bound (Proposition BS-trace, ε > 1/2): ||K_ε(z)||_S1 ≤ (1/2)(Im z)^{-1} ||V_ε||_L1
- Mourre estimate (Theorem mourre-H0): [H_0, iA] = I (optimal with c_I = 1)
- Uniform strip bounds (Proposition BS-uniform): ||K_ε(z)||_Sp ≤ C_p η_0^{-1/p} ||V_ε||_Lp for Im z ≥ η_0
- LAP boundary extension (Corollary BK-boundary): Birman-Krein derivative extends continuously to Im z = 0

**Reference:** See `改善案/論文/riemann_hypothesis_main.tex` Section "Birman--Schwinger operator and Schatten bounds" and "Mourre Estimate and Limiting Absorption Principle"

### 2. Prime-Bump Potential Module

**Purpose:** Initialize model with structured potential V_ε(x) = Σ_p α_{p,k}(ε) ψ_ε(x - log p) that encodes prime number distribution.

**Interface:**
```python
class PrimeBumpPotential(nn.Module):
    """
    Prime-Bump potential with GUE eigenvalue statistics.
    
    Args:
        n_seq: sequence length
        epsilon: cutoff width (ε ∈ [0.5, 1.0])
        k_max: maximum prime power (default: 3)
    """
    def __init__(self, n_seq: int, epsilon: float = 1.0, k_max: int = 3)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Compute potential V_ε(x) with prime bumps.
        
        Args:
            x: (B, N, D) input features
        
        Returns:
            v: (B, N) potential values
        """
    
    def get_prime_indices(self) -> List[int]:
        """Return list of prime positions < n_seq."""
    
    def compute_alpha_coefficients(self, p: int, k: int) -> float:
        """Compute α_{p,k}(ε) = (log p) / p^{k(1/2+ε)}."""
    
    def verify_gue_statistics(self) -> Dict[str, float]:
        """Verify eigenvalue spacing follows Wigner surmise."""
```

**Key Design Decisions:**

1. **Initialization Strategy:** Add prime bumps to position embeddings (not learned from scratch)
2. **Cutoff Function:** ψ_ε(x) = ε^{-1/2} exp(-x²/(2ε)) with Gaussian profile
3. **Coefficient Scaling:** α_{p,k}(ε) = (log p) / p^{k(1/2+ε)} ensures finite L2 norm
4. **Epsilon Schedule:** Start ε=1.0, anneal to ε=0.5 over training

**Mathematical Properties (from Paper):**
- Finite overlap: supp(ψ_ε(· - log p)) ∩ supp(ψ_ε(· - log q)) = ∅ for |log p - log q| > 2√ε
- GUE statistics: eigenvalue spacing follows s·exp(-πs²/4) (Wigner surmise)
- Spectral shift: ξ(λ) = (1/π) Im log D_ε(λ + i0) matches prime counting function
- Canonical coefficients (Corollary canonical-V): α_{p,k}(ε) = (log p) p^{-k(1/2+ε)}
- Weil formula matching (eq:explicit-formula): Prime-bump potential exactly reproduces Weil's explicit formula for band-limited test functions

**Reference:** See `改善案/論文/riemann_hypothesis_main.tex` Section "Birman--Krein determinant and the Weil explicit formula"

### 3. Scattering-Based Router Module

**Purpose:** Replace learned MLP gating with parameter-free routing based on scattering phase δ_ε(λ).

**Interface:**
```python
class ScatteringRouter(nn.Module):
    """
    Parameter-free MoE routing using scattering phase.
    
    Args:
        num_experts: number of experts
        use_clark_measure: use Clark measure for routing
    """
    def __init__(self, num_experts: int, use_clark_measure: bool = False)
    
    def forward(self, G_ii: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Route tokens based on scattering phase.
        
        Args:
            G_ii: (B, N) complex resolvent diagonal
        
        Returns:
            expert_indices: (B, N, top_k) selected experts
            routing_weights: (B, N, top_k) mixing weights
        """
    
    def compute_scattering_phase(self, G_ii: Tensor) -> Tensor:
        """Compute δ_ε(λ) = arg(det_2(I + K_ε(λ + i0)))."""
    
    def compute_spectral_shift(self, lambda_: Tensor) -> Tensor:
        """Compute ξ(λ) = (1/π) Im log D_ε(λ + i0)."""
    
    def detect_resonances(self, D_eps: Tensor) -> Tensor:
        """Identify λ where |D_ε(λ + i0)| is small."""
```

**Key Design Decisions:**

1. **Phase-Based Routing:** Route to expert e if δ_ε(λ_i) ∈ [(e-1)π/E, eπ/E]
2. **Resonance Handling:** Use top-2/top-3 routing near resonances (|D_ε| small)
3. **Zero Parameters:** Purely physics-based, no learnable weights
4. **Birman-Krein Formula (Proposition BK-formula):** d/dλ log D_ε(λ) = -Tr((H_ε - λ)^{-1} - (H_0 - λ)^{-1})
5. **Boundary Extension (Corollary BK-boundary):** Formula extends continuously to Im z = 0 via LAP

**Mathematical Foundation:**
The scattering phase δ_ε(λ) = arg(det_2(I + K_ε(λ + i0))) is well-defined on the boundary due to:
- LAP ensuring uniform invertibility (Corollary lap-Heps)
- Unimodular boundary values of D_ε(λ) (Clark measure theory)
- Uniform resolvent bounds (Theorem uniform-resolvent-bounds)

**Reference:** See `改善案/論文/riemann_hypothesis_main.tex` Section "Mourre Estimate and Limiting Absorption Principle"

**Performance Expectations:**
- 10× faster than MLP gating (no forward pass)
- Equal or better routing quality (physics-informed)
- Interpretable: scattering phase correlates with linguistic difficulty


### 4. Semiseparable Matrix Structure

**Purpose:** Exploit H = tridiag + low_rank structure to reduce memory from O(N²) to O(N log N).

**Interface:**
```python
class SemiseparableMatrix:
    """
    Semiseparable matrix: H = T + U·V^T where rank(UV^T) << N.
    
    Args:
        n_seq: sequence length
        rank: low-rank component rank (default: log(n_seq))
    """
    def __init__(self, n_seq: int, rank: Optional[int] = None)
    
    def matvec(self, x: Tensor) -> Tensor:
        """O(N) matrix-vector product: y = H·x."""
    
    def factorize(self, H: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Decompose H into tridiagonal + low-rank.
        
        Returns:
            T: (N, N) tridiagonal part
            U: (N, r) left low-rank factor
            V: (N, r) right low-rank factor
        """
    
    def checkpoint_recompute(self, x: Tensor, k: int = 4) -> Tensor:
        """
        Gradient checkpointing with semiseparable structure.
        Store only tridiagonal, recompute low-rank.
        """
```

**Key Design Decisions:**

1. **Rank Selection:** r = ⌈log₂(N)⌉ for logarithmic memory growth
2. **Partitioning Strategy:** Keep tridiagonal on GPU, offload low-rank to CPU
3. **Checkpointing:** Store tridiagonal (O(N)), recompute low-rank (O(Nr))
4. **Memory Savings:** 70% reduction vs. dense attention

**Complexity Analysis:**
- Dense attention: O(N²) memory, O(N²D) compute
- Semiseparable: O(N log N) memory, O(ND log N) compute
- With checkpointing: O(N) memory, O(ND log N) compute

### 5. Adaptive Computation Time (ACT) Module

**Purpose:** Enable dynamic depth based on scattering phase for compute efficiency.

**Interface:**
```python
class ACTModule(nn.Module):
    """
    Adaptive Computation Time with scattering-based halting.
    
    Args:
        n_layers: maximum number of layers
        halt_threshold: scattering phase threshold for early exit
    """
    def __init__(self, n_layers: int, halt_threshold: float = 0.2)
    
    def forward(self, x: Tensor, scattering_phases: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute with adaptive depth.
        
        Args:
            x: (B, N, D) input
            scattering_phases: (B, N) per-token phases
        
        Returns:
            output: (B, N, D) processed features
            avg_depth: (B, N) average layers used per token
        """
    
    def compute_halting_probability(self, phase: Tensor) -> Tensor:
        """Compute p_halt from scattering phase."""
```

**Key Design Decisions:**

1. **Halting Criterion:** Exit early when δ_ε < 0.2 (low scattering = easy token)
2. **Full Depth:** Use all layers when δ_ε > 0.8 (high scattering = hard token)
3. **Expected Savings:** 40% FLOPs reduction while maintaining PPL within 5%

### 6. Clark Measure Compression

**Purpose:** Compress model via ε→0 limit while preserving spectral distribution.

**Interface:**
```python
class ClarkMeasureCompression:
    """
    Compress model using ε-parametrized family with Clark measure preservation.
    
    Args:
        epsilon_schedule: [1.0, 0.75, 0.5, 0.25, 0.1]
    """
    def __init__(self, epsilon_schedule: List[float])
    
    def compress(self, model: nn.Module, target_epsilon: float) -> nn.Module:
        """
        Compress model from current ε to target ε.
        
        Returns:
            compressed_model: smaller model with preserved μ_ε
        """
    
    def compute_clark_measure(self, D_eps: Tensor, lambda_: Tensor) -> Tensor:
        """Compute μ_ε(E) = (1/2π) ∫_E |D_ε(λ + i0)|^{-2} dλ."""
    
    def measure_total_variation(self, mu1: Tensor, mu2: Tensor) -> float:
        """Compute ||μ_1 - μ_2||_TV."""
```

**Key Design Decisions:**

1. **Progressive Compression:** ε = 1.0 → 0.75 → 0.5 → 0.25 → 0.1
2. **Distillation Loss:** L = L_CE + λ_Clark · ||μ_teacher - μ_student||²
3. **Target:** 10× parameter reduction with <15% PPL degradation

## Data Models

### Configuration Schema

```python
@dataclass
class MambaKillerConfig:
    """Configuration for Mamba-Killer ResNet-BK."""
    
    # Model architecture
    vocab_size: int = 30000
    d_model: int = 256
    n_layers: int = 8
    n_seq: int = 2048  # Support up to 1M
    
    # Birman-Schwinger parameters
    epsilon: float = 1.0  # Regularization parameter
    use_prime_bump: bool = True
    prime_bump_scale: float = 0.02
    k_max: int = 3  # Maximum prime power
    
    # Scattering router
    use_scattering_router: bool = True
    num_experts: int = 8
    top_k: int = 2
    use_clark_measure: bool = False
    
    # Numerical stability
    use_mourre: bool = True
    use_lap: bool = True
    schatten_threshold: float = 100.0
    precision_upgrade_threshold: float = 1e6
    
    # Semiseparable structure
    use_semiseparable: bool = True
    low_rank: Optional[int] = None  # Default: log(n_seq)
    use_cpu_offload: bool = False
    
    # Adaptive computation
    use_act: bool = True
    act_halt_threshold: float = 0.2
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 8
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    
    # Compression
    epsilon_schedule: List[float] = field(default_factory=lambda: [1.0, 0.75, 0.5, 0.25, 0.1])
    clark_measure_weight: float = 0.1
```

### Training State

```python
@dataclass
class TrainingState:
    """Training state for checkpointing and recovery."""
    
    epoch: int
    step: int
    epsilon: float  # Current ε value
    best_ppl: float
    
    # Numerical health
    schatten_norms: List[Tuple[float, float]]  # [(||K||_S1, ||K||_S2)]
    condition_numbers: List[float]
    gradient_norms: List[float]
    
    # Routing statistics
    routing_entropy: List[float]
    expert_usage: Dict[int, float]  # expert_id -> usage_rate
    
    # Failure recovery
    nan_count: int
    oom_count: int
    last_stable_checkpoint: str
```


### Benchmark Results Schema

```python
@dataclass
class BenchmarkResults:
    """Results from Mamba comparison benchmarks."""
    
    # Long-context stability
    sequence_lengths: List[int]  # [8k, 32k, 128k, 512k, 1M]
    resnetbk_loss: List[float]
    mamba_loss: List[float]
    resnetbk_diverged: List[bool]
    mamba_diverged: List[bool]
    
    # Quantization robustness
    bit_widths: List[int]  # [32, 16, 8, 4, 2]
    resnetbk_ppl: List[float]
    mamba_ppl: List[float]
    
    # Dynamic efficiency
    flops_budgets: List[float]
    resnetbk_ppl_at_flops: List[float]
    mamba_ppl_at_flops: List[float]
    
    # Statistical tests
    p_values: Dict[str, float]  # metric -> p-value
    confidence_intervals: Dict[str, Tuple[float, float]]
```

## Error Handling

### Numerical Stability Monitoring

```python
class StabilityMonitor:
    """Real-time monitoring of numerical health."""
    
    def check_tensors(self, tensors: Dict[str, Tensor]) -> Dict[str, bool]:
        """Check for NaN/Inf in all tensors."""
        return {name: torch.isfinite(t).all().item() for name, t in tensors.items()}
    
    def check_condition_number(self, H: Tensor) -> float:
        """Compute κ(H) = σ_max / σ_min."""
        eigenvalues = torch.linalg.eigvalsh(H)
        return (eigenvalues.max() / eigenvalues.min()).item()
    
    def check_schatten_bounds(self, K: Tensor, epsilon: float, z: complex) -> bool:
        """Verify ||K||_S2 ≤ (1/2)(Im z)^{-1/2} ||V||_L2."""
        pass
    
    def suggest_recovery(self, failure_type: str) -> str:
        """Suggest recovery action based on failure mode."""
        recovery_map = {
            "nan_detected": "rollback_checkpoint",
            "gradient_explosion": "reduce_lr_10x",
            "loss_divergence": "increase_epsilon",
            "oom": "reduce_batch_size",
            "condition_number_high": "upgrade_precision",
        }
        return recovery_map.get(failure_type, "halt_training")
```

### Automatic Recovery System

```python
class AutoRecovery:
    """Automatic failure detection and recovery."""
    
    def __init__(self, checkpoint_dir: str, max_retries: int = 3):
        self.checkpoint_dir = checkpoint_dir
        self.max_retries = max_retries
        self.retry_count = 0
    
    def detect_failure(self, state: TrainingState) -> Optional[str]:
        """Detect failure mode from training state."""
        if state.nan_count > 0:
            return "nan_detected"
        if state.gradient_norms[-1] > 10 * np.median(state.gradient_norms):
            return "gradient_explosion"
        if state.condition_numbers[-1] > 1e6:
            return "condition_number_high"
        return None
    
    def recover(self, failure_type: str, model: nn.Module, optimizer: Optimizer) -> bool:
        """Attempt recovery from failure."""
        if self.retry_count >= self.max_retries:
            return False
        
        action = StabilityMonitor().suggest_recovery(failure_type)
        
        if action == "rollback_checkpoint":
            self.load_last_stable_checkpoint(model, optimizer)
        elif action == "reduce_lr_10x":
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        elif action == "increase_epsilon":
            # Increase ε to improve stability
            for module in model.modules():
                if hasattr(module, 'epsilon'):
                    module.epsilon = min(module.epsilon * 1.5, 1.0)
        
        self.retry_count += 1
        return True
```

## Testing Strategy

### Unit Tests

1. **Birman-Schwinger Core Tests**
   - Verify Hilbert-Schmidt bound: ||K_ε||_S2 ≤ (1/2)(Im z)^{-1/2} ||V_ε||_L2
   - Verify trace-class bound: ||K_ε||_S1 ≤ (1/2)(Im z)^{-1} ||V_ε||_L1
   - Test Mourre estimate: [H_0, iA] = I
   - Test LAP: uniform bound as η → 0

2. **Prime-Bump Potential Tests**
   - Verify prime positions match sieve
   - Verify α_{p,k} coefficients
   - Test GUE eigenvalue spacing (Wigner surmise)
   - Test finite overlap condition

3. **Scattering Router Tests**
   - Verify phase computation: δ_ε(λ) = arg(det_2(I + K_ε))
   - Test resonance detection
   - Verify routing is deterministic (no randomness)
   - Compare routing speed vs. MLP gating

4. **Semiseparable Structure Tests**
   - Verify O(N) matvec complexity
   - Test factorization accuracy: ||H - (T + UV^T)||_F < ε
   - Test checkpointing memory savings

### Integration Tests

1. **End-to-End Training**
   - Train 1M parameter model on WikiText-2
   - Verify no NaN/Inf over 10 epochs
   - Verify convergence (PPL decreases)

2. **Long-Context Scaling**
   - Train on N ∈ {512, 2048, 8192, 32768}
   - Verify memory scales as O(N log N)
   - Verify no divergence at N=32768

3. **Quantization Pipeline**
   - Train FP32 model
   - Apply INT8 QAT
   - Verify PPL degradation < 5%

### Benchmark Tests

1. **Mamba Comparison**
   - Implement fair comparison harness
   - Use identical hyperparameters
   - Run on same hardware (T4 GPU)
   - Generate all three "killer graphs"

2. **Ablation Studies**
   - Disable Prime-Bump init → measure convergence speed
   - Disable scattering router → measure routing quality
   - Disable LAP → measure numerical stability
   - Disable semiseparable → measure memory usage

3. **Statistical Validation**
   - Run 5 seeds per configuration
   - Compute mean ± std
   - Perform paired t-tests (p < 0.01)
   - Apply Bonferroni correction


## Implementation Strategy

### Phase 1: Mathematical Foundations (Weeks 1-2)

**Goal:** Implement core mathematical components with rigorous verification.

**Tasks:**
1. Implement Birman-Schwinger kernel with Schatten norm monitoring
2. Implement Prime-Bump potential with GUE verification
3. Implement Mourre estimate and LAP
4. Write comprehensive unit tests for all mathematical properties

**Success Criteria:**
- All Schatten norm bounds verified
- GUE eigenvalue spacing matches Wigner surmise (p < 0.01)
- Mourre estimate holds: [H_0, iA] = I (error < 1e-6)
- LAP uniform bound verified as η → 0

**Risks:**
- Numerical instability in resolvent computation
- Mitigation: Use LAP with weighted spaces, automatic precision upgrade

### Phase 2: Scattering Router (Weeks 3-4)

**Goal:** Replace learned MLP gating with parameter-free scattering-based routing.

**Tasks:**
1. Implement scattering phase computation: δ_ε(λ) = arg(det_2(I + K_ε))
2. Implement spectral shift function: ξ(λ) = (1/π) Im log D_ε(λ)
3. Implement resonance detection and adaptive top-k
4. Benchmark routing speed vs. MLP gating

**Success Criteria:**
- Routing is 10× faster than MLP gating
- Routing quality equal or better (measured by PPL)
- Scattering phase correlates with linguistic difficulty
- Zero learnable parameters in routing

**Risks:**
- Scattering-based routing may underperform learned routing
- Mitigation: Hybrid approach (scattering + small learned correction)

### Phase 3: Semiseparable Structure (Weeks 5-6)

**Goal:** Exploit H = tridiag + low_rank to enable ultra-large scale training.

**Tasks:**
1. Implement semiseparable factorization
2. Implement O(N) matvec and gradient computation
3. Implement gradient checkpointing with structure awareness
4. Implement CPU offloading for low-rank factors

**Success Criteria:**
- Memory usage: O(N log N) instead of O(N²)
- 70% memory reduction vs. dense attention
- Train 1B parameters on single T4 GPU
- Train 10B parameters on 4× T4 GPUs

**Risks:**
- Factorization may not be accurate enough
- Mitigation: Adaptive rank selection, iterative refinement

### Phase 4: Long-Context Stability (Weeks 7-8)

**Goal:** Demonstrate stable training on 128k-1M token sequences.

**Tasks:**
1. Implement hierarchical semiseparable structure
2. Implement streaming evaluation for ultra-long sequences
3. Train on N ∈ {8k, 32k, 128k, 512k, 1M}
4. Compare to Mamba baseline

**Success Criteria:**
- No divergence at N=128k (while Mamba diverges at N=32k)
- Gradient norms remain stable (no spikes)
- PPL degradation < 30% from N=1k to N=128k
- Generate "Long-Context Stability Graph"

**Risks:**
- Memory may still be insufficient for N=1M
- Mitigation: Use model parallelism, sequence parallelism

### Phase 5: Quantization Robustness (Weeks 9-10)

**Goal:** Demonstrate superior quantization performance vs. Mamba.

**Tasks:**
1. Implement quantization-aware training (QAT)
2. Implement INT8 and INT4 quantization
3. Implement mixed-precision quantization
4. Compare to Mamba across bit widths

**Success Criteria:**
- INT8: PPL degradation < 5% (Mamba: >10%)
- INT4: PPL degradation < 15% (Mamba: >30%)
- Generate "Quantization Robustness Graph"
- Show 4× lower PPL than Mamba at INT4

**Risks:**
- Complex-valued operations may not quantize well
- Mitigation: Separate quantization for real/imaginary parts

### Phase 6: Dynamic Efficiency (Weeks 11-12)

**Goal:** Achieve 2× lower FLOPs than Mamba at equal PPL.

**Tasks:**
1. Implement ACT with scattering-based halting
2. Implement learned sparsity for G_ii computation
3. Implement multi-scale processing
4. Measure FLOPs and generate efficiency graph

**Success Criteria:**
- 40% FLOPs reduction with ACT (PPL within 5%)
- 2× lower FLOPs than Mamba at equal PPL
- Generate "Dynamic Efficiency Graph"
- Show Pareto dominance over Mamba

**Risks:**
- ACT may not save enough FLOPs
- Mitigation: Combine with learned sparsity, multi-scale

### Phase 7: Benchmark Pipeline (Weeks 13-14)

**Goal:** Create automated benchmark system for fair Mamba comparison.

**Tasks:**
1. Implement fair comparison harness (identical hyperparameters)
2. Implement multi-dataset evaluation (WikiText-2, WikiText-103, C4, Pile)
3. Implement statistical testing (bootstrap, permutation tests)
4. Generate all three "killer graphs"

**Success Criteria:**
- All comparisons use identical settings
- Statistical significance: p < 0.01 with Bonferroni correction
- Reproducibility: variance < 2% across 5 runs
- Publication-quality figures (300 DPI, vector graphics)

### Phase 8: Reproducibility Package (Weeks 15-16)

**Goal:** Enable independent verification of all results.

**Tasks:**
1. Create Docker container with all dependencies
2. Create Google Colab notebooks for all experiments
3. Upload checkpoints to Hugging Face Hub
4. Write comprehensive documentation

**Success Criteria:**
- One-click execution on Google Colab
- Complete in < 48 hours on free tier
- Independent researchers can reproduce within 2% variance
- 1000+ GitHub stars within 6 months

## Performance Targets

### Computational Complexity

| Operation | Current | Target | Improvement |
|-----------|---------|--------|-------------|
| Forward pass | O(N) | O(N) | - |
| Backward pass | O(N) | O(N) | - |
| Memory (dense) | O(N²) | O(N log N) | 100× @ N=10k |
| Memory (semisep) | - | O(N) | - |
| Routing | O(ND) | O(N) | D× |

### Benchmark Targets

| Metric | Current | Target | Mamba |
|--------|---------|--------|-------|
| WikiText-2 PPL | 1122 | < 50 | ~30 |
| Max stable N | 2048 | 131072 | 32768 |
| INT8 PPL degradation | - | < 5% | > 10% |
| INT4 PPL degradation | - | < 15% | > 30% |
| FLOPs @ PPL=30 | - | 50% | 100% |
| Training speed | 1× | 2× | - |

### Scalability Targets

| Configuration | Parameters | Hardware | Time |
|---------------|------------|----------|------|
| Small | 10M | 1× T4 | 2 hours |
| Medium | 100M | 1× T4 | 12 hours |
| Large | 1B | 1× T4 | 48 hours |
| XL | 10B | 4× T4 | 7 days |
| XXL | 100B | 8× A100 | 30 days |

## Dependencies and Integration

### External Dependencies

```python
# Core dependencies
torch >= 2.0.0
numpy >= 1.24.0
scipy >= 1.10.0

# Numerical libraries
mpmath >= 1.3.0  # High-precision arithmetic
sympy >= 1.12  # Symbolic math for verification

# Optimization
triton >= 2.0.0  # Custom CUDA kernels
deepspeed >= 0.10.0  # ZeRO optimizer

# Benchmarking
datasets >= 2.14.0  # Hugging Face datasets
transformers >= 4.30.0  # Mamba baseline
wandb >= 0.15.0  # Experiment tracking

# Visualization
matplotlib >= 3.7.0
seaborn >= 0.12.0
plotly >= 5.15.0
```

### Integration Points

1. **Existing BK-Core:** Extend with Birman-Schwinger operator
2. **Existing MoE:** Replace with ScatteringMoE
3. **Existing Training Loop:** Add stability monitoring and auto-recovery
4. **Existing Benchmarks:** Extend with Mamba comparison

### Backward Compatibility

- Maintain existing BKCoreFunction interface
- Add `use_birman_schwinger` flag for gradual migration
- Support both learned and scattering-based routing
- Preserve checkpoint format


## Security and Privacy Considerations

### Data Security

1. **Dataset Handling:** All datasets (WikiText-2, C4, Pile) are public and properly licensed
2. **Checkpoint Security:** Encrypt model checkpoints before uploading to Hugging Face Hub
3. **API Keys:** Store W&B and HF tokens in environment variables, never in code

### Computational Security

1. **Resource Limits:** Implement OOM protection to prevent system crashes
2. **Timeout Handling:** Graceful shutdown when Colab timeout approaches
3. **Checkpoint Integrity:** Verify checksums before loading checkpoints

### Reproducibility Security

1. **Random Seeds:** Fix all random seeds for reproducibility
2. **Deterministic Operations:** Use `torch.use_deterministic_algorithms(True)`
3. **Version Pinning:** Pin all dependency versions in requirements.txt

## Monitoring and Observability

### Real-Time Dashboards

```python
class TrainingDashboard:
    """Real-time monitoring dashboard."""
    
    def __init__(self, wandb_project: str):
        self.wandb = wandb.init(project=wandb_project)
    
    def log_metrics(self, step: int, metrics: Dict[str, float]):
        """Log training metrics."""
        self.wandb.log(metrics, step=step)
    
    def log_stability(self, step: int, state: TrainingState):
        """Log numerical stability metrics."""
        self.wandb.log({
            "stability/schatten_s1": state.schatten_norms[-1][0],
            "stability/schatten_s2": state.schatten_norms[-1][1],
            "stability/condition_number": state.condition_numbers[-1],
            "stability/gradient_norm": state.gradient_norms[-1],
        }, step=step)
    
    def log_routing(self, step: int, state: TrainingState):
        """Log routing statistics."""
        self.wandb.log({
            "routing/entropy": state.routing_entropy[-1],
            **{f"routing/expert_{i}_usage": usage 
               for i, usage in state.expert_usage.items()},
        }, step=step)
    
    def create_visualization(self, results: BenchmarkResults):
        """Create and upload visualization."""
        fig = self.plot_killer_graphs(results)
        self.wandb.log({"killer_graphs": wandb.Image(fig)})
```

### Alerting System

```python
class AlertSystem:
    """Alert on critical failures."""
    
    def __init__(self, email: Optional[str] = None):
        self.email = email
    
    def check_health(self, state: TrainingState) -> List[str]:
        """Check training health and return alerts."""
        alerts = []
        
        if state.nan_count > 0:
            alerts.append("NaN detected in tensors")
        
        if state.condition_numbers[-1] > 1e6:
            alerts.append("Condition number exceeds 1e6")
        
        if state.gradient_norms[-1] > 1000:
            alerts.append("Gradient explosion detected")
        
        return alerts
    
    def send_alert(self, message: str):
        """Send alert via email or logging."""
        if self.email:
            # Send email (implementation omitted)
            pass
        else:
            logging.critical(f"ALERT: {message}")
```

## Documentation Plan

### User Documentation

1. **README.md:** Overview, quick start, installation
2. **TUTORIAL.md:** Step-by-step guide for training and evaluation
3. **API_REFERENCE.md:** Complete API documentation
4. **FAQ.md:** Common questions and troubleshooting

### Developer Documentation

1. **ARCHITECTURE.md:** Detailed architecture description
2. **CONTRIBUTING.md:** Contribution guidelines
3. **TESTING.md:** Testing strategy and how to run tests
4. **BENCHMARKING.md:** How to run benchmarks and interpret results

### Research Documentation

1. **THEORY.md:** Mathematical foundations and proofs
2. **EXPERIMENTS.md:** Experimental setup and results
3. **ABLATIONS.md:** Ablation study results
4. **COMPARISON.md:** Detailed Mamba comparison

### Colab Notebooks

1. **quick_start.ipynb:** 5-minute introduction
2. **full_training.ipynb:** Complete training pipeline
3. **benchmarking.ipynb:** Run all benchmarks
4. **visualization.ipynb:** Generate killer graphs
5. **ablation_studies.ipynb:** Run ablation experiments

## Deployment Strategy

### Hugging Face Hub Integration

```python
from transformers import PreTrainedModel, PretrainedConfig

class ResNetBKConfig(PretrainedConfig):
    """Configuration for Hugging Face integration."""
    model_type = "resnet-bk"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Copy from MambaKillerConfig

class ResNetBKForCausalLM(PreTrainedModel):
    """Hugging Face compatible model."""
    config_class = ResNetBKConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.model = LanguageModel(**config.to_dict())
    
    def forward(self, input_ids, **kwargs):
        return self.model(input_ids)

# Upload to Hub
model.push_to_hub("resnet-bk/mamba-killer-1b")
```

### PyTorch Hub Integration

```python
# hubconf.py
dependencies = ['torch', 'numpy']

def resnet_bk_1b(pretrained=True, **kwargs):
    """Load 1B parameter ResNet-BK model."""
    model = LanguageModel(vocab_size=30000, d_model=1024, n_layers=24, **kwargs)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            'https://github.com/neko-jpg/Project-ResNet-BK-An-O-N-Language-Model-Architecture/releases/download/v1.0/resnet_bk_1b.pt'
        )
        model.load_state_dict(checkpoint)
    return model
```

### ONNX Export

```python
def export_to_onnx(model: nn.Module, output_path: str):
    """Export model to ONNX format."""
    dummy_input = torch.randint(0, 30000, (1, 128))
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={'input_ids': {0: 'batch', 1: 'sequence'}},
        opset_version=14,
    )
```

## Risk Assessment and Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Numerical instability | High | High | LAP, Mourre estimate, auto-recovery |
| Scattering router underperforms | Medium | Medium | Hybrid learned + physics approach |
| Memory insufficient for 1M tokens | Medium | High | Hierarchical semiseparable, streaming |
| Quantization degrades performance | Low | Medium | QAT, mixed-precision quantization |
| Mamba comparison unfair | Low | High | Use official implementation, identical settings |

### Research Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Results not reproducible | Low | Critical | Docker, Colab, fixed seeds, version pinning |
| Statistical significance not achieved | Medium | High | Increase sample size, use stronger baselines |
| Paper rejected | Medium | Medium | Incorporate feedback, run additional experiments |
| Community skepticism | Medium | Medium | Open source, reproducibility package, theory |

### Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Colab timeout | High | Low | Auto-save, resume from checkpoint |
| OOM on Colab | Medium | Medium | Gradient checkpointing, CPU offload |
| Hardware failure | Low | Medium | Cloud backup, multiple checkpoints |
| Dependency conflicts | Low | Low | Docker container, version pinning |

## Success Metrics

### Primary Metrics (Must Achieve)

1. **Long-Context Stability:** Stable training at N=128k (Mamba diverges at N=32k)
2. **Quantization Robustness:** INT4 PPL < 50 (Mamba: >200)
3. **Dynamic Efficiency:** 2× lower FLOPs at equal PPL
4. **Statistical Significance:** p < 0.01 on all three axes

### Secondary Metrics (Should Achieve)

1. **Scalability:** Train 10B parameters on 4× T4 GPUs
2. **Speed:** 2× faster training than current implementation
3. **Memory:** 70% reduction vs. dense attention
4. **Reproducibility:** <2% variance across runs

### Stretch Goals (Nice to Have)

1. **Ultra-Scale:** Train 100B parameters on 8× A100 GPUs
2. **Ultra-Long:** Stable training at N=1M tokens
3. **Ultra-Efficient:** 5× lower FLOPs than Mamba
4. **Community Impact:** 1000+ GitHub stars, 100+ citations

## Timeline and Milestones

### Month 1: Foundations
- Week 1-2: Birman-Schwinger core + Prime-Bump potential
- Week 3-4: Scattering router + unit tests
- **Milestone:** All mathematical properties verified

### Month 2: Scalability
- Week 5-6: Semiseparable structure + checkpointing
- Week 7-8: Long-context training (up to 128k)
- **Milestone:** 1B parameters on single T4 GPU

### Month 3: Optimization
- Week 9-10: Quantization (INT8, INT4)
- Week 11-12: ACT + dynamic efficiency
- **Milestone:** All three "killer graphs" generated

### Month 4: Validation
- Week 13-14: Benchmark pipeline + statistical tests
- Week 15-16: Reproducibility package + documentation
- **Milestone:** Paper submission ready

## Conclusion

This design provides a comprehensive roadmap for transforming ResNet-BK into a Mamba-killer architecture. The key innovations are:

1. **Mathematical Rigor:** Birman-Schwinger operator theory provides provable stability guarantees
2. **Parameter-Free Routing:** Scattering phase eliminates learned gating overhead
3. **Semiseparable Structure:** Enables training at unprecedented scale (10B+ parameters on Colab)
4. **Three-Axis Superiority:** Dominates Mamba on stability, quantization, and efficiency

The implementation follows a phased approach with clear milestones and success criteria. Each phase builds on the previous, with comprehensive testing and validation at every step. The final deliverable will be a complete reproducibility package that enables independent verification of all claims.

**Next Steps:**
1. Review this design document with stakeholders
2. Set up development environment and CI/CD pipeline
3. Begin Phase 1 implementation (Birman-Schwinger core)
4. Establish baseline benchmarks for comparison
