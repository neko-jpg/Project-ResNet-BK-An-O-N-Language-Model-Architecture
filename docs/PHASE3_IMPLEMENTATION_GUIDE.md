# Phase 3 Implementation Guide: Physics Transcendence

## Introduction

Phase 3 "Physics Transcendence" transforms the discrete, layer-wise architecture of LLMs into a continuous, physics-based thinking process. This guide explains the core components and how to use them.

## Architecture Overview

The Phase 3 model integrates the following key technologies:

1.  **Complex Dynamics**: Using complex numbers to model amplitude (meaning) and phase (context/nuance).
2.  **Hamiltonian Neural ODE**: Ensuring logical consistency through energy conservation.
3.  **Koopman Operator**: Global linearization of non-linear dynamics for fast multi-step reasoning.
4.  **MERA Router**: Hierarchical entanglement renormalization for handling infinite context.
5.  **Dialectic Loop**: Self-correction mechanism based on Hegelian dialectics.

## Core Components

### 1. Complex Embedding (`src/models/phase3/complex_embedding.py`)

Represents tokens as complex vectors.
- **Real Part**: Basic semantic meaning.
- **Imaginary Part**: Contextual phase (e.g., negation, sarcasm).
- **Usage**:
  ```python
  from src.models.phase3.complex_embedding import ComplexEmbedding
  emb = ComplexEmbedding(vocab_size=50000, d_model=512, use_complex32=True)
  z = emb(input_ids) # Returns ComplexTensor
  ```

### 2. Hamiltonian Neural ODE (`src/models/phase3/hamiltonian_ode.py`)

Evolves the "thought state" (position $q$ and momentum $p$) over continuous time $t$.
- **Physics**: $H(q, p) = T(p) + V(q)$. Energy $H$ is conserved.
- **Fallback**: Automatically switches from Symplectic Adjoint (O(1) mem) to Checkpointing if numerical instability is detected.
- **Usage**:
  ```python
  ode = HamiltonianNeuralODE(d_model=512)
  state_t = ode(state_0, t_span=(0, 1))
  ```

### 3. Koopman Operator (`src/models/phase3/koopman.py`)

Linearizes the dynamics in a high-dimensional observable space.
- **Mechanism**: $x \to \Psi(x) \to K \cdot \Psi(x) \to \Psi^{-1}$.
- **Benefit**: Allows $K^n$ for extremely fast multi-step prediction.
- **Usage**:
  ```python
  koopman = KoopmanOperator(d_model=512, d_koopman=1024)
  x_next, g, g_next = koopman(x)
  ```

### 4. MERA Router (`src/models/phase3/mera.py`)

Aggregates information hierarchically using Disentanglers and Isometries.
- **Structure**: Tree-like tensor network (Log N depth).
- **Usage**:
  ```python
  mera = MERARouter(d_model=512)
  global_context, hierarchy = mera(x)
  ```

### 5. Dialectic Loop (`src/models/phase3/dialectic_loop.py`)

Generates hypotheses and critiques them using energy variance.
- **Thesis**: High-temp generation (Gumbel-Softmax).
- **Antithesis**: Hamiltonian critique (Energy Drift).
- **Synthesis**: Minimize contradiction score.
- **Usage**:
  ```python
  loop = DialecticLoop(d_model=512, vocab_size=50000, hamiltonian_ode=ode)
  logits, contradiction_loss, diag = loop(x)
  ```

## Training Strategy

Phase 3 requires a multi-stage training approach:

1.  **Stage 1**: Complex Dynamics (Train Complex Layers).
2.  **Stage 2**: Hamiltonian Integration (Train Potential V(q)).
3.  **Stage 3**: Full Integration (Train Koopman & Dialectic Loop).

Use `scripts/train_phase3.py` to start training.

## Troubleshooting

-   **NaN/Inf**: Usually indicates instability in Symplectic Adjoint. The model should auto-fallback, but if it persists, try reducing `dt` or increasing `recon_threshold`.
-   **High Energy Drift**: Indicates the learned potential is too sharp. Add regularization to V(q).
-   **Triton Errors**: Ensure GPU is available. The code falls back to PyTorch automatically on CPU.
