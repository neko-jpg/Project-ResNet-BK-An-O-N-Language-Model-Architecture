# Phase 2 Docstring Quick Reference

**目的**: Phase 2モジュールのdocstring構造と物理的直観の素早い参照

---

## 📚 モジュール構成

### 1. BK-Core Triton Kernel
**ファイル**: `src/kernels/bk_scan.py`

**物理的直観**:
```
Birman-Schwinger核 = 三重対角行列の逆行列対角要素
量子散乱問題の解 = G_ii = diag((H - zI)^(-1))
```

**主要数式**:
```
Forward:  theta_i = (V_i - z - |h0|^2 / theta_{i-1})^(-1)
Backward: phi_i = (V_i - z - |h0|^2 / phi_{i+1})^(-1)
Result:   G_ii = theta_i * phi_i / det
```

**性能目標**: 3.0x+ speedup, MSE < 1e-6

---

### 2. Non-Hermitian Potential
**ファイル**: `src/models/phase2/non_hermitian.py`

**物理的直観**:
```
開放量子系 = 環境との相互作用でエネルギー散逸
H_eff = H_0 + V - iΓ
Γ > 0 → 情報の自然な忘却
```

**主要数式**:
```
Time evolution: ||ψ(t)||² = exp(-2Γt) ||ψ(0)||²
Overdamping: Γ >> |V| → Pure dissipation
```

**使用例**:
```python
potential = NonHermitianPotential(d_model=512, n_seq=1024)
V_complex = potential(x)  # (B, N) complex64
gamma = -V_complex.imag   # Positive decay rate
```

---

### 3. Dissipative Hebbian Layer
**ファイル**: `src/models/phase2/dissipative_hebbian.py`

**物理的直観**:
```
Hebbの法則 + 散逸 = 記憶形成と忘却の統合
dW/dt = η(k^T v) - ΓW
生物のシナプス可塑性を完全複製
```

**主要数式**:
```
Continuous: dW/dt = η(k^T v) - ΓW
Discrete:   W_new = exp(-Γ*dt) * W_old + η * (k^T v)
Lyapunov:   E = ||W||², dE/dt ≤ 0 (stable)
```

**Key Innovation**:
```
Memory → Potential Feedback:
W → V(x, M) → BK-Core → Output
Phase 2 = "Dynamically adjusting Phase 1's H based on M"
```

---

### 4. SNR Memory Filter
**ファイル**: `src/models/phase2/memory_selection.py`

**物理的直観**:
```
脳 = 重要な記憶だけを長期保持
SNR = 信号強度 / ノイズレベル
高SNR → 保持・強化
低SNR → 急速忘却
```

**主要数式**:
```
SNR_i = |W_i| / σ_noise
σ_noise = std(W) + ε

Adaptive:
  SNR < τ → Γ *= gamma_boost (forget)
  SNR > τ → η *= eta_boost (learn)
```

**デフォルト値**:
- τ (threshold) = 2.0
- gamma_boost = 2.0
- eta_boost = 1.5

---

### 5. Memory Resonance Layer
**ファイル**: `src/models/phase2/memory_resonance.py`

**物理的直観**:
```
量子固有状態 = 互いに直交 → 干渉最小化
ゼータ零点 = 最も規則的なランダム性 (GUE統計)
この基底で対角化 → 記憶の干渉最小化
```

**主要数式**:
```
Diagonalization: W' = U^(-1) W U
Basis matrix:    U[i,j] = exp(2πi * gamma_j * i / N)
Zeta zeros:      gamma_j = j-th zero imaginary part

Energy filter:   Keep only |W'_ii| > threshold
```

**最適化**:
```
U is model-fixed (input-independent)
→ Compute once, cache per (dim, device)
→ Dramatically reduces per-step cost
```

---

### 6. Zeta Initialization
**ファイル**: `src/models/phase2/zeta_init.py`

**物理的直観**:
```
リーマンゼータ零点 = 量子カオス系のエネルギー準位
GUE統計 = 最大エントロピー分布
特異値分布 = 情報の分散度合い
→ 干渉最小化、効率的な分散表現
```

**主要数式**:
```
Linear Init:
  W = U S V^T (SVD)
  S_i = scale / zero_i
  W_new = U S_new V^T

Position Embedding:
  PE(pos, 2i) = sin(pos * gamma_i / (2π))
  PE(pos, 2i+1) = cos(pos * gamma_i / (2π))
```

**ゼータ零点**:
```
Precise (n ≤ 10): 14.13, 21.02, 25.01, ...
Approximate (n > 10): GUE statistics
```

---

### 7. Gradient Safety
**ファイル**: `src/models/phase2/gradient_safety.py`

**物理的直観**:
```
複素勾配 = 実部と虚部の両方に勾配
NaN/Inf = 数値不安定性の兆候
クリッピング = 勾配爆発の防止
```

**Safety Mechanisms**:
```
1. NaN/Inf Detection: torch.isfinite()
2. Replacement: NaN/Inf → 0
3. Clipping: ||grad|| > threshold → scale down
4. Monitoring: Track norm, NaN count, clip count
```

**デフォルト閾値**: max_grad_norm = 1000.0

---

### 8. Integrated Model
**ファイル**: `src/models/phase2/integrated_model.py`

**アーキテクチャ**:
```
Input → ZetaEmbedding → Phase2Block × N → Output

Phase2Block:
  x → [LN] → NonHermitian+BK-Core → [Residual]
    → [LN] → DissipativeHebbian → SNRFilter → MemoryResonance → [Residual]
    → [LN] → FFN → [Residual]
```

**物理的解釈**:
```
Static Phase 1 → Dynamic Phase 2
H (fixed) → H(t, M) (memory-dependent)
V (static) → V(x, M) (adaptive)
```

---

### 9. Factory and Configuration
**ファイル**: `src/models/phase2/factory.py`

**主要機能**:
```
1. create_phase2_model: Create from config
2. convert_phase1_to_phase2: Convert Phase 1 → Phase 2
3. Presets: small, base, large
```

**設定例**:
```python
# Default
config = Phase2Config()

# From Phase 1
config = Phase2Config.from_phase1(phase1_config)

# Custom
config = Phase2Config(
    d_model=1024,
    n_layers=12,
    base_decay=0.02,
    hebbian_eta=0.15
)
```

---

## 🔬 物理的直観マップ

### 散逸 (Dissipation)
```
NonHermitian: Γ > 0 → Energy loss
DissipativeHebbian: -ΓW → Synaptic decay
SNRFilter: Low SNR → Increase Γ
```

### 記憶 (Memory)
```
DissipativeHebbian: η(k^T v) → Memory formation
SNRFilter: High SNR → Increase η
MemoryResonance: Diagonalization → Organization
```

### 安定性 (Stability)
```
Lyapunov: dE/dt ≤ 0 → Stable
GradientSafety: Clipping → Prevent explosion
NonHermitian: Γ/|V| < 10 → Not overdamped
```

### 効率性 (Efficiency)
```
BK-Core Triton: 3x+ speedup
MemoryResonance: Basis caching
SNRFilter: Selective retention (80% filtered)
```

---

## 📊 数式クイックリファレンス

### 時間発展
```
Continuous: dW/dt = η(k^T v) - ΓW
Discrete:   W(t+dt) = exp(-Γ*dt) * W(t) + η * (k^T v)
Quantum:    ||ψ(t)||² = exp(-2Γt) ||ψ(0)||²
```

### 記憶選択
```
SNR = |W| / σ_noise
Importance = w_snr * SNR + w_energy * E + w_recency * R
```

### 共鳴
```
Diagonalization: W' = U^(-1) W U
Basis: U[i,j] = exp(2πi * gamma_j * i / N) / sqrt(N)
Filter: Keep |W'_ii| > threshold
```

### 初期化
```
Singular values: S_i = scale / zero_i
Position encoding: PE(pos, 2i) = sin(pos / zero_i)
```

---

## 🎯 性能目標

### BK-Core Triton
- Speedup: **3.0x+** vs PyTorch vmap
- Numerical error: **MSE < 1e-6**
- NaN rate: **0%** (100 trials)

### Memory Efficiency
- VRAM: **< 8.0 GB** (Batch=1, Seq=4096, fp16)
- Fast Weights: **< 1.5 GB** additional

### Computational Cost
- Resonance layer: **< 20%** of total time
- Sparsity: **80%+** filtered

### Accuracy
- PPL degradation: **< +5%** vs Phase 1
- Gradient norm: **> 1e-5** (Seq=4096, end→start)
- Γ variation: **> 0.1** (initial vs trained)

---

## 💡 使用例テンプレート

### 基本使用
```python
from src.models.phase2 import Phase2IntegratedModel, Phase2Config

config = Phase2Config(vocab_size=50257, d_model=512, n_layers=6)
model = Phase2IntegratedModel(config)

input_ids = torch.randint(0, 50257, (4, 1024))
logits = model(input_ids)
```

### 診断情報
```python
logits, diag = model(input_ids, return_diagnostics=True)

print(f"Mean Γ: {diag['gamma_values'][0].mean():.4f}")
print(f"Mean SNR: {diag['snr_stats'][0]['mean_snr']:.4f}")
print(f"Resonant modes: {diag['resonance_info'][0]['num_resonant']:.1f}")
```

### Phase 1変換
```python
from src.models.phase2 import convert_phase1_to_phase2

phase2_model = convert_phase1_to_phase2(phase1_model, phase2_config)
```

---

## 📖 Requirements Coverage

| Requirement | Module | Status |
|------------|--------|--------|
| 1.1-1.8 | BK-Core Triton | ✅ |
| 2.1-2.5 | Gradient Safety | ✅ |
| 3.1-3.6 | Non-Hermitian | ✅ |
| 4.1-4.10 | Dissipative Hebbian | ✅ |
| 5.1-5.6 | Zeta Init | ✅ |
| 6.1-6.5 | Integrated Model | ✅ |
| 8.1-8.7 | Lyapunov Stability | ✅ |
| 9.1-9.7 | SNR Filter | ✅ |
| 10.1-10.7 | Memory Resonance | ✅ |
| 11.8 | Docstrings | ✅ |

---

## 🔍 デバッグチェックリスト

### Γ (Gamma) 関連
- [ ] Γ > 0 (always positive)
- [ ] Γ/|V| < 10 (not overdamped)
- [ ] Γ varies during training (> 0.1 change)

### Fast Weights 関連
- [ ] dE/dt ≤ 0 (Lyapunov stable)
- [ ] ||W|| bounded (no explosion)
- [ ] SNR > threshold for important memories

### 勾配関連
- [ ] No NaN/Inf in gradients
- [ ] ||grad|| < max_grad_norm
- [ ] Gradient flows to first layer (> 1e-5)

### 性能関連
- [ ] VRAM < 8.0 GB
- [ ] Triton speedup > 3.0x
- [ ] PPL degradation < +5%

---

**最終更新**: 2025-01-20  
**バージョン**: Phase 2.0  
**ステータス**: Complete
