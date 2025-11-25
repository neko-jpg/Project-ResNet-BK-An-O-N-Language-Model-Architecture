# MUSE Phase 6: Task Checklist

## Step 1: The Sacred Geometry (Hyperbolic Initialization)
- [ ] **Research:** Identify a fast algorithm for generating {7,3} hyperbolic tiling coordinates for $N$ points.
- [ ] **Implementation:** Create `src/models/phase6/geometry/hyperbolic.py`.
    - [ ] Implement `PoincareDisk` class.
    - [ ] Implement `generate_tiling_coordinates(N)`.
- [ ] **Integration:** Create `HyperbolicEmbedding` class inheriting from `nn.Embedding` but using the hyperbolic distance metric.
- [ ] **Verification:** Verify that the distance distribution follows an exponential law (capacity check).

## Step 2: Flow Resonance Injection (Viscous Data Loading)
- [ ] **Design:** Define the "Viscosity Score" heuristic (e.g., inverse frequency or gradient norm).
- [ ] **Implementation:** Create `src/models/phase6/physics/fluid_dynamics.py`.
    - [ ] Implement `ViscousDiffusion` kernel (simplified Navier-Stokes on graph).
- [ ] **Modification:** Update `PhysicsInformedTrainer` to accept viscosity parameters.
- [ ] **Verification:** Confirm that high-viscosity tokens drift towards the hyperbolic origin (index 0) over time.

## Step 3: Harmonic Percolation (Criticality Control)
- [ ] **Implementation:** Create `src/models/phase6/physics/percolation.py`.
    - [ ] Implement Union-Find or BFS to detect cluster sizes in the attention graph.
    - [ ] Implement `CriticalityController` (extends `HomeostasisController`).
- [ ] **Integration:** Hook into the training loop to adjust `gamma` or `temperature` based on percolation status.
- [ ] **Verification:** Plot cluster size vs. time; confirm it hovers around the phase transition point (Power law distribution).

## Step 4: Topological Polishing (Ricci Flow)
- [ ] **Research:** Adapt "Discrete Ollivier-Ricci Curvature" for attention graphs (simplified approximation needed for speed).
- [ ] **Implementation:** Create `src/models/phase6/geometry/ricci_flow.py`.
    - [ ] Implement `compute_curvature(adj_matrix)`.
    - [ ] Implement `evolve_metric(adj_matrix, curvature)`.
- [ ] **Workflow:** Create a `sleep_cycle.py` script that loads a checkpoint, runs Ricci Flow smoothing, and saves the polished model.
- [ ] **Verification:** Measure "Sheaf Energy" (Phase 5) before and after polishing; confirm reduction.

## Step 5: Final Integration (The Artifact)
- [ ] **Integration:** Create `Phase6ArtifactModel` wrapping Phase 5 with Phase 6 initialization and dynamics.
- [ ] **Demo:** Create `scripts/demo_phase6_artifact.py` demonstrating the "Crystallization" process.
