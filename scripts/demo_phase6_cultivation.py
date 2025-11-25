import torch
from src.models.phase6.geometry.hyperbolic import HyperbolicInitializer
from src.models.phase6.physics.fluid_dynamics import ViscousFlow
from src.models.phase6.physics.percolation import PercolationMonitor
from src.models.phase6.geometry.ricci_flow import RicciFlowSmoother

def demo_phase6_cultivation():
    print("--- MUSE Phase 6: Cultivation Process Demo ---")

    # 1. The Sacred Geometry
    print("\n[Step 1] Initializing Hyperbolic Vessel...")
    d_model = 16
    n_vocab = 100
    initializer = HyperbolicInitializer(d_model=d_model)
    coords = initializer.generate_tiling_coordinates(n_vocab)
    print(f"  Generated {len(coords)} hyperbolic coordinates.")
    print(f"  Root node norm: {coords[0].norm():.4f} (Should be small)")
    print(f"  Leaf node norm: {coords[-1].norm():.4f} (Should be larger)")

    # 2. Flow Resonance Injection
    print("\n[Step 2] Viscous Data Injection...")
    flow = ViscousFlow(vocab_size=n_vocab)

    # Simulate some data: Token 0 is frequent, Token 99 is rare
    batch = torch.cat([torch.zeros(10, dtype=torch.long), torch.tensor([99], dtype=torch.long)])
    flow.update_counts(batch)
    viscosity = flow.compute_viscosity(batch)

    print(f"  Viscosity for Frequent Token (0): {viscosity[0]:.4f} (Low)")
    print(f"  Viscosity for Rare Token (99): {viscosity[-1]:.4f} (High)")

    # 3. Harmonic Percolation
    print("\n[Step 3] Checking Knowledge Percolation...")
    monitor = PercolationMonitor()
    adj = torch.rand(5, 20, 20) # Batch=5, N=20
    # Threshold it to make it sparse
    adj = (adj > 0.8).float()

    status = monitor.check_criticality(adj)
    print(f"  Network State: {status['state']}")
    print(f"  Giant Component Ratio: {status['giant_component_ratio']:.2f}")

    # 4. Topological Polishing (Sleep)
    print("\n[Step 4] Offline Sleep Cycle (Ricci Flow)...")
    smoother = RicciFlowSmoother()
    raw_adj = torch.rand(20, 20)
    polished = smoother.evolve(raw_adj)

    change = (raw_adj - polished).norm().item()
    print(f"  Polished graph. Total metric change: {change:.4f}")

    print("\n--- Cultivation Demo Complete ---")

if __name__ == "__main__":
    demo_phase6_cultivation()
