import torch
from src.models.phase8.topology import TopologicalNorm

def main():
    print("=== Topological Norm Demo ===")
    d_model = 16
    topo_norm = TopologicalNorm(d_model)

    # 1. Clustered Data (High Variance in Distances)
    c1 = torch.zeros(1, 10, d_model)
    c2 = torch.ones(1, 10, d_model) * 10
    x_clustered = torch.cat([c1, c2], dim=1)

    out_c = topo_norm(x_clustered)
    metric_c = topo_norm.get_diagnostics()['topo_metric']

    print(f"Clustered Data Metric (Variance): {metric_c:.4f}")

    # 2. Uniform Data (Low Variance)
    x_uniform = torch.rand(1, 20, d_model) * 10
    out_u = topo_norm(x_uniform)
    metric_u = topo_norm.get_diagnostics()['topo_metric']

    print(f"Uniform Data Metric (Variance):   {metric_u:.4f}")

    if metric_c > metric_u:
        print("SUCCESS: Clustered data detected as topologically more complex (higher variance).")
    else:
        print("FAIL: Metrics indistinguishable.")

if __name__ == "__main__":
    main()
