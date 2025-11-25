import torch
import torch.nn as nn

class PercolationMonitor(nn.Module):
    """
    Harmonic Percolation Monitor.

    Estimates the connectivity of the Knowledge Graph (Attention Matrix or similar).
    Detects Phase Transition (Criticality).
    """

    def __init__(self, threshold: float = 0.1):
        super().__init__()
        self.threshold = threshold

    def compute_cluster_size(self, adj_matrix: torch.Tensor) -> float:
        """
        Compute the size of the largest connected component (Giant Component)
        relative to the total size N.

        Args:
            adj_matrix: (B, N, N) adjacency (e.g. attention weights)

        Returns:
            giant_ratio: (B,) size of largest cluster / N
        """
        B, N, _ = adj_matrix.shape

        # Binarize
        mask = (adj_matrix > self.threshold).float()

        # Simple BFS or matrix power to find connectivity
        # For small N, matrix power is okay.
        # Reachability matrix R = (I + A)^N > 0

        # Ideally use Union-Find, but dense matrix multiplication is faster on GPU for small N
        # For N < 1024, we can do a few steps of propagation.

        # Propagate connectivity: X_{t+1} = sign(X_t @ X_t)
        # This approximates transitive closure.

        conn = mask
        # log(N) steps
        steps = int(torch.log2(torch.tensor(float(N))).item()) + 1
        for _ in range(steps):
            conn = torch.matmul(conn, conn)
            conn = (conn > 0).float()

        # Row sum roughly indicates cluster size reachable from node i
        cluster_sizes = conn.sum(dim=-1) # (B, N)
        max_cluster = cluster_sizes.max(dim=-1).values # (B,)

        return max_cluster / N

    def check_criticality(self, adj_matrix: torch.Tensor) -> dict:
        """
        Check if system is Critical, Sub-critical, or Super-critical.
        """
        ratio = self.compute_cluster_size(adj_matrix).mean().item()

        # Percolation threshold p_c for many graphs is roughly 1/mean_degree or related to size
        # We define "Critical" as having a giant component that covers ~50% of the nodes
        # (This is heuristic for small world networks)

        if ratio < 0.3:
            state = "sub_critical" # Fragmented
        elif ratio > 0.8:
            state = "super_critical" # Over-connected
        else:
            state = "critical" # Just right

        return {
            "state": state,
            "giant_component_ratio": ratio
        }
