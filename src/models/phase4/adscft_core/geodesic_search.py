import torch

def fast_marching_method_cpu(
    bulk_coords: torch.Tensor,
    ads_radius: float = 1.0
) -> torch.Tensor:
    """
    Simplified Fast Marching Method (Dynamic Programming) for Bulk Geodesics.

    Args:
        bulk_coords: (B, N, bulk_dim, D)
        ads_radius: AdS curvature radius

    Returns:
        geodesics: (B, N, bulk_dim, D)
    """
    B, N, bulk_dim, D = bulk_coords.shape
    device = bulk_coords.device

    # Arrival Time map (Cost to reach each point from z=0)
    arrival_time = torch.full((B, N, bulk_dim), float('inf'), device=device)
    arrival_time[:, :, 0] = 0.0

    # Geodesic path points
    geodesics = bulk_coords.clone()

    # DP forward pass through z dimension
    # Note: This assumes causal/directed flow in z, which is a simplification of true FMM
    # but sufficient for "layer-wise" bulk generation.

    for z in range(1, bulk_dim):
        # Previous layer coords (from original bulk or updated geodesics?)
        # Design says bulk_coords for coords, but arrival time updates.
        # Wait, if we update geodesics, we should use the updated ones?
        # Design snippet uses `bulk_coords` for `prev` and `curr` but updates `geodesics`.
        # Let's follow design.

        prev_coords = bulk_coords[:, :, z-1] # (B, N, D)
        curr_coords = bulk_coords[:, :, z]   # (B, N, D)

        # AdS Metric Factor: g_zz = (L/z)^2
        # We use z index as proxy for depth z.
        # Avoid z=0 singularity by adding epsilon or offset
        z_val = float(z)
        metric_factor = (ads_radius / (z_val + 1e-6)) ** 2

        # Euclidean distance in embedding space
        dist = torch.norm(curr_coords - prev_coords, dim=-1) # (B, N)

        # Update cost
        # Cost = Cost_prev + Metric * Dist
        cost = arrival_time[:, :, z-1] + metric_factor * dist
        arrival_time[:, :, z] = cost

        # Update geodesic path (interpolate)
        # ratio acts as a weighting factor
        ratio = arrival_time[:, :, z-1] / (arrival_time[:, :, z] + 1e-6)

        geodesics[:, :, z] = prev_coords + (curr_coords - prev_coords) * ratio.unsqueeze(-1)

    return geodesics
