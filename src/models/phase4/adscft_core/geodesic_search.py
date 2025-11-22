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

    # We build lists to avoid in-place modification issues with autograd
    arrival_times = []
    geodesic_slices = []

    # Initial condition (z=0)
    # arrival_time[0] = 0
    arrival_times.append(torch.zeros(B, N, device=device))
    geodesic_slices.append(bulk_coords[:, :, 0])

    for z in range(1, bulk_dim):
        # Previous layer
        prev_time = arrival_times[z-1]
        prev_coords = bulk_coords[:, :, z-1]
        curr_coords = bulk_coords[:, :, z]

        # AdS Metric Factor: g_zz = (L/z)^2
        z_val = float(z)
        metric_factor = (ads_radius / (z_val + 1e-6)) ** 2

        # Euclidean distance
        dist = torch.norm(curr_coords - prev_coords, dim=-1) # (B, N)

        # Update cost
        cost = prev_time + metric_factor * dist
        arrival_times.append(cost)

        # Update geodesic path
        # ratio acts as a weighting factor
        ratio = prev_time / (cost + 1e-6)
        ratio = ratio.unsqueeze(-1) # (B, N, 1)

        geodesic_slice = prev_coords + (curr_coords - prev_coords) * ratio
        geodesic_slices.append(geodesic_slice)

    # Stack results
    geodesics = torch.stack(geodesic_slices, dim=2) # (B, N, bulk_dim, D)

    return geodesics
