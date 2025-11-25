import torch
import torch.nn as nn
import math

class HyperbolicInitializer(nn.Module):
    """
    Hyperbolic Initializer using the Poincaré Disk Model.

    Generates embedding coordinates distributed according to a hyperbolic tiling (e.g., {7,3}).
    This ensures that the embedding space has exponential capacity and a natural hierarchical structure.

    The coordinates are mapped to the embedding dimension D.
    """

    def __init__(self, d_model: int, curvature: float = -1.0):
        super().__init__()
        self.d_model = d_model
        self.c = -curvature # c = 1/R^2 usually, here curvature K = -c

    def poincare_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute hyperbolic distance between x and y in Poincaré ball.
        d(x, y) = arccosh(1 + 2 * ||x-y||^2 / ((1-||x||^2)(1-||y||^2)))
        """
        sq_norm_x = x.norm(dim=-1).pow(2)
        sq_norm_y = y.norm(dim=-1).pow(2)
        sq_dist = (x - y).norm(dim=-1).pow(2)

        denom = (1 - sq_norm_x) * (1 - sq_norm_y)
        arg = 1 + 2 * sq_dist / (denom + 1e-7)
        return torch.acosh(arg + 1e-7)

    def generate_tiling_coordinates(self, n_nodes: int) -> torch.Tensor:
        """
        Generate N coordinates distributed roughly like a hyperbolic tree/tiling.

        Since exact {7,3} tiling generation is complex, we approximate it by:
        1. Placing root at origin.
        2. Placing layer 1 at radius r1.
        3. Placing layer 2 at radius r2, with exponential branching factor.

        The number of nodes at radius R scales as e^R.
        """
        coords = torch.zeros(n_nodes, self.d_model)

        # Branching factor roughly e (2.718) for hyperbolic space area growth
        branching = 3

        current_idx = 1 # 0 is root at (0,0,...)
        layer = 1

        while current_idx < n_nodes:
            # Radius in hyperbolic metric
            # Equal steps in radius cover exponential area
            r_hyper = float(layer) * 1.0

            # Convert hyperbolic radius r to Poincaré radius R
            # R = tanh(r/2)
            R_poincare = math.tanh(r_hyper / 2.0)

            # Number of nodes in this layer
            n_layer = branching ** layer

            for i in range(n_layer):
                if current_idx >= n_nodes:
                    break

                # Distribute uniformly on the sphere at radius R_poincare
                # Simple random direction for high dimensions
                direction = torch.randn(self.d_model)
                direction = direction / direction.norm()

                coords[current_idx] = direction * R_poincare
                current_idx += 1

            layer += 1

        return coords

    def initialize_embeddings(self, embedding_layer: nn.Embedding):
        """
        In-place initialization of an embedding layer.
        """
        n_vocab, d_model = embedding_layer.weight.shape
        assert d_model == self.d_model

        with torch.no_grad():
            coords = self.generate_tiling_coordinates(n_vocab)
            embedding_layer.weight.copy_(coords)

            # Scale to match typical variance if needed, or keep as is for Poincaré
            # Standard embeddings are often N(0, 1).
            # Poincaré coordinates are bounded < 1. This might be too small for dot products.
            # We project them to tangent space (log map) at origin if the model expects Euclidean vectors?
            # OR we rely on the model to learn the metric.
            # MUSE's BK-Core expects a Potential V.
            # Let's assume these are raw vectors that will be projected to scalar potential V.
            pass
