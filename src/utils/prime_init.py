"""
Prime-Bump Initialization Utility

This utility provides a function to initialize weight matrices according
to the Prime-Bump Potential distribution described in the Riemann AI paper.

The core idea is to create a weight matrix whose singular value distribution
mimics the distribution of prime numbers, thereby embedding a natural
hierarchical structure into the model from the start.

Algorithm:
1.  Create a target singular value distribution by sampling from the
    Prime-Bump Potential function (Eq. 5 of the paper).
2.  Generate random orthogonal matrices U and V.
3.  Construct the weight matrix W = U * diag(Sigma) * V^T.
"""
import torch
import numpy as np
from sympy import primerange

def _get_prime_bump_distribution(num_samples: int, epsilon: float = 0.1, device: str ='cpu'):
    """
    Generates a sample distribution based on the Prime-Bump Potential.
    """
    # 1. Generate primes
    # We need enough primes to create a meaningful distribution.
    # The number of primes up to x is roughly x / log(x).
    # Let's generate primes up to a reasonable limit.
    max_prime = num_samples * 2  # Heuristic
    primes = list(primerange(2, max_prime))
    if len(primes) == 0:
        primes = [2, 3, 5, 7, 11, 13, 17, 19]

    log_primes = np.log(primes)

    # 2. Define the components of the potential V_epsilon(x) from Eq. (5)
    # alpha_{p,k}(epsilon) = (log p) / p^(k * (1/2 + epsilon))
    # psi_epsilon(x) = epsilon^(-1/2) * exp(-x^2 / (2*epsilon))
    # We'll use k=1 for simplicity

    weights = (np.log(primes)) / (np.array(primes)**(0.5 + epsilon))
    weights /= weights.sum() # Normalize to a probability distribution

    # 3. Sample from this distribution
    # We sample the log_primes according to the calculated weights.
    # This gives us the locations of the "bumps".
    sampled_log_primes = np.random.choice(
        log_primes,
        size=num_samples,
        p=weights,
        replace=True
    )

    # 4. Add Gaussian noise to simulate the psi_epsilon function
    # The width of the Gaussian is related to epsilon.
    noise = np.random.normal(0, np.sqrt(epsilon), size=num_samples)

    samples = sampled_log_primes + noise

    # Convert to a tensor, sort in descending order, and normalize
    sigma = torch.from_numpy(samples).to(torch.float32).to(device)
    sigma = torch.sort(sigma, descending=True).values
    sigma = sigma / torch.max(sigma) # Normalize to have max value 1

    return sigma

def prime_bump_init_(tensor: torch.Tensor, epsilon: float = 0.1):
    """
    Initializes a weight tensor in-place using the Prime-Bump method.

    Args:
        tensor: The weight tensor to initialize.
        epsilon: The epsilon parameter for the prime-bump potential.
    """
    if tensor.dim() < 2:
        # For vectors (biases), standard initialization is more appropriate.
        torch.nn.init.normal_(tensor)
        return

    d_out, d_in = tensor.shape
    num_singular_values = min(d_in, d_out)

    # 1. Create the target singular value distribution
    sigma = _get_prime_bump_distribution(num_singular_values, epsilon, device=tensor.device)

    # 2. Generate random orthogonal matrices
    # Using QR decomposition of a random Gaussian matrix
    u_shape = (d_out, num_singular_values)
    v_shape = (d_in, num_singular_values)

    u = torch.randn(*u_shape, device=tensor.device)
    u, _ = torch.linalg.qr(u)

    v = torch.randn(*v_shape, device=tensor.device)
    v, _ = torch.linalg.qr(v)

    # 3. Construct the weight matrix
    with torch.no_grad():
        weight = u @ torch.diag(sigma) @ v.T

        # Ensure the constructed weight has the correct shape
        if weight.shape != tensor.shape:
             # This can happen if d_out != d_in
            padded_weight = torch.zeros_like(tensor)
            padded_weight[:weight.shape[0], :weight.shape[1]] = weight
            tensor.copy_(padded_weight)
        else:
            tensor.copy_(weight)

    return tensor
