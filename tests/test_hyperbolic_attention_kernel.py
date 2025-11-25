import pytest
import torch
import triton

from src.models.phase7.hyperbolic_attention import HyperbolicMultiHeadAttention

# Mark the entire module as skipping if Triton is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not triton.is_available(), reason="Requires CUDA and Triton")

@pytest.mark.parametrize("d_model", [32, 64])
@pytest.mark.parametrize("num_heads", [2, 4])
@pytest.mark.parametrize("seq_len", [64, 128])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_forward_pass_correctness(d_model, num_heads, seq_len, batch_size):
    """
    Compares the output of the Triton kernel with the PyTorch implementation to ensure correctness.
    """
    # Set a seed for reproducibility
    torch.manual_seed(42)
    device = "cuda"

    # Create random input tensor
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16)

    # 1. Get the output from the reference PyTorch implementation
    model_pytorch = HyperbolicMultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
        use_triton_kernel=False
    ).to(device).eval().half()

    # The forward pass returns a tuple (output, diagnostics)
    output_pytorch, _ = model_pytorch(x)

    # 2. Get the output from the Triton kernel implementation
    # Note: Triton kernel currently doesn't support backprop, so we use no_grad
    model_triton = HyperbolicMultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
        use_triton_kernel=True
    ).to(device).eval().half()

    # We need to ensure the parameters are identical
    model_triton.load_state_dict(model_pytorch.state_dict())

    with torch.no_grad():
        output_triton, _ = model_triton(x)

    # 3. Compare the outputs
    # The numerical precision might differ slightly due to different computation orders and approximations.
    # We use a relatively tolerant atol.
    assert torch.allclose(output_pytorch, output_triton, atol=1e-2, rtol=1e-2), \
        "The output of the Triton kernel does not match the PyTorch implementation."

    print(f"Test passed for shape ({batch_size}, {seq_len}, {d_model}) with {num_heads} heads.")
    # Optional: Print max difference for debugging
    max_diff = (output_pytorch - output_triton).abs().max().item()
    print(f"Max difference: {max_diff}")


@pytest.mark.parametrize("d_head", [16, 32])
@pytest.mark.parametrize("seq_len", [4, 8])
@pytest.mark.parametrize("batch_size", [1]) # gradcheck is slow
def test_backward_pass_gradient_correctness(d_head, seq_len, batch_size):
    """
    Uses torch.autograd.gradcheck to verify the correctness of the backward pass.
    """
    from src.kernels.hyperbolic_attention_kernel import HyperbolicAttention

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float64  # gradcheck requires double precision

    # gradcheck is very sensitive, so we use a smaller, controlled setup
    num_heads = 1

    # Create inputs
    q = torch.randn(batch_size, num_heads, seq_len, d_head, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, num_heads, seq_len, d_head, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, num_heads, seq_len, d_head, device=device, dtype=dtype, requires_grad=True)

    # Hyperparameters
    c = torch.tensor(0.5, device=device, dtype=dtype)
    # The parameters inside the model are float32, so we create them separately
    # and cast for the gradcheck function.
    beta_param = torch.nn.Parameter(torch.randn((), device=device, dtype=dtype))
    bias_param = torch.nn.Parameter(torch.randn((), device=device, dtype=dtype))

    # The function to be tested by gradcheck
    def run_hyperbolic_attention(query, key, value, beta, bias):
        # We need to ensure the autograd function is called directly
        return HyperbolicAttention.apply(query, key, value, c, beta, bias)

    # Perform the gradient check
    # Note: The backward pass for distance is a placeholder, so this is expected to fail initially.
    # The goal is to use the failure reports to correctly implement the backward pass.
    try:
        is_correct = torch.autograd.gradcheck(run_hyperbolic_attention, (q, k, v, beta_param, bias_param), eps=1e-6, atol=1e-4)
        assert is_correct, "Gradient check failed!"
        print(f"Gradcheck passed for shape ({batch_size}, {seq_len}, {d_head})")
    except RuntimeError as e:
        # This allows us to see the error from gradcheck without completely halting the test suite
        pytest.fail(f"Gradient check failed with error: {e}")
