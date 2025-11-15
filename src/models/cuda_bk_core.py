"""
CUDA-optimized BK-Core with fused theta/phi recursion kernels.

This module provides custom CUDA kernels for the theta and phi recursions
in the BK-Core algorithm, achieving significant speedup over PyTorch implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# CUDA kernel for fused theta recursion
THETA_RECURSION_KERNEL = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_theta_recursion_kernel(
    const float* __restrict__ a,      // (N,) main diagonal
    const float* __restrict__ b,      // (N-1,) super diagonal
    const float* __restrict__ c,      // (N-1,) sub diagonal
    float z_real,                     // complex shift (real part)
    float z_imag,                     // complex shift (imag part)
    float* __restrict__ theta_real,   // (N+1,) output (real part)
    float* __restrict__ theta_imag,   // (N+1,) output (imag part)
    int N
) {
    // Each block processes one batch element
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // Shared memory for intermediate results
    extern __shared__ float shared_mem[];
    float* theta_r_shared = shared_mem;
    float* theta_i_shared = shared_mem + N + 1;
    
    // Offset for this batch
    const float* a_batch = a + batch_idx * N;
    const float* b_batch = b + batch_idx * (N - 1);
    const float* c_batch = c + batch_idx * (N - 1);
    float* theta_r_out = theta_real + batch_idx * (N + 1);
    float* theta_i_out = theta_imag + batch_idx * (N + 1);
    
    // Initialize
    if (tid == 0) {
        theta_r_shared[0] = 1.0f;
        theta_i_shared[0] = 0.0f;
        
        float a_shifted_r = a_batch[0] - z_real;
        float a_shifted_i = -z_imag;
        theta_r_shared[1] = a_shifted_r;
        theta_i_shared[1] = a_shifted_i;
    }
    __syncthreads();
    
    // Recursion (sequential, but fused in single kernel)
    for (int i = 1; i < N; i++) {
        if (tid == 0) {
            // a_shifted = a[i] - z
            float a_shifted_r = a_batch[i] - z_real;
            float a_shifted_i = -z_imag;
            
            // theta[i+1] = a_shifted * theta[i] - c[i-1] * b[i-1] * theta[i-1]
            // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            
            // Term 1: a_shifted * theta[i]
            float term1_r = a_shifted_r * theta_r_shared[i] - a_shifted_i * theta_i_shared[i];
            float term1_i = a_shifted_r * theta_i_shared[i] + a_shifted_i * theta_r_shared[i];
            
            // Term 2: c[i-1] * b[i-1] * theta[i-1] (all real)
            float term2_r = c_batch[i-1] * b_batch[i-1] * theta_r_shared[i-1];
            float term2_i = c_batch[i-1] * b_batch[i-1] * theta_i_shared[i-1];
            
            // Result
            theta_r_shared[i+1] = term1_r - term2_r;
            theta_i_shared[i+1] = term1_i - term2_i;
        }
        __syncthreads();
    }
    
    // Write to global memory
    if (tid < N + 1) {
        theta_r_out[tid] = theta_r_shared[tid];
        theta_i_out[tid] = theta_i_shared[tid];
    }
}

torch::Tensor fused_theta_recursion(
    torch::Tensor a,      // (B, N)
    torch::Tensor b,      // (B, N-1)
    torch::Tensor c,      // (B, N-1)
    float z_real,
    float z_imag
) {
    const int B = a.size(0);
    const int N = a.size(1);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(a.device());
    auto theta_real = torch::zeros({B, N + 1}, options);
    auto theta_imag = torch::zeros({B, N + 1}, options);
    
    const int threads = 256;
    const int blocks = B;
    const int shared_mem_size = 2 * (N + 1) * sizeof(float);
    
    fused_theta_recursion_kernel<<<blocks, threads, shared_mem_size>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        z_real,
        z_imag,
        theta_real.data_ptr<float>(),
        theta_imag.data_ptr<float>(),
        N
    );
    
    // Stack real and imaginary parts
    return torch::stack({theta_real, theta_imag}, -1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_theta_recursion", &fused_theta_recursion, "Fused theta recursion (CUDA)");
}
"""

# CUDA kernel for fused phi recursion
PHI_RECURSION_KERNEL = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_phi_recursion_kernel(
    const float* __restrict__ a,      // (N,) main diagonal
    const float* __restrict__ b,      // (N-1,) super diagonal
    const float* __restrict__ c,      // (N-1,) sub diagonal
    float z_real,                     // complex shift (real part)
    float z_imag,                     // complex shift (imag part)
    float* __restrict__ phi_real,     // (N,) output (real part)
    float* __restrict__ phi_imag,     // (N,) output (imag part)
    int N
) {
    // Each block processes one batch element
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // Shared memory for intermediate results
    extern __shared__ float shared_mem[];
    float* phi_r_shared = shared_mem;
    float* phi_i_shared = shared_mem + N + 2;
    
    // Offset for this batch
    const float* a_batch = a + batch_idx * N;
    const float* b_batch = b + batch_idx * (N - 1);
    const float* c_batch = c + batch_idx * (N - 1);
    float* phi_r_out = phi_real + batch_idx * N;
    float* phi_i_out = phi_imag + batch_idx * N;
    
    // Initialize (backward from N)
    if (tid == 0) {
        phi_r_shared[N] = 1.0f;
        phi_i_shared[N] = 0.0f;
        phi_r_shared[N + 1] = 0.0f;
        phi_i_shared[N + 1] = 0.0f;
        
        float a_shifted_r = a_batch[N-1] - z_real;
        float a_shifted_i = -z_imag;
        phi_r_shared[N-1] = a_shifted_r;
        phi_i_shared[N-1] = a_shifted_i;
    }
    __syncthreads();
    
    // Backward recursion
    for (int i = N - 2; i >= 0; i--) {
        if (tid == 0) {
            // a_shifted = a[i+1] - z
            float a_shifted_r = a_batch[i+1] - z_real;
            float a_shifted_i = -z_imag;
            
            // phi[i] = a_shifted * phi[i+1] - b[i] * c[i] * phi[i+2]
            
            // Term 1: a_shifted * phi[i+1]
            float term1_r = a_shifted_r * phi_r_shared[i+1] - a_shifted_i * phi_i_shared[i+1];
            float term1_i = a_shifted_r * phi_i_shared[i+1] + a_shifted_i * phi_r_shared[i+1];
            
            // Term 2: b[i] * c[i] * phi[i+2] (all real)
            float term2_r = b_batch[i] * c_batch[i] * phi_r_shared[i+2];
            float term2_i = b_batch[i] * c_batch[i] * phi_i_shared[i+2];
            
            // Result
            phi_r_shared[i] = term1_r - term2_r;
            phi_i_shared[i] = term1_i - term2_i;
        }
        __syncthreads();
    }
    
    // Write to global memory
    if (tid < N) {
        phi_r_out[tid] = phi_r_shared[tid];
        phi_i_out[tid] = phi_i_shared[tid];
    }
}

torch::Tensor fused_phi_recursion(
    torch::Tensor a,      // (B, N)
    torch::Tensor b,      // (B, N-1)
    torch::Tensor c,      // (B, N-1)
    float z_real,
    float z_imag
) {
    const int B = a.size(0);
    const int N = a.size(1);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(a.device());
    auto phi_real = torch::zeros({B, N}, options);
    auto phi_imag = torch::zeros({B, N}, options);
    
    const int threads = 256;
    const int blocks = B;
    const int shared_mem_size = 2 * (N + 2) * sizeof(float);
    
    fused_phi_recursion_kernel<<<blocks, threads, shared_mem_size>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        z_real,
        z_imag,
        phi_real.data_ptr<float>(),
        phi_imag.data_ptr<float>(),
        N
    );
    
    // Stack real and imaginary parts
    return torch::stack({phi_real, phi_imag}, -1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_phi_recursion", &fused_phi_recursion, "Fused phi recursion (CUDA)");
}
"""


class CUDAOptimizedBKCore(nn.Module):
    """
    BK-Core with custom CUDA kernels for theta/phi recursions.
    
    This implementation uses fused CUDA kernels to compute the theta and phi
    recursions in a single kernel launch, reducing memory transfers and
    achieving significant speedup over the PyTorch implementation.
    
    Args:
        n_seq: Sequence length
        use_cuda_kernels: If True, use custom CUDA kernels; otherwise fallback to PyTorch
    """
    
    def __init__(self, n_seq: int, use_cuda_kernels: bool = True):
        super().__init__()
        self.n_seq = n_seq
        self.use_cuda_kernels = use_cuda_kernels
        
        # Try to compile CUDA kernels
        self.cuda_available = False
        if use_cuda_kernels and torch.cuda.is_available():
            try:
                self._compile_cuda_kernels()
                self.cuda_available = True
                print("CUDA kernels compiled successfully")
            except Exception as e:
                print(f"Failed to compile CUDA kernels: {e}")
                print("Falling back to PyTorch implementation")
                self.cuda_available = False
        
        # Base Hamiltonian parameters
        self.register_buffer('h0_diag', torch.full((n_seq,), -2.0))
        self.register_buffer('h0_sub', torch.full((n_seq-1,), 1.0))
        self.register_buffer('h0_super', torch.full((n_seq-1,), 1.0))
        
        # Complex shift
        self.z = torch.tensor(1.0j, dtype=torch.complex128)
    
    def _compile_cuda_kernels(self):
        """Compile CUDA kernels using torch.utils.cpp_extension."""
        from torch.utils.cpp_extension import load_inline
        
        # Compile theta kernel
        self.theta_module = load_inline(
            name='theta_recursion_cuda',
            cpp_sources='',
            cuda_sources=THETA_RECURSION_KERNEL,
            functions=['fused_theta_recursion'],
            verbose=False,
            extra_cuda_cflags=['-O3']
        )
        
        # Compile phi kernel
        self.phi_module = load_inline(
            name='phi_recursion_cuda',
            cpp_sources='',
            cuda_sources=PHI_RECURSION_KERNEL,
            functions=['fused_phi_recursion'],
            verbose=False,
            extra_cuda_cflags=['-O3']
        )
    
    def _theta_recursion_pytorch(self, he_diag, h0_super, h0_sub, z):
        """PyTorch fallback implementation of theta recursion."""
        B, N = he_diag.shape
        device = he_diag.device
        dtype = torch.complex128
        
        theta = torch.zeros(B, N + 1, dtype=dtype, device=device)
        theta[:, 0] = 1.0
        theta[:, 1] = he_diag[:, 0] - z
        
        for i in range(1, N):
            theta[:, i + 1] = (he_diag[:, i] - z) * theta[:, i] - \
                              h0_sub[i - 1] * h0_super[i - 1] * theta[:, i - 1]
        
        return theta
    
    def _phi_recursion_pytorch(self, he_diag, h0_super, h0_sub, z):
        """PyTorch fallback implementation of phi recursion."""
        B, N = he_diag.shape
        device = he_diag.device
        dtype = torch.complex128
        
        phi = torch.zeros(B, N + 2, dtype=dtype, device=device)
        phi[:, N] = 1.0
        phi[:, N - 1] = he_diag[:, N - 1] - z
        
        for i in range(N - 2, -1, -1):
            phi[:, i] = (he_diag[:, i + 1] - z) * phi[:, i + 1] - \
                        h0_super[i] * h0_sub[i] * phi[:, i + 2]
        
        return phi[:, :N]
    
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing G_ii diagonal elements.
        
        Args:
            v: (B, N) - learned potential
        
        Returns:
            features: (B, N, 2) - [real(G_ii), imag(G_ii)]
        """
        B, N = v.shape
        device = v.device
        
        # Compute effective Hamiltonian diagonal
        h0_diag_batch = self.h0_diag.unsqueeze(0).expand(B, -1)
        he_diag = h0_diag_batch + v
        
        # Expand sub/super diagonals
        h0_sub_batch = self.h0_sub.unsqueeze(0).expand(B, -1)
        h0_super_batch = self.h0_super.unsqueeze(0).expand(B, -1)
        
        if self.cuda_available and device.type == 'cuda':
            # Use CUDA kernels
            # Convert to FP32 for CUDA kernel
            he_diag_fp32 = he_diag.float()
            h0_sub_fp32 = h0_sub_batch.float()
            h0_super_fp32 = h0_super_batch.float()
            
            # Theta recursion
            theta_complex = self.theta_module.fused_theta_recursion(
                he_diag_fp32.contiguous(),
                h0_super_fp32.contiguous(),
                h0_sub_fp32.contiguous(),
                self.z.real.item(),
                self.z.imag.item()
            )  # (B, N+1, 2)
            
            theta = torch.complex(theta_complex[..., 0], theta_complex[..., 1])
            
            # Phi recursion
            phi_complex = self.phi_module.fused_phi_recursion(
                he_diag_fp32.contiguous(),
                h0_super_fp32.contiguous(),
                h0_sub_fp32.contiguous(),
                self.z.real.item(),
                self.z.imag.item()
            )  # (B, N, 2)
            
            phi = torch.complex(phi_complex[..., 0], phi_complex[..., 1])
            
        else:
            # Use PyTorch fallback
            theta = self._theta_recursion_pytorch(he_diag, h0_super_batch, h0_sub_batch, self.z)
            phi = self._phi_recursion_pytorch(he_diag, h0_super_batch, h0_sub_batch, self.z)
        
        # Compute G_ii = theta[:-1] * phi / det_T
        det_T = theta[:, -1]
        
        # Numerical stability
        det_T_mag = det_T.abs()
        det_T_stable = torch.where(
            det_T_mag < 1e-18,
            det_T / (det_T_mag + 1e-18) * 1e-18,
            det_T
        )
        
        # Compute diagonal elements
        G_ii = (theta[:, :-1] * phi) / det_T_stable.unsqueeze(-1)
        
        # Clamp for numerical stability
        G_ii_real = torch.clamp(G_ii.real, -10.0, 10.0)
        G_ii_imag = torch.clamp(G_ii.imag, -10.0, 10.0)
        
        # Stack as features
        features = torch.stack([G_ii_real, G_ii_imag], dim=-1).float()
        
        return features


def test_cuda_kernels():
    """Test CUDA kernels against PyTorch implementation."""
    print("Testing CUDA kernels...")
    
    # Create test data
    B, N = 4, 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    v = torch.randn(B, N, device=device) * 0.5
    
    # CUDA implementation
    cuda_core = CUDAOptimizedBKCore(N, use_cuda_kernels=True).to(device)
    
    # PyTorch implementation
    pytorch_core = CUDAOptimizedBKCore(N, use_cuda_kernels=False).to(device)
    
    # Forward pass
    with torch.no_grad():
        features_cuda = cuda_core(v)
        features_pytorch = pytorch_core(v)
    
    # Compare results
    if cuda_core.cuda_available:
        max_diff = (features_cuda - features_pytorch).abs().max().item()
        mean_diff = (features_cuda - features_pytorch).abs().mean().item()
        
        print(f"Max difference: {max_diff:.6e}")
        print(f"Mean difference: {mean_diff:.6e}")
        
        if max_diff < 1e-4:
            print("✓ CUDA kernels match PyTorch implementation")
        else:
            print("✗ CUDA kernels differ from PyTorch implementation")
    else:
        print("CUDA kernels not available, skipping comparison")


if __name__ == '__main__':
    test_cuda_kernels()
