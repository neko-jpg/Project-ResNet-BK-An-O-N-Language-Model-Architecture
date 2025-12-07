#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// External CUDA kernel launchers
extern "C" {
    void launch_exp_map_f32(const float* v, float* y, float c, int batch_size, int seq_len, int dim, cudaStream_t stream);
    void launch_log_map_f32(const float* y, float* v, float c, int batch_size, int seq_len, int dim, cudaStream_t stream);
    void launch_poincare_distance_f32(const float* Q, const float* K, float* dist, float c, int batch_size, int num_heads, int seq_len, int dim, cudaStream_t stream);
    void launch_fused_hyperbolic_attention_f32(const float* Q, const float* K, const float* V, float* output, float c, float beta, bool causal, int batch_size, int num_heads, int seq_len, int dim, cudaStream_t stream);
}

// exp_map: tangent space -> Poincaré ball
torch::Tensor exp_map_cuda(torch::Tensor v, float c) {
    // TORCH_CHECK(v.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(v.dim() >= 2, "Input must have at least 2 dimensions");
    
    auto v_contig = v.contiguous().to(torch::kFloat32);
    auto y = torch::empty_like(v_contig);
    
    int64_t dim = v_contig.size(-1);
    int64_t total = v_contig.numel() / dim;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    launch_exp_map_f32(
        v_contig.data_ptr<float>(),
        y.data_ptr<float>(),
        c,
        1,  // batch_size (flattened)
        total,  // seq_len (flattened vectors)
        dim,
        stream
    );
    
    return y.to(v.dtype());
}

// log_map: Poincaré ball -> tangent space
torch::Tensor log_map_cuda(torch::Tensor y, float c) {
    // TORCH_CHECK(y.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(y.dim() >= 2, "Input must have at least 2 dimensions");
    
    auto y_contig = y.contiguous().to(torch::kFloat32);
    auto v = torch::empty_like(y_contig);
    
    int64_t dim = y_contig.size(-1);
    int64_t total = y_contig.numel() / dim;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    launch_log_map_f32(
        y_contig.data_ptr<float>(),
        v.data_ptr<float>(),
        c,
        1,
        total,
        dim,
        stream
    );
    
    return v.to(y.dtype());
}

// poincare_distance: compute pairwise distances
torch::Tensor poincare_distance_cuda(torch::Tensor Q, torch::Tensor K, float c) {
    // TORCH_CHECK(Q.is_cuda() && K.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Inputs must be 4D [B, H, N, D]");
    TORCH_CHECK(Q.sizes() == K.sizes(), "Q and K must have same shape");
    
    auto Q_contig = Q.contiguous().to(torch::kFloat32);
    auto K_contig = K.contiguous().to(torch::kFloat32);
    
    int64_t B = Q_contig.size(0);
    int64_t H = Q_contig.size(1);
    int64_t N = Q_contig.size(2);
    int64_t D = Q_contig.size(3);
    
    auto dist = torch::empty({B, H, N, N}, Q_contig.options());
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    launch_poincare_distance_f32(
        Q_contig.data_ptr<float>(),
        K_contig.data_ptr<float>(),
        dist.data_ptr<float>(),
        c,
        B, H, N, D,
        stream
    );
    
    return dist.to(Q.dtype());
}

// fused_hyperbolic_attention: full fused attention
torch::Tensor fused_hyperbolic_attention_cuda(
    torch::Tensor Q,
    torch::Tensor K, 
    torch::Tensor V,
    float c,
    float beta,
    bool causal
) {
    // TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D [B, H, N, D]");
    
    auto Q_contig = Q.contiguous().to(torch::kFloat32);
    auto K_contig = K.contiguous().to(torch::kFloat32);
    auto V_contig = V.contiguous().to(torch::kFloat32);
    
    int64_t B = Q_contig.size(0);
    int64_t H = Q_contig.size(1);
    int64_t N = Q_contig.size(2);
    int64_t D = Q_contig.size(3);
    
    auto output = torch::empty_like(Q_contig);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    launch_fused_hyperbolic_attention_f32(
        Q_contig.data_ptr<float>(),
        K_contig.data_ptr<float>(),
        V_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        c, beta, causal,
        B, H, N, D,
        stream
    );
    
    return output.to(Q.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("exp_map", &exp_map_cuda, "Exponential map (CUDA)",
          py::arg("v"), py::arg("c") = 1.0f);
    m.def("log_map", &log_map_cuda, "Logarithmic map (CUDA)", 
          py::arg("y"), py::arg("c") = 1.0f);
    m.def("poincare_distance", &poincare_distance_cuda, "Poincare distance (CUDA)",
          py::arg("Q"), py::arg("K"), py::arg("c") = 1.0f);
    m.def("fused_hyperbolic_attention", &fused_hyperbolic_attention_cuda, 
          "Fused hyperbolic attention (CUDA)",
          py::arg("Q"), py::arg("K"), py::arg("V"), 
          py::arg("c") = 1.0f, py::arg("beta") = 1.0f, py::arg("causal") = true);
}
