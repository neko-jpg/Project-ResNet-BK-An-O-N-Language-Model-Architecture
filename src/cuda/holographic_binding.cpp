/*
 * PyTorch C++ Extension for Holographic Synthesis
 * 
 * Provides Python bindings for the CUDA holographic kernel.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>

// Forward declarations from holographic_kernel.cu
extern "C" {
    void* holographic_create(int size, float lr);
    void holographic_destroy(void* synth);
    float holographic_synthesize(void* synth, const float* x, const float* y, float* output, int n);
}

// Global synthesizer instance (lazy initialization)
static void* g_synthesizer = nullptr;
static int g_size = 0;
static float g_lr = 0.01f;

void ensure_synthesizer(int size, float lr) {
    if (g_synthesizer == nullptr || g_size != size || g_lr != lr) {
        if (g_synthesizer != nullptr) {
            holographic_destroy(g_synthesizer);
        }
        g_synthesizer = holographic_create(size, lr);
        g_size = size;
        g_lr = lr;
    }
}

/*
 * Holographic phasor binding
 * 
 * Args:
 *   x: Input tensor (1D, float32, CUDA)
 *   y: Input tensor (1D, float32, CUDA)
 *   lr: Learning rate
 * 
 * Returns:
 *   Tuple of (output tensor, time_ms)
 */
std::tuple<torch::Tensor, float> holographic_bind(
    torch::Tensor x,
    torch::Tensor y,
    float lr
) {
    // Validate inputs
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");
    TORCH_CHECK(x.dim() == 1, "x must be 1D");
    TORCH_CHECK(y.dim() == 1, "y must be 1D");
    
    int n = x.size(0);
    int m = y.size(0);
    int size = (n < m) ? n : m;
    
    // Ensure synthesizer is ready
    ensure_synthesizer(size, lr);
    
    // Create output tensor
    auto output = torch::empty({size}, x.options());
    
    // Get raw pointers
    const float* x_ptr = x.data_ptr<float>();
    const float* y_ptr = y.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();
    
    // Run synthesis
    float time_ms = holographic_synthesize(g_synthesizer, x_ptr, y_ptr, out_ptr, size);
    
    return std::make_tuple(output, time_ms);
}

// Module registration
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("holographic_bind", &holographic_bind, 
          "Holographic phasor binding (CUDA)",
          py::arg("x"), py::arg("y"), py::arg("lr") = 0.01f);
}
