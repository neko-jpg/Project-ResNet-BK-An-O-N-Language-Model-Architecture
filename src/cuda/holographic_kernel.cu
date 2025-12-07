/*
 * Holographic Weight Synthesis - CUDA C++ Implementation
 * 
 * Ultra-fast phasor binding using cuFFT for 0.105ms target.
 * 
 * Key optimizations:
 * - cuFFT with pre-planned transforms
 * - Fused phasor normalization kernel
 * - Minimal memory allocations
 * - Power-of-2 sizes for optimal cuFFT
 * 
 * Author: Project MUSE Team
 */

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            fprintf(stderr, "cuFFT error at %s:%d: %d\n", \
                    __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Constants
constexpr float EPS = 1e-8f;
constexpr int BLOCK_SIZE = 256;

/*
 * Kernel: Phasor normalization in frequency domain
 * Computes: Z[i] = (X[i] / |X[i]|) * conj(Y[i] / |Y[i]|)
 */
__global__ void phasor_normalize_kernel(
    cufftComplex* X,
    cufftComplex* Y,
    cufftComplex* Z,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Get X[i]
    float x_re = X[idx].x;
    float x_im = X[idx].y;
    float x_mag = sqrtf(x_re * x_re + x_im * x_im) + EPS;
    
    // Normalize X to unit magnitude
    float x_phasor_re = x_re / x_mag;
    float x_phasor_im = x_im / x_mag;
    
    // Get Y[i]
    float y_re = Y[idx].x;
    float y_im = Y[idx].y;
    float y_mag = sqrtf(y_re * y_re + y_im * y_im) + EPS;
    
    // Normalize Y and conjugate
    float y_phasor_re = y_re / y_mag;
    float y_phasor_im = -y_im / y_mag;  // Conjugate
    
    // Multiply: Z = X_phasor * conj(Y_phasor)
    Z[idx].x = x_phasor_re * y_phasor_re - x_phasor_im * y_phasor_im;
    Z[idx].y = x_phasor_re * y_phasor_im + x_phasor_im * y_phasor_re;
}

/*
 * Kernel: Scale output by learning rate
 */
__global__ void scale_output_kernel(
    float* output,
    float lr,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    output[idx] *= lr;
}

/*
 * HolographicSynthesizer class
 * Manages cuFFT plans and device memory for fast synthesis
 */
class HolographicSynthesizer {
public:
    int n;              // Signal size
    int n_complex;      // Complex array size (n/2 + 1)
    float lr;           // Learning rate
    
    // cuFFT plans
    cufftHandle plan_forward;
    cufftHandle plan_inverse;
    
    // Device memory (pre-allocated)
    float* d_x;         // Input x
    float* d_y;         // Input y
    float* d_output;    // Output
    cufftComplex* d_X;  // FFT of x
    cufftComplex* d_Y;  // FFT of y
    cufftComplex* d_Z;  // Result in frequency domain
    
    // CUDA events for timing
    cudaEvent_t start, stop;
    
    HolographicSynthesizer(int size, float learning_rate = 0.01f) 
        : n(size), lr(learning_rate) {
        
        // Ensure power of 2
        n = 1;
        while (n < size) n <<= 1;
        
        n_complex = n / 2 + 1;
        
        // Create cuFFT plans
        CUFFT_CHECK(cufftPlan1d(&plan_forward, n, CUFFT_R2C, 1));
        CUFFT_CHECK(cufftPlan1d(&plan_inverse, n, CUFFT_C2R, 1));
        
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_X, n_complex * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_Y, n_complex * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_Z, n_complex * sizeof(cufftComplex)));
        
        // Zero initialize
        CUDA_CHECK(cudaMemset(d_x, 0, n * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_y, 0, n * sizeof(float)));
        
        // Create CUDA events
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    
    ~HolographicSynthesizer() {
        cufftDestroy(plan_forward);
        cufftDestroy(plan_inverse);
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_output);
        cudaFree(d_X);
        cudaFree(d_Y);
        cudaFree(d_Z);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    /*
     * Perform holographic synthesis
     * 
     * Args:
     *   h_x: Host pointer to input x (size n)
     *   h_y: Host pointer to input y (size n)
     *   h_output: Host pointer to output (size n)
     *   actual_n: Actual input size (may be < n)
     * 
     * Returns: Time in milliseconds
     */
    float synthesize(
        const float* h_x,
        const float* h_y,
        float* h_output,
        int actual_n
    ) {
        // Copy inputs to device (pad with zeros if needed)
        int copy_n = (actual_n < n) ? actual_n : n;
        CUDA_CHECK(cudaMemcpy(d_x, h_x, copy_n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, h_y, copy_n * sizeof(float), cudaMemcpyHostToDevice));
        
        // Start timing
        CUDA_CHECK(cudaEventRecord(start));
        
        // Forward FFT: x -> X, y -> Y
        CUFFT_CHECK(cufftExecR2C(plan_forward, d_x, d_X));
        CUFFT_CHECK(cufftExecR2C(plan_forward, d_y, d_Y));
        
        // Phasor normalization: Z = (X/|X|) * conj(Y/|Y|)
        int blocks = (n_complex + BLOCK_SIZE - 1) / BLOCK_SIZE;
        phasor_normalize_kernel<<<blocks, BLOCK_SIZE>>>(d_X, d_Y, d_Z, n_complex);
        
        // Inverse FFT: Z -> output
        CUFFT_CHECK(cufftExecC2R(plan_inverse, d_Z, d_output));
        
        // Scale by learning rate (and normalize by n for cuFFT)
        float scale = lr / n;
        blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        scale_output_kernel<<<blocks, BLOCK_SIZE>>>(d_output, scale, n);
        
        // Stop timing
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        
        // Copy output to host
        CUDA_CHECK(cudaMemcpy(h_output, d_output, copy_n * sizeof(float), cudaMemcpyDeviceToHost));
        
        return elapsed_ms;
    }
    
    /*
     * Synthesize with device pointers (no host-device copy)
     * Even faster for integration with PyTorch tensors
     */
    float synthesize_device(
        float* d_input_x,
        float* d_input_y,
        float* d_result,
        int actual_n
    ) {
        // Start timing
        CUDA_CHECK(cudaEventRecord(start));
        
        // Forward FFT
        CUFFT_CHECK(cufftExecR2C(plan_forward, d_input_x, d_X));
        CUFFT_CHECK(cufftExecR2C(plan_forward, d_input_y, d_Y));
        
        // Phasor normalization
        int blocks = (n_complex + BLOCK_SIZE - 1) / BLOCK_SIZE;
        phasor_normalize_kernel<<<blocks, BLOCK_SIZE>>>(d_X, d_Y, d_Z, n_complex);
        
        // Inverse FFT
        CUFFT_CHECK(cufftExecC2R(plan_inverse, d_Z, d_result));
        
        // Scale
        float scale = lr / n;
        blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        scale_output_kernel<<<blocks, BLOCK_SIZE>>>(d_result, scale, n);
        
        // Stop timing
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        
        return elapsed_ms;
    }
};

// C interface for Python binding
extern "C" {
    HolographicSynthesizer* holographic_create(int size, float lr) {
        return new HolographicSynthesizer(size, lr);
    }
    
    void holographic_destroy(HolographicSynthesizer* synth) {
        delete synth;
    }
    
    float holographic_synthesize(
        HolographicSynthesizer* synth,
        const float* x,
        const float* y,
        float* output,
        int n
    ) {
        return synth->synthesize(x, y, output, n);
    }
}
