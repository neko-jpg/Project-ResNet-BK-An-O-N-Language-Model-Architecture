/*
 * Hyperbolic Operations CUDA Kernel
 * 
 * Fused implementation of Poincaré ball operations:
 * - exp_map: Tangent space -> Poincaré ball
 * - log_map: Poincaré ball -> Tangent space  
 * - poincare_distance: Geodesic distance in hyperbolic space
 *
 * Optimizations:
 * - Fused operations to minimize memory bandwidth
 * - Warp-level reductions for norm computations
 * - Fast math intrinsics for exp/log/tanh
 *
 * Author: Project MUSE Team
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

#define EPS 1e-6f
#define MAX_TANGENT_NORM 5.0f

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Fast rsqrt with Newton-Raphson refinement
__device__ __forceinline__ float fast_rsqrt(float x) {
    float y = __frsqrt_rn(x);
    // One Newton-Raphson iteration for better precision
    y = y * (1.5f - 0.5f * x * y * y);
    return y;
}

/*
 * Fused exp_map kernel
 * 
 * Formula: exp_map(v, c) = (1/sqrt(c)) * tanh(sqrt(c) * ||v||) * (v / ||v||)
 * 
 * Input: v [B, N, D] - tangent vectors
 * Output: y [B, N, D] - points in Poincaré ball
 */
template<typename scalar_t>
__global__ void exp_map_kernel(
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ y,
    const float c,
    const int batch_size,
    const int seq_len,
    const int dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_vectors = batch_size * seq_len;
    
    if (idx >= total_vectors) return;
    
    const int vec_offset = idx * dim;
    const scalar_t* v_ptr = v + vec_offset;
    scalar_t* y_ptr = y + vec_offset;
    
    // Compute ||v||^2 using warp reduction
    float norm_sq = 0.0f;
    for (int d = 0; d < dim; d++) {
        float val = static_cast<float>(v_ptr[d]);
        norm_sq += val * val;
    }
    
    float norm = sqrtf(norm_sq + EPS);
    
    // Clamp norm to prevent explosion
    float clamped_norm = fminf(norm, MAX_TANGENT_NORM);
    float scale = (clamped_norm / (norm + EPS));
    scale = fminf(scale, 1.0f);
    
    // Compute exp_map scale: (1/sqrt(c)) * tanh(sqrt(c) * ||v||) / ||v||
    float sqrt_c = sqrtf(c);
    float tanh_arg = sqrt_c * clamped_norm;
    tanh_arg = fminf(tanh_arg, 15.0f);  // Prevent overflow
    float tanh_val = tanhf(tanh_arg);
    
    float exp_scale = tanh_val / (sqrt_c * clamped_norm + EPS);
    
    // Apply to each dimension
    for (int d = 0; d < dim; d++) {
        float v_d = static_cast<float>(v_ptr[d]) * scale;
        y_ptr[d] = static_cast<scalar_t>(v_d * exp_scale);
    }
}

/*
 * Fused log_map kernel
 * 
 * Formula: log_map(y, c) = (1/sqrt(c)) * atanh(sqrt(c) * ||y||) * (y / ||y||)
 */
template<typename scalar_t>
__global__ void log_map_kernel(
    const scalar_t* __restrict__ y,
    scalar_t* __restrict__ v,
    const float c,
    const int batch_size,
    const int seq_len,
    const int dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_vectors = batch_size * seq_len;
    
    if (idx >= total_vectors) return;
    
    const int vec_offset = idx * dim;
    const scalar_t* y_ptr = y + vec_offset;
    scalar_t* v_ptr = v + vec_offset;
    
    // Compute ||y||
    float norm_sq = 0.0f;
    for (int d = 0; d < dim; d++) {
        float val = static_cast<float>(y_ptr[d]);
        norm_sq += val * val;
    }
    
    float norm = sqrtf(norm_sq + EPS);
    float sqrt_c = sqrtf(c);
    
    // Clamp to stay inside ball: ||y|| < 1/sqrt(c)
    float max_norm = (1.0f / sqrt_c) - EPS;
    float clamped_norm = fminf(norm, max_norm);
    
    // atanh argument: sqrt(c) * ||y||
    float atanh_arg = sqrt_c * clamped_norm;
    atanh_arg = fminf(atanh_arg, 0.999f);  // Clamp for stability
    
    // atanh(x) = 0.5 * log((1+x)/(1-x))
    float atanh_val = 0.5f * logf((1.0f + atanh_arg + EPS) / (1.0f - atanh_arg + EPS));
    
    // Scale: (1/sqrt(c)) * atanh(...) / ||y||
    float log_scale = atanh_val / (sqrt_c * norm + EPS);
    
    for (int d = 0; d < dim; d++) {
        float y_d = static_cast<float>(y_ptr[d]);
        v_ptr[d] = static_cast<scalar_t>(y_d * log_scale);
    }
}

/*
 * Poincaré distance kernel
 * 
 * Computes pairwise distances between Q and K in Poincaré ball.
 * d(x,y) = (1/sqrt(c)) * acosh(1 + 2*c*||x-y||^2 / ((1-c*||x||^2)(1-c*||y||^2)))
 * 
 * Input: Q [B, H, N, D], K [B, H, N, D]
 * Output: dist [B, H, N, N]
 */
template<typename scalar_t>
__global__ void poincare_distance_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    scalar_t* __restrict__ dist,
    const float c,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int dim
) {
    // Each thread computes one distance value
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // Query index
    const int j = threadIdx.y;  // Key index (within block)
    
    if (b >= batch_size || h >= num_heads || i >= seq_len) return;
    
    // Pointers to Q[b,h,i,:] and K[b,h,j,:]
    const int stride_b = num_heads * seq_len * dim;
    const int stride_h = seq_len * dim;
    const int stride_n = dim;
    
    const scalar_t* q_ptr = Q + b * stride_b + h * stride_h + i * stride_n;
    
    // For each key position j
    for (int jj = j; jj < seq_len; jj += blockDim.y) {
        const scalar_t* k_ptr = K + b * stride_b + h * stride_h + jj * stride_n;
        
        // Compute norms and dot product
        float q_norm_sq = 0.0f;
        float k_norm_sq = 0.0f;
        float diff_norm_sq = 0.0f;
        
        for (int d = 0; d < dim; d++) {
            float q_d = static_cast<float>(q_ptr[d]);
            float k_d = static_cast<float>(k_ptr[d]);
            float diff = q_d - k_d;
            
            q_norm_sq += q_d * q_d;
            k_norm_sq += k_d * k_d;
            diff_norm_sq += diff * diff;
        }
        
        // Denominators: (1 - c*||x||^2)(1 - c*||y||^2)
        float denom_q = 1.0f - c * q_norm_sq;
        float denom_k = 1.0f - c * k_norm_sq;
        float denom = denom_q * denom_k;
        denom = fmaxf(denom, EPS);  // Prevent division by zero
        
        // Argument of acosh
        float numerator = 2.0f * c * diff_norm_sq;
        float acosh_arg = 1.0f + numerator / denom;
        acosh_arg = fmaxf(acosh_arg, 1.0f + EPS);  // Clamp for stability
        
        // acosh(x) = log(x + sqrt(x^2 - 1))
        float sqrt_c = sqrtf(c);
        float distance = logf(acosh_arg + sqrtf(acosh_arg * acosh_arg - 1.0f)) / sqrt_c;
        
        // Store result
        const int out_idx = b * (num_heads * seq_len * seq_len) + 
                           h * (seq_len * seq_len) + 
                           i * seq_len + jj;
        dist[out_idx] = static_cast<scalar_t>(distance);
    }
}

/*
 * Fused hyperbolic attention kernel
 * 
 * Combines exp_map + poincare_distance + softmax + weighted sum
 * This is the main workhorse kernel for hyperbolic attention.
 */
template<typename scalar_t>
__global__ void fused_hyperbolic_attention_kernel(
    const scalar_t* __restrict__ Q_tangent,  // [B, H, N, D] in tangent space
    const scalar_t* __restrict__ K_tangent,  // [B, H, N, D] in tangent space
    const scalar_t* __restrict__ V,          // [B, H, N, D]
    scalar_t* __restrict__ output,           // [B, H, N, D]
    const float c,                           // curvature
    const float beta,                        // temperature
    const bool causal,                       // causal masking
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int dim
) {
    extern __shared__ float smem[];
    
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int i = blockIdx.x;  // Query position
    
    if (b >= batch_size || h >= num_heads || i >= seq_len) return;
    
    const int tid = threadIdx.x;
    const int stride_bhnd = num_heads * seq_len * dim;
    const int stride_hnd = seq_len * dim;
    const int stride_nd = dim;
    
    // Pointers
    const scalar_t* q_ptr = Q_tangent + b * stride_bhnd + h * stride_hnd + i * stride_nd;
    
    // Step 1: Compute exp_map for query (store in shared memory)
    float* q_hyp = smem;  // [dim]
    float q_norm_sq = 0.0f;
    
    for (int d = tid; d < dim; d += blockDim.x) {
        float q_d = static_cast<float>(q_ptr[d]);
        q_norm_sq += q_d * q_d;
    }
    // Reduce q_norm_sq across threads
    q_norm_sq = warp_reduce_sum(q_norm_sq);
    if (tid == 0) smem[dim] = q_norm_sq;  // Store in shared
    __syncthreads();
    q_norm_sq = smem[dim];
    
    float q_norm = sqrtf(q_norm_sq + EPS);
    float sqrt_c = sqrtf(c);
    float q_scale = tanhf(fminf(sqrt_c * fminf(q_norm, MAX_TANGENT_NORM), 15.0f)) / 
                    (sqrt_c * fminf(q_norm, MAX_TANGENT_NORM) + EPS);
    
    for (int d = tid; d < dim; d += blockDim.x) {
        float q_d = static_cast<float>(q_ptr[d]);
        q_hyp[d] = q_d * q_scale;
    }
    __syncthreads();
    
    // Step 2: Compute attention scores for all keys
    float* scores = smem + dim + 1;  // [seq_len]
    float max_score = -INFINITY;
    
    for (int j = tid; j < seq_len; j += blockDim.x) {
        if (causal && j > i) {
            scores[j] = -INFINITY;
            continue;
        }
        
        // Get K[j] and compute exp_map
        const scalar_t* k_ptr = K_tangent + b * stride_bhnd + h * stride_hnd + j * stride_nd;
        
        float k_norm_sq = 0.0f;
        float q_norm_sq_local = 0.0f;
        float diff_norm_sq = 0.0f;
        
        for (int d = 0; d < dim; d++) {
            float k_d = static_cast<float>(k_ptr[d]);
            float k_scale = tanhf(fminf(sqrt_c * sqrtf(k_norm_sq + k_d * k_d + EPS), 15.0f));
            float k_hyp_d = k_d * k_scale / (sqrt_c * sqrtf(k_norm_sq + k_d * k_d + EPS) + EPS);
            
            k_norm_sq += k_d * k_d;
            
            float q_d = q_hyp[d];
            float diff = q_d - k_hyp_d;
            diff_norm_sq += diff * diff;
            q_norm_sq_local += q_d * q_d;
        }
        
        // Recompute k_hyp properly
        float k_norm = sqrtf(k_norm_sq + EPS);
        float k_scale_final = tanhf(fminf(sqrt_c * fminf(k_norm, MAX_TANGENT_NORM), 15.0f)) / 
                              (sqrt_c * fminf(k_norm, MAX_TANGENT_NORM) + EPS);
        
        // Recompute diff_norm_sq with proper k_hyp
        diff_norm_sq = 0.0f;
        float k_hyp_norm_sq = 0.0f;
        for (int d = 0; d < dim; d++) {
            float k_d = static_cast<float>(k_ptr[d]) * k_scale_final;
            float diff = q_hyp[d] - k_d;
            diff_norm_sq += diff * diff;
            k_hyp_norm_sq += k_d * k_d;
        }
        
        // Poincaré distance
        float denom = (1.0f - c * q_norm_sq_local) * (1.0f - c * k_hyp_norm_sq);
        denom = fmaxf(denom, EPS);
        float acosh_arg = 1.0f + 2.0f * c * diff_norm_sq / denom;
        acosh_arg = fmaxf(acosh_arg, 1.0f + EPS);
        float dist = logf(acosh_arg + sqrtf(acosh_arg * acosh_arg - 1.0f)) / sqrt_c;
        
        float score = -beta * dist;
        scores[j] = score;
        max_score = fmaxf(max_score, score);
    }
    __syncthreads();
    
    // Reduce max_score
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            max_score = fmaxf(max_score, __shfl_down_sync(0xffffffff, max_score, offset));
        }
    }
    if (tid == 0) smem[dim] = max_score;
    __syncthreads();
    max_score = smem[dim];
    
    // Step 3: Softmax
    float sum_exp = 0.0f;
    for (int j = tid; j < seq_len; j += blockDim.x) {
        float exp_val = expf(scores[j] - max_score);
        scores[j] = exp_val;
        sum_exp += exp_val;
    }
    // Reduce sum_exp
    sum_exp = warp_reduce_sum(sum_exp);
    if (tid == 0) smem[dim] = sum_exp;
    __syncthreads();
    sum_exp = smem[dim] + EPS;
    
    // Normalize
    for (int j = tid; j < seq_len; j += blockDim.x) {
        scores[j] /= sum_exp;
    }
    __syncthreads();
    
    // Step 4: Weighted sum of values
    scalar_t* out_ptr = output + b * stride_bhnd + h * stride_hnd + i * stride_nd;
    
    for (int d = tid; d < dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            const scalar_t* v_ptr = V + b * stride_bhnd + h * stride_hnd + j * stride_nd;
            acc += scores[j] * static_cast<float>(v_ptr[d]);
        }
        out_ptr[d] = static_cast<scalar_t>(acc);
    }
}

// C++ wrapper functions for PyTorch binding
extern "C" {

void launch_exp_map_f32(
    const float* v, float* y, float c,
    int batch_size, int seq_len, int dim,
    cudaStream_t stream
) {
    int total = batch_size * seq_len;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    exp_map_kernel<float><<<blocks, threads, 0, stream>>>(
        v, y, c, batch_size, seq_len, dim
    );
}

void launch_log_map_f32(
    const float* y, float* v, float c,
    int batch_size, int seq_len, int dim,
    cudaStream_t stream
) {
    int total = batch_size * seq_len;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    log_map_kernel<float><<<blocks, threads, 0, stream>>>(
        y, v, c, batch_size, seq_len, dim
    );
}

void launch_poincare_distance_f32(
    const float* Q, const float* K, float* dist, float c,
    int batch_size, int num_heads, int seq_len, int dim,
    cudaStream_t stream
) {
    dim3 blocks((seq_len + 31) / 32, num_heads, batch_size);
    dim3 threads(32, 32);
    poincare_distance_kernel<float><<<blocks, threads, 0, stream>>>(
        Q, K, dist, c, batch_size, num_heads, seq_len, dim
    );
}

void launch_fused_hyperbolic_attention_f32(
    const float* Q, const float* K, const float* V, float* output,
    float c, float beta, bool causal,
    int batch_size, int num_heads, int seq_len, int dim,
    cudaStream_t stream
) {
    int smem_size = (dim + 1 + seq_len) * sizeof(float);
    dim3 blocks(seq_len, num_heads, batch_size);
    int threads = min(256, dim);
    fused_hyperbolic_attention_kernel<float><<<blocks, threads, smem_size, stream>>>(
        Q, K, V, output, c, beta, causal,
        batch_size, num_heads, seq_len, dim
    );
}

}  // extern "C"
