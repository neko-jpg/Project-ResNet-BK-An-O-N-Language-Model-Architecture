// gradient_feeder_cuda.cu
// CUDA extension for fast gradient scaling
// Simplified version for maximum compatibility

#include <torch/extension.h>
#include <vector>

// Scale all gradients using PyTorch's CUDA tensors
// This is a simple wrapper that leverages PyTorch's existing CUDA ops
void scale_all_gradients_cuda(
    std::vector<torch::Tensor>& grads,
    float scale
) {
    // Early exit
    if (scale <= 1.0f + 1e-6f || grads.empty()) {
        return;
    }
    
    // Use PyTorch's in-place multiply (already CUDA-optimized)
    for (auto& grad : grads) {
        if (grad.defined() && grad.numel() > 0) {
            grad.mul_(scale);
        }
    }
}

// Feeder state structure
struct FeederState {
    float clip_threshold = 50.0f;
    float scale_factor = 1.0f;
    float velocity = 0.0f;
    float last_grad = 0.0f;
    int step_count = 0;
    
    // Config
    float target_low = 0.5f;
    float target_high = 3.0f;
    float critical_threshold = 0.2f;
    float emergency_threshold = 0.1f;
    float max_scale = 3.0f;
    float min_threshold = 5.0f;
    float max_threshold = 200.0f;
    float reaction_speed = 0.4f;
    float prediction_weight = 0.6f;
};

// C++ implementation of feed() - no Python GIL overhead
std::tuple<float, float, std::string> feed_cpp(
    FeederState& state,
    float grad_norm
) {
    state.step_count++;
    std::string action = "hold";
    
    // Velocity calculation
    if (state.step_count > 1) {
        state.velocity = grad_norm - state.last_grad;
    }
    state.last_grad = grad_norm;
    
    // Prediction
    float predicted = grad_norm + state.velocity * state.prediction_weight;
    
    // Upper bound control (explosion)
    if (grad_norm > state.target_high || predicted > state.target_high) {
        float excess = std::max(grad_norm, predicted) / state.target_high;
        float reduction = 1.0f - state.reaction_speed * (excess - 1.0f) * 0.5f;
        reduction = std::max(reduction, 0.7f);
        state.clip_threshold *= reduction;
        action = "threshold_lower";
        state.scale_factor = 1.0f;
    }
    // Lower bound control (vanishing)
    else if (grad_norm < state.critical_threshold || predicted < state.emergency_threshold) {
        if (grad_norm < state.emergency_threshold) {
            float target_grad = state.target_low * 1.5f;
            float needed_boost = target_grad / (grad_norm + 1e-8f);
            state.scale_factor = std::min(needed_boost, state.max_scale);
            action = "EMERGENCY_SCALE";
        } else if (grad_norm < state.critical_threshold) {
            float deficit_ratio = state.target_low / (grad_norm + 1e-8f);
            state.scale_factor = 1.0f + (deficit_ratio - 1.0f) * 0.5f;
            state.scale_factor = std::min(state.scale_factor, state.max_scale * 0.7f);
            action = "scale_boost";
        } else {
            state.scale_factor = 1.0f;
        }
        state.clip_threshold = std::min(state.clip_threshold * 1.05f, state.max_threshold);
    }
    // Healthy range
    else {
        state.scale_factor = state.scale_factor * 0.9f + 1.0f * 0.1f;
        state.clip_threshold = state.clip_threshold * 0.98f + 50.0f * 0.02f;
        action = "hold";
    }
    
    // Clamp
    state.clip_threshold = std::max(state.min_threshold, std::min(state.max_threshold, state.clip_threshold));
    state.scale_factor = std::max(1.0f, std::min(state.max_scale, state.scale_factor));
    
    return std::make_tuple(state.clip_threshold, state.scale_factor, action);
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scale_all_gradients", &scale_all_gradients_cuda, 
          "Scale all gradients");
    
    pybind11::class_<FeederState>(m, "FeederState")
        .def(pybind11::init<>())
        .def_readwrite("clip_threshold", &FeederState::clip_threshold)
        .def_readwrite("scale_factor", &FeederState::scale_factor)
        .def_readwrite("velocity", &FeederState::velocity)
        .def_readwrite("step_count", &FeederState::step_count)
        .def_readwrite("target_low", &FeederState::target_low)
        .def_readwrite("target_high", &FeederState::target_high)
        .def_readwrite("max_scale", &FeederState::max_scale);
    
    m.def("feed", &feed_cpp, "Feed gradient norm and get threshold/scale");
}
