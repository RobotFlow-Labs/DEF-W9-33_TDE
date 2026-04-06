/**
 * ANIMA TDE: Custom CUDA Kernels for Spiking Neural Network Operations
 *
 * 1. fused_lif_forward — Fused LIF neuron (all T timesteps in one launch)
 * 2. fused_sda_attention — Spike-Driven Attention (accumulation-only)
 * 3. fused_spike_encode — Spiking Encoder temporal mixing
 *
 * Target: sm_89 (L4 GPU), CUDA 12.8, PyTorch 2.11+
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ============================================================================
// 1. Fused LIF Forward: processes all T timesteps without Python loop
// ============================================================================

/**
 * Fused LIF neuron forward pass.
 * Processes (T, B, C, H, W) input in a single kernel launch.
 * Each thread handles one spatial/channel element across all timesteps.
 *
 * input: (T, B*C*H*W) flattened per-timestep
 * output: (T, B*C*H*W) binary spikes
 * membrane_out: (T, B*C*H*W) membrane potentials (for backward)
 */
__global__ void fused_lif_forward_kernel(
    const float* __restrict__ input,    // (T * N)
    float* __restrict__ spikes_out,     // (T * N)
    float* __restrict__ membrane_out,   // (T * N)
    int T, int N,
    float beta, float threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float membrane = 0.0f;

    for (int t = 0; t < T; t++) {
        int offset = t * N + idx;
        // Charge
        membrane = beta * membrane + input[offset];
        // Store membrane for backward
        membrane_out[offset] = membrane;
        // Fire
        float spike = (membrane >= threshold) ? 1.0f : 0.0f;
        spikes_out[offset] = spike;
        // Soft reset
        membrane = membrane - threshold * spike;
    }
}

/**
 * Fused LIF backward with atan surrogate gradient.
 * grad_input[t] = grad_spikes[t] * surrogate(membrane[t]) + beta * grad_from_future
 */
__global__ void fused_lif_backward_kernel(
    const float* __restrict__ grad_spikes,  // (T * N)
    const float* __restrict__ membrane,     // (T * N)
    float* __restrict__ grad_input,         // (T * N)
    int T, int N,
    float beta, float threshold, float surrogate_alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float grad_mem = 0.0f;
    float pi = 3.14159265358979f;

    // Backward: iterate from T-1 to 0
    for (int t = T - 1; t >= 0; t--) {
        int offset = t * N + idx;
        float mem = membrane[offset];

        // Surrogate gradient: atan
        float x = pi * surrogate_alpha * (mem - threshold);
        float surrogate = surrogate_alpha / 2.0f / (1.0f + x * x);

        // Spike was fired if membrane >= threshold
        float spike = (mem >= threshold) ? 1.0f : 0.0f;

        // grad through spike function
        grad_mem += grad_spikes[offset] * surrogate;

        // grad_input
        grad_input[offset] = grad_mem;

        // Propagate through reset: membrane = beta * prev - threshold * spike
        grad_mem = grad_mem * beta;
        // Through soft reset
        grad_mem -= grad_spikes[offset] * surrogate * threshold;
    }
}


// ============================================================================
// 2. Fused Spike-Driven Attention (SDA)
// ============================================================================

/**
 * SDA formula (accumulation only — no float*float multiply):
 * H_Att = g_temp_spike * g_ch_float * g_spa_spike
 *       + g_ch_spike * g_spa_float * g_temp_spike
 *       + g_spa_spike * g_temp_float * g_ch_spike
 *       + H
 *
 * Since spike tensors are binary {0,1}, spike*float = conditional copy (AC op).
 * This kernel fuses all 3 terms + residual in a single pass.
 */
__global__ void fused_sda_kernel(
    const float* __restrict__ H,             // (N,) hidden states
    const float* __restrict__ g_temp_spike,  // (N,) binary {0,1}
    const float* __restrict__ g_temp_float,  // (N,) float membrane
    const float* __restrict__ g_ch_spike,    // (N,) binary {0,1}
    const float* __restrict__ g_ch_float,    // (N,) float membrane
    const float* __restrict__ g_spa_spike,   // (N,) binary {0,1}
    const float* __restrict__ g_spa_float,   // (N,) float membrane
    float* __restrict__ output,              // (N,)
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float h = H[idx];
    float ts = g_temp_spike[idx];
    float tf = g_temp_float[idx];
    float cs = g_ch_spike[idx];
    float cf = g_ch_float[idx];
    float ss = g_spa_spike[idx];
    float sf = g_spa_float[idx];

    // Term 1: temp_spike * ch_float * spa_spike (AC only: spike masks float)
    float term1 = ts * cf * ss;
    // Term 2: ch_spike * spa_float * temp_spike
    float term2 = cs * sf * ts;
    // Term 3: spa_spike * temp_float * ch_spike
    float term3 = ss * tf * cs;

    output[idx] = term1 + term2 + term3 + h;
}


// ============================================================================
// 3. Fused Spiking Encoder
// ============================================================================

/**
 * Fused spiking encoder: alpha mixing + LIF in one pass.
 * At t=0: feat = input_feat
 * At t>0: feat = alpha * input_feat + (1-alpha) * temporal_conv(prev_feat)
 * Then LIF on the result.
 *
 * Note: temporal_conv is Conv2d which must run in PyTorch, but we fuse
 * the alpha mixing + LIF into a single kernel per timestep.
 */
__global__ void fused_alpha_mix_kernel(
    const float* __restrict__ input_feat,     // (N,)
    const float* __restrict__ temporal_feat,  // (N,)
    float* __restrict__ output,               // (N,)
    float alpha_sigmoid,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    output[idx] = alpha_sigmoid * input_feat[idx]
                + (1.0f - alpha_sigmoid) * temporal_feat[idx];
}


// ============================================================================
// Python bindings
// ============================================================================

torch::Tensor fused_lif_forward(
    torch::Tensor input,
    float beta, float threshold
) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dims (T, N)");

    int T = input.size(0);
    int N = input.numel() / T;

    auto input_flat = input.reshape({T, N}).contiguous();
    auto spikes = torch::zeros_like(input_flat);
    auto membrane = torch::zeros_like(input_flat);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    fused_lif_forward_kernel<<<blocks, threads>>>(
        input_flat.data_ptr<float>(),
        spikes.data_ptr<float>(),
        membrane.data_ptr<float>(),
        T, N, beta, threshold
    );

    return spikes.reshape_as(input);
}

torch::Tensor fused_lif_backward(
    torch::Tensor grad_spikes,
    torch::Tensor membrane,
    float beta, float threshold, float surrogate_alpha
) {
    TORCH_CHECK(grad_spikes.is_cuda(), "grad_spikes must be CUDA tensor");

    int T = grad_spikes.size(0);
    int N = grad_spikes.numel() / T;

    auto grad_flat = grad_spikes.reshape({T, N}).contiguous();
    auto mem_flat = membrane.reshape({T, N}).contiguous();
    auto grad_input = torch::zeros_like(grad_flat);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    fused_lif_backward_kernel<<<blocks, threads>>>(
        grad_flat.data_ptr<float>(),
        mem_flat.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        T, N, beta, threshold, surrogate_alpha
    );

    return grad_input.reshape_as(grad_spikes);
}

torch::Tensor fused_sda(
    torch::Tensor H,
    torch::Tensor g_temp_spike, torch::Tensor g_temp_float,
    torch::Tensor g_ch_spike, torch::Tensor g_ch_float,
    torch::Tensor g_spa_spike, torch::Tensor g_spa_float
) {
    TORCH_CHECK(H.is_cuda(), "H must be CUDA tensor");
    int N = H.numel();

    auto H_flat = H.reshape({N}).contiguous();
    auto output = torch::zeros_like(H_flat);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    fused_sda_kernel<<<blocks, threads>>>(
        H_flat.data_ptr<float>(),
        g_temp_spike.reshape({N}).contiguous().data_ptr<float>(),
        g_temp_float.reshape({N}).contiguous().data_ptr<float>(),
        g_ch_spike.reshape({N}).contiguous().data_ptr<float>(),
        g_ch_float.reshape({N}).contiguous().data_ptr<float>(),
        g_spa_spike.reshape({N}).contiguous().data_ptr<float>(),
        g_spa_float.reshape({N}).contiguous().data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );

    return output.reshape_as(H);
}

torch::Tensor fused_alpha_mix(
    torch::Tensor input_feat,
    torch::Tensor temporal_feat,
    float alpha_sigmoid
) {
    TORCH_CHECK(input_feat.is_cuda(), "input_feat must be CUDA tensor");
    int N = input_feat.numel();

    auto output = torch::zeros_like(input_feat);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    fused_alpha_mix_kernel<<<blocks, threads>>>(
        input_feat.reshape({N}).contiguous().data_ptr<float>(),
        temporal_feat.reshape({N}).contiguous().data_ptr<float>(),
        output.reshape({N}).data_ptr<float>(),
        alpha_sigmoid,
        N
    );

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_lif_forward", &fused_lif_forward,
          "Fused LIF forward (CUDA) — all T timesteps in one launch");
    m.def("fused_lif_backward", &fused_lif_backward,
          "Fused LIF backward with atan surrogate (CUDA)");
    m.def("fused_sda", &fused_sda,
          "Fused Spike-Driven Attention — accumulation only (CUDA)");
    m.def("fused_alpha_mix", &fused_alpha_mix,
          "Fused alpha mixing for Spiking Encoder (CUDA)");
}
