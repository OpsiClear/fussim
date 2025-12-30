#pragma once
#include <torch/extension.h>
#include <tuple>

// FP16 forward pass (inference only)
// Accepts FP16 or FP32 input (auto-converts), returns FP32 SSIM map
torch::Tensor fusedssim_fp16(
    int window_size,
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2
);

// FP16 forward pass (training mode)
// Returns: (ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
// Uses mixed precision: FP16 inputs, FP32 intermediate/output
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedssim_fp16_train(
    int window_size,
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2
);

// FP16 backward pass
// Uses mixed precision: FP16 inputs, FP32 gradients
torch::Tensor fusedssim_fp16_backward(
    int window_size,
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::Tensor &dL_dmap,
    torch::Tensor &dm_dmu1,
    torch::Tensor &dm_dsigma1_sq,
    torch::Tensor &dm_dsigma12
);
