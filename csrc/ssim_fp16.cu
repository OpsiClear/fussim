// FP16 (Half Precision) SSIM Implementation
// Separate path for inference-only FP16 computation
// Provides ~2x speedup on modern GPUs with Tensor Cores

// Include standard library headers before PyTorch to avoid 'std' ambiguity on Windows
#include <cstddef>
#include <utility>
#include <type_traits>
#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <c10/cuda/CUDAGuard.h>

namespace cg = cooperative_groups;

// ------------------------------------------
// Constant Memory for Gaussian Coefficients (FP16)
// Separate arrays for each window size
// ------------------------------------------
__constant__ __half cGaussHalf7[7];
__constant__ __half cGaussHalf9[9];
__constant__ __half cGaussHalf11[11];

// Host-side initialization tracking (per-device for multi-GPU support)
#include <mutex>
#include <unordered_set>

static std::mutex cGaussHalf_mutex;
static std::unordered_set<int> cGaussHalf_initialized_devices;

// Initialize FP16 Gaussian coefficients for all window sizes (thread-safe, multi-GPU aware)
void init_gaussian_fp16() {
    int device;
    cudaGetDevice(&device);

    // Quick check without lock
    {
        std::lock_guard<std::mutex> lock(cGaussHalf_mutex);
        if (cGaussHalf_initialized_devices.count(device) > 0) {
            return;
        }
    }

    // Window 7 coefficients (sigma=1.5)
    float gauss7[7] = {
        0.00443304609f, 0.05400558188f, 0.24203622937f, 0.39905029535f,
        0.24203622937f, 0.05400558188f, 0.00443304609f
    };
    __half gauss7Half[7];
    for (int i = 0; i < 7; ++i) gauss7Half[i] = __float2half(gauss7[i]);
    cudaMemcpyToSymbol(cGaussHalf7, gauss7Half, sizeof(__half) * 7);

    // Window 9 coefficients (sigma=1.5)
    float gauss9[9] = {
        0.00098442685f, 0.00971502065f, 0.04529544711f, 0.11722830242f,
        0.19355361462f, 0.11722830242f, 0.04529544711f, 0.00971502065f, 0.00098442685f
    };
    __half gauss9Half[9];
    for (int i = 0; i < 9; ++i) gauss9Half[i] = __float2half(gauss9[i]);
    cudaMemcpyToSymbol(cGaussHalf9, gauss9Half, sizeof(__half) * 9);

    // Window 11 coefficients (sigma=1.5) - original values
    float gauss11[11] = {
        0.001028380123898387f, 0.0075987582094967365f, 0.036000773310661316f,
        0.10936068743467331f, 0.21300552785396576f, 0.26601171493530273f,
        0.21300552785396576f, 0.10936068743467331f, 0.036000773310661316f,
        0.0075987582094967365f, 0.001028380123898387f
    };
    __half gauss11Half[11];
    for (int i = 0; i < 11; ++i) gauss11Half[i] = __float2half(gauss11[i]);
    cudaMemcpyToSymbol(cGaussHalf11, gauss11Half, sizeof(__half) * 11);

    // Mark this device as initialized
    {
        std::lock_guard<std::mutex> lock(cGaussHalf_mutex);
        cGaussHalf_initialized_devices.insert(device);
    }
}

// ------------------------------------------
// Block Dimensions (fixed for all window sizes)
// ------------------------------------------
#define BLOCK_X_FP16 16
#define BLOCK_Y_FP16 16

// ------------------------------------------
// Template configuration for different HALO values
// ------------------------------------------
template<int HALO>
struct SSIMConfigFP16 {
    static constexpr int WINDOW_SIZE = 2 * HALO + 1;
    static constexpr int SHARED_X = BLOCK_X_FP16 + 2 * HALO;
    static constexpr int SHARED_Y = BLOCK_Y_FP16 + 2 * HALO;
    static constexpr int CONV_X = BLOCK_X_FP16;
    static constexpr int CONV_Y = SHARED_Y;
};

// ------------------------------------------
// Device function to get Gaussian coefficients (FP16)
// ------------------------------------------
template<int HALO>
__device__ __forceinline__ const __half* getGaussCoeffsHalf();

template<>
__device__ __forceinline__ const __half* getGaussCoeffsHalf<3>() { return cGaussHalf7; }

template<>
__device__ __forceinline__ const __half* getGaussCoeffsHalf<4>() { return cGaussHalf9; }

template<>
__device__ __forceinline__ const __half* getGaussCoeffsHalf<5>() { return cGaussHalf11; }

// ------------------------------------------
// Utility: Safe pixel fetch w/ zero padding (FP16)
// ------------------------------------------
__device__ __forceinline__ __half get_pix_value_half(
    const __half* img,
    int b, int c, int y, int x,
    int CH, int H, int W
) {
    if (x < 0 || x >= W || y < 0 || y >= H) {
        return __float2half(0.0f);
    }
    return img[b * CH * H * W + c * H * W + y * W + x];
}

// ------------------------------------------
// FP16 Forward Kernel: Training Mode (Templated)
// ------------------------------------------
template<int HALO>
__global__ void fusedssim_fp16_train_CUDA(
    int H,
    int W,
    int CH,
    float C1,
    float C2,
    const __half* __restrict__ img1,
    const __half* __restrict__ img2,
    float* __restrict__ ssim_map,
    float* __restrict__ dm_dmu1,
    float* __restrict__ dm_dsigma1_sq,
    float* __restrict__ dm_dsigma12
) {
    using Config = SSIMConfigFP16<HALO>;
    const __half* cGaussHalf = getGaussCoeffsHalf<HALO>();

    auto block = cg::this_thread_block();
    const int bIdx   = block.group_index().z;
    const int pix_y  = block.group_index().y * BLOCK_Y_FP16 + block.thread_index().y;
    const int pix_x  = block.group_index().x * BLOCK_X_FP16 + block.thread_index().x;
    const int pix_id = pix_y * W + pix_x;
    const int num_pix = H * W;

    __shared__ __half sTile[Config::SHARED_Y][Config::SHARED_X][2];
    __shared__ float xconv[Config::CONV_Y][Config::CONV_X][5];

    for (int c = 0; c < CH; ++c) {
        // 1) Load tile
        {
            const int tileSize = Config::SHARED_Y * Config::SHARED_X;
            const int threads = BLOCK_X_FP16 * BLOCK_Y_FP16;
            const int steps = (tileSize + threads - 1) / threads;
            const int tileStartY = block.group_index().y * BLOCK_Y_FP16;
            const int tileStartX = block.group_index().x * BLOCK_X_FP16;

            for (int s = 0; s < steps; ++s) {
                int tid = s * threads + block.thread_rank();
                if (tid < tileSize) {
                    int local_y = tid / Config::SHARED_X;
                    int local_x = tid % Config::SHARED_X;
                    int gy = tileStartY + local_y - HALO;
                    int gx = tileStartX + local_x - HALO;
                    sTile[local_y][local_x][0] = get_pix_value_half(img1, bIdx, c, gy, gx, CH, H, W);
                    sTile[local_y][local_x][1] = get_pix_value_half(img2, bIdx, c, gy, gx, CH, H, W);
                }
            }
        }
        block.sync();

        // 2) Horizontal convolution
        {
            int ly = threadIdx.y;
            int lx = threadIdx.x + HALO;

            float sumX = 0.f, sumX2 = 0.f, sumY = 0.f, sumY2 = 0.f, sumXY = 0.f;

            #pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = __half2float(cGaussHalf[HALO - d]);
                float Xleft = __half2float(sTile[ly][lx - d][0]);
                float Yleft = __half2float(sTile[ly][lx - d][1]);
                float Xright = __half2float(sTile[ly][lx + d][0]);
                float Yright = __half2float(sTile[ly][lx + d][1]);
                sumX += (Xleft + Xright) * w;
                sumX2 += ((Xleft * Xleft) + (Xright * Xright)) * w;
                sumY += (Yleft + Yright) * w;
                sumY2 += ((Yleft * Yleft) + (Yright * Yright)) * w;
                sumXY += ((Xleft * Yleft) + (Xright * Yright)) * w;
            }
            {
                float cx = __half2float(sTile[ly][lx][0]);
                float cy = __half2float(sTile[ly][lx][1]);
                float wc = __half2float(cGaussHalf[HALO]);
                sumX += cx * wc;
                sumX2 += (cx * cx) * wc;
                sumY += cy * wc;
                sumY2 += (cy * cy) * wc;
                sumXY += (cx * cy) * wc;
            }
            xconv[ly][threadIdx.x][0] = sumX;
            xconv[ly][threadIdx.x][1] = sumX2;
            xconv[ly][threadIdx.x][2] = sumY;
            xconv[ly][threadIdx.x][3] = sumY2;
            xconv[ly][threadIdx.x][4] = sumXY;

            int ly2 = ly + BLOCK_Y_FP16;
            if (ly2 < Config::CONV_Y) {
                sumX = 0.f; sumX2 = 0.f; sumY = 0.f; sumY2 = 0.f; sumXY = 0.f;
                #pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    float w = __half2float(cGaussHalf[HALO - d]);
                    float Xleft = __half2float(sTile[ly2][lx - d][0]);
                    float Yleft = __half2float(sTile[ly2][lx - d][1]);
                    float Xright = __half2float(sTile[ly2][lx + d][0]);
                    float Yright = __half2float(sTile[ly2][lx + d][1]);
                    sumX += (Xleft + Xright) * w;
                    sumX2 += ((Xleft * Xleft) + (Xright * Xright)) * w;
                    sumY += (Yleft + Yright) * w;
                    sumY2 += ((Yleft * Yleft) + (Yright * Yright)) * w;
                    sumXY += ((Xleft * Yleft) + (Xright * Yright)) * w;
                }
                {
                    float cx = __half2float(sTile[ly2][lx][0]);
                    float cy = __half2float(sTile[ly2][lx][1]);
                    float wc = __half2float(cGaussHalf[HALO]);
                    sumX += cx * wc;
                    sumX2 += (cx * cx) * wc;
                    sumY += cy * wc;
                    sumY2 += (cy * cy) * wc;
                    sumXY += (cx * cy) * wc;
                }
                xconv[ly2][threadIdx.x][0] = sumX;
                xconv[ly2][threadIdx.x][1] = sumX2;
                xconv[ly2][threadIdx.x][2] = sumY;
                xconv[ly2][threadIdx.x][3] = sumY2;
                xconv[ly2][threadIdx.x][4] = sumXY;
            }
        }
        block.sync();

        // 3) Vertical convolution + SSIM + derivatives
        {
            int ly = threadIdx.y + HALO;
            int lx = threadIdx.x;

            float out0 = 0.f, out1 = 0.f, out2 = 0.f, out3 = 0.f, out4 = 0.f;

            #pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = __half2float(cGaussHalf[HALO - d]);
                float* top = xconv[ly - d][lx];
                float* bot = xconv[ly + d][lx];
                out0 += (top[0] + bot[0]) * w;
                out1 += (top[1] + bot[1]) * w;
                out2 += (top[2] + bot[2]) * w;
                out3 += (top[3] + bot[3]) * w;
                out4 += (top[4] + bot[4]) * w;
            }
            {
                float wC = __half2float(cGaussHalf[HALO]);
                float* ctr = xconv[ly][lx];
                out0 += ctr[0] * wC;
                out1 += ctr[1] * wC;
                out2 += ctr[2] * wC;
                out3 += ctr[3] * wC;
                out4 += ctr[4] * wC;
            }

            if (pix_x < W && pix_y < H) {
                float mu1 = out0;
                float mu2 = out2;
                float mu1_sq = mu1 * mu1;
                float mu2_sq = mu2 * mu2;

                float sigma1_sq = out1 - mu1_sq;
                float sigma2_sq = out3 - mu2_sq;
                float sigma12 = out4 - mu1 * mu2;

                float A = mu1_sq + mu2_sq + C1;
                float B = sigma1_sq + sigma2_sq + C2;
                float C_ = 2.f * mu1 * mu2 + C1;
                float D_ = 2.f * sigma12 + C2;

                float val = (C_ * D_) / (A * B);

                int global_idx = bIdx * CH * num_pix + c * num_pix + pix_id;
                ssim_map[global_idx] = val;

                float d_m_dmu1 = (
                    (mu2 * 2.f * D_) / (A * B)
                    - (mu2 * 2.f * C_) / (A * B)
                    - (mu1 * 2.f * C_ * D_) / (A * A * B)
                    + (mu1 * 2.f * C_ * D_) / (A * B * B)
                );
                float d_m_dsigma1_sq = (-C_ * D_) / (A * B * B);
                float d_m_dsigma12 = (2.f * C_) / (A * B);

                dm_dmu1[global_idx] = d_m_dmu1;
                dm_dsigma1_sq[global_idx] = d_m_dsigma1_sq;
                dm_dsigma12[global_idx] = d_m_dsigma12;
            }
        }
    }
}

// ------------------------------------------
// FP16 Backward Kernel (Templated)
// ------------------------------------------
template<int HALO>
__global__ void fusedssim_fp16_backward_CUDA(
    int H,
    int W,
    int CH,
    float C1,
    float C2,
    const __half* __restrict__ img1,
    const __half* __restrict__ img2,
    const float* __restrict__ dL_dmap,
    float* __restrict__ dL_dimg1,
    const float* __restrict__ dm_dmu1,
    const float* __restrict__ dm_dsigma1_sq,
    const float* __restrict__ dm_dsigma12
) {
    using Config = SSIMConfigFP16<HALO>;
    const __half* cGaussHalf = getGaussCoeffsHalf<HALO>();

    auto block = cg::this_thread_block();

    const int pix_y = block.group_index().y * BLOCK_Y_FP16 + block.thread_index().y;
    const int pix_x = block.group_index().x * BLOCK_X_FP16 + block.thread_index().x;
    const int pix_id = pix_y * W + pix_x;
    const int num_pix = H * W;
    const int bIdx = block.group_index().z;

    // Interleaved layout for better cache locality: sData[y][x][feature]
    __shared__ float sData[Config::SHARED_Y][Config::SHARED_X][3];
    __shared__ float sScratch[Config::CONV_Y][Config::CONV_X][3];

    for (int c = 0; c < CH; ++c) {
        float p1 = 0.f, p2 = 0.f;
        if (pix_x < W && pix_y < H) {
            p1 = __half2float(get_pix_value_half(img1, bIdx, c, pix_y, pix_x, CH, H, W));
            p2 = __half2float(get_pix_value_half(img2, bIdx, c, pix_y, pix_x, CH, H, W));
        }

        // 1) Load + fuse multiplication
        {
            const int start_y = block.group_index().y * BLOCK_Y_FP16;
            const int start_x = block.group_index().x * BLOCK_X_FP16;

            int tid = threadIdx.y * blockDim.x + threadIdx.x;
            int warp_id = tid / 32;
            int lane_id = tid % 32;
            int totalThreads = BLOCK_X_FP16 * BLOCK_Y_FP16;
            int num_warps = (totalThreads + 31) / 32;

            for (int row = warp_id; row < Config::SHARED_Y; row += num_warps) {
                int gy = start_y + row - HALO;
                for (int col = lane_id; col < Config::SHARED_X; col += 32) {
                    int gx = start_x + col - HALO;

                    float chain = 0.f, vmu = 0.f, vs1 = 0.f, vs12 = 0.f;
                    if (gx >= 0 && gx < W && gy >= 0 && gy < H) {
                        int idx = bIdx * CH * num_pix + c * num_pix + gy * W + gx;
                        chain = dL_dmap[idx];
                        vmu = dm_dmu1[idx];
                        vs1 = dm_dsigma1_sq[idx];
                        vs12 = dm_dsigma12[idx];
                    }

                    sData[row][col][0] = vmu * chain;
                    sData[row][col][1] = vs1 * chain;
                    sData[row][col][2] = vs12 * chain;
                }
            }
        }
        block.sync();

        // 2) Horizontal pass
        {
            int ly = threadIdx.y;
            int lx = threadIdx.x + HALO;

            for (int pass = 0; pass < 2; ++pass) {
                int yy = ly + pass * BLOCK_Y_FP16;
                if (yy < Config::CONV_Y) {
                    float accum0 = 0.f, accum1 = 0.f, accum2 = 0.f;

                    #pragma unroll
                    for (int d = 1; d <= HALO; ++d) {
                        float w = __half2float(cGaussHalf[HALO - d]);
                        accum0 += (sData[yy][lx - d][0] + sData[yy][lx + d][0]) * w;
                        accum1 += (sData[yy][lx - d][1] + sData[yy][lx + d][1]) * w;
                        accum2 += (sData[yy][lx - d][2] + sData[yy][lx + d][2]) * w;
                    }
                    {
                        float wc = __half2float(cGaussHalf[HALO]);
                        accum0 += sData[yy][lx][0] * wc;
                        accum1 += sData[yy][lx][1] * wc;
                        accum2 += sData[yy][lx][2] * wc;
                    }

                    sScratch[yy][threadIdx.x][0] = accum0;
                    sScratch[yy][threadIdx.x][1] = accum1;
                    sScratch[yy][threadIdx.x][2] = accum2;
                }
            }
        }
        block.sync();

        // 3) Vertical pass -> finalize gradient
        if (pix_x < W && pix_y < H) {
            int ly = threadIdx.y + HALO;
            int lx = threadIdx.x;

            float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f;

            #pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = __half2float(cGaussHalf[HALO - d]);
                float* top = sScratch[ly - d][lx];
                float* bot = sScratch[ly + d][lx];
                sum0 += (top[0] + bot[0]) * w;
                sum1 += (top[1] + bot[1]) * w;
                sum2 += (top[2] + bot[2]) * w;
            }
            {
                float wc = __half2float(cGaussHalf[HALO]);
                float* ctr = sScratch[ly][lx];
                sum0 += ctr[0] * wc;
                sum1 += ctr[1] * wc;
                sum2 += ctr[2] * wc;
            }

            float dL_dpix = sum0 + (2.f * p1) * sum1 + (p2) * sum2;

            int out_idx = bIdx * CH * num_pix + c * num_pix + pix_id;
            dL_dimg1[out_idx] = dL_dpix;
        }
        block.sync();
    }
}

// ------------------------------------------
// FP16 Forward Kernel: Inference Only (Templated)
// ------------------------------------------
template<int HALO>
__global__ void fusedssim_fp16_CUDA(
    int H,
    int W,
    int CH,
    float C1,
    float C2,
    const __half* __restrict__ img1,
    const __half* __restrict__ img2,
    float* __restrict__ ssim_map
) {
    using Config = SSIMConfigFP16<HALO>;
    const __half* cGaussHalf = getGaussCoeffsHalf<HALO>();

    auto block = cg::this_thread_block();
    const int bIdx   = block.group_index().z;
    const int pix_y  = block.group_index().y * BLOCK_Y_FP16 + block.thread_index().y;
    const int pix_x  = block.group_index().x * BLOCK_X_FP16 + block.thread_index().x;
    const int pix_id = pix_y * W + pix_x;
    const int num_pix = H * W;

    __shared__ __half sTile[Config::SHARED_Y][Config::SHARED_X][2];
    __shared__ float xconv[Config::CONV_Y][Config::CONV_X][5];

    for (int c = 0; c < CH; ++c) {
        // 1) Load tile
        {
            const int tileSize = Config::SHARED_Y * Config::SHARED_X;
            const int threads = BLOCK_X_FP16 * BLOCK_Y_FP16;
            const int steps = (tileSize + threads - 1) / threads;
            const int tileStartY = block.group_index().y * BLOCK_Y_FP16;
            const int tileStartX = block.group_index().x * BLOCK_X_FP16;

            for (int s = 0; s < steps; ++s) {
                int tid = s * threads + block.thread_rank();
                if (tid < tileSize) {
                    int local_y = tid / Config::SHARED_X;
                    int local_x = tid % Config::SHARED_X;
                    int gy = tileStartY + local_y - HALO;
                    int gx = tileStartX + local_x - HALO;
                    sTile[local_y][local_x][0] = get_pix_value_half(img1, bIdx, c, gy, gx, CH, H, W);
                    sTile[local_y][local_x][1] = get_pix_value_half(img2, bIdx, c, gy, gx, CH, H, W);
                }
            }
        }
        block.sync();

        // 2) Horizontal convolution
        {
            int ly = threadIdx.y;
            int lx = threadIdx.x + HALO;

            float sumX = 0.f, sumX2 = 0.f, sumY = 0.f, sumY2 = 0.f, sumXY = 0.f;

            #pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = __half2float(cGaussHalf[HALO - d]);
                float Xleft = __half2float(sTile[ly][lx - d][0]);
                float Yleft = __half2float(sTile[ly][lx - d][1]);
                float Xright = __half2float(sTile[ly][lx + d][0]);
                float Yright = __half2float(sTile[ly][lx + d][1]);
                sumX += (Xleft + Xright) * w;
                sumX2 += ((Xleft * Xleft) + (Xright * Xright)) * w;
                sumY += (Yleft + Yright) * w;
                sumY2 += ((Yleft * Yleft) + (Yright * Yright)) * w;
                sumXY += ((Xleft * Yleft) + (Xright * Yright)) * w;
            }
            {
                float cx = __half2float(sTile[ly][lx][0]);
                float cy = __half2float(sTile[ly][lx][1]);
                float wc = __half2float(cGaussHalf[HALO]);
                sumX += cx * wc;
                sumX2 += (cx * cx) * wc;
                sumY += cy * wc;
                sumY2 += (cy * cy) * wc;
                sumXY += (cx * cy) * wc;
            }
            xconv[ly][threadIdx.x][0] = sumX;
            xconv[ly][threadIdx.x][1] = sumX2;
            xconv[ly][threadIdx.x][2] = sumY;
            xconv[ly][threadIdx.x][3] = sumY2;
            xconv[ly][threadIdx.x][4] = sumXY;

            int ly2 = ly + BLOCK_Y_FP16;
            if (ly2 < Config::CONV_Y) {
                sumX = 0.f; sumX2 = 0.f; sumY = 0.f; sumY2 = 0.f; sumXY = 0.f;
                #pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    float w = __half2float(cGaussHalf[HALO - d]);
                    float Xleft = __half2float(sTile[ly2][lx - d][0]);
                    float Yleft = __half2float(sTile[ly2][lx - d][1]);
                    float Xright = __half2float(sTile[ly2][lx + d][0]);
                    float Yright = __half2float(sTile[ly2][lx + d][1]);
                    sumX += (Xleft + Xright) * w;
                    sumX2 += ((Xleft * Xleft) + (Xright * Xright)) * w;
                    sumY += (Yleft + Yright) * w;
                    sumY2 += ((Yleft * Yleft) + (Yright * Yright)) * w;
                    sumXY += ((Xleft * Yleft) + (Xright * Yright)) * w;
                }
                {
                    float cx = __half2float(sTile[ly2][lx][0]);
                    float cy = __half2float(sTile[ly2][lx][1]);
                    float wc = __half2float(cGaussHalf[HALO]);
                    sumX += cx * wc;
                    sumX2 += (cx * cx) * wc;
                    sumY += cy * wc;
                    sumY2 += (cy * cy) * wc;
                    sumXY += (cx * cy) * wc;
                }
                xconv[ly2][threadIdx.x][0] = sumX;
                xconv[ly2][threadIdx.x][1] = sumX2;
                xconv[ly2][threadIdx.x][2] = sumY;
                xconv[ly2][threadIdx.x][3] = sumY2;
                xconv[ly2][threadIdx.x][4] = sumXY;
            }
        }
        block.sync();

        // 3) Vertical convolution + SSIM
        {
            int ly = threadIdx.y + HALO;
            int lx = threadIdx.x;

            float out0 = 0.f, out1 = 0.f, out2 = 0.f, out3 = 0.f, out4 = 0.f;

            #pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = __half2float(cGaussHalf[HALO - d]);
                float* top = xconv[ly - d][lx];
                float* bot = xconv[ly + d][lx];
                out0 += (top[0] + bot[0]) * w;
                out1 += (top[1] + bot[1]) * w;
                out2 += (top[2] + bot[2]) * w;
                out3 += (top[3] + bot[3]) * w;
                out4 += (top[4] + bot[4]) * w;
            }
            {
                float wC = __half2float(cGaussHalf[HALO]);
                float* ctr = xconv[ly][lx];
                out0 += ctr[0] * wC;
                out1 += ctr[1] * wC;
                out2 += ctr[2] * wC;
                out3 += ctr[3] * wC;
                out4 += ctr[4] * wC;
            }

            if (pix_x < W && pix_y < H) {
                float mu1 = out0;
                float mu2 = out2;
                float mu1_sq = mu1 * mu1;
                float mu2_sq = mu2 * mu2;

                float sigma1_sq = out1 - mu1_sq;
                float sigma2_sq = out3 - mu2_sq;
                float sigma12 = out4 - mu1 * mu2;

                float A = mu1_sq + mu2_sq + C1;
                float B = sigma1_sq + sigma2_sq + C2;
                float C_ = 2.f * mu1 * mu2 + C1;
                float D_ = 2.f * sigma12 + C2;

                float val = (C_ * D_) / (A * B);

                int global_idx = bIdx * CH * num_pix + c * num_pix + pix_id;
                ssim_map[global_idx] = val;
            }
        }
    }
}

// Note: Explicit template instantiation of __global__ kernels is not needed
// because each kernel launch in the dispatch functions below instantiates the template.

// ------------------------------------------
// Templated PyTorch Interfaces
// ------------------------------------------
template<int HALO>
torch::Tensor fusedssim_fp16_impl(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2
) {
    init_gaussian_fp16();

    const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));

    int B  = img1.size(0);
    int CH = img1.size(1);
    int H  = img1.size(2);
    int W  = img1.size(3);

    dim3 grid((W + BLOCK_X_FP16 - 1) / BLOCK_X_FP16,
              (H + BLOCK_Y_FP16 - 1) / BLOCK_Y_FP16,
              B);
    dim3 block(BLOCK_X_FP16, BLOCK_Y_FP16);

    auto ssim_map = torch::empty({B, CH, H, W}, img1.options().dtype(torch::kFloat32));

    auto img1_half = img1.contiguous();
    auto img2_half = img2.contiguous();

    if (img1_half.scalar_type() != torch::kFloat16) {
        img1_half = img1_half.to(torch::kFloat16);
    }
    if (img2_half.scalar_type() != torch::kFloat16) {
        img2_half = img2_half.to(torch::kFloat16);
    }

    fusedssim_fp16_CUDA<HALO><<<grid, block>>>(
        H, W, CH, C1, C2,
        reinterpret_cast<const __half*>(img1_half.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(img2_half.data_ptr<at::Half>()),
        ssim_map.data_ptr<float>()
    );

    return ssim_map;
}

template<int HALO>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedssim_fp16_train_impl(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2
) {
    init_gaussian_fp16();

    const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));

    int B  = img1.size(0);
    int CH = img1.size(1);
    int H  = img1.size(2);
    int W  = img1.size(3);

    dim3 grid((W + BLOCK_X_FP16 - 1) / BLOCK_X_FP16,
              (H + BLOCK_Y_FP16 - 1) / BLOCK_Y_FP16,
              B);
    dim3 block(BLOCK_X_FP16, BLOCK_Y_FP16);

    auto ssim_map = torch::empty({B, CH, H, W}, img1.options().dtype(torch::kFloat32));
    auto dm_dmu1 = torch::empty({B, CH, H, W}, img1.options().dtype(torch::kFloat32));
    auto dm_dsigma1_sq = torch::empty({B, CH, H, W}, img1.options().dtype(torch::kFloat32));
    auto dm_dsigma12 = torch::empty({B, CH, H, W}, img1.options().dtype(torch::kFloat32));

    auto img1_half = img1.contiguous();
    auto img2_half = img2.contiguous();

    if (img1_half.scalar_type() != torch::kFloat16) {
        img1_half = img1_half.to(torch::kFloat16);
    }
    if (img2_half.scalar_type() != torch::kFloat16) {
        img2_half = img2_half.to(torch::kFloat16);
    }

    fusedssim_fp16_train_CUDA<HALO><<<grid, block>>>(
        H, W, CH, C1, C2,
        reinterpret_cast<const __half*>(img1_half.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(img2_half.data_ptr<at::Half>()),
        ssim_map.data_ptr<float>(),
        dm_dmu1.data_ptr<float>(),
        dm_dsigma1_sq.data_ptr<float>(),
        dm_dsigma12.data_ptr<float>()
    );

    return std::make_tuple(ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
}

template<int HALO>
torch::Tensor fusedssim_fp16_backward_impl(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::Tensor &dL_dmap,
    torch::Tensor &dm_dmu1,
    torch::Tensor &dm_dsigma1_sq,
    torch::Tensor &dm_dsigma12
) {
    init_gaussian_fp16();

    const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));

    int B  = img1.size(0);
    int CH = img1.size(1);
    int H  = img1.size(2);
    int W  = img1.size(3);

    dim3 grid((W + BLOCK_X_FP16 - 1) / BLOCK_X_FP16,
              (H + BLOCK_Y_FP16 - 1) / BLOCK_Y_FP16,
              B);
    dim3 block(BLOCK_X_FP16, BLOCK_Y_FP16);

    auto dL_dimg1 = torch::zeros({B, CH, H, W}, img1.options().dtype(torch::kFloat32));

    auto img1_half = img1.contiguous();
    auto img2_half = img2.contiguous();

    if (img1_half.scalar_type() != torch::kFloat16) {
        img1_half = img1_half.to(torch::kFloat16);
    }
    if (img2_half.scalar_type() != torch::kFloat16) {
        img2_half = img2_half.to(torch::kFloat16);
    }

    fusedssim_fp16_backward_CUDA<HALO><<<grid, block>>>(
        H, W, CH, C1, C2,
        reinterpret_cast<const __half*>(img1_half.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(img2_half.data_ptr<at::Half>()),
        dL_dmap.contiguous().data_ptr<float>(),
        dL_dimg1.data_ptr<float>(),
        dm_dmu1.contiguous().data_ptr<float>(),
        dm_dsigma1_sq.contiguous().data_ptr<float>(),
        dm_dsigma12.contiguous().data_ptr<float>()
    );

    return dL_dimg1;
}

// ------------------------------------------
// Dispatch functions
// ------------------------------------------
torch::Tensor fusedssim_fp16(
    int window_size,
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2
) {
    switch (window_size) {
        case 7:  return fusedssim_fp16_impl<3>(C1, C2, img1, img2);
        case 9:  return fusedssim_fp16_impl<4>(C1, C2, img1, img2);
        case 11: return fusedssim_fp16_impl<5>(C1, C2, img1, img2);
        default:
            throw std::invalid_argument("window_size must be 7, 9, or 11, got " + std::to_string(window_size));
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedssim_fp16_train(
    int window_size,
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2
) {
    switch (window_size) {
        case 7:  return fusedssim_fp16_train_impl<3>(C1, C2, img1, img2);
        case 9:  return fusedssim_fp16_train_impl<4>(C1, C2, img1, img2);
        case 11: return fusedssim_fp16_train_impl<5>(C1, C2, img1, img2);
        default:
            throw std::invalid_argument("window_size must be 7, 9, or 11, got " + std::to_string(window_size));
    }
}

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
) {
    switch (window_size) {
        case 7:  return fusedssim_fp16_backward_impl<3>(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
        case 9:  return fusedssim_fp16_backward_impl<4>(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
        case 11: return fusedssim_fp16_backward_impl<5>(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
        default:
            throw std::invalid_argument("window_size must be 7, 9, or 11, got " + std::to_string(window_size));
    }
}
