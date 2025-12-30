// Include standard library headers before PyTorch to avoid 'std' ambiguity on Windows
#include <cstddef>
#include <utility>
#include <type_traits>
#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <torch/extension.h>
#include <cooperative_groups.h>
#include <c10/cuda/CUDAGuard.h>

namespace cg = cooperative_groups;

// ------------------------------------------
// Constant Memory for Gaussian Coefficients
// Pre-computed for sigma=1.5, normalized
// ------------------------------------------

// Window 7 (HALO=3): G(x) = exp(-x^2 / (2 * 1.5^2)) for x in [-3, 3]
__constant__ float cGauss7[7] = {
    0.00443304609f,
    0.05400558188f,
    0.24203622937f,
    0.39905029535f,
    0.24203622937f,
    0.05400558188f,
    0.00443304609f
};

// Window 9 (HALO=4): G(x) for x in [-4, 4]
__constant__ float cGauss9[9] = {
    0.00098442685f,
    0.00971502065f,
    0.04529544711f,
    0.11722830242f,
    0.19355361462f,
    0.11722830242f,
    0.04529544711f,
    0.00971502065f,
    0.00098442685f
};

// Window 11 (HALO=5): G(x) for x in [-5, 5] - original values
__constant__ float cGauss11[11] = {
    0.001028380123898387f,
    0.0075987582094967365f,
    0.036000773310661316f,
    0.10936068743467331f,
    0.21300552785396576f,
    0.26601171493530273f,
    0.21300552785396576f,
    0.10936068743467331f,
    0.036000773310661316f,
    0.0075987582094967365f,
    0.001028380123898387f
};

// ------------------------------------------
// Block Dimensions (fixed for all window sizes)
// ------------------------------------------
#define BLOCK_X 16
#define BLOCK_Y 16

// ------------------------------------------
// Template configuration for different HALO values
// ------------------------------------------
template<int HALO>
struct SSIMConfig {
    static constexpr int WINDOW_SIZE = 2 * HALO + 1;
    static constexpr int SHARED_X = BLOCK_X + 2 * HALO;
    static constexpr int SHARED_Y = BLOCK_Y + 2 * HALO;
    static constexpr int CONV_X = BLOCK_X;
    static constexpr int CONV_Y = SHARED_Y;
    // Padding to avoid bank conflicts (32 banks, 4-byte words)
    static constexpr int SHARED_X_PAD = SHARED_X + 1;
    static constexpr int CONV_X_PAD = CONV_X + 1;
};

// ------------------------------------------
// Device function to get Gaussian coefficients
// Uses template specialization for compile-time dispatch
// ------------------------------------------
template<int HALO>
__device__ __forceinline__ const float* getGaussCoeffs();

template<>
__device__ __forceinline__ const float* getGaussCoeffs<3>() { return cGauss7; }

template<>
__device__ __forceinline__ const float* getGaussCoeffs<4>() { return cGauss9; }

template<>
__device__ __forceinline__ const float* getGaussCoeffs<5>() { return cGauss11; }

// ------------------------------------------
// Utility: Safe pixel fetch w/ zero padding
// ------------------------------------------
__device__ __forceinline__ float get_pix_value(
    const float* img,
    int b, int c, int y, int x,
    int CH, int H, int W
) {
    if (x < 0 || x >= W || y < 0 || y >= H) {
        return 0.0f;
    }
    return img[b * CH * H * W + c * H * W + y * W + x];
}

// ------------------------------------------
// Forward Kernel: Fused SSIM (Templated)
// ------------------------------------------
template<int HALO>
__global__ void fusedssimCUDA(
    int H,
    int W,
    int CH,
    float C1,
    float C2,
    const float* __restrict__ img1,
    const float* __restrict__ img2,
    float* __restrict__ ssim_map,
    float* __restrict__ dm_dmu1,
    float* __restrict__ dm_dsigma1_sq,
    float* __restrict__ dm_dsigma12
) {
    using Config = SSIMConfig<HALO>;
    const float* cGauss = getGaussCoeffs<HALO>();

    auto block = cg::this_thread_block();
    const int bIdx   = block.group_index().z;
    const int pix_y  = block.group_index().y * BLOCK_Y + block.thread_index().y;
    const int pix_x  = block.group_index().x * BLOCK_X + block.thread_index().x;
    const int pix_id = pix_y * W + pix_x;
    const int num_pix = H * W;

    // Shared memory for the tile - using template config
    // Interleaved layout for better cache locality: sTile[y][x][img_idx]
    __shared__ float sTile[Config::SHARED_Y][Config::SHARED_X_PAD][2];
    // Features-innermost layout for better cache locality: xconv[y][x][feature]
    __shared__ float xconv[Config::CONV_Y][Config::CONV_X_PAD][5];

    for (int c = 0; c < CH; ++c) {
        // 1) Load tile + halo into shared memory
        {
            const int tileStartY = block.group_index().y * BLOCK_Y;
            const int tileStartX = block.group_index().x * BLOCK_X;

            const int gMinY = tileStartY - HALO;
            const int gMaxY = tileStartY + BLOCK_Y + HALO - 1;
            const int gMinX = tileStartX - HALO;
            const int gMaxX = tileStartX + BLOCK_X + HALO - 1;
            const bool fullTileInBounds = (gMinY >= 0) && (gMaxY < H) && (gMinX >= 0) && (gMaxX < W);

            const float* img1_base = img1 + bIdx * CH * H * W + c * H * W;
            const float* img2_base = img2 + bIdx * CH * H * W + c * H * W;

            if (fullTileInBounds) {
                for (int row = threadIdx.y; row < Config::SHARED_Y; row += BLOCK_Y) {
                    int gy = gMinY + row;
                    const float* row_ptr1 = img1_base + gy * W + gMinX;
                    const float* row_ptr2 = img2_base + gy * W + gMinX;

                    for (int col = threadIdx.x; col < Config::SHARED_X; col += BLOCK_X) {
                        sTile[row][col][0] = __ldg(row_ptr1 + col);
                        sTile[row][col][1] = __ldg(row_ptr2 + col);
                    }
                }
            } else {
                const int tileSize = Config::SHARED_Y * Config::SHARED_X;
                const int threads = BLOCK_X * BLOCK_Y;
                const int steps = (tileSize + threads - 1) / threads;

                for (int s = 0; s < steps; ++s) {
                    int tid = s * threads + block.thread_rank();
                    if (tid < tileSize) {
                        int local_y = tid / Config::SHARED_X;
                        int local_x = tid % Config::SHARED_X;
                        int gy = tileStartY + local_y - HALO;
                        int gx = tileStartX + local_x - HALO;

                        float X = get_pix_value(img1, bIdx, c, gy, gx, CH, H, W);
                        float Y = get_pix_value(img2, bIdx, c, gy, gx, CH, H, W);

                        sTile[local_y][local_x][0] = X;
                        sTile[local_y][local_x][1] = Y;
                    }
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
                float w = cGauss[HALO - d];
                float Xleft  = sTile[ly][lx - d][0];
                float Yleft  = sTile[ly][lx - d][1];
                float Xright = sTile[ly][lx + d][0];
                float Yright = sTile[ly][lx + d][1];

                sumX  += (Xleft + Xright) * w;
                sumX2 += ((Xleft * Xleft) + (Xright * Xright)) * w;
                sumY  += (Yleft + Yright) * w;
                sumY2 += ((Yleft * Yleft) + (Yright * Yright)) * w;
                sumXY += ((Xleft * Yleft) + (Xright * Yright)) * w;
            }
            // center
            {
                float centerX = sTile[ly][lx][0];
                float centerY = sTile[ly][lx][1];
                float wc = cGauss[HALO];
                sumX  += centerX * wc;
                sumX2 += (centerX * centerX) * wc;
                sumY  += centerY * wc;
                sumY2 += (centerY * centerY) * wc;
                sumXY += (centerX * centerY) * wc;
            }

            xconv[ly][threadIdx.x][0] = sumX;
            xconv[ly][threadIdx.x][1] = sumX2;
            xconv[ly][threadIdx.x][2] = sumY;
            xconv[ly][threadIdx.x][3] = sumY2;
            xconv[ly][threadIdx.x][4] = sumXY;

            // Handle additional rows if needed
            int ly2 = ly + BLOCK_Y;
            if (ly2 < Config::CONV_Y) {
                sumX = 0.f; sumX2 = 0.f; sumY = 0.f; sumY2 = 0.f; sumXY = 0.f;

                #pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    float w = cGauss[HALO - d];
                    float Xleft  = sTile[ly2][lx - d][0];
                    float Yleft  = sTile[ly2][lx - d][1];
                    float Xright = sTile[ly2][lx + d][0];
                    float Yright = sTile[ly2][lx + d][1];

                    sumX  += (Xleft + Xright) * w;
                    sumX2 += ((Xleft * Xleft) + (Xright * Xright)) * w;
                    sumY  += (Yleft + Yright) * w;
                    sumY2 += ((Yleft * Yleft) + (Yright * Yright)) * w;
                    sumXY += ((Xleft * Yleft) + (Xright * Yright)) * w;
                }
                {
                    float cx = sTile[ly2][lx][0];
                    float cy = sTile[ly2][lx][1];
                    float wc = cGauss[HALO];
                    sumX  += cx * wc;
                    sumX2 += (cx * cx) * wc;
                    sumY  += cy * wc;
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

        // 3) Vertical convolution + final SSIM
        {
            int ly = threadIdx.y + HALO;
            int lx = threadIdx.x;

            float out0 = 0.f, out1 = 0.f, out2 = 0.f, out3 = 0.f, out4 = 0.f;

            #pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                out0 += (xconv[ly - d][lx][0] + xconv[ly + d][lx][0]) * w;
                out1 += (xconv[ly - d][lx][1] + xconv[ly + d][lx][1]) * w;
                out2 += (xconv[ly - d][lx][2] + xconv[ly + d][lx][2]) * w;
                out3 += (xconv[ly - d][lx][3] + xconv[ly + d][lx][3]) * w;
                out4 += (xconv[ly - d][lx][4] + xconv[ly + d][lx][4]) * w;
            }
            {
                float wC = cGauss[HALO];
                out0 += xconv[ly][lx][0] * wC;
                out1 += xconv[ly][lx][1] * wC;
                out2 += xconv[ly][lx][2] * wC;
                out3 += xconv[ly][lx][3] * wC;
                out4 += xconv[ly][lx][4] * wC;
            }

            if (pix_x < W && pix_y < H) {
                float mu1 = out0;
                float mu2 = out2;
                float mu1_sq = mu1 * mu1;
                float mu2_sq = mu2 * mu2;

                float sigma1_sq = out1 - mu1_sq;
                float sigma2_sq = out3 - mu2_sq;
                float sigma12   = out4 - mu1 * mu2;

                float A = mu1_sq + mu2_sq + C1;
                float B = sigma1_sq + sigma2_sq + C2;
                float C_ = 2.f * mu1 * mu2 + C1;
                float D_ = 2.f * sigma12 + C2;

                float val = (C_ * D_) / (A * B);

                int global_idx = bIdx * CH * num_pix + c * num_pix + pix_id;
                ssim_map[global_idx] = val;

                if (dm_dmu1) {
                    float d_m_dmu1 = (
                        (mu2 * 2.f * D_) / (A * B)
                        - (mu2 * 2.f * C_) / (A * B)
                        - (mu1 * 2.f * C_ * D_) / (A * A * B)
                        + (mu1 * 2.f * C_ * D_) / (A * B * B)
                    );
                    float d_m_dsigma1_sq = (-C_ * D_) / (A * B * B);
                    float d_m_dsigma12   = (2.f * C_) / (A * B);

                    dm_dmu1[global_idx]       = d_m_dmu1;
                    dm_dsigma1_sq[global_idx] = d_m_dsigma1_sq;
                    dm_dsigma12[global_idx]   = d_m_dsigma12;
                }
            }
        }
    }
}

// ------------------------------------------
// Backward Kernel: Templated
// ------------------------------------------
template<int HALO>
__global__ void fusedssim_backwardCUDA(
    int H,
    int W,
    int CH,
    float C1,
    float C2,
    const float* __restrict__ img1,
    const float* __restrict__ img2,
    const float* __restrict__ dL_dmap,
    float* __restrict__ dL_dimg1,
    const float* __restrict__ dm_dmu1,
    const float* __restrict__ dm_dsigma1_sq,
    const float* __restrict__ dm_dsigma12
) {
    using Config = SSIMConfig<HALO>;
    const float* cGauss = getGaussCoeffs<HALO>();

    auto block = cg::this_thread_block();

    const int pix_y  = block.group_index().y * BLOCK_Y + block.thread_index().y;
    const int pix_x  = block.group_index().x * BLOCK_X + block.thread_index().x;
    const int pix_id = pix_y * W + pix_x;
    const int num_pix = H * W;
    const int bIdx   = block.group_index().z;

    // Features-outermost layout for sData (better for horizontal pass access pattern)
    __shared__ float sData[3][Config::SHARED_Y][Config::SHARED_X];
    // Features-innermost layout for sScratch (better for vertical pass access pattern)
    __shared__ float sScratch[Config::CONV_Y][Config::CONV_X][3];

    for (int c = 0; c < CH; ++c) {
        float p1 = 0.f, p2 = 0.f;
        if (pix_x < W && pix_y < H) {
            p1 = get_pix_value(img1, bIdx, c, pix_y, pix_x, CH, H, W);
            p2 = get_pix_value(img2, bIdx, c, pix_y, pix_x, CH, H, W);
        }

        // (1) Load + fuse multiplication
        {
            const int start_y = block.group_index().y * BLOCK_Y;
            const int start_x = block.group_index().x * BLOCK_X;

            const int gMinY = start_y - HALO;
            const int gMaxY = start_y + BLOCK_Y + HALO - 1;
            const int gMinX = start_x - HALO;
            const int gMaxX = start_x + BLOCK_X + HALO - 1;
            const bool fullTileInBounds = (gMinY >= 0) && (gMaxY < H) && (gMinX >= 0) && (gMaxX < W);

            const size_t offset = bIdx * CH * H * W + c * H * W;
            const float* dL_base = dL_dmap + offset;
            const float* mu_base = dm_dmu1 + offset;
            const float* s1_base = dm_dsigma1_sq + offset;
            const float* s12_base = dm_dsigma12 + offset;

            if (fullTileInBounds) {
                for (int row = threadIdx.y; row < Config::SHARED_Y; row += BLOCK_Y) {
                    int gy = gMinY + row;
                    const float* dL_row = dL_base + gy * W + gMinX;
                    const float* mu_row = mu_base + gy * W + gMinX;
                    const float* s1_row = s1_base + gy * W + gMinX;
                    const float* s12_row = s12_base + gy * W + gMinX;

                    for (int col = threadIdx.x; col < Config::SHARED_X; col += BLOCK_X) {
                        float chain = __ldg(dL_row + col);
                        sData[0][row][col] = __ldg(mu_row + col) * chain;
                        sData[1][row][col] = __ldg(s1_row + col) * chain;
                        sData[2][row][col] = __ldg(s12_row + col) * chain;
                    }
                }
            } else {
                int tid = threadIdx.y * blockDim.x + threadIdx.x;
                int warp_id = tid / 32;
                int lane_id = tid % 32;
                int totalThreads = BLOCK_X * BLOCK_Y;
                int num_warps = (totalThreads + 31) / 32;

                for (int row = warp_id; row < Config::SHARED_Y; row += num_warps) {
                    int gy = start_y + row - HALO;
                    for (int col = lane_id; col < Config::SHARED_X; col += 32) {
                        int gx = start_x + col - HALO;

                        float chain = get_pix_value(dL_dmap,      bIdx, c, gy, gx, CH, H, W);
                        float vmu   = get_pix_value(dm_dmu1,      bIdx, c, gy, gx, CH, H, W);
                        float vs1   = get_pix_value(dm_dsigma1_sq,bIdx, c, gy, gx, CH, H, W);
                        float vs12  = get_pix_value(dm_dsigma12,  bIdx, c, gy, gx, CH, H, W);

                        sData[0][row][col] = vmu  * chain;
                        sData[1][row][col] = vs1  * chain;
                        sData[2][row][col] = vs12 * chain;
                    }
                }
            }
        }
        block.sync();

        // (2) Horizontal pass
        {
            int ly = threadIdx.y;
            int lx = threadIdx.x + HALO;

            for (int pass = 0; pass < 2; ++pass) {
                int yy = ly + pass * BLOCK_Y;
                if (yy < Config::CONV_Y) {
                    float accum0 = 0.f, accum1 = 0.f, accum2 = 0.f;

                    #pragma unroll
                    for (int d = 1; d <= HALO; ++d) {
                        float w = cGauss[HALO - d];
                        float left0  = sData[0][yy][lx - d];
                        float left1  = sData[1][yy][lx - d];
                        float left2  = sData[2][yy][lx - d];

                        float right0 = sData[0][yy][lx + d];
                        float right1 = sData[1][yy][lx + d];
                        float right2 = sData[2][yy][lx + d];

                        accum0 += (left0 + right0) * w;
                        accum1 += (left1 + right1) * w;
                        accum2 += (left2 + right2) * w;
                    }
                    {
                        float wc = cGauss[HALO];
                        float c0 = sData[0][yy][lx];
                        float c1 = sData[1][yy][lx];
                        float c2 = sData[2][yy][lx];
                        accum0 += c0 * wc;
                        accum1 += c1 * wc;
                        accum2 += c2 * wc;
                    }

                    sScratch[yy][threadIdx.x][0] = accum0;
                    sScratch[yy][threadIdx.x][1] = accum1;
                    sScratch[yy][threadIdx.x][2] = accum2;
                }
            }
        }
        block.sync();

        // (3) Vertical pass -> finalize dL/d(img1)
        if (pix_x < W && pix_y < H) {
            int ly = threadIdx.y + HALO;
            int lx = threadIdx.x;

            float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f;

            #pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                sum0 += (sScratch[ly - d][lx][0] + sScratch[ly + d][lx][0]) * w;
                sum1 += (sScratch[ly - d][lx][1] + sScratch[ly + d][lx][1]) * w;
                sum2 += (sScratch[ly - d][lx][2] + sScratch[ly + d][lx][2]) * w;
            }
            {
                float wc = cGauss[HALO];
                sum0 += sScratch[ly][lx][0] * wc;
                sum1 += sScratch[ly][lx][1] * wc;
                sum2 += sScratch[ly][lx][2] * wc;
            }

            float dL_dpix = sum0 + (2.f * p1) * sum1 + (p2) * sum2;

            int out_idx = bIdx * CH * num_pix + c * num_pix + pix_id;
            dL_dimg1[out_idx] = dL_dpix;
        }
        block.sync();
    }
}

// Note: Explicit template instantiation of __global__ kernels is not needed
// because each kernel launch in the dispatch functions below instantiates the template.

// ------------------------------------------
// Templated PyTorch Interface (Forward)
// ------------------------------------------
template<int HALO>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedssim_impl(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    bool train
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));
    int B  = img1.size(0);
    int CH = img1.size(1);
    int H  = img1.size(2);
    int W  = img1.size(3);

    dim3 grid((W + BLOCK_X - 1) / BLOCK_X,
              (H + BLOCK_Y - 1) / BLOCK_Y,
              B);
    dim3 block(BLOCK_X, BLOCK_Y);

    auto ssim_map = torch::zeros_like(img1, img1.options()).contiguous();

    auto dm_dmu1       = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma1_sq = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma12   = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());

    fusedssimCUDA<HALO><<<grid, block>>>(
        H, W, CH, C1, C2,
        img1.contiguous().data_ptr<float>(),
        img2.contiguous().data_ptr<float>(),
        ssim_map.data_ptr<float>(),
        train ? dm_dmu1.data_ptr<float>()       : nullptr,
        train ? dm_dsigma1_sq.data_ptr<float>() : nullptr,
        train ? dm_dsigma12.data_ptr<float>()   : nullptr
    );

    return std::make_tuple(ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
}

// ------------------------------------------
// Templated PyTorch Interface (Backward)
// ------------------------------------------
template<int HALO>
torch::Tensor
fusedssim_backward_impl(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::Tensor &dL_dmap,
    torch::Tensor &dm_dmu1,
    torch::Tensor &dm_dsigma1_sq,
    torch::Tensor &dm_dsigma12
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));
    int B  = img1.size(0);
    int CH = img1.size(1);
    int H  = img1.size(2);
    int W  = img1.size(3);

    auto dL_dimg1 = torch::zeros_like(img1);

    dim3 grid((W + BLOCK_X - 1) / BLOCK_X,
              (H + BLOCK_Y - 1) / BLOCK_Y,
              B);
    dim3 block(BLOCK_X, BLOCK_Y);

    fusedssim_backwardCUDA<HALO><<<grid, block>>>(
        H, W, CH, C1, C2,
        img1.contiguous().data_ptr<float>(),
        img2.contiguous().data_ptr<float>(),
        dL_dmap.contiguous().data_ptr<float>(),
        dL_dimg1.data_ptr<float>(),
        dm_dmu1.contiguous().data_ptr<float>(),
        dm_dsigma1_sq.contiguous().data_ptr<float>(),
        dm_dsigma12.contiguous().data_ptr<float>()
    );

    return dL_dimg1;
}

// ------------------------------------------
// Dispatch functions (select kernel by window_size)
// ------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedssim(
    int window_size,
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    bool train
) {
    switch (window_size) {
        case 7:  return fusedssim_impl<3>(C1, C2, img1, img2, train);
        case 9:  return fusedssim_impl<4>(C1, C2, img1, img2, train);
        case 11: return fusedssim_impl<5>(C1, C2, img1, img2, train);
        default:
            throw std::invalid_argument("window_size must be 7, 9, or 11, got " + std::to_string(window_size));
    }
}

torch::Tensor
fusedssim_backward(
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
        case 7:  return fusedssim_backward_impl<3>(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
        case 9:  return fusedssim_backward_impl<4>(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
        case 11: return fusedssim_backward_impl<5>(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
        default:
            throw std::invalid_argument("window_size must be 7, 9, or 11, got " + std::to_string(window_size));
    }
}
