# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2024-12-30

First release of `fussim`, based on [MrNeRF/optimized-fused-ssim](https://github.com/MrNeRF/optimized-fused-ssim).

### Changes from [MrNeRF/optimized-fused-ssim](https://github.com/MrNeRF/optimized-fused-ssim)

- **Renamed package** from `fused-ssim` to `fussim`
- **Renamed module** from `fused_ssim` to `fussim`
- **Renamed CUDA extension** from `fused_ssim_cuda` to `fussim_cuda`
- **Simplified installation**: `pip install fussim`
- **Cleaner README** with concise documentation

### Features inherited from optimized-fused-ssim

- **~7x faster than pytorch-msssim** on RTX 4090
- **FP16/AMP support** for 1.2-1.4x additional speedup
- **Configurable window sizes**: 7, 9, or 11 (original fused-ssim only supports 11)
- **pytorch-msssim compatible API**: `ssim()` function and `SSIM` module
- **Cross-platform**: Linux and Windows support
- **Pre-built wheels** for common PyTorch/CUDA combinations
- **Blackwell GPU support**: sm_100/sm_120 for B100/B200, RTX 50xx

### CUDA Optimizations (from optimized-fused-ssim)

- Fast tile path with `fullTileInBounds` (~30-40% speedup)
- Texture cache reads via `__ldg()` (~10-15% speedup)
- Interleaved shared memory layout for L1 cache locality
- Fused single-pass computation of all SSIM statistics
- Gaussian symmetry exploitation (halves multiplications)
- 16×16 thread blocks for better GPU occupancy

### Original Attribution

This project is based on:
- [MrNeRF/optimized-fused-ssim](https://github.com/MrNeRF/optimized-fused-ssim) by Janusch Patas
- [rahul-goel/fused-ssim](https://github.com/rahul-goel/fused-ssim) by Rahul Goel (original implementation)
