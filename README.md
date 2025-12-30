# Optimized Fully Fused Differentiable SSIM

An enhanced fork of [fused-ssim](https://github.com/rahul-goel/fused-ssim) with additional features: **FP16/AMP support**, **configurable window sizes** (7, 9, 11), **pre-built wheels**, and **pytorch-msssim compatibility**.

This is a **drop-in replacement** for both [fused-ssim](https://github.com/rahul-goel/fused-ssim) and [pytorch-msssim](https://github.com/VainF/pytorch-msssim).

## Performance (RTX 4090)

**~7x faster than pytorch-msssim** with full forward+backward support.

| Implementation | Forward | Backward | Total | vs Reference |
|----------------|---------|----------|-------|--------------|
| Reference (F.conv2d) | 33.6 ms | 30.6 ms | 64.2 ms | 1.0x |
| pytorch_msssim | 28.7 ms | 28.9 ms | 57.5 ms | 1.1x |
| **fused-ssim** | **4.38 ms** | **4.66 ms** | **9.04 ms** | **7.1x** |

*Benchmark: 5×5×1080×1920, 100 iterations with 20 warmup iterations.*

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support
- Python 3.10+
- PyTorch 2.5+ with CUDA support

### Quick Install (One-Liner)

**Pre-built wheel (no compilation needed):**
```bash
pip install fused-ssim --extra-index-url https://mrnerf.github.io/optimized-fused-ssim/whl/
```

**From PyPI (requires CUDA Toolkit for compilation):**
```bash
pip install fused-ssim
```

**From GitHub (requires CUDA Toolkit):**
```bash
pip install git+https://github.com/MrNeRF/optimized-fused-ssim.git
```

### Pre-built Wheels

Pre-built wheels are available for common PyTorch and CUDA combinations:

| PyTorch | CUDA 11.8 | CUDA 12.1 | CUDA 12.4 | CUDA 12.6 | CUDA 12.8 |
|---------|-----------|-----------|-----------|-----------|-----------|
| 2.9.0   | -         | -         | -         | pt29cu126 | pt29cu128 |
| 2.8.0   | -         | -         | -         | pt28cu126 | pt28cu128 |
| 2.7.1   | pt27cu118 | -         | -         | pt27cu126 | pt27cu128 |
| 2.6.0   | pt26cu118 | -         | pt26cu124 | pt26cu126 | -         |
| 2.5.1   | pt25cu118 | pt25cu121 | pt25cu124 | -         | -         |

To install a specific variant:
```bash
# Example: PyTorch 2.9 + CUDA 12.8
pip install fused-ssim --extra-index-url https://mrnerf.github.io/optimized-fused-ssim/whl/pt29cu128/
```

### Automatic Installation (Recommended)

Use the installation helper script that auto-detects your environment:

```bash
# Download and run the installer
curl -sSL https://raw.githubusercontent.com/MrNeRF/optimized-fused-ssim/main/scripts/install.py | python

# Or clone and run locally
git clone https://github.com/MrNeRF/optimized-fused-ssim.git
cd optimized-fused-ssim
python scripts/install.py
```

Options:
```bash
python scripts/install.py --check   # Check environment without installing
python scripts/install.py --source  # Force build from source
```

### Install from PyPI (Source Build)

```bash
pip install fused-ssim
```

This requires CUDA Toolkit to be installed for compilation.

### Install from Source

```bash
git clone https://github.com/MrNeRF/optimized-fused-ssim.git
cd optimized-fused-ssim
pip install .
```

For development:
```bash
pip install -e .
```

### Verify Installation

After installation, verify it works:

```bash
# CLI check
fused-ssim-check

# Or in Python
python -c "from fused_ssim import get_build_info; print(get_build_info())"
```

### GPU Architecture

The build system automatically detects your GPU architecture. If auto-detection fails or you want to target specific architectures, use the `TORCH_CUDA_ARCH_LIST` environment variable:

```bash
# For RTX 4090 (Ada Lovelace)
TORCH_CUDA_ARCH_LIST="8.9" pip install .

# For RTX 3090 (Ampere)
TORCH_CUDA_ARCH_LIST="8.6" pip install .

# For multiple architectures
TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9" pip install .

# With PTX for forward compatibility
TORCH_CUDA_ARCH_LIST="8.0+PTX" pip install .
```

#### Common GPU Architectures

| GPU Series | Architecture | Compute Capability | Min CUDA Version |
|------------|--------------|-------------------|------------------|
| RTX 50xx (5090, 5080, etc.) | Blackwell | 12.0 | CUDA 12.9+ |
| B100, B200 | Blackwell | 10.0 | CUDA 12.8+ |
| RTX 40xx (4090, 4080, etc.) | Ada Lovelace | 8.9 | CUDA 11.8+ |
| H100, H200 | Hopper | 9.0 | CUDA 12.0+ |
| RTX 30xx (3090, 3080, etc.) | Ampere | 8.6 | CUDA 11.1+ |
| A100, A30 | Ampere | 8.0 | CUDA 11.0+ |
| RTX 20xx, Quadro RTX | Turing | 7.5 | CUDA 10.0+ |
| V100 | Volta | 7.0 | CUDA 9.0+ |

### Platform-Specific Notes

#### Linux

Standard installation should work. Ensure CUDA Toolkit is installed and `nvcc` is in your PATH:

```bash
nvcc --version  # Should show CUDA version
```

#### Windows

1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) with C++ development tools
2. Install CUDA Toolkit
3. Open "x64 Native Tools Command Prompt" or ensure MSVC is in PATH
4. Install the package:

```cmd
pip install .
```

#### Troubleshooting

**Check your installation:**
```bash
fused-ssim-check
# Or: python -c "from fused_ssim import get_build_info, check_compatibility; print(get_build_info()); check_compatibility()"
```

**"CUDA not found" error:**
```bash
# Set CUDA path explicitly
export CUDA_HOME=/usr/local/cuda  # Linux
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1  # Windows
```

**"No kernel image available" error:**
The pre-built wheel doesn't include your GPU architecture. Options:
```bash
# Option 1: Build from source with your GPU arch
TORCH_CUDA_ARCH_LIST="8.6" pip install --force-reinstall --no-cache-dir --no-binary fused-ssim fused-ssim

# Option 2: Use the installer script
python scripts/install.py --source
```

**"undefined symbol" or ABI mismatch error:**
The installed wheel was built for a different PyTorch version. Fix:
```bash
# Check your environment
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

# Reinstall matching wheel (example for PyTorch 2.9 + CUDA 12.8)
pip install fused-ssim --force-reinstall --extra-index-url https://mrnerf.github.io/optimized-fused-ssim/whl/pt29cu128/

# Or build from source
pip install --force-reinstall --no-cache-dir --no-binary fused-ssim fused-ssim
```

**Version mismatch warning at runtime:**
If you see warnings about PyTorch/CUDA version mismatch, the package may still work but could have issues. Reinstall with the correct wheel or build from source.

**PyTorch updated after installing fused-ssim:**
```bash
pip install --force-reinstall fused-ssim
```

## Usage

```python
import torch
from fused_ssim import fused_ssim

# Create sample images (must be on CUDA)
img1 = torch.rand(1, 3, 256, 256, device="cuda", requires_grad=True)
img2 = torch.rand(1, 3, 256, 256, device="cuda")

# Compute SSIM (training mode - supports backpropagation)
ssim_value = fused_ssim(img1, img2)

# Compute SSIM (inference mode - faster, less memory)
with torch.no_grad():
    ssim_value = fused_ssim(img1, img2)

# Use as a loss function
loss = 1.0 - fused_ssim(pred_img, target_img)
loss.backward()
```

### API Reference

```python
fused_ssim(img1, img2, padding="same", train=True, window_size=11)
```

**Parameters:**
- `img1`: First image tensor (B, C, H, W). **Gradients computed for this tensor.**
- `img2`: Second image tensor (B, C, H, W).
- `padding`: `"same"` (default) or `"valid"`.
- `train`: If `True` (default), compute gradients. If `False`, inference-only mode (faster).
- `window_size`: Gaussian window size. Must be `7`, `9`, or `11` (default).

**Returns:** Scalar mean SSIM value.

**Important:** Only `img1` receives gradients. Always pass your prediction as `img1`:
```python
loss = 1.0 - fused_ssim(pred, target)  # Correct: pred gets gradients
loss = 1.0 - fused_ssim(target, pred)  # Wrong: pred gets no gradients!
```

### FP16 / AMP Support

For mixed precision training with `torch.autocast`:

```python
import torch
from fused_ssim import fused_ssim

# Automatic Mixed Precision (recommended)
with torch.autocast(device_type="cuda"):
    ssim_value = fused_ssim(img1, img2)  # Automatically uses FP16 kernel
    loss = 1.0 - ssim_value

# With GradScaler for training
scaler = torch.amp.GradScaler()
with torch.autocast(device_type="cuda"):
    loss = 1.0 - fused_ssim(pred, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Drop-in Replacement for pytorch-msssim

This package provides a **drop-in replacement** for [pytorch-msssim](https://github.com/VainF/pytorch-msssim), the most widely used SSIM library in CV research. Just change your import:

```python
# Before (pytorch-msssim)
from pytorch_msssim import ssim, SSIM

# After (fused-ssim) - 6-7x faster!
from fused_ssim import ssim, SSIM
```

**Function API:**
```python
ssim(X, Y, data_range=255, size_average=True, nonnegative_ssim=False)
```

**Module API:**
```python
ssim_module = SSIM(data_range=1.0, size_average=True, channel=3)
loss = 1 - ssim_module(pred, target)
loss.backward()
```

Both APIs produce identical results to pytorch-msssim. The CUDA kernel uses the standard SSIM parameters (11×11 Gaussian, σ=1.5, K₁=0.01, K₂=0.03).

## API Compatibility

This package is designed as a **fully compatible drop-in replacement** for both the original [fused-ssim](https://github.com/rahul-goel/fused-ssim) and [pytorch-msssim](https://github.com/VainF/pytorch-msssim).

### Compatibility with Original fused-ssim

| Feature | Status | Notes |
|---------|--------|-------|
| Function signature | Identical | `fused_ssim(img1, img2, padding, train, window_size)` |
| Default parameters | Identical | `padding="same"`, `train=True`, `window_size=11` |
| Return value | Identical | Scalar mean SSIM |
| Gradient behavior | Identical | Only `img1` receives gradients |
| Numerical accuracy | Verified | Values and gradients match reference implementations |

**Additional features** (not in original):
- Window sizes 7 and 9 (original only supports 11)
- FP16/AMP support for faster training
- Better error messages and input validation

### Compatibility with pytorch-msssim

| Feature | Status | Notes |
|---------|--------|-------|
| `ssim()` function | Compatible | Same signature and behavior |
| `SSIM` module | Compatible | Same constructor args and forward() |
| `data_range` | Supported | Any positive value |
| `size_average` | Supported | True (scalar) or False (per-batch) |
| `K` constants | Supported | Configurable (K₁, K₂) |
| `nonnegative_ssim` | Supported | Clamps negative values to 0 |
| Numerical accuracy | Verified | Matches within `rtol=1e-4`, `atol=1e-4` |

### Limitations

The optimized CUDA kernels have some constraints:

| Parameter | Constraint | Reason |
|-----------|------------|--------|
| `win_size` | Must be 7, 9, or 11 | CUDA kernel template instantiation |
| `win_sigma` | Must be 1.5 | Hardcoded in CUDA constant memory |
| `win` (custom window) | Not supported | Kernel uses built-in Gaussian |
| `spatial_dims` | Must be 2 (images only) | 2D convolution kernels |

These limitations cover 99%+ of use cases since the standard SSIM uses an 11×11 Gaussian window with σ=1.5.

**Error handling:** Unsupported parameters raise `ValueError` with clear messages explaining the constraint.

### Enhancements over Original fused-ssim

| Category | Feature | Description |
|----------|---------|-------------|
| **New Features** | FP16/AMP support | 1.2-1.4x faster training with `torch.autocast()` |
| | Window sizes 7, 9, 11 | Original only supports 11 |
| | pytorch-msssim API | Drop-in `ssim()` and `SSIM` module |
| **Build System** | Pre-built wheels | No CUDA Toolkit needed for common setups |
| | Auto GPU detection | Detects architecture at build time |
| | Windows/Blackwell | MSVC support, sm_100/sm_120 |
| **Developer UX** | `fused-ssim-check` CLI | Installation verification |
| | `get_build_info()` | Query build configuration |
| | Input validation | Helpful error messages |

## CUDA Optimizations

### Summary

| Optimization | Category | Est. Contribution |
|-------------|----------|-------------------|
| **Fast tile path (`fullTileInBounds`)** | Control flow | **~30-40%** (primary) |
| **Texture cache (`__ldg`)** | Memory | **~10-15%** |
| Interleaved shared memory layout | Memory | ~10% |
| Fused computations | Algorithmic | ~10% |
| Gaussian symmetry | Compute | ~5% |
| Constant memory | Memory | ~5% |
| 16x16 block size | Occupancy | ~5% |
| Single convolution pass | Compute | ~5% |
| Padded shared memory | Memory | ~1% |
| FP16 kernels | Precision | ~20-40% additional |

### Details

#### 1. Fast Tile Path (Primary Optimization)
For interior tiles (most of the image), we skip boundary checking entirely:
```cpp
const bool fullTileInBounds = (gMinY >= 0) && (gMaxY < H) && (gMinX >= 0) && (gMaxX < W);
if (fullTileInBounds) {
    // Fast path: direct __ldg() reads, no boundary checks
    sTile[row][col][0] = __ldg(row_ptr1 + col);
    sTile[row][col][1] = __ldg(row_ptr2 + col);
} else {
    // Slow path: boundary-checked get_pix_value() for edge tiles
}
```
This eliminates branch overhead and enables `__ldg()` texture cache reads for ~95% of tiles.

#### 2. Fused Computations
The original implementation computes multiple statistics (mean, variance, covariance) in separate passes. This implementation fuses all computations into a **single pass**, reducing redundant memory accesses.

#### 3. Gaussian Symmetry Exploitation
The Gaussian filter is symmetric. By pairing symmetric elements, we **halve the multiplications** from 11 to 6 per pixel.

#### 4. Constant Memory for Coefficients
Gaussian coefficients are stored in **CUDA constant memory** (`__constant__ float cGauss[11]`), reducing register pressure and enabling broadcast reads.

#### 5. Optimized Block Size
Uses **16x16** thread blocks (vs 32x32), improving GPU occupancy by reducing per-block resource usage.

#### 6. Efficient Shared Memory Layout
- **Interleaved layout** `sTile[y][x][2]` keeps img1/img2 for same pixel adjacent in memory, improving L1 cache locality
- **Features-innermost layout** `xconv[y][x][5]` keeps all 5 statistics for same pixel in one cache line
- Padded dimensions (`SHARED_X_PAD`, `CONV_X_PAD`) to reduce bank conflicts

#### 7. Single Convolution Pass
All statistics (μ₁, μ₂, σ₁², σ₂², σ₁₂) computed in one separable convolution pass.

#### 8. Texture Cache Reads
Uses `__ldg()` intrinsic for read-only global memory in the fast tile path, routing through texture cache. Provides **up to 15% speedup** on 1080p+ images.

#### 9. FP16 Kernels with AMP
Dedicated half-precision kernels with proper PyTorch AMP integration via `@torch.amp.custom_fwd/bwd` decorators.

## Acknowledgments

Special thanks to [Florian Hahlbohm](https://github.com/fhahlbohm) for helping verify that optimizations don't break correctness.

## Radiance Fields Dev Discord
[![](https://dcbadge.limes.pink/api/server/https://discord.gg/TbxJST2BbC)](https://discord.gg/TbxJST2BbC)

## Citation

If you use this optimized fused-SSIM implementation for your research, please cite both the original paper and this implementation:

```bibtex
@inproceedings{optimized-fused-ssim,
    author = {Janusch Patas},
    title = {Optimized Fused-SSIM},
    year = {2025},
    url = {https://github.com/MrNeRF/optimized-fused-ssim},
}
@inproceedings{taming3dgs,
    author = {Mallick, Saswat Subhajyoti and Goel, Rahul and Kerbl, Bernhard and Steinberger, Markus and Carrasco, Francisco Vicente and De La Torre, Fernando},
    title = {Taming 3DGS: High-Quality Radiance Fields with Limited Resources},
    year = {2024},
    url = {https://doi.org/10.1145/3680528.3687694},
    doi = {10.1145/3680528.3687694},
    booktitle = {SIGGRAPH Asia 2024 Conference Papers},
    series = {SA '24}
}
```
