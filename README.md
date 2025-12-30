# Optimized Fully Fused Differentiable SSIM

> **Based on [MrNeRF/optimized-fused-ssim](https://github.com/MrNeRF/optimized-fused-ssim)** by Janusch Patas, which optimizes [rahul-goel/fused-ssim](https://github.com/rahul-goel/fused-ssim).

A fast CUDA implementation of differentiable SSIM for PyTorch with **FP16/AMP support**, **configurable window sizes** (7, 9, 11), and **pytorch-msssim compatibility**.

Drop-in replacement for [fused-ssim](https://github.com/rahul-goel/fused-ssim) and [pytorch-msssim](https://github.com/VainF/pytorch-msssim).

## Performance (RTX 4090)

**~7x faster than pytorch-msssim** with full forward+backward support.

| Implementation | Forward | Backward | Total | vs Reference |
|----------------|---------|----------|-------|--------------|
| Reference (F.conv2d) | 33.6 ms | 30.6 ms | 64.2 ms | 1.0x |
| pytorch_msssim | 28.7 ms | 28.9 ms | 57.5 ms | 1.1x |
| **fused-ssim** | **4.38 ms** | **4.66 ms** | **9.04 ms** | **7.1x** |

*Benchmark: 5×5×1080×1920, 100 iterations with 20 warmup iterations.*

## Installation

**Prerequisites:** NVIDIA GPU, Python 3.10+, PyTorch 2.5+ with CUDA

```bash
pip install fussim
```

<details>
<summary><b>Build from source</b></summary>

```bash
git clone https://github.com/OpsiClear/fussim.git
cd fussim
pip install .  # or pip install -e . for development
```

**GPU architecture override:**
```bash
TORCH_CUDA_ARCH_LIST="8.9" pip install .  # RTX 4090
TORCH_CUDA_ARCH_LIST="8.6" pip install .  # RTX 3090
```
</details>

## Usage

```python
import torch
from fussim import fused_ssim

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
from fussim import fused_ssim

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
from fussim import ssim, SSIM
```

**Function API:**
```python
ssim(X, Y, data_range=255, size_average=True, win_size=11, K=(0.01, 0.03), nonnegative_ssim=False)
```

**Module API:**
```python
ssim_module = SSIM(data_range=1.0, size_average=True, win_size=11, K=(0.01, 0.03))
loss = 1 - ssim_module(pred, target)
loss.backward()
```

Both use standard SSIM parameters (σ=1.5 fixed). Only `X` receives gradients.

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

<details>
<summary><h2>CUDA Optimizations</h2></summary>

| Optimization | Est. Contribution |
|-------------|-------------------|
| Fast tile path (`fullTileInBounds`) | ~30-40% |
| Texture cache (`__ldg`) | ~10-15% |
| Interleaved shared memory | ~10% |
| Fused computations | ~10% |
| FP16 kernels | ~20-40% additional |

**Key optimizations:**
- **Fast tile path**: Skips boundary checks for interior tiles (~95% of image)
- **Texture cache**: Uses `__ldg()` for read-only global memory
- **Fused computation**: All statistics (μ₁, μ₂, σ₁², σ₂², σ₁₂) in single pass
- **Gaussian symmetry**: Halves multiplications by pairing symmetric elements
- **16×16 blocks**: Better GPU occupancy than 32×32
</details>

## Attribution

This project is built upon the work of:

| Project | Author | Contribution |
|---------|--------|--------------|
| [optimized-fused-ssim](https://github.com/MrNeRF/optimized-fused-ssim) | [Janusch Patas](https://github.com/MrNeRF) | CUDA optimizations, FP16 support, pytorch-msssim API |
| [fused-ssim](https://github.com/rahul-goel/fused-ssim) | [Rahul Goel](https://github.com/rahul-goel) | Original fused CUDA implementation |

Special thanks to [Florian Hahlbohm](https://github.com/fhahlbohm) for helping verify correctness.

## Citation

```bibtex
@software{fussim,
    author = {Janusch Patas},
    title = {Optimized Fused-SSIM},
    year = {2025},
    url = {https://github.com/MrNeRF/optimized-fused-ssim},
}
@inproceedings{taming3dgs,
    author = {Mallick, Saswat Subhajyoti and Goel, Rahul and Kerbl, Bernhard and Steinberger, Markus and Carrasco, Francisco Vicente and De La Torre, Fernando},
    title = {Taming 3DGS: High-Quality Radiance Fields with Limited Resources},
    year = {2024},
    doi = {10.1145/3680528.3687694},
    booktitle = {SIGGRAPH Asia 2024 Conference Papers},
}
```
