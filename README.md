# fussim

[![PyPI](https://img.shields.io/pypi/v/fussim)](https://pypi.org/project/fussim/)
[![Downloads](https://img.shields.io/pypi/dm/fussim)](https://pypistats.org/packages/fussim)
[![Python](https://img.shields.io/pypi/pyversions/fussim)](https://pypi.org/project/fussim/)
[![License](https://img.shields.io/pypi/l/fussim)](LICENSE)

**The fastest SSIM for PyTorch.** Pre-built wheels, zero compilation required.

```bash
pip install fussim
```

**Requirements:** Python 3.10+, PyTorch 2.6+, NVIDIA GPU (Turing or newer)

|  | fussim | pytorch-msssim | fused-ssim |
|:-|:------:|:--------------:|:----------:|
| `pip install` | Yes | Yes | Needs compiler |
| CUDA kernels | Yes | No | Yes |
| Native FP16 | Yes | No | No |
| Speed (vs msssim) | **6.4x** | 1x | ~5x |

> Used in 3D Gaussian Splatting training. Based on [Taming3DGS](https://github.com/MrNeRF/optimized-fused-ssim).

---

## Quick Start

```python
import torch
from fussim import fused_ssim

# Images must be on CUDA, shape (B, C, H, W), range [0, 1]
img1 = torch.rand(1, 3, 256, 256, device="cuda", requires_grad=True)
img2 = torch.rand(1, 3, 256, 256, device="cuda")

# Compute SSIM (returns scalar mean)
ssim_value = fused_ssim(img1, img2)

# Use as loss (only img1 receives gradients)
loss = 1.0 - ssim_value
loss.backward()
```

**FP16 / Mixed Precision:**

```python
with torch.autocast(device_type="cuda"):
    ssim_value = fused_ssim(img1, img2)  # Native FP16 CUDA kernel
```

**Drop-in replacement for pytorch-msssim:**

```python
# Before
from pytorch_msssim import ssim, SSIM
# After (no other code changes needed)
from fussim import ssim, SSIM
```

---

## Installation

### Recommended: Fat Wheel (auto-detection)

```bash
pip install fussim
```

This installs a **single wheel containing all CUDA variants** (~10MB). At runtime, fussim automatically detects your PyTorch's CUDA version and loads the correct extension.

| Platform | Python | PyTorch | CUDA (auto-detected) |
|:---------|:------:|:-------:|:---------------------|
| Linux    | 3.10-3.13 | 2.6 - 2.9 | 11.8, 12.4, 12.6, 12.8 |
| Windows  | 3.10-3.13 | 2.6 - 2.9 | 11.8, 12.4, 12.6, 12.8 |

> **PyTorch 2.5 or older?** The fat wheel requires PyTorch 2.6+. For older versions, use [version-specific wheels](#alternative-version-specific-wheels) or [build from source](#build-from-source).

**No manual version selection needed.** Just install and use.

<details>
<summary><b>Fat wheel compatibility matrix</b></summary>

The fat wheel contains extensions built with these PyTorch versions:

| CUDA | Built with PyTorch | Compatible with |
|:-----|:-------------------|:----------------|
| 11.8 | 2.7.1 | 2.7+ |
| 12.4 | 2.6.0 | 2.6+ |
| 12.6 | 2.8.0 (Win) / 2.9.1 (Linux) | 2.8+ (or 2.6+ via cu124 fallback) |
| 12.8 | 2.8.0 (Win) / 2.9.1 (Linux) | 2.8+ |

PyTorch maintains forward ABI compatibility, so extensions built with older versions work with newer PyTorch. If your exact CUDA version isn't compatible, fussim automatically falls back to a compatible variant.

</details>

---

### Alternative: Version-Specific Wheels

For exact PyTorch ABI matching or smaller downloads (~2MB each), you can install wheels built for specific PyTorch versions.

> **Important:** You must specify the exact variant. Pip cannot auto-select the PyTorch/CUDA combination.

**Step 1:** Find your PyTorch and CUDA versions:
```python
import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
# Example output: PyTorch: 2.9.1, CUDA: 12.8
```

**Step 2:** Install the matching wheel:
```bash
# Format: fussim==VERSION+ptXXcuYYY
# pt29 = PyTorch 2.9, cu128 = CUDA 12.8

pip install "fussim==0.3.14+pt29cu128" --extra-index-url https://opsiclear.github.io/fussim/whl/
```

<details>
<summary><b>Available combinations</b></summary>

| PyTorch | Version Tag | CUDA 11.8 | CUDA 12.1 | CUDA 12.4 | CUDA 12.6 | CUDA 12.8 |
|:-------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| 2.5.1   | `pt25`      | `cu118` | `cu121` | `cu124` | - | - |
| 2.6.0   | `pt26`      | `cu118` | - | `cu124` | `cu126` | - |
| 2.7.1   | `pt27`      | `cu118` | - | - | `cu126` | `cu128` |
| 2.8.0   | `pt28`      | - | - | - | `cu126` | `cu128` |
| 2.9.1   | `pt29`      | - | - | - | `cu126`* | `cu128`* |

*Linux only. Windows has a [known PyTorch bug](https://github.com/pytorch/pytorch/issues/141026).

**Examples:**
```bash
pip install "fussim==0.3.14+pt27cu118" --extra-index-url https://opsiclear.github.io/fussim/whl/
pip install "fussim==0.3.14+pt28cu126" --extra-index-url https://opsiclear.github.io/fussim/whl/
pip install "fussim==0.3.14+pt29cu128" --extra-index-url https://opsiclear.github.io/fussim/whl/
```

**[Interactive Configurator](https://opsiclear.github.io/fussim/)** - generates the exact command for your setup.

</details>

---

### Build from Source

<details>
<summary>Requires CUDA Toolkit and C++ compiler</summary>

```bash
git clone https://github.com/OpsiClear/fussim.git && cd fussim
pip install .

# For specific GPU architecture:
TORCH_CUDA_ARCH_LIST="8.9" pip install .  # RTX 4090
```

</details>

---

## GPU Support

| Architecture | GPUs | Compute Capability |
|:-------------|:-----|:-------------------|
| Turing | RTX 20xx, GTX 16xx | 7.5 |
| Ampere | RTX 30xx, A100 | 8.0, 8.6 |
| Ada Lovelace | RTX 40xx | 8.9 |
| Hopper | H100, H200 | 9.0 |
| Blackwell | RTX 50xx, B100/B200 | 10.0, 12.0 |

---

## API Reference

### `fused_ssim`

```python
fused_ssim(img1, img2, padding="same", train=True, window_size=11) -> Tensor
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `img1` | Tensor | - | First image `(B, C, H, W)`. **Receives gradients.** |
| `img2` | Tensor | - | Second image `(B, C, H, W)` |
| `padding` | str | `"same"` | `"same"` (output = input size) or `"valid"` (cropped) |
| `train` | bool | `True` | Enable gradient computation |
| `window_size` | int | `11` | Gaussian window: `7`, `9`, or `11` |

**Returns:** Scalar mean SSIM value (range: -1 to 1, typically 0 to 1).

> **Note:** Only `img1` receives gradients. For training, pass your prediction as `img1`:
> ```python
> loss = 1 - fused_ssim(prediction, target)  # Correct
> ```

### `ssim` (pytorch-msssim compatible)

```python
ssim(X, Y, data_range=255, size_average=True, win_size=11, K=(0.01, 0.03), nonnegative_ssim=False) -> Tensor
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `X`, `Y` | Tensor | - | Images `(B, C, H, W)`. Gradients computed for `X`. |
| `data_range` | float | `255` | Value range (`255` for uint8, `1.0` for normalized) |
| `size_average` | bool | `True` | Return scalar mean or per-batch `(B,)` values |
| `win_size` | int | `11` | Gaussian window: `7`, `9`, or `11` |
| `K` | tuple | `(0.01, 0.03)` | SSIM constants (K1, K2) |
| `nonnegative_ssim` | bool | `False` | Clamp negative values to 0 |

### `SSIM` Module

```python
from fussim import SSIM

module = SSIM(data_range=1.0)
ssim_val = module(pred, target)
loss = 1 - ssim_val
loss.backward()
```

### Utility Functions

```python
from fussim import get_build_info, check_compatibility

# Check installation details
info = get_build_info()
print(info)  # {'version': '0.3.14', 'runtime_torch_version': '2.9.1', ...}

# Verify compatibility
compatible, issues = check_compatibility()
```

---

## Performance

RTX 4090, batch 5x5x1080x1920, 100 iterations:

| Implementation | Forward | Backward | Total | Speedup |
|:---------------|--------:|---------:|------:|--------:|
| pytorch-msssim | 28.7 ms | 28.9 ms | 57.5 ms | 1.0x |
| **fussim** | **4.38 ms** | **4.66 ms** | **9.04 ms** | **6.4x** |

**Memory:** Fused kernels avoid intermediate allocations, reducing VRAM usage compared to unfused implementations.

---

## Troubleshooting

<details>
<summary><b>ImportError: No compatible fussim CUDA extension found</b></summary>

This usually means your PyTorch version is too old for the fat wheel.

**Check your versions:**
```python
import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
```

**Solutions:**

| PyTorch Version | Solution |
|:----------------|:---------|
| 2.6 - 2.9 | Should work. Run `pip install --upgrade fussim` |
| 2.5 or older | Use [version-specific wheel](#alternative-version-specific-wheels) or upgrade PyTorch |

</details>

<details>
<summary><b>DLL load failed / undefined symbol</b></summary>

This is a PyTorch ABI mismatch. The extension was built with a different PyTorch version.

**Fix:** Install a version-specific wheel that matches your exact PyTorch version:

```bash
# Check your version first
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# Install matching wheel (example for PyTorch 2.7.1 + CUDA 11.8)
pip install "fussim==0.3.14+pt27cu118" --extra-index-url https://opsiclear.github.io/fussim/whl/
```

</details>

<details>
<summary><b>CUDA extension not loading</b></summary>

Check your installation:

```bash
python -c "import fussim; print(fussim.get_build_info())"
```

Or use the compatibility check:

```python
from fussim import check_compatibility
compatible, issues = check_compatibility()
print(f"Compatible: {compatible}")
print(f"Issues: {issues}")
```

</details>

<details>
<summary><b>Wrong CUDA version detected</b></summary>

The fat wheel auto-detects from `torch.version.cuda`. If a fallback warning appears:

```python
import torch
print(torch.version.cuda)  # Check PyTorch's CUDA version
```

Install a version-specific wheel for exact matching.

</details>

<details>
<summary><b>Windows build errors with PyTorch 2.9.x</b></summary>

PyTorch 2.9.x has a [Windows compilation bug](https://github.com/pytorch/pytorch/issues/141026) that prevents building extensions from source.

**Note:** Pre-built wheels work fine on Windows with PyTorch 2.9.x. This only affects building from source.

</details>

---

## Limitations

| Constraint | Reason |
|:-----------|:-------|
| PyTorch 2.6+ (fat wheel) | ABI compatibility; use version-specific wheels for 2.5 |
| NVIDIA GPU required | No CPU fallback |
| `window_size`: 7, 9, or 11 only | CUDA kernel templates |
| `win_sigma`: 1.5 (fixed) | Hardcoded in optimized kernel |
| Custom `win` not supported | Uses built-in Gaussian |
| No MS-SSIM | Single-scale SSIM only |

---

## Attribution

- [optimized-fused-ssim](https://github.com/MrNeRF/optimized-fused-ssim) by Janusch Patas (Taming3DGS)
- [fused-ssim](https://github.com/rahul-goel/fused-ssim) by Rahul Goel

## Citation

```bibtex
@software{optimized-fused-ssim,
    author = {Janusch Patas},
    title = {Optimized Fused-SSIM},
    year = {2025},
    url = {https://github.com/MrNeRF/optimized-fused-ssim},
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
