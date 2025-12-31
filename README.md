# fussim

[![PyPI](https://img.shields.io/pypi/v/fussim)](https://pypi.org/project/fussim/)
[![Python](https://img.shields.io/pypi/pyversions/fussim)](https://pypi.org/project/fussim/)

Fast CUDA SSIM for PyTorch — **`pip install fussim`** with pre-built wheels.

> ~7x faster than pytorch-msssim | FP16/AMP support | Drop-in replacement

Based on [MrNeRF/optimized-fused-ssim](https://github.com/MrNeRF/optimized-fused-ssim).

---

## Installation

```bash
pip install fussim
```

Works out of the box for **Linux** (PyTorch 2.9 + CUDA 12.8) and **Windows** (PyTorch 2.8 + CUDA 12.8).

<details>
<summary><b>Other PyTorch/CUDA versions</b></summary>

Pre-built wheels for PyTorch 2.5–2.9 and CUDA 11.8–12.8:

```bash
pip install fussim --extra-index-url https://opsiclear.github.io/fussim/whl/
```

This auto-selects the correct wheel for your installed PyTorch version.

| PyTorch | Python | CUDA 11.8 | CUDA 12.1 | CUDA 12.4 | CUDA 12.6 | CUDA 12.8 |
|:-------:|:------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| 2.5.1   | 3.10–3.12 | ✓ | ✓ | ✓ | — | — |
| 2.6.0   | 3.10–3.12 | ✓ | — | ✓ | ✓ | — |
| 2.7.1   | 3.10–3.12 | ✓ | — | — | ✓ | ✓ |
| 2.8.0   | 3.10–3.13 | — | — | — | ✓ | ✓ |
| 2.9.0   | 3.10–3.13 | — | — | — | ✓* | ✓* |

*\*Linux only. Windows has a [known PyTorch bug](https://github.com/pytorch/pytorch/issues/141026).*

**[→ Open Installation Configurator](https://opsiclear.github.io/fussim/)**

</details>

<details>
<summary><b>Build from source</b></summary>

Requires CUDA Toolkit and C++ compiler.

```bash
git clone https://github.com/OpsiClear/fussim.git && cd fussim
pip install .

# For specific GPU architecture:
TORCH_CUDA_ARCH_LIST="8.9" pip install .  # RTX 4090
```

</details>

<details>
<summary><b>Other platforms</b></summary>

For macOS, older CUDA versions, or unsupported configurations, use the original [fused-ssim](https://github.com/rahul-goel/fused-ssim) which builds from source.

</details>

---

## Quick Start

```python
import torch
from fussim import fused_ssim

img1 = torch.rand(1, 3, 256, 256, device="cuda", requires_grad=True)
img2 = torch.rand(1, 3, 256, 256, device="cuda")

# Compute SSIM
ssim_value = fused_ssim(img1, img2)

# Use as loss
loss = 1.0 - fused_ssim(img1, img2)
loss.backward()
```

FP16 / AMP:

```python
with torch.autocast(device_type="cuda"):
    ssim_value = fused_ssim(img1, img2)  # Uses FP16 kernel automatically
```

---

## API

### `fused_ssim`

```python
fused_ssim(img1, img2, padding="same", train=True, window_size=11) -> Tensor
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `img1` | Tensor | — | First image `(B, C, H, W)`. **Receives gradients.** |
| `img2` | Tensor | — | Second image `(B, C, H, W)` |
| `padding` | str | `"same"` | `"same"` or `"valid"` |
| `train` | bool | `True` | Enable gradient computation |
| `window_size` | int | `11` | Gaussian window: `7`, `9`, or `11` |

**Returns:** Scalar mean SSIM value.

### `ssim` (pytorch-msssim compatible)

```python
ssim(X, Y, data_range=255, size_average=True, win_size=11, K=(0.01, 0.03), nonnegative_ssim=False) -> Tensor
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X`, `Y` | Tensor | — | Images `(B, C, H, W)`. Gradients computed for `X`. |
| `data_range` | float | `255` | Value range (`255` for uint8, `1.0` for normalized) |
| `size_average` | bool | `True` | Return scalar mean or per-batch values |
| `win_size` | int | `11` | `7`, `9`, or `11` |
| `K` | tuple | `(0.01, 0.03)` | SSIM constants (K1, K2) |
| `nonnegative_ssim` | bool | `False` | Clamp negative values to 0 |

### `SSIM` Module

```python
from fussim import SSIM

module = SSIM(data_range=1.0)
loss = 1 - module(pred, target)
loss.backward()
```

---

## Performance

RTX 4090, batch 5×5×1080×1920, 100 iterations:

| Implementation | Forward | Backward | Total | Speedup |
|----------------|--------:|---------:|------:|--------:|
| pytorch-msssim | 28.7 ms | 28.9 ms | 57.5 ms | 1.0× |
| **fussim** | **4.38 ms** | **4.66 ms** | **9.04 ms** | **6.4×** |

---

## Limitations

| Constraint | Reason |
|------------|--------|
| `window_size`: 7, 9, or 11 | CUDA kernel templates |
| `win_sigma`: 1.5 (fixed) | Hardcoded in kernel |
| Custom `win` not supported | Uses built-in Gaussian |

---

## Attribution

- [optimized-fused-ssim](https://github.com/MrNeRF/optimized-fused-ssim) by Janusch Patas
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
