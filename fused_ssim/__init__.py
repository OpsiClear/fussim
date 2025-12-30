"""
Optimized Fused SSIM - A fast CUDA implementation of differentiable SSIM for PyTorch.

This module provides an optimized implementation of the Structural Similarity Index
(SSIM) that fuses multiple operations into efficient CUDA kernels.
"""

import re
import warnings

import torch

__version__ = "0.1.0"
__all__ = [
    # Primary API
    "fused_ssim",
    # pytorch-msssim compatible API
    "ssim",
    "SSIM",
    # Advanced/internal
    "FusedSSIMMap",
    "FusedSSIMMapFP16",
    # Utilities
    "__version__",
    "get_build_info",
    "check_compatibility",
    "SUPPORTED_WINDOW_SIZES",
]

# Supported window sizes (must match CUDA kernel instantiations)
SUPPORTED_WINDOW_SIZES = (7, 9, 11)


def _cli_check():
    """CLI entry point to check installation and compatibility."""
    print("fused-ssim Installation Check")
    print("=" * 50)

    info = get_build_info()
    print(f"Package version: {info['version']}")
    print(f"Build type: {'pre-built wheel' if info['is_prebuilt'] else 'source build'}")

    if info["is_prebuilt"]:
        print(
            f"Built for: PyTorch {info['build_torch_version']} + CUDA {info['build_cuda_version']}"
        )

    print("\nRuntime environment:")
    print(f"  PyTorch: {info['runtime_torch_version']}")
    print(f"  CUDA: {info['runtime_cuda_version'] or 'not available'}")

    compatible, issues = check_compatibility(warn=False)
    if issues:
        print("\nCompatibility warnings:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nCompatibility: OK")

    # Try to load extension
    print("\nTesting CUDA extension...")
    try:
        _ensure_cuda_extension()
        print("  CUDA extension loaded successfully!")

        # Quick test
        if torch.cuda.is_available():
            img = torch.rand(1, 1, 32, 32, device="cuda")
            ssim_val = fused_ssim(img, img)
            print(f"  Quick test passed! (SSIM of identical images = {ssim_val.item():.4f})")
        else:
            print("  CUDA not available for runtime test")

        return 0
    except Exception as e:
        print(f"  Failed: {e}")
        return 1


# Lazy import of CUDA extension with helpful error messages
_cuda_extension = None
_import_error = None
_compatibility_checked = False


def _parse_version(version_str):
    """Parse version string to extract base version and build metadata."""
    # Pattern: 0.1.0+pt29cu128
    match = re.match(r"^(\d+\.\d+\.\d+)(?:\+(.+))?$", version_str)
    if match:
        return match.group(1), match.group(2)
    return version_str, None


def _parse_build_tag(build_tag):
    """Parse build tag like 'pt29cu128' into components."""
    if not build_tag:
        return None, None

    match = re.match(r"pt(\d+)cu(\d+)", build_tag)
    if match:
        torch_major_minor = match.group(1)  # e.g., "29" for 2.9
        cuda_version = match.group(2)  # e.g., "128" for 12.8

        # Convert to proper version format
        torch_ver = f"{torch_major_minor[0]}.{torch_major_minor[1:]}"  # "29" -> "2.9"
        cuda_ver = f"{cuda_version[:-1]}.{cuda_version[-1]}"  # "128" -> "12.8"

        return torch_ver, cuda_ver
    return None, None


def get_build_info():
    """
    Get information about how this package was built.

    Returns:
        dict: Build information including:
            - version: Package version
            - build_torch_version: PyTorch version used during build (if pre-built wheel)
            - build_cuda_version: CUDA version used during build (if pre-built wheel)
            - runtime_torch_version: Currently installed PyTorch version
            - runtime_cuda_version: Currently available CUDA version
            - is_prebuilt: Whether this is a pre-built wheel or source build
    """
    base_version, build_tag = _parse_version(__version__)
    build_torch, build_cuda = _parse_build_tag(build_tag)

    runtime_torch = torch.__version__.split("+")[0]  # Remove +cu121 suffix if present
    runtime_cuda = torch.version.cuda if torch.cuda.is_available() else None

    return {
        "version": __version__,
        "base_version": base_version,
        "build_torch_version": build_torch,
        "build_cuda_version": build_cuda,
        "runtime_torch_version": runtime_torch,
        "runtime_cuda_version": runtime_cuda,
        "is_prebuilt": build_tag is not None,
    }


def check_compatibility(warn=True):
    """
    Check if the installed package is compatible with the runtime environment.

    Args:
        warn: If True, emit warnings for potential compatibility issues.

    Returns:
        tuple: (is_compatible, list of issues)
    """
    info = get_build_info()
    issues = []

    if not info["is_prebuilt"]:
        # Source build - should be compatible
        return True, []

    # Check PyTorch version compatibility
    if info["build_torch_version"] and info["runtime_torch_version"]:
        build_major_minor = ".".join(info["build_torch_version"].split(".")[:2])
        runtime_major_minor = ".".join(info["runtime_torch_version"].split(".")[:2])

        if build_major_minor != runtime_major_minor:
            issues.append(
                f"PyTorch version mismatch: package built for PyTorch {info['build_torch_version']}, "
                f"but {info['runtime_torch_version']} is installed. "
                f"This may cause compatibility issues."
            )

    # Check CUDA version compatibility
    if info["build_cuda_version"] and info["runtime_cuda_version"]:
        build_cuda_major = info["build_cuda_version"].split(".")[0]
        runtime_cuda_major = info["runtime_cuda_version"].split(".")[0]

        if build_cuda_major != runtime_cuda_major:
            issues.append(
                f"CUDA major version mismatch: package built for CUDA {info['build_cuda_version']}, "
                f"but CUDA {info['runtime_cuda_version']} is available. "
                f"This may cause compatibility issues."
            )

    is_compatible = len(issues) == 0

    if warn and issues:
        for issue in issues:
            warnings.warn(issue, RuntimeWarning, stacklevel=2)

    return is_compatible, issues


def _ensure_cuda_extension():
    """Lazy load the CUDA extension with helpful error messages."""
    global _cuda_extension, _import_error, _compatibility_checked

    if _cuda_extension is not None:
        return _cuda_extension

    if _import_error is not None:
        raise _import_error

    # Check compatibility on first load
    if not _compatibility_checked:
        _compatibility_checked = True
        check_compatibility(warn=True)

    try:
        import fused_ssim_cuda

        _cuda_extension = fused_ssim_cuda
        return _cuda_extension
    except ImportError as e:
        error_msg = str(e)
        info = get_build_info()

        # Build helpful context
        context = []
        if info["is_prebuilt"]:
            context.append(
                f"  Installed wheel: built for PyTorch {info['build_torch_version']} + CUDA {info['build_cuda_version']}"
            )
        context.append(f"  Runtime PyTorch: {info['runtime_torch_version']}")
        context.append(f"  Runtime CUDA: {info['runtime_cuda_version'] or 'not available'}")
        context_str = "\n".join(context)

        # Provide helpful error messages based on the error type
        if "No module named" in error_msg:
            _import_error = ImportError(
                "fused_ssim_cuda extension not found.\n\n"
                f"Environment:\n{context_str}\n\n"
                "Solutions:\n"
                "  1. Install a pre-built wheel matching your environment:\n"
                "     pip install fused-ssim --extra-index-url https://mrnerf.github.io/optimized-fused-ssim/whl/\n"
                "  2. Or build from source (requires CUDA Toolkit):\n"
                "     pip install fused-ssim --no-binary fused-ssim\n"
                f"\nOriginal error: {error_msg}"
            )
        elif "undefined symbol" in error_msg or "cannot open shared object" in error_msg:
            _import_error = ImportError(
                "fused_ssim_cuda extension failed to load - ABI mismatch detected.\n\n"
                f"Environment:\n{context_str}\n\n"
                "This happens when the installed wheel doesn't match your PyTorch/CUDA version.\n\n"
                "Solutions:\n"
                "  1. Install the correct wheel for your environment:\n"
                f"     pip install fused-ssim --extra-index-url https://mrnerf.github.io/optimized-fused-ssim/whl/pt{info['runtime_torch_version'].replace('.', '')[:2]}cu{info['runtime_cuda_version'].replace('.', '') if info['runtime_cuda_version'] else 'XXX'}/\n"
                "  2. Or rebuild from source:\n"
                "     pip install --force-reinstall --no-cache-dir --no-binary fused-ssim fused-ssim\n"
                f"\nOriginal error: {error_msg}"
            )
        elif "CUDA" in error_msg or "cuda" in error_msg:
            _import_error = ImportError(
                "CUDA-related error loading fused_ssim_cuda.\n\n"
                f"Environment:\n{context_str}\n\n"
                "Please check:\n"
                "  1. NVIDIA drivers are installed and up to date\n"
                "  2. The installed wheel matches your CUDA version\n"
                f"\nOriginal error: {error_msg}"
            )
        else:
            _import_error = ImportError(
                f"Failed to import fused_ssim_cuda.\n\n"
                f"Environment:\n{context_str}\n\n"
                f"Original error: {error_msg}\n\n"
                "Try reinstalling: pip install --force-reinstall fused-ssim"
            )

        raise _import_error


def _get_halo(window_size: int) -> int:
    """Get the HALO (half-window) size for a given window size."""
    return (window_size - 1) // 2


ALLOWED_PADDING = ("same", "valid")


class FusedSSIMMap(torch.autograd.Function):
    """
    Autograd Function for computing SSIM with automatic differentiation support.

    This function computes the SSIM map between two images and supports
    backpropagation through the first image (img1).
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx, C1, C2, img1, img2, padding="same", train=True, window_size=11):
        cuda_ext = _ensure_cuda_extension()
        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = cuda_ext.fusedssim(
            window_size, C1, C2, img1, img2, train
        )

        halo = _get_halo(window_size)
        if padding == "valid":
            ssim_map = ssim_map[:, :, halo:-halo, halo:-halo]

        ctx.save_for_backward(img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.C1 = C1
        ctx.C2 = C2
        ctx.padding = padding
        ctx.window_size = window_size

        return ssim_map

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, opt_grad):
        cuda_ext = _ensure_cuda_extension()
        img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        C1, C2, padding, window_size = ctx.C1, ctx.C2, ctx.padding, ctx.window_size
        dL_dmap = opt_grad

        halo = _get_halo(window_size)
        if padding == "valid":
            dL_dmap = torch.zeros_like(img1)
            dL_dmap[:, :, halo:-halo, halo:-halo] = opt_grad

        grad = cuda_ext.fusedssim_backward(
            window_size, C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12
        )
        return None, None, grad, None, None, None, None


class FusedSSIMMapFP16(torch.autograd.Function):
    """
    Autograd Function for computing SSIM with FP16 mixed precision.

    Uses FP16 for input images and FP32 for gradient computation to balance
    speed and numerical stability.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, C1, C2, img1, img2, padding="same", window_size=11):
        cuda_ext = _ensure_cuda_extension()

        # Convert to FP16 for forward pass (only if not already FP16)
        img1_half = img1 if img1.dtype == torch.float16 else img1.half()
        img2_half = img2 if img2.dtype == torch.float16 else img2.half()

        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = cuda_ext.fusedssim_fp16_train(
            window_size, C1, C2, img1_half, img2_half
        )

        halo = _get_halo(window_size)
        if padding == "valid":
            ssim_map = ssim_map[:, :, halo:-halo, halo:-halo]

        # Save FP16 images and FP32 derivatives for backward
        ctx.save_for_backward(img1_half, img2_half, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.C1 = C1
        ctx.C2 = C2
        ctx.padding = padding
        ctx.window_size = window_size

        return ssim_map

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, opt_grad):
        cuda_ext = _ensure_cuda_extension()
        img1_half, img2_half, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        C1, C2, padding, window_size = ctx.C1, ctx.C2, ctx.padding, ctx.window_size
        dL_dmap = opt_grad.float()  # Ensure FP32 for gradient computation

        halo = _get_halo(window_size)
        if padding == "valid":
            dL_dmap = torch.zeros(img1_half.shape, dtype=torch.float32, device=img1_half.device)
            dL_dmap[:, :, halo:-halo, halo:-halo] = opt_grad.float()

        grad = cuda_ext.fusedssim_fp16_backward(
            window_size, C1, C2, img1_half, img2_half, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12
        )
        return None, None, grad, None, None, None


def fused_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    padding: str = "same",
    train: bool = True,
    window_size: int = 11,
) -> torch.Tensor:
    """
    Compute the Structural Similarity Index (SSIM) between two images.

    This is an optimized CUDA implementation that fuses multiple convolution
    operations for better performance. Drop-in replacement for the original
    fused-ssim library (https://github.com/rahul-goel/fused-ssim).

    Args:
        img1: First image tensor of shape (B, C, H, W). Must be on CUDA device.
              Gradients are computed with respect to this tensor.
        img2: Second image tensor of shape (B, C, H, W). Must be on CUDA device.
        padding: Padding mode for convolution. Either "same" (default) or "valid".
        train: If True, compute gradients (default). If False, inference-only mode.
        window_size: Size of the Gaussian window. Must be 7, 9, or 11 (default).
            Larger windows capture more structure but may miss fine details.

    Returns:
        Scalar SSIM value (mean over all pixels).

    Raises:
        RuntimeError: If CUDA is not available or tensors are not on CUDA device.
        ValueError: If padding mode is invalid, tensor shapes don't match, or
            window_size is not supported.

    Note:
        Only img1 receives gradients. Ensure your prediction is img1, not img2.
        Use torch.autocast() for FP16 mixed precision (faster on modern GPUs).

    Example:
        >>> import torch
        >>> from fused_ssim import fused_ssim
        >>> img1 = torch.rand(1, 3, 256, 256, device="cuda", requires_grad=True)
        >>> img2 = torch.rand(1, 3, 256, 256, device="cuda")
        >>> ssim_value = fused_ssim(img1, img2)
        >>> loss = 1.0 - ssim_value
        >>> loss.backward()
        >>>
        >>> # For FP16 mixed precision:
        >>> with torch.autocast(device_type="cuda"):
        ...     ssim_value = fused_ssim(img1, img2)
        >>>
        >>> # Use smaller window for fine-grained distortions:
        >>> ssim_value = fused_ssim(img1, img2, window_size=7)
    """
    # Input validation
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. fused_ssim requires a CUDA-capable GPU.\n"
            "If you have a GPU, make sure:\n"
            "  1. NVIDIA drivers are installed\n"
            "  2. PyTorch is installed with CUDA support"
        )

    if not img1.is_cuda:
        raise RuntimeError(
            f"img1 must be on a CUDA device, but got device: {img1.device}\n"
            "Move your tensor to GPU with: img1 = img1.cuda()"
        )

    if not img2.is_cuda:
        raise RuntimeError(
            f"img2 must be on a CUDA device, but got device: {img2.device}\n"
            "Move your tensor to GPU with: img2 = img2.cuda()"
        )

    if img1.device != img2.device:
        raise RuntimeError(
            f"img1 and img2 must be on the same device.\n"
            f"img1 device: {img1.device}, img2 device: {img2.device}"
        )

    if img1.shape != img2.shape:
        raise ValueError(
            f"img1 and img2 must have the same shape.\n"
            f"img1 shape: {img1.shape}, img2 shape: {img2.shape}"
        )

    if len(img1.shape) != 4:
        raise ValueError(
            f"Expected 4D tensor (B, C, H, W), but got {len(img1.shape)}D tensor "
            f"with shape {img1.shape}"
        )

    if padding not in ALLOWED_PADDING:
        raise ValueError(f"padding must be one of {ALLOWED_PADDING}, but got '{padding}'")

    if window_size not in SUPPORTED_WINDOW_SIZES:
        raise ValueError(
            f"window_size must be one of {SUPPORTED_WINDOW_SIZES}, got {window_size}"
        )

    # SSIM constants (standard values for [0, 1] range)
    C1 = 0.01**2
    C2 = 0.03**2

    # Respect train parameter, but also check gradient context
    train = train and torch.is_grad_enabled() and img1.requires_grad

    # Auto-detect FP16 mode from autocast context
    use_fp16 = torch.is_autocast_enabled()

    # Use FP16 path when autocast is enabled
    if use_fp16:
        if train:
            # FP16 training path with mixed precision
            img1 = img1.contiguous()
            ssim_map = FusedSSIMMapFP16.apply(C1, C2, img1, img2, padding, window_size)
        else:
            # FP16 inference path (no gradient computation)
            cuda_ext = _ensure_cuda_extension()
            img1_cont = img1.contiguous()
            img2_cont = img2.contiguous()
            ssim_map = cuda_ext.fusedssim_fp16(window_size, C1, C2, img1_cont, img2_cont)
            halo = _get_halo(window_size)
            if padding == "valid":
                ssim_map = ssim_map[:, :, halo:-halo, halo:-halo]
    else:
        # Standard FP32 path
        img1 = img1.contiguous()
        ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train, window_size)

    # Always return mean (matches original fused-ssim behavior)
    return ssim_map.mean()


# =============================================================================
# pytorch-msssim Compatible API
# =============================================================================
# These functions provide a drop-in replacement for pytorch-msssim.
# Usage:
#   # Before (pytorch-msssim):
#   from pytorch_msssim import ssim, SSIM
#
#   # After (fused-ssim - just change the import):
#   from fused_ssim import ssim, SSIM


def ssim(
    X: torch.Tensor,
    Y: torch.Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: torch.Tensor = None,
    K: tuple = (0.01, 0.03),
    nonnegative_ssim: bool = False,
) -> torch.Tensor:
    """
    Compute SSIM between two images - pytorch-msssim compatible API.

    This is a drop-in replacement for pytorch_msssim.ssim(). Simply change:
        from pytorch_msssim import ssim  ->  from fused_ssim import ssim

    Args:
        X: First image (B, C, H, W). Gradients computed for this tensor.
        Y: Second image (B, C, H, W).
        data_range: Value range of images. Default 255 for [0-255], use 1.0 for [0-1].
        size_average: If True, return scalar mean. If False, return per-image values.
        win_size: Gaussian window size. Supported: 7, 9, 11 (default).
        win_sigma: Gaussian sigma (must be 1.5, other values not supported).
        win: Custom window tensor (not supported, must be None).
        K: SSIM constants (K1, K2). Default (0.01, 0.03). Values must be in (0, 1).
        nonnegative_ssim: If True, clamp negative SSIM values to 0.

    Returns:
        SSIM value(s). Scalar if size_average=True, tensor (B,) if size_average=False.

    Note:
        This implementation uses optimized CUDA kernels with fixed sigma=1.5.
        Window size can be 7, 9, or 11. K values are configurable.

    Example:
        >>> from fused_ssim import ssim
        >>> X = torch.rand(4, 3, 256, 256, device="cuda")
        >>> Y = torch.rand(4, 3, 256, 256, device="cuda")
        >>> ssim_val = ssim(X, Y, data_range=1.0)
    """
    # Validate parameters
    if win_size not in SUPPORTED_WINDOW_SIZES:
        raise ValueError(
            f"win_size must be one of {SUPPORTED_WINDOW_SIZES}, got {win_size}. "
            "These are the window sizes supported by the optimized CUDA kernels."
        )
    if win_sigma != 1.5:
        raise ValueError(
            f"win_sigma must be 1.5 (CUDA kernel limitation), got {win_sigma}. "
            "This is the standard SSIM sigma used by most implementations."
        )
    if win is not None:
        raise ValueError(
            "Custom window (win) is not supported. The CUDA kernel uses an "
            f"optimized built-in {win_size}x{win_size} Gaussian window with sigma=1.5."
        )

    # Validate K
    if not (isinstance(K, (tuple, list)) and len(K) == 2):
        raise ValueError(f"K must be a tuple/list of 2 values (K1, K2), got {K}")
    K1, K2 = K
    if not (0 < K1 < 1 and 0 < K2 < 1):
        raise ValueError(f"K values must be in (0, 1), got K1={K1}, K2={K2}")

    # Compute SSIM constants scaled by data_range
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # Check if training mode
    train = torch.is_grad_enabled() and X.requires_grad

    # Auto-detect FP16 mode
    use_fp16 = torch.is_autocast_enabled()

    # Compute SSIM map using CUDA kernels directly
    X_cont = X.contiguous()
    halo = _get_halo(win_size)

    if use_fp16:
        if train:
            ssim_map = FusedSSIMMapFP16.apply(C1, C2, X_cont, Y, "valid", win_size)
        else:
            cuda_ext = _ensure_cuda_extension()
            Y_cont = Y.contiguous()
            ssim_map = cuda_ext.fusedssim_fp16(win_size, C1, C2, X_cont, Y_cont)
            ssim_map = ssim_map[:, :, halo:-halo, halo:-halo]
    else:
        ssim_map = FusedSSIMMap.apply(C1, C2, X_cont, Y, "valid", train, win_size)

    # Apply nonnegative_ssim if requested
    if nonnegative_ssim:
        ssim_map = torch.clamp(ssim_map, min=0)

    # Apply reduction
    if size_average:
        return ssim_map.mean()
    else:
        # pytorch-msssim returns (B,) shape when size_average=False
        # Mean over C, H, W dimensions
        return ssim_map.mean(dim=(1, 2, 3))


class SSIM(torch.nn.Module):
    """
    SSIM as a PyTorch module - pytorch-msssim compatible API.

    This is a drop-in replacement for pytorch_msssim.SSIM(). Simply change:
        from pytorch_msssim import SSIM  ->  from fused_ssim import SSIM

    Args:
        data_range: Value range of images. Default 255 for [0-255], use 1.0 for [0-1].
        size_average: If True, return scalar mean. If False, return per-image values.
        win_size: Gaussian window size. Supported: 7, 9, 11 (default).
        win_sigma: Gaussian sigma (must be 1.5).
        channel: Number of channels (ignored, inferred from input).
        spatial_dims: Spatial dimensions (must be 2 for 2D images).
        K: SSIM constants (K1, K2). Default (0.01, 0.03). Values must be in (0, 1).
        nonnegative_ssim: If True, clamp negative SSIM values to 0.

    Example:
        >>> from fused_ssim import SSIM
        >>> ssim_module = SSIM(data_range=1.0)
        >>> X = torch.rand(4, 3, 256, 256, device="cuda", requires_grad=True)
        >>> Y = torch.rand(4, 3, 256, 256, device="cuda")
        >>> ssim_val = ssim_module(X, Y)
        >>> loss = 1 - ssim_val
        >>> loss.backward()
    """

    def __init__(
        self,
        data_range: float = 255,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        spatial_dims: int = 2,
        K: tuple = (0.01, 0.03),
        nonnegative_ssim: bool = False,
    ):
        super().__init__()

        # Validate parameters
        if win_size not in SUPPORTED_WINDOW_SIZES:
            raise ValueError(
                f"win_size must be one of {SUPPORTED_WINDOW_SIZES}, got {win_size}"
            )
        if win_sigma != 1.5:
            raise ValueError(f"win_sigma must be 1.5 (CUDA kernel limitation), got {win_sigma}")
        if spatial_dims != 2:
            raise ValueError(
                f"spatial_dims must be 2 (only 2D images supported), got {spatial_dims}"
            )
        if not (isinstance(K, (tuple, list)) and len(K) == 2):
            raise ValueError(f"K must be a tuple/list of 2 values (K1, K2), got {K}")
        if not (0 < K[0] < 1 and 0 < K[1] < 1):
            raise ValueError(f"K values must be in (0, 1), got K1={K[0]}, K2={K[1]}")

        self.data_range = data_range
        self.size_average = size_average
        self.win_size = win_size
        self.K = tuple(K)
        self.nonnegative_ssim = nonnegative_ssim
        # channel is ignored - we infer from input

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM between X and Y.

        Args:
            X: First image (B, C, H, W). Gradients computed for this tensor.
            Y: Second image (B, C, H, W).

        Returns:
            SSIM value(s).
        """
        return ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win_size=self.win_size,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        )
