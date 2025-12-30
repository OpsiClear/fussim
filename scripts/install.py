#!/usr/bin/env python3
"""
Automatic installer for fused-ssim.

This script detects your PyTorch and CUDA versions and installs the appropriate
pre-built wheel, or falls back to building from source if no matching wheel exists.

Usage:
    python install.py              # Auto-detect and install
    python install.py --source     # Force build from source
    python install.py --check      # Just check environment, don't install
"""

import argparse
import subprocess
import sys


def get_torch_info():
    """Get PyTorch version and CUDA version."""
    try:
        import torch

        torch_version = torch.__version__.split("+")[0]  # Remove local version
        cuda_version = torch.version.cuda if torch.cuda.is_available() else None
        return torch_version, cuda_version
    except ImportError:
        return None, None


def get_wheel_tag(torch_version, cuda_version):
    """
    Convert PyTorch and CUDA versions to wheel tag.

    Returns:
        str: Wheel tag like 'pt29cu128', or None if no matching wheel.
    """
    if not torch_version or not cuda_version:
        return None

    # Parse versions
    torch_parts = torch_version.split(".")
    torch_major_minor = f"{torch_parts[0]}{torch_parts[1]}"  # e.g., "29" for 2.9.0

    cuda_parts = cuda_version.split(".")
    cuda_tag = f"{cuda_parts[0]}{cuda_parts[1]}"  # e.g., "128" for 12.8

    # Map of supported combinations
    # Format: (torch_major_minor, cuda_tag) -> wheel_tag
    supported = {
        # PyTorch 2.9.x
        ("29", "126"): "pt29cu126",
        ("29", "128"): "pt29cu128",
        # PyTorch 2.8.x
        ("28", "126"): "pt28cu126",
        ("28", "128"): "pt28cu128",
        # PyTorch 2.7.x
        ("27", "118"): "pt27cu118",
        ("27", "126"): "pt27cu126",
        ("27", "128"): "pt27cu128",
        # PyTorch 2.6.x
        ("26", "118"): "pt26cu118",
        ("26", "124"): "pt26cu124",
        ("26", "126"): "pt26cu126",
        # PyTorch 2.5.x
        ("25", "118"): "pt25cu118",
        ("25", "121"): "pt25cu121",
        ("25", "124"): "pt25cu124",
    }

    key = (torch_major_minor, cuda_tag)
    return supported.get(key)


def print_environment():
    """Print detected environment."""
    torch_version, cuda_version = get_torch_info()

    print("=" * 60)
    print("Environment Detection")
    print("=" * 60)

    if torch_version:
        print(f"  PyTorch version: {torch_version}")
    else:
        print("  PyTorch: NOT INSTALLED")
        print("  Please install PyTorch first: https://pytorch.org/get-started/locally/")

    if cuda_version:
        print(f"  CUDA version: {cuda_version}")
    else:
        print("  CUDA: NOT AVAILABLE")

    wheel_tag = get_wheel_tag(torch_version, cuda_version)
    if wheel_tag:
        print(f"  Matching wheel: {wheel_tag}")
    else:
        print("  Matching wheel: None (will build from source)")

    print("=" * 60)
    return torch_version, cuda_version, wheel_tag


def install_from_wheel(wheel_tag):
    """Install from pre-built wheel."""
    index_url = f"https://mrnerf.github.io/optimized-fused-ssim/whl/{wheel_tag}/"

    print(f"\nInstalling from pre-built wheel ({wheel_tag})...")
    print(f"Using index: {index_url}")

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--extra-index-url",
        index_url,
        "fused-ssim",
    ]

    result = subprocess.run(cmd)
    return result.returncode == 0


def install_from_source():
    """Install by building from source."""
    print("\nBuilding from source...")
    print("This requires CUDA Toolkit to be installed.")

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-cache-dir",
        "--no-binary",
        "fused-ssim",
        "fused-ssim",
    ]

    result = subprocess.run(cmd)
    return result.returncode == 0


def install_from_git():
    """Install directly from git repository."""
    print("\nInstalling from git repository...")

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "git+https://github.com/MrNeRF/optimized-fused-ssim.git",
    ]

    result = subprocess.run(cmd)
    return result.returncode == 0


def verify_installation():
    """Verify the installation works."""
    print("\nVerifying installation...")
    try:
        import fused_ssim

        info = fused_ssim.get_build_info()
        print(f"  Installed version: {info['version']}")
        print(f"  Build type: {'pre-built wheel' if info['is_prebuilt'] else 'source build'}")

        # Try to load the CUDA extension
        compatible, issues = fused_ssim.check_compatibility(warn=False)
        if issues:
            print("\n  Warnings:")
            for issue in issues:
                print(f"    - {issue}")

        # Quick functionality test
        import torch

        if torch.cuda.is_available():
            img1 = torch.rand(1, 1, 32, 32, device="cuda")
            img2 = torch.rand(1, 1, 32, 32, device="cuda")
            ssim = fused_ssim.fused_ssim(img1, img2)
            print(f"\n  Quick test passed! SSIM = {ssim.item():.4f}")
            return True
        else:
            print("\n  CUDA not available - cannot run functionality test")
            return True  # Still consider success if import works

    except Exception as e:
        print(f"\n  Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Install fused-ssim with automatic environment detection"
    )
    parser.add_argument(
        "--source",
        action="store_true",
        help="Force build from source instead of using pre-built wheel",
    )
    parser.add_argument("--git", action="store_true", help="Install directly from git repository")
    parser.add_argument(
        "--check", action="store_true", help="Only check environment, don't install"
    )
    args = parser.parse_args()

    # Detect environment
    torch_version, cuda_version, wheel_tag = print_environment()

    if args.check:
        return 0

    if not torch_version:
        print("\nError: PyTorch is not installed.")
        print("Please install PyTorch first: https://pytorch.org/get-started/locally/")
        return 1

    # Install
    success = False

    if args.git:
        success = install_from_git()
    elif args.source or not wheel_tag:
        if not wheel_tag and not args.source:
            print("\nNo matching pre-built wheel found.")
        success = install_from_source()
    else:
        success = install_from_wheel(wheel_tag)
        if not success:
            print("\nPre-built wheel installation failed. Trying source build...")
            success = install_from_source()

    if not success:
        print("\nInstallation failed!")
        return 1

    # Verify
    if verify_installation():
        print("\n" + "=" * 60)
        print("Installation successful!")
        print("=" * 60)
        print("\nUsage:")
        print("  from fused_ssim import fused_ssim")
        print("  ssim = fused_ssim(img1, img2)")
        return 0
    else:
        print("\nInstallation completed but verification failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
