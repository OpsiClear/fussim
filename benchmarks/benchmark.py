#!/usr/bin/env python3
"""
Performance benchmarks for fused-ssim.

Compares performance against reference implementations:
- Reference SSIM (pytorch-ssim style)
- pytorch-msssim

Usage:
    python benchmarks/benchmark.py
    python benchmarks/benchmark.py --iterations 200
    python benchmarks/benchmark.py --size 1920x1080
"""

import argparse
import time
from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F

from fused_ssim import fused_ssim


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def reference_ssim(img1, img2, window_size=11):
    """Reference SSIM implementation."""
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean()


def benchmark_forward(func, img1, img2, iterations, warmup=10, **kwargs):
    """Benchmark forward pass."""
    # Warmup
    for _ in range(warmup):
        _ = func(img1, img2, **kwargs)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        _ = func(img1, img2, **kwargs)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / iterations * 1000  # ms


def benchmark_backward(func, img1, img2, iterations, warmup=10, **kwargs):
    """Benchmark forward + backward pass."""
    # Warmup
    for _ in range(warmup):
        result = func(img1, img2, **kwargs)
        result.backward()
        img1.grad = None
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        result = func(img1, img2, **kwargs)
        result.backward()
        img1.grad = None
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / iterations * 1000  # ms


def run_benchmarks(batch_size, channels, height, width, iterations):
    """Run all benchmarks."""
    print("=" * 70)
    print("Benchmark Configuration")
    print("=" * 70)
    print(f"  Image size: {batch_size} x {channels} x {height} x {width}")
    print(f"  Iterations: {iterations}")
    print(f"  Device: {torch.cuda.get_device_name()}")
    print("=" * 70)
    print()

    torch.manual_seed(0)
    img1 = nn.Parameter(torch.rand(batch_size, channels, height, width, device="cuda"))
    img2 = torch.rand(batch_size, channels, height, width, device="cuda")

    results = {}

    # Reference SSIM
    print("Benchmarking Reference SSIM...")
    ref_forward = benchmark_forward(reference_ssim, img1, img2, iterations)
    ref_total = benchmark_backward(reference_ssim, img1, img2, iterations)
    ref_backward = ref_total - ref_forward
    results["Reference SSIM"] = {
        "forward": ref_forward,
        "backward": ref_backward,
        "total": ref_total,
    }

    # pytorch-msssim (if available)
    try:
        from pytorch_msssim import SSIM

        pm_ssim = SSIM(data_range=1.0, channel=channels)
        print("Benchmarking pytorch-msssim...")
        pm_forward = benchmark_forward(pm_ssim, img1, img2, iterations)
        pm_total = benchmark_backward(pm_ssim, img1, img2, iterations)
        pm_backward = pm_total - pm_forward
        results["pytorch-msssim"] = {
            "forward": pm_forward,
            "backward": pm_backward,
            "total": pm_total,
        }
    except ImportError:
        print("  pytorch-msssim not installed, skipping...")

    # Fused SSIM (training mode)
    print("Benchmarking fused-ssim (train=True)...")
    fused_forward = benchmark_forward(fused_ssim, img1, img2, iterations)
    fused_total = benchmark_backward(fused_ssim, img1, img2, iterations)
    fused_backward = fused_total - fused_forward
    results["fused-ssim (train)"] = {
        "forward": fused_forward,
        "backward": fused_backward,
        "total": fused_total,
    }

    # Fused SSIM (inference mode)
    print("Benchmarking fused-ssim (train=False)...")
    with torch.no_grad():
        fused_infer = benchmark_forward(fused_ssim, img1, img2, iterations, train=False)
    results["fused-ssim (inference)"] = {
        "forward": fused_infer,
        "backward": 0.0,
        "total": fused_infer,
    }

    print()
    print("=" * 70)
    print("Results (times in milliseconds)")
    print("=" * 70)
    print(f"{'Implementation':<25} {'Forward':>12} {'Backward':>12} {'Total':>12}")
    print("-" * 70)

    for name, times in results.items():
        print(
            f"{name:<25} {times['forward']:>12.2f} {times['backward']:>12.2f} {times['total']:>12.2f}"
        )

    print("=" * 70)

    # Calculate speedups
    if "Reference SSIM" in results:
        ref_total = results["Reference SSIM"]["total"]
        fused_total = results["fused-ssim (train)"]["total"]
        speedup = ref_total / fused_total
        print(f"\nSpeedup vs Reference: {speedup:.2f}x")

    if "pytorch-msssim" in results:
        pm_total = results["pytorch-msssim"]["total"]
        fused_total = results["fused-ssim (train)"]["total"]
        speedup = pm_total / fused_total
        print(f"Speedup vs pytorch-msssim: {speedup:.2f}x")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark fused-ssim performance")
    parser.add_argument("--batch", type=int, default=5, help="Batch size")
    parser.add_argument("--channels", type=int, default=5, help="Number of channels")
    parser.add_argument(
        "--size",
        type=str,
        default="1920x1080",
        help="Image size as WIDTHxHEIGHT (default: 1920x1080)",
    )
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    args = parser.parse_args()

    # Parse size
    width, height = map(int, args.size.lower().split("x"))

    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        return 1

    run_benchmarks(args.batch, args.channels, height, width, args.iterations)
    return 0


if __name__ == "__main__":
    exit(main())
