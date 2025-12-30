"""Comprehensive benchmark comparing optimized fused-ssim against original implementation.

Uses subprocess isolation to avoid namespace conflicts.
Reports detailed performance statistics with speedup ratios.
"""

import argparse
import json
import subprocess
import sys
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Paths
OPTIMIZED_PATH = Path(__file__).parent.parent
ORIGINAL_VENV = Path("C:/temp/fused-ssim-original/.venv/Scripts/python.exe")


@dataclass
class BenchmarkConfig:
    batch: int
    channels: int
    height: int
    width: int
    window_size: int = 11
    dtype: str = "float32"
    warmup: int = 10
    iterations: int = 100

    @property
    def name(self) -> str:
        return f"{self.batch}x{self.channels}x{self.height}x{self.width}"


@dataclass
class TimingResult:
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


def run_benchmark_optimized(config: BenchmarkConfig, include_backward: bool = True) -> dict:
    """Benchmark optimized implementation."""
    dtype_str = "torch.float16" if config.dtype == "float16" else "torch.float32"

    script = f"""
import torch
import json
import statistics

from fused_ssim import fused_ssim

# Config
B, C, H, W = {config.batch}, {config.channels}, {config.height}, {config.width}
warmup = {config.warmup}
iterations = {config.iterations}
dtype = {dtype_str}
include_backward = {include_backward}

# Create inputs
torch.manual_seed(42)
img1 = torch.rand(B, C, H, W, device="cuda", dtype=dtype, requires_grad=include_backward)
img2 = torch.rand(B, C, H, W, device="cuda", dtype=dtype)

# Warmup
for _ in range(warmup):
    result = fused_ssim(img1, img2, window_size={config.window_size})
    if include_backward:
        result.backward()
        img1.grad = None
torch.cuda.synchronize()

# Benchmark forward
fwd_times = []
for _ in range(iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fused_ssim(img1, img2, window_size={config.window_size})
    end.record()
    torch.cuda.synchronize()
    fwd_times.append(start.elapsed_time(end))

# Benchmark backward
bwd_times = []
if include_backward:
    for _ in range(iterations):
        result = fused_ssim(img1, img2, window_size={config.window_size})
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result.backward()
        end.record()
        torch.cuda.synchronize()
        bwd_times.append(start.elapsed_time(end))
        img1.grad = None

fwd_times.sort()
bwd_times.sort() if bwd_times else None

output = {{
    "forward": {{
        "mean": statistics.mean(fwd_times),
        "std": statistics.stdev(fwd_times) if len(fwd_times) > 1 else 0,
        "min": min(fwd_times),
        "max": max(fwd_times),
        "p50": fwd_times[len(fwd_times)//2],
        "p95": fwd_times[int(len(fwd_times)*0.95)],
        "p99": fwd_times[int(len(fwd_times)*0.99)],
    }}
}}

if bwd_times:
    output["backward"] = {{
        "mean": statistics.mean(bwd_times),
        "std": statistics.stdev(bwd_times) if len(bwd_times) > 1 else 0,
        "min": min(bwd_times),
        "max": max(bwd_times),
        "p50": bwd_times[len(bwd_times)//2],
        "p95": bwd_times[int(len(bwd_times)*0.95)],
        "p99": bwd_times[int(len(bwd_times)*0.99)],
    }}

print(json.dumps(output))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(OPTIMIZED_PATH),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Optimized benchmark failed:\n{result.stderr}")
    return json.loads(result.stdout)


def run_benchmark_original(config: BenchmarkConfig, include_backward: bool = True) -> Optional[dict]:
    """Benchmark original implementation."""
    if not ORIGINAL_VENV.exists():
        return None

    # Original only supports window_size=11
    if config.window_size != 11:
        return None

    # Original doesn't support FP16
    if config.dtype == "float16":
        return None

    script = f"""
import torch
import json
import statistics
import sys
sys.path.insert(0, "C:/temp/fused-ssim-original")

from fused_ssim import fused_ssim

# Config
B, C, H, W = {config.batch}, {config.channels}, {config.height}, {config.width}
warmup = {config.warmup}
iterations = {config.iterations}
include_backward = {include_backward}

# Create inputs
torch.manual_seed(42)
img1 = torch.rand(B, C, H, W, device="cuda", requires_grad=include_backward)
img2 = torch.rand(B, C, H, W, device="cuda")

# Warmup
for _ in range(warmup):
    result = fused_ssim(img1, img2)
    if include_backward:
        result.backward()
        img1.grad = None
torch.cuda.synchronize()

# Benchmark forward
fwd_times = []
for _ in range(iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fused_ssim(img1, img2)
    end.record()
    torch.cuda.synchronize()
    fwd_times.append(start.elapsed_time(end))

# Benchmark backward
bwd_times = []
if include_backward:
    for _ in range(iterations):
        result = fused_ssim(img1, img2)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result.backward()
        end.record()
        torch.cuda.synchronize()
        bwd_times.append(start.elapsed_time(end))
        img1.grad = None

fwd_times.sort()
bwd_times.sort() if bwd_times else None

output = {{
    "forward": {{
        "mean": statistics.mean(fwd_times),
        "std": statistics.stdev(fwd_times) if len(fwd_times) > 1 else 0,
        "min": min(fwd_times),
        "max": max(fwd_times),
        "p50": fwd_times[len(fwd_times)//2],
        "p95": fwd_times[int(len(fwd_times)*0.95)],
        "p99": fwd_times[int(len(fwd_times)*0.99)],
    }}
}}

if bwd_times:
    output["backward"] = {{
        "mean": statistics.mean(bwd_times),
        "std": statistics.stdev(bwd_times) if len(bwd_times) > 1 else 0,
        "min": min(bwd_times),
        "max": max(bwd_times),
        "p50": bwd_times[len(bwd_times)//2],
        "p95": bwd_times[int(len(bwd_times)*0.95)],
        "p99": bwd_times[int(len(bwd_times)*0.99)],
    }}

print(json.dumps(output))
"""
    result = subprocess.run(
        [str(ORIGINAL_VENV), "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        cwd="C:/temp/fused-ssim-original",
    )
    if result.returncode != 0:
        print(f"Warning: Original benchmark failed for {config.name}:\n{result.stderr}")
        return None
    return json.loads(result.stdout)


def format_speedup(optimized: float, original: float) -> str:
    """Format speedup ratio."""
    if original <= 0:
        return "N/A"
    ratio = original / optimized
    if ratio >= 1.0:
        return f"{ratio:.2f}x faster"
    else:
        return f"{1/ratio:.2f}x slower"


def run_benchmarks(configs: list[BenchmarkConfig], include_backward: bool = True, output_file: Optional[str] = None):
    """Run all benchmarks and print results."""
    results = []

    print("=" * 100)
    print("COMPREHENSIVE FUSED-SSIM BENCHMARK")
    print("=" * 100)
    print()

    # Header
    print(f"{'Configuration':<25} {'Direction':<10} {'Optimized':<12} {'Original':<12} {'Speedup':<15}")
    print("-" * 100)

    for config in configs:
        try:
            opt_result = run_benchmark_optimized(config, include_backward)
            orig_result = run_benchmark_original(config, include_backward)

            result_entry = {
                "config": {
                    "batch": config.batch,
                    "channels": config.channels,
                    "height": config.height,
                    "width": config.width,
                    "window_size": config.window_size,
                    "dtype": config.dtype,
                },
                "optimized": opt_result,
                "original": orig_result,
            }

            # Print forward
            opt_fwd = opt_result["forward"]["mean"]
            orig_fwd = orig_result["forward"]["mean"] if orig_result else None
            speedup_fwd = format_speedup(opt_fwd, orig_fwd) if orig_fwd else "N/A"
            orig_fwd_str = f"{orig_fwd:>8.3f}ms" if orig_fwd else "      N/A"
            print(f"{config.name:<25} {'Forward':<10} {opt_fwd:>8.3f}ms {orig_fwd_str} {speedup_fwd:<15}")

            # Print backward
            if include_backward and "backward" in opt_result:
                opt_bwd = opt_result["backward"]["mean"]
                orig_bwd = orig_result["backward"]["mean"] if orig_result and "backward" in orig_result else None
                speedup_bwd = format_speedup(opt_bwd, orig_bwd) if orig_bwd else "N/A"
                orig_bwd_str = f"{orig_bwd:>8.3f}ms" if orig_bwd else "      N/A"
                print(f"{'':<25} {'Backward':<10} {opt_bwd:>8.3f}ms {orig_bwd_str} {speedup_bwd:<15}")

                # Total
                opt_total = opt_fwd + opt_bwd
                orig_total = (orig_fwd + orig_bwd) if orig_fwd and orig_bwd else None
                speedup_total = format_speedup(opt_total, orig_total) if orig_total else "N/A"
                orig_total_str = f"{orig_total:>8.3f}ms" if orig_total else "      N/A"
                print(f"{'':<25} {'Total':<10} {opt_total:>8.3f}ms {orig_total_str} {speedup_total:<15}")

            print()
            results.append(result_entry)

        except Exception as e:
            print(f"{config.name:<25} ERROR: {e}")
            print()

    # Summary statistics
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    speedups = []
    for r in results:
        if r["original"] and "forward" in r["optimized"] and "forward" in r["original"]:
            opt_total = r["optimized"]["forward"]["mean"]
            orig_total = r["original"]["forward"]["mean"]
            if "backward" in r["optimized"] and "backward" in r["original"]:
                opt_total += r["optimized"]["backward"]["mean"]
                orig_total += r["original"]["backward"]["mean"]
            speedups.append(orig_total / opt_total)

    if speedups:
        print(f"Configurations tested: {len(results)}")
        print(f"Average speedup: {statistics.mean(speedups):.2f}x")
        print(f"Min speedup: {min(speedups):.2f}x")
        print(f"Max speedup: {max(speedups):.2f}x")
        print(f"Median speedup: {statistics.median(speedups):.2f}x")

    # Save to file
    if output_file:
        with open(output_file, "w") as f:
            json.dump({"results": results, "speedups": speedups}, f, indent=2)
        print(f"\nResults saved to: {output_file}")


def get_standard_configs() -> list[BenchmarkConfig]:
    """Get standard benchmark configurations."""
    configs = []

    # Small sizes
    for b in [1, 4]:
        for c in [1, 3]:
            configs.append(BenchmarkConfig(b, c, 256, 256))
            configs.append(BenchmarkConfig(b, c, 512, 512))

    # Medium (HD)
    for b in [1, 4]:
        for c in [1, 3, 5]:
            configs.append(BenchmarkConfig(b, c, 1080, 1920))

    # Large (4K) - smaller batches due to memory
    for b in [1, 2]:
        for c in [1, 3]:
            configs.append(BenchmarkConfig(b, c, 2160, 3840))

    return configs


def get_quick_configs() -> list[BenchmarkConfig]:
    """Get quick benchmark configurations for fast testing."""
    return [
        BenchmarkConfig(1, 3, 256, 256, warmup=5, iterations=50),
        BenchmarkConfig(1, 3, 512, 512, warmup=5, iterations=50),
        BenchmarkConfig(1, 3, 1080, 1920, warmup=5, iterations=50),
        BenchmarkConfig(4, 3, 512, 512, warmup=5, iterations=50),
        BenchmarkConfig(1, 5, 1080, 1920, warmup=5, iterations=50),
    ]


def get_window_size_configs() -> list[BenchmarkConfig]:
    """Configs to test different window sizes."""
    configs = []
    for ws in [7, 9, 11]:
        configs.append(BenchmarkConfig(1, 3, 512, 512, window_size=ws, warmup=5, iterations=50))
        configs.append(BenchmarkConfig(1, 3, 1080, 1920, window_size=ws, warmup=5, iterations=50))
    return configs


def get_fp16_configs() -> list[BenchmarkConfig]:
    """Configs to test FP16 performance."""
    return [
        BenchmarkConfig(1, 3, 512, 512, dtype="float16", warmup=5, iterations=50),
        BenchmarkConfig(1, 3, 1080, 1920, dtype="float16", warmup=5, iterations=50),
        BenchmarkConfig(4, 3, 512, 512, dtype="float16", warmup=5, iterations=50),
    ]


def main():
    parser = argparse.ArgumentParser(description="Comprehensive fused-ssim benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--standard", action="store_true", help="Run standard benchmark")
    parser.add_argument("--window-sizes", action="store_true", help="Test different window sizes")
    parser.add_argument("--fp16", action="store_true", help="Test FP16 performance")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--forward-only", action="store_true", help="Only benchmark forward pass")
    parser.add_argument("-o", "--output", help="Output JSON file")
    args = parser.parse_args()

    # Default to quick if nothing specified
    if not any([args.quick, args.standard, args.window_sizes, args.fp16, args.all]):
        args.quick = True

    configs = []
    if args.all or args.standard:
        configs.extend(get_standard_configs())
    if args.quick and not args.standard:
        configs.extend(get_quick_configs())
    if args.all or args.window_sizes:
        configs.extend(get_window_size_configs())
    if args.all or args.fp16:
        configs.extend(get_fp16_configs())

    # Remove duplicates while preserving order
    seen = set()
    unique_configs = []
    for c in configs:
        key = (c.batch, c.channels, c.height, c.width, c.window_size, c.dtype)
        if key not in seen:
            seen.add(key)
            unique_configs.append(c)

    run_benchmarks(unique_configs, include_backward=not args.forward_only, output_file=args.output)


if __name__ == "__main__":
    main()
