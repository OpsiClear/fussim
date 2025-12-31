#!/usr/bin/env python3
"""
Detailed performance benchmarks for fused-ssim optimization tracking.

Features:
- CUDA event-based timing for accurate GPU measurements
- Statistical analysis (mean, std, min, max, percentiles)
- Memory bandwidth utilization calculation
- Comparison across multiple configurations
- JSON output for tracking optimization progress
- Baseline comparison support

Usage:
    python benchmarks/benchmark_detailed.py
    python benchmarks/benchmark_detailed.py --save baseline
    python benchmarks/benchmark_detailed.py --compare baseline_results.json
"""

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    batch_size: int
    channels: int
    height: int
    width: int
    mode: str  # 'fp32_train', 'fp32_infer', 'fp16_train', 'fp16_infer'
    padding: str  # 'same', 'valid'
    iterations: int
    warmup: int


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    config: dict
    forward_ms: float
    forward_std: float
    backward_ms: float
    backward_std: float
    total_ms: float
    throughput_mpix_sec: float
    memory_mb: float
    timestamp: str


class CUDATimer:
    """Accurate GPU timing using CUDA events."""

    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.start_event.record()

    def stop(self):
        self.end_event.record()
        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)


def get_gpu_info() -> dict:
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gb": props.total_memory / (1024**3),
        "sm_count": props.multi_processor_count,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
    }


def benchmark_forward(
    func, img1: torch.Tensor, img2: torch.Tensor, iterations: int, warmup: int, **kwargs
) -> tuple[float, float, list[float]]:
    """Benchmark forward pass with CUDA events."""
    timer = CUDATimer()
    times = []

    # Warmup
    for _ in range(warmup):
        _ = func(img1, img2, **kwargs)
    torch.cuda.synchronize()

    # Benchmark
    for _ in range(iterations):
        timer.start()
        _ = func(img1, img2, **kwargs)
        elapsed = timer.stop()
        times.append(elapsed)

    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    return mean, std, times


def benchmark_backward(
    func, img1: torch.Tensor, img2: torch.Tensor, iterations: int, warmup: int, **kwargs
) -> tuple[float, float, list[float]]:
    """Benchmark forward + backward pass with CUDA events."""
    timer = CUDATimer()
    times = []

    # Warmup
    for _ in range(warmup):
        result = func(img1, img2, **kwargs)
        result.backward()
        img1.grad = None
    torch.cuda.synchronize()

    # Benchmark
    for _ in range(iterations):
        timer.start()
        result = func(img1, img2, **kwargs)
        result.backward()
        elapsed = timer.stop()
        times.append(elapsed)
        img1.grad = None

    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    return mean, std, times


def run_single_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    from fussim import fussim

    torch.manual_seed(42)

    # Determine mode settings
    use_fp16 = "fp16" in config.mode
    train = "train" in config.mode

    # Create tensors
    dtype = torch.float32  # Input always float32, kernel handles conversion
    img1 = torch.rand(
        config.batch_size,
        config.channels,
        config.height,
        config.width,
        device="cuda",
        dtype=dtype,
        requires_grad=train,
    )
    img2 = torch.rand(
        config.batch_size, config.channels, config.height, config.width, device="cuda", dtype=dtype
    )

    # Clear cache and measure memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark forward
    forward_mean, forward_std, _ = benchmark_forward(
        fused_ssim,
        img1,
        img2,
        config.iterations,
        config.warmup,
        padding=config.padding,
        train=train,
        use_fp16=use_fp16,
    )

    # Benchmark backward (only for training modes)
    if train:
        total_mean, total_std, _ = benchmark_backward(
            fused_ssim,
            img1,
            img2,
            config.iterations,
            config.warmup,
            padding=config.padding,
            train=train,
            use_fp16=use_fp16,
        )
        backward_mean = total_mean - forward_mean
        backward_std = (total_std**2 + forward_std**2) ** 0.5
    else:
        total_mean = forward_mean
        total_std = forward_std
        backward_mean = 0.0
        backward_std = 0.0

    # Calculate throughput
    pixels = config.batch_size * config.channels * config.height * config.width
    throughput = (pixels / 1e6) / (total_mean / 1000)  # megapixels/sec

    # Get memory usage
    memory_mb = torch.cuda.max_memory_allocated() / (1024**2)

    return BenchmarkResult(
        config=asdict(config),
        forward_ms=forward_mean,
        forward_std=forward_std,
        backward_ms=backward_mean,
        backward_std=backward_std,
        total_ms=total_mean,
        throughput_mpix_sec=throughput,
        memory_mb=memory_mb,
        timestamp=datetime.now().isoformat(),
    )


def run_benchmark_suite(
    iterations: int = 100, warmup: int = 20, quick: bool = False
) -> list[BenchmarkResult]:
    """Run the full benchmark suite."""
    results = []

    # Define test configurations
    if quick:
        # Quick mode for testing
        sizes = [(512, 512), (1920, 1080)]
        batch_sizes = [1, 4]
        channels_list = [3]
        modes = ["fp32_train", "fp16_train"]
    else:
        # Full benchmark suite
        sizes = [
            (256, 256),
            (512, 512),
            (1280, 720),  # 720p
            (1920, 1080),  # 1080p
            (3840, 2160),  # 4K
        ]
        batch_sizes = [1, 2, 4, 8]
        channels_list = [1, 3]
        modes = ["fp32_train", "fp32_infer", "fp16_train", "fp16_infer"]

    paddings = ["same"]

    total_configs = len(sizes) * len(batch_sizes) * len(channels_list) * len(modes) * len(paddings)
    current = 0

    for height, width in sizes:
        for batch_size in batch_sizes:
            for channels in channels_list:
                for mode in modes:
                    for padding in paddings:
                        current += 1
                        config = BenchmarkConfig(
                            batch_size=batch_size,
                            channels=channels,
                            height=height,
                            width=width,
                            mode=mode,
                            padding=padding,
                            iterations=iterations,
                            warmup=warmup,
                        )

                        print(
                            f"[{current}/{total_configs}] "
                            f"{batch_size}x{channels}x{height}x{width} "
                            f"{mode} {padding}...",
                            end=" ",
                            flush=True,
                        )

                        try:
                            result = run_single_benchmark(config)
                            print(
                                f"{result.total_ms:.2f}ms ({result.throughput_mpix_sec:.0f} MPix/s)"
                            )
                            results.append(result)
                        except Exception as e:
                            print(f"FAILED: {e}")

                        # Clear memory between runs
                        torch.cuda.empty_cache()

    return results


def compare_results(current: list[BenchmarkResult], baseline: list[BenchmarkResult]) -> None:
    """Compare current results against baseline."""
    print("\n" + "=" * 80)
    print("COMPARISON: Current vs Baseline")
    print("=" * 80)

    # Create lookup for baseline results
    baseline_lookup = {}
    for r in baseline:
        key = (
            r.config["batch_size"],
            r.config["channels"],
            r.config["height"],
            r.config["width"],
            r.config["mode"],
            r.config["padding"],
        )
        baseline_lookup[key] = r

    print(f"{'Configuration':<40} {'Baseline':>12} {'Current':>12} {'Speedup':>10}")
    print("-" * 80)

    speedups = []
    for r in current:
        key = (
            r.config["batch_size"],
            r.config["channels"],
            r.config["height"],
            r.config["width"],
            r.config["mode"],
            r.config["padding"],
        )

        if key in baseline_lookup:
            b = baseline_lookup[key]
            speedup = b.total_ms / r.total_ms

            config_str = (
                f"{r.config['batch_size']}x{r.config['channels']}x"
                f"{r.config['height']}x{r.config['width']} "
                f"{r.config['mode']}"
            )

            color = "\033[92m" if speedup > 1.0 else "\033[91m"
            reset = "\033[0m"

            print(
                f"{config_str:<40} {b.total_ms:>10.2f}ms {r.total_ms:>10.2f}ms "
                f"{color}{speedup:>9.2f}x{reset}"
            )

            speedups.append(speedup)

    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        min_speedup = min(speedups)
        max_speedup = max(speedups)

        print("-" * 80)
        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Min speedup: {min_speedup:.2f}x")
        print(f"Max speedup: {max_speedup:.2f}x")


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print summary of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Group by mode
    modes = {}
    for r in results:
        mode = r.config["mode"]
        if mode not in modes:
            modes[mode] = []
        modes[mode].append(r)

    for mode, mode_results in sorted(modes.items()):
        print(f"\n{mode.upper()}:")
        print(f"{'Config':<35} {'Forward':>10} {'Backward':>10} {'Total':>10} {'Throughput':>12}")
        print("-" * 80)

        for r in sorted(
            mode_results,
            key=lambda x: (x.config["height"] * x.config["width"], x.config["batch_size"]),
        ):
            config_str = (
                f"{r.config['batch_size']}x{r.config['channels']}x"
                f"{r.config['height']}x{r.config['width']}"
            )
            print(
                f"{config_str:<35} {r.forward_ms:>8.2f}ms {r.backward_ms:>8.2f}ms "
                f"{r.total_ms:>8.2f}ms {r.throughput_mpix_sec:>10.0f} MP/s"
            )


def save_results(results: list[BenchmarkResult], filepath: Path) -> None:
    """Save results to JSON file."""
    data = {
        "gpu_info": get_gpu_info(),
        "timestamp": datetime.now().isoformat(),
        "results": [asdict(r) for r in results],
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {filepath}")


def load_results(filepath: Path) -> list[BenchmarkResult]:
    """Load results from JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    results = []
    for r in data["results"]:
        config = r.pop("config")
        results.append(BenchmarkResult(config=config, **r))

    return results


def main():
    parser = argparse.ArgumentParser(description="Detailed fused-ssim performance benchmarks")
    parser.add_argument(
        "--iterations", type=int, default=100, help="Number of benchmark iterations (default: 100)"
    )
    parser.add_argument(
        "--warmup", type=int, default=20, help="Number of warmup iterations (default: 20)"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmark with fewer configurations"
    )
    parser.add_argument(
        "--save", type=str, metavar="NAME", help="Save results to benchmarks/<NAME>_results.json"
    )
    parser.add_argument(
        "--compare", type=str, metavar="FILE", help="Compare against baseline results file"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        return 1

    # Print GPU info
    gpu_info = get_gpu_info()
    print("=" * 80)
    print("GPU INFORMATION")
    print("=" * 80)
    for key, value in gpu_info.items():
        print(f"  {key}: {value}")
    print("=" * 80)
    print()

    # Run benchmarks
    print("Running benchmarks...")
    results = run_benchmark_suite(
        iterations=args.iterations,
        warmup=args.warmup,
        quick=args.quick,
    )

    # Print summary
    print_summary(results)

    # Compare with baseline if provided
    if args.compare:
        baseline_path = Path(args.compare)
        if baseline_path.exists():
            baseline = load_results(baseline_path)
            compare_results(results, baseline)
        else:
            print(f"Warning: Baseline file not found: {baseline_path}")

    # Save results if requested
    if args.save:
        benchmarks_dir = Path(__file__).parent
        filepath = benchmarks_dir / f"{args.save}_results.json"
        save_results(results, filepath)

    return 0


if __name__ == "__main__":
    exit(main())
