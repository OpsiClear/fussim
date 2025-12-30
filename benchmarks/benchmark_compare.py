"""Benchmark comparison: optimized-fused-ssim vs pytorch-msssim vs original fused-ssim."""

import os
import sys

# Use local development version instead of installed package
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import torch  # noqa: E402

# Check CUDA availability
if not torch.cuda.is_available():
    print("CUDA not available!")
    sys.exit(1)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print()


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


def benchmark(func, img1, img2, iterations=100, warmup=20, backward=False):
    """Benchmark a function and return mean time in ms."""
    timer = CUDATimer()

    # Warmup
    for _ in range(warmup):
        if backward and img1.requires_grad:
            img1_w = img1.clone().requires_grad_(True)
            result = func(img1_w, img2)
            result.backward() if result.dim() == 0 else result.mean().backward()
        else:
            with torch.no_grad():
                func(img1, img2)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        if backward:
            img1_b = img1.clone().requires_grad_(True)
            timer.start()
            result = func(img1_b, img2)
            result.backward() if result.dim() == 0 else result.mean().backward()
            elapsed = timer.stop()
        else:
            timer.start()
            with torch.no_grad():
                result = func(img1, img2)
            elapsed = timer.stop()
        times.append(elapsed)

    return sum(times) / len(times)


def run_benchmarks():
    # Test configurations
    configs = [
        ("Small (1, 3, 512, 512)", (1, 3, 512, 512)),
        ("1080p (1, 3, 1080, 1920)", (1, 3, 1080, 1920)),
        ("Batch (5, 5, 1080, 1920)", (5, 5, 1080, 1920)),
        ("4K (1, 3, 2160, 3840)", (1, 3, 2160, 3840)),
    ]

    # Import implementations
    implementations = {}

    # 1. Our optimized implementation
    try:
        from fused_ssim import fused_ssim

        implementations["optimized-fused-ssim"] = lambda x, y: fused_ssim(
            x, y, padding="valid", data_range=1.0
        )
        print("[OK] optimized-fused-ssim loaded")

        # Try compat API if available
        try:
            from fused_ssim import ssim as compat_ssim

            implementations["optimized (compat API)"] = lambda x, y: compat_ssim(
                x, y, data_range=1.0
            )
            print("[OK] optimized-fused-ssim compat API loaded")
        except ImportError:
            pass
    except ImportError as e:
        print(f"[FAIL] optimized-fused-ssim not available: {e}")

    # 2. pytorch-msssim
    try:
        from pytorch_msssim import ssim as pm_ssim

        implementations["pytorch-msssim"] = lambda x, y: pm_ssim(x, y, data_range=1.0)
        print("[OK] pytorch-msssim loaded")
    except ImportError:
        print("[FAIL] pytorch-msssim not installed (pip install pytorch-msssim)")

    # 3. Original fused-ssim (if installed separately)
    # Note: This would conflict with our package, so we skip it
    # Users can test by installing original in a separate env

    if len(implementations) < 2:
        print("\nNeed at least 2 implementations to compare!")
        print("Install pytorch-msssim: pip install pytorch-msssim")
        return

    print()
    print("=" * 80)
    print("INFERENCE BENCHMARK (forward only, no gradients)")
    print("=" * 80)

    for config_name, shape in configs:
        print(f"\n{config_name}")
        print("-" * 60)

        torch.manual_seed(42)
        img1 = torch.rand(*shape, device="cuda")
        img2 = torch.rand(*shape, device="cuda")

        results = {}
        for name, func in implementations.items():
            try:
                time_ms = benchmark(func, img1, img2, iterations=100, warmup=20, backward=False)
                results[name] = time_ms
            except Exception as e:
                results[name] = None
                print(f"  {name}: ERROR - {e}")

        # Print results sorted by time
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            baseline = valid_results.get("pytorch-msssim", list(valid_results.values())[0])
            for name, time_ms in sorted(valid_results.items(), key=lambda x: x[1]):
                speedup = baseline / time_ms if baseline else 1.0
                print(f"  {name:25s}: {time_ms:8.3f} ms  ({speedup:.2f}x)")

    print()
    print("=" * 80)
    print("TRAINING BENCHMARK (forward + backward)")
    print("=" * 80)

    for config_name, shape in configs:
        print(f"\n{config_name}")
        print("-" * 60)

        torch.manual_seed(42)
        img1 = torch.rand(*shape, device="cuda", requires_grad=True)
        img2 = torch.rand(*shape, device="cuda")

        results = {}
        for name, func in implementations.items():
            try:
                time_ms = benchmark(func, img1, img2, iterations=50, warmup=10, backward=True)
                results[name] = time_ms
            except Exception as e:
                results[name] = None
                print(f"  {name}: ERROR - {e}")

        # Print results sorted by time
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            baseline = valid_results.get("pytorch-msssim", list(valid_results.values())[0])
            for name, time_ms in sorted(valid_results.items(), key=lambda x: x[1]):
                speedup = baseline / time_ms if baseline else 1.0
                print(f"  {name:25s}: {time_ms:8.3f} ms  ({speedup:.2f}x)")

    # FP16 benchmark
    print()
    print("=" * 80)
    print("FP16/AMP TRAINING BENCHMARK (forward + backward with autocast)")
    print("=" * 80)

    for config_name, shape in configs[:3]:  # Skip 4K for FP16 to save time
        print(f"\n{config_name}")
        print("-" * 60)

        torch.manual_seed(42)
        img1 = torch.rand(*shape, device="cuda", requires_grad=True)
        img2 = torch.rand(*shape, device="cuda")

        results = {}

        # Optimized FP16
        if "optimized-fused-ssim" in implementations:

            def opt_fp16(x, y):
                with torch.autocast(device_type="cuda"):
                    return fused_ssim(x, y, padding="valid", data_range=1.0)

            try:
                time_ms = benchmark(opt_fp16, img1, img2, iterations=50, warmup=10, backward=True)
                results["optimized-fused-ssim FP16"] = time_ms
            except Exception as e:
                print(f"  optimized-fused-ssim FP16: ERROR - {e}")

        # pytorch-msssim FP16
        if "pytorch-msssim" in implementations:

            def pm_fp16(x, y):
                with torch.autocast(device_type="cuda"):
                    return pm_ssim(x, y, data_range=1.0)

            try:
                time_ms = benchmark(pm_fp16, img1, img2, iterations=50, warmup=10, backward=True)
                results["pytorch-msssim FP16"] = time_ms
            except Exception as e:
                print(f"  pytorch-msssim FP16: ERROR - {e}")

        # Print results
        if results:
            baseline = results.get("pytorch-msssim FP16", list(results.values())[0])
            for name, time_ms in sorted(results.items(), key=lambda x: x[1]):
                speedup = baseline / time_ms if baseline else 1.0
                print(f"  {name:25s}: {time_ms:8.3f} ms  ({speedup:.2f}x)")


if __name__ == "__main__":
    run_benchmarks()
