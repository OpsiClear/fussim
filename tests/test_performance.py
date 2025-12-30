"""Performance regression tests for fused-ssim.

These tests ensure that kernel optimizations don't cause significant
performance regressions. They run a subset of benchmarks and compare
against minimum expected throughput thresholds.

Note: These tests are GPU-dependent and may need threshold adjustments
for different GPU architectures.
"""

import pytest
import torch


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


def benchmark_throughput(func, img1, img2, iterations=50, warmup=10, **kwargs):
    """Measure throughput in megapixels per second."""
    timer = CUDATimer()

    # Warmup
    for _ in range(warmup):
        result = func(img1, img2, **kwargs)
        if img1.requires_grad:
            result.backward()
            img1.grad = None
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        timer.start()
        result = func(img1, img2, **kwargs)
        if img1.requires_grad:
            result.backward()
            img1.grad = None
        elapsed = timer.stop()
        times.append(elapsed)

    mean_ms = sum(times) / len(times)
    pixels = img1.shape[0] * img1.shape[1] * img1.shape[2] * img1.shape[3]
    throughput_mpix_sec = (pixels / 1e6) / (mean_ms / 1000)

    return throughput_mpix_sec, mean_ms


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPerformanceRegression:
    """Performance regression tests.

    Thresholds are set conservatively to avoid flaky tests while still
    catching significant regressions. Based on RTX 4090 baseline.
    """

    def test_forward_throughput_small(self):
        """Forward pass throughput for small images."""
        from fused_ssim import fused_ssim

        torch.manual_seed(42)
        img1 = torch.rand(1, 3, 512, 512, device="cuda")
        img2 = torch.rand(1, 3, 512, 512, device="cuda")

        # Inference mode via no_grad
        with torch.no_grad():
            throughput, _ = benchmark_throughput(
                fused_ssim, img1, img2, iterations=50, warmup=10, padding="same"
            )

        # Minimum expected: 20000 MPix/s (conservative threshold)
        assert throughput > 20000, (
            f"Forward throughput {throughput:.0f} MPix/s below threshold 20000"
        )

    def test_forward_throughput_large(self):
        """Forward pass throughput for large images."""
        from fused_ssim import fused_ssim

        torch.manual_seed(42)
        img1 = torch.rand(1, 3, 1920, 1080, device="cuda")
        img2 = torch.rand(1, 3, 1920, 1080, device="cuda")

        # Inference mode via no_grad
        with torch.no_grad():
            throughput, _ = benchmark_throughput(
                fused_ssim, img1, img2, iterations=50, warmup=10, padding="same"
            )

        # Minimum expected: 30000 MPix/s
        assert throughput > 30000, (
            f"Forward throughput {throughput:.0f} MPix/s below threshold 30000"
        )

    def test_training_throughput_small(self):
        """Training (forward + backward) throughput for small images."""
        from fused_ssim import fused_ssim

        torch.manual_seed(42)
        img1 = torch.rand(1, 3, 512, 512, device="cuda", requires_grad=True)
        img2 = torch.rand(1, 3, 512, 512, device="cuda")

        throughput, _ = benchmark_throughput(
            fused_ssim, img1, img2, iterations=50, warmup=10, padding="same"
        )

        # Minimum expected: 8000 MPix/s (training is slower)
        assert throughput > 8000, (
            f"Training throughput {throughput:.0f} MPix/s below threshold 8000"
        )

    def test_training_throughput_large(self):
        """Training (forward + backward) throughput for large images."""
        from fused_ssim import fused_ssim

        torch.manual_seed(42)
        img1 = torch.rand(1, 3, 1920, 1080, device="cuda", requires_grad=True)
        img2 = torch.rand(1, 3, 1920, 1080, device="cuda")

        throughput, _ = benchmark_throughput(
            fused_ssim, img1, img2, iterations=30, warmup=5, padding="same"
        )

        # Minimum expected: 8000 MPix/s
        assert throughput > 8000, (
            f"Training throughput {throughput:.0f} MPix/s below threshold 8000"
        )

    def test_fp16_throughput(self):
        """FP16 mode (via autocast) should not be slower than FP32."""
        from fused_ssim import fused_ssim

        torch.manual_seed(42)
        img1 = torch.rand(4, 3, 512, 512, device="cuda", requires_grad=True)
        img2 = torch.rand(4, 3, 512, 512, device="cuda")

        # FP32 throughput
        fp32_throughput, _ = benchmark_throughput(
            fused_ssim, img1, img2, iterations=30, warmup=10, padding="same"
        )

        # Reset grad
        img1.grad = None

        # FP16 throughput via autocast wrapper
        def fused_ssim_fp16(img1, img2, **kwargs):
            with torch.autocast(device_type="cuda"):
                return fused_ssim(img1, img2, **kwargs)

        fp16_throughput, _ = benchmark_throughput(
            fused_ssim_fp16, img1, img2, iterations=30, warmup=10, padding="same"
        )

        # FP16 should be at least 90% of FP32 speed
        ratio = fp16_throughput / fp32_throughput
        assert ratio >= 0.9, (
            f"FP16 throughput ({fp16_throughput:.0f}) should be >= 90% of FP32 ({fp32_throughput:.0f}), got {ratio:.2f}"
        )

    def test_batch_scaling(self):
        """Throughput should scale reasonably with batch size."""
        from fused_ssim import fused_ssim

        torch.manual_seed(42)

        # Batch size 1 (inference mode)
        img1_b1 = torch.rand(1, 3, 512, 512, device="cuda")
        img2_b1 = torch.rand(1, 3, 512, 512, device="cuda")
        with torch.no_grad():
            throughput_b1, _ = benchmark_throughput(
                fused_ssim, img1_b1, img2_b1, iterations=30, warmup=10, padding="same"
            )

        # Batch size 4 (inference mode)
        img1_b4 = torch.rand(4, 3, 512, 512, device="cuda")
        img2_b4 = torch.rand(4, 3, 512, 512, device="cuda")
        with torch.no_grad():
            throughput_b4, _ = benchmark_throughput(
                fused_ssim, img1_b4, img2_b4, iterations=30, warmup=10, padding="same"
            )

        # Batch 4 should have at least 80% of batch 1's per-image throughput
        # (accounting for some overhead)
        ratio = throughput_b4 / throughput_b1
        assert ratio >= 0.8, (
            f"Batch 4 throughput ({throughput_b4:.0f}) should scale well from batch 1 ({throughput_b1:.0f}), got ratio {ratio:.2f}"
        )
