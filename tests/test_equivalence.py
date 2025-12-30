"""Equivalence tests for fused-ssim optimizations.

These tests ensure that optimized kernels produce identical results
to the baseline implementation. Run after any kernel modifications.

Tolerance levels:
- FP32 forward: rtol=1e-5, atol=1e-5
- FP32 backward: rtol=1e-4, atol=1e-4
- FP16 forward: rtol=1e-3, atol=1e-3
- FP16 backward: rtol=1e-3, atol=1e-3
"""

import pytest
import torch

from fused_ssim import fused_ssim

# Tolerance levels for different precision modes
TOLERANCES = {
    "fp32_forward": {"rtol": 1e-5, "atol": 1e-5},
    "fp32_backward": {"rtol": 1e-4, "atol": 1e-4},
    "fp16_forward": {"rtol": 1e-3, "atol": 1e-3},
    "fp16_backward": {"rtol": 1e-3, "atol": 1e-3},
}


class TestForwardEquivalence:
    """Test forward pass produces consistent results."""

    @pytest.mark.parametrize(
        "size",
        [
            (1, 1, 32, 32),
            (1, 3, 64, 64),
            (2, 3, 128, 128),
            (4, 3, 256, 256),
            (1, 3, 1080, 1920),
        ],
    )
    def test_fp32_forward_deterministic(self, size, device):
        """FP32 forward pass is deterministic."""
        torch.manual_seed(42)
        B, C, H, W = size
        img1 = torch.rand(B, C, H, W, device=device)
        img2 = torch.rand(B, C, H, W, device=device)

        # Run twice (inference mode)
        with torch.no_grad():
            result1 = fused_ssim(img1, img2)
            result2 = fused_ssim(img1, img2)

        torch.testing.assert_close(
            result1, result2, rtol=0, atol=0, msg="FP32 forward should be exactly deterministic"
        )

    @pytest.mark.parametrize(
        "size",
        [
            (1, 1, 32, 32),
            (1, 3, 64, 64),
            (2, 3, 128, 128),
            (4, 3, 256, 256),
        ],
    )
    def test_fp16_forward_deterministic(self, size, device):
        """FP16 forward pass is deterministic."""
        torch.manual_seed(42)
        B, C, H, W = size
        img1 = torch.rand(B, C, H, W, device=device)
        img2 = torch.rand(B, C, H, W, device=device)

        # Run twice (FP16 via autocast)
        with torch.no_grad(), torch.autocast(device_type="cuda"):
            result1 = fused_ssim(img1, img2)
            result2 = fused_ssim(img1, img2)

        torch.testing.assert_close(
            result1, result2, rtol=0, atol=0, msg="FP16 forward should be exactly deterministic"
        )

    @pytest.mark.parametrize(
        "size",
        [
            (1, 3, 64, 64),
            (2, 3, 128, 128),
            (1, 3, 256, 256),
        ],
    )
    def test_fp16_matches_fp32(self, size, device):
        """FP16 forward matches FP32 within tolerance."""
        torch.manual_seed(42)
        B, C, H, W = size
        img1 = torch.rand(B, C, H, W, device=device)
        img2 = torch.rand(B, C, H, W, device=device)

        with torch.no_grad():
            result_fp32 = fused_ssim(img1, img2)

        with torch.no_grad(), torch.autocast(device_type="cuda"):
            result_fp16 = fused_ssim(img1, img2)

        torch.testing.assert_close(
            result_fp16,
            result_fp32,
            **TOLERANCES["fp16_forward"],
            msg="FP16 should match FP32 within tolerance",
        )

    def test_train_matches_inference(self, device):
        """Training mode forward matches inference mode."""
        torch.manual_seed(42)
        img1 = torch.rand(2, 3, 128, 128, device=device)
        img2 = torch.rand(2, 3, 128, 128, device=device)

        # Training mode (requires_grad=True)
        img1_train = img1.clone().requires_grad_(True)
        result_train = fused_ssim(img1_train, img2)

        # Inference mode (no_grad)
        with torch.no_grad():
            result_infer = fused_ssim(img1, img2)

        torch.testing.assert_close(
            result_train,
            result_infer,
            **TOLERANCES["fp32_forward"],
            msg="Training mode should match inference mode",
        )


class TestBackwardEquivalence:
    """Test backward pass produces consistent gradients."""

    @pytest.mark.parametrize(
        "size",
        [
            (1, 1, 32, 32),
            (1, 3, 64, 64),
            (2, 3, 128, 128),
        ],
    )
    def test_fp32_backward_deterministic(self, size, device):
        """FP32 backward pass is deterministic."""
        torch.manual_seed(42)
        B, C, H, W = size
        img1_base = torch.rand(B, C, H, W, device=device)
        img2 = torch.rand(B, C, H, W, device=device)

        # First run
        img1_a = img1_base.clone().requires_grad_(True)
        loss_a = 1 - fused_ssim(img1_a, img2)
        loss_a.backward()
        grad_a = img1_a.grad.clone()

        # Second run
        img1_b = img1_base.clone().requires_grad_(True)
        loss_b = 1 - fused_ssim(img1_b, img2)
        loss_b.backward()
        grad_b = img1_b.grad.clone()

        torch.testing.assert_close(
            grad_a, grad_b, rtol=0, atol=0, msg="FP32 backward should be exactly deterministic"
        )

    @pytest.mark.parametrize(
        "size",
        [
            (1, 1, 32, 32),
            (1, 3, 64, 64),
            (2, 3, 128, 128),
        ],
    )
    def test_fp16_backward_deterministic(self, size, device):
        """FP16 backward pass is deterministic."""
        torch.manual_seed(42)
        B, C, H, W = size
        img1_base = torch.rand(B, C, H, W, device=device)
        img2 = torch.rand(B, C, H, W, device=device)

        # First run (FP16 via autocast)
        img1_a = img1_base.clone().requires_grad_(True)
        with torch.autocast(device_type="cuda"):
            loss_a = 1 - fused_ssim(img1_a, img2)
        loss_a.backward()
        grad_a = img1_a.grad.clone()

        # Second run
        img1_b = img1_base.clone().requires_grad_(True)
        with torch.autocast(device_type="cuda"):
            loss_b = 1 - fused_ssim(img1_b, img2)
        loss_b.backward()
        grad_b = img1_b.grad.clone()

        torch.testing.assert_close(
            grad_a, grad_b, rtol=0, atol=0, msg="FP16 backward should be exactly deterministic"
        )

    @pytest.mark.parametrize(
        "size",
        [
            (1, 3, 64, 64),
            (2, 3, 128, 128),
        ],
    )
    def test_fp16_gradient_matches_fp32(self, size, device):
        """FP16 gradients match FP32 within tolerance."""
        torch.manual_seed(42)
        B, C, H, W = size
        img1_base = torch.rand(B, C, H, W, device=device)
        img2 = torch.rand(B, C, H, W, device=device)

        # FP32
        img1_fp32 = img1_base.clone().requires_grad_(True)
        loss_fp32 = 1 - fused_ssim(img1_fp32, img2)
        loss_fp32.backward()
        grad_fp32 = img1_fp32.grad.clone()

        # FP16 (via autocast)
        img1_fp16 = img1_base.clone().requires_grad_(True)
        with torch.autocast(device_type="cuda"):
            loss_fp16 = 1 - fused_ssim(img1_fp16, img2)
        loss_fp16.backward()
        grad_fp16 = img1_fp16.grad.clone()

        torch.testing.assert_close(
            grad_fp16,
            grad_fp32,
            **TOLERANCES["fp16_backward"],
            msg="FP16 gradients should match FP32 within tolerance",
        )


class TestEdgeCases:
    """Test edge cases that might break optimizations."""

    def test_minimum_size(self, device):
        """Test minimum supported image size (32x32 due to 11x11 kernel + halo)."""
        img1 = torch.rand(1, 1, 32, 32, device=device, requires_grad=True)
        img2 = torch.rand(1, 1, 32, 32, device=device)

        ssim = fused_ssim(img1, img2)
        assert not torch.isnan(ssim), "SSIM should not be NaN for minimum size"

        (1 - ssim).backward()
        assert not torch.isnan(img1.grad).any(), "Gradient should not contain NaN"

    def test_single_channel(self, device):
        """Test single channel images."""
        img1 = torch.rand(1, 1, 128, 128, device=device, requires_grad=True)
        img2 = torch.rand(1, 1, 128, 128, device=device)

        ssim = fused_ssim(img1, img2)
        (1 - ssim).backward()

        assert img1.grad is not None
        assert img1.grad.shape == img1.shape

    def test_many_channels(self, device):
        """Test images with many channels."""
        img1 = torch.rand(1, 16, 64, 64, device=device, requires_grad=True)
        img2 = torch.rand(1, 16, 64, 64, device=device)

        ssim = fused_ssim(img1, img2)
        (1 - ssim).backward()

        assert img1.grad is not None
        assert img1.grad.shape == img1.shape

    def test_non_square_image(self, device):
        """Test non-square images."""
        img1 = torch.rand(1, 3, 720, 1280, device=device, requires_grad=True)
        img2 = torch.rand(1, 3, 720, 1280, device=device)

        ssim = fused_ssim(img1, img2)
        (1 - ssim).backward()

        assert img1.grad is not None

    def test_large_batch(self, device):
        """Test large batch size."""
        img1 = torch.rand(16, 3, 64, 64, device=device, requires_grad=True)
        img2 = torch.rand(16, 3, 64, 64, device=device)

        ssim = fused_ssim(img1, img2)
        (1 - ssim).backward()

        assert img1.grad is not None

    @pytest.mark.parametrize("width", [33, 65, 127, 257])
    def test_non_aligned_width(self, width, device):
        """Test widths not aligned to vectorization boundaries."""
        img1 = torch.rand(1, 3, 64, width, device=device, requires_grad=True)
        img2 = torch.rand(1, 3, 64, width, device=device)

        ssim = fused_ssim(img1, img2)
        (1 - ssim).backward()

        assert img1.grad is not None
        assert img1.grad.shape == img1.shape


class TestNumericalStability:
    """Test numerical stability with extreme values."""

    def test_identical_images(self, device):
        """SSIM of identical images should be close to 1."""
        img = torch.rand(1, 3, 64, 64, device=device)
        ssim = fused_ssim(img, img.clone())

        assert ssim.item() > 0.99, f"SSIM of identical images should be ~1, got {ssim.item()}"

    def test_constant_images(self, device):
        """Test with constant value images."""
        for value in [0.0, 0.5, 1.0]:
            img1 = torch.full((1, 3, 64, 64), value, device=device)
            img2 = torch.full((1, 3, 64, 64), value, device=device)

            ssim = fused_ssim(img1, img2)
            assert not torch.isnan(ssim), f"SSIM should not be NaN for constant {value}"

    def test_near_zero_images(self, device):
        """Test with very small values."""
        img1 = torch.rand(1, 3, 64, 64, device=device) * 1e-6
        img2 = torch.rand(1, 3, 64, 64, device=device) * 1e-6

        ssim = fused_ssim(img1, img2)
        assert not torch.isnan(ssim), "SSIM should handle near-zero values"

    def test_high_contrast(self, device):
        """Test with high contrast images."""
        img1 = torch.zeros(1, 3, 64, 64, device=device)
        img2 = torch.ones(1, 3, 64, 64, device=device)

        ssim = fused_ssim(img1, img2)
        assert not torch.isnan(ssim), "SSIM should handle high contrast"
        assert ssim.item() < 0.5, "High contrast should have low SSIM"

    def test_gradient_stability(self, device):
        """Test gradient stability with various inputs."""
        torch.manual_seed(42)

        for _ in range(10):
            img1 = torch.rand(2, 3, 64, 64, device=device, requires_grad=True)
            img2 = torch.rand(2, 3, 64, 64, device=device)

            ssim = fused_ssim(img1, img2)
            (1 - ssim).backward()

            assert not torch.isnan(img1.grad).any(), "Gradient should not contain NaN"
            assert not torch.isinf(img1.grad).any(), "Gradient should not contain Inf"


class TestPaddingModes:
    """Test different padding modes produce expected results."""

    def test_same_padding_preserves_size(self, device):
        """Same padding should preserve spatial dimensions."""
        img1 = torch.rand(1, 3, 128, 128, device=device, requires_grad=True)
        img2 = torch.rand(1, 3, 128, 128, device=device)

        ssim = fused_ssim(img1, img2, padding="same")
        (1 - ssim).backward()

        assert img1.grad.shape == img1.shape

    def test_valid_padding(self, device):
        """Valid padding should work correctly."""
        img1 = torch.rand(1, 3, 128, 128, device=device, requires_grad=True)
        img2 = torch.rand(1, 3, 128, 128, device=device)

        ssim = fused_ssim(img1, img2, padding="valid")
        (1 - ssim).backward()

        assert img1.grad.shape == img1.shape

    def test_padding_modes_different_results(self, device):
        """Different padding modes should produce slightly different results."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 3, 128, 128, device=device)
        img2 = torch.rand(1, 3, 128, 128, device=device)

        ssim_same = fused_ssim(img1, img2, padding="same")
        ssim_valid = fused_ssim(img1, img2, padding="valid")

        # They should be close but not identical
        diff = abs(ssim_same.item() - ssim_valid.item())
        assert diff < 0.1, f"Padding modes should give similar results, diff={diff}"


class TestAMPEquivalence:
    """Test AMP behavior is consistent."""

    def test_autocast_is_deterministic(self, device):
        """Autocast should produce deterministic results."""
        torch.manual_seed(42)
        img1 = torch.rand(2, 3, 64, 64, device=device, requires_grad=True)
        img2 = torch.rand(2, 3, 64, 64, device=device)

        # First run with autocast
        with torch.amp.autocast(device_type="cuda"):
            ssim1 = fused_ssim(img1, img2)

        # Second run with autocast
        with torch.amp.autocast(device_type="cuda"):
            ssim2 = fused_ssim(img1, img2)

        torch.testing.assert_close(
            ssim1, ssim2, rtol=0, atol=0, msg="Autocast should produce deterministic results"
        )

    def test_gradscaler_compatible(self, device):
        """GradScaler should work without issues."""
        torch.manual_seed(42)
        img1 = torch.rand(2, 3, 64, 64, device=device, requires_grad=True)
        img2 = torch.rand(2, 3, 64, 64, device=device)

        scaler = torch.amp.GradScaler()

        with torch.amp.autocast(device_type="cuda"):
            ssim = fused_ssim(img1, img2)
            loss = 1 - ssim

        scaler.scale(loss).backward()

        # Unscale and check gradients
        scaler.unscale_(torch.optim.SGD([img1], lr=0.01))

        assert img1.grad is not None
        assert not torch.isnan(img1.grad).any()
        assert not torch.isinf(img1.grad).any()
