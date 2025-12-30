"""Tests for multi-window-size SSIM support."""
import pytest
import torch

from fused_ssim import SUPPORTED_WINDOW_SIZES, fused_ssim


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestWindowSizeSupport:
    """Test window size configuration."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.img1 = torch.rand(1, 3, 64, 64, device="cuda")
        self.img2 = torch.rand(1, 3, 64, 64, device="cuda")

    def test_supported_sizes_are_7_9_11(self):
        """Verify the supported window sizes."""
        assert SUPPORTED_WINDOW_SIZES == (7, 9, 11)

    @pytest.mark.parametrize("window_size", SUPPORTED_WINDOW_SIZES)
    def test_forward_pass_all_sizes(self, window_size):
        """Forward pass works for all supported sizes."""
        result = fused_ssim(self.img1, self.img2, window_size=window_size)
        assert result.dim() == 0
        assert -1 <= result <= 1

    @pytest.mark.parametrize("window_size", SUPPORTED_WINDOW_SIZES)
    def test_identical_images_ssim_one(self, window_size):
        """SSIM of identical images should be ~1.0 for all sizes."""
        result = fused_ssim(self.img1, self.img1, window_size=window_size)
        assert torch.isclose(result, torch.tensor(1.0, device="cuda"), atol=1e-5)

    @pytest.mark.parametrize("window_size", SUPPORTED_WINDOW_SIZES)
    def test_gradients_work_all_sizes(self, window_size):
        """Gradients flow correctly for all window sizes."""
        img = self.img1.clone().requires_grad_(True)
        result = fused_ssim(img, self.img2, window_size=window_size)
        result.backward()
        assert img.grad is not None
        assert img.grad.shape == img.shape
        assert not torch.isnan(img.grad).any()
        assert not torch.isinf(img.grad).any()

    @pytest.mark.parametrize("window_size", SUPPORTED_WINDOW_SIZES)
    def test_fp16_autocast_all_sizes(self, window_size):
        """FP16 autocast works for all window sizes."""
        with torch.autocast(device_type="cuda"):
            result = fused_ssim(self.img1, self.img2, window_size=window_size)
        assert result.dtype == torch.float32  # Output is always FP32
        assert -1 <= result <= 1

    @pytest.mark.parametrize("window_size", SUPPORTED_WINDOW_SIZES)
    def test_fp16_training_all_sizes(self, window_size):
        """FP16 training with gradients works for all window sizes."""
        img = self.img1.clone().requires_grad_(True)
        with torch.autocast(device_type="cuda"):
            result = fused_ssim(img, self.img2, window_size=window_size)
        result.backward()
        assert img.grad is not None
        assert not torch.isnan(img.grad).any()

    def test_default_is_11(self):
        """Default window size should be 11."""
        # Call without window_size - should use 11 (default)
        result_default = fused_ssim(self.img1, self.img2)
        result_11 = fused_ssim(self.img1, self.img2, window_size=11)
        assert torch.isclose(result_default, result_11)

    def test_invalid_window_size_raises(self):
        """Unsupported window sizes should raise ValueError."""
        for invalid_size in [3, 5, 8, 10, 13, 15]:
            with pytest.raises(ValueError, match="window_size"):
                fused_ssim(self.img1, self.img2, window_size=invalid_size)

    def test_different_sizes_give_different_results(self):
        """Different window sizes should produce different SSIM values."""
        results = {}
        for ws in SUPPORTED_WINDOW_SIZES:
            results[ws] = fused_ssim(self.img1, self.img2, window_size=ws).item()

        # At least some should be different
        values = list(results.values())
        assert not all(abs(v - values[0]) < 1e-6 for v in values[1:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestWindowSizeEdgeCases:
    """Test edge cases for different window sizes."""

    @pytest.mark.parametrize("window_size", SUPPORTED_WINDOW_SIZES)
    def test_minimum_image_size(self, window_size):
        """Test with minimum valid image size for each window size."""
        # Minimum size is window_size for valid padding
        min_size = window_size + 10  # Add some margin
        img1 = torch.rand(1, 1, min_size, min_size, device="cuda")
        img2 = torch.rand(1, 1, min_size, min_size, device="cuda")
        result = fused_ssim(img1, img2, window_size=window_size, padding="same")
        assert -1 <= result <= 1

    @pytest.mark.parametrize("window_size", SUPPORTED_WINDOW_SIZES)
    def test_large_batch(self, window_size):
        """Test with larger batch sizes for each window size."""
        img1 = torch.rand(8, 3, 64, 64, device="cuda")
        img2 = torch.rand(8, 3, 64, 64, device="cuda")
        result = fused_ssim(img1, img2, window_size=window_size)
        assert -1 <= result <= 1

    @pytest.mark.parametrize("window_size", SUPPORTED_WINDOW_SIZES)
    def test_non_square_images(self, window_size):
        """Test with non-square images for each window size."""
        img1 = torch.rand(1, 3, 64, 128, device="cuda")
        img2 = torch.rand(1, 3, 64, 128, device="cuda")
        result = fused_ssim(img1, img2, window_size=window_size)
        assert -1 <= result <= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestWindowSizeNumericalStability:
    """Test numerical stability across window sizes."""

    @pytest.mark.parametrize("window_size", SUPPORTED_WINDOW_SIZES)
    def test_constant_images(self, window_size):
        """Constant images should have SSIM=1.0."""
        img = torch.ones(1, 3, 64, 64, device="cuda") * 0.5
        result = fused_ssim(img, img, window_size=window_size)
        assert torch.isclose(result, torch.tensor(1.0, device="cuda"), atol=1e-5)

    @pytest.mark.parametrize("window_size", SUPPORTED_WINDOW_SIZES)
    def test_near_zero_images(self, window_size):
        """Near-zero images should not cause numerical issues."""
        img1 = torch.ones(1, 3, 64, 64, device="cuda") * 1e-6
        img2 = torch.ones(1, 3, 64, 64, device="cuda") * 1e-6
        result = fused_ssim(img1, img2, window_size=window_size)
        assert not torch.isnan(result)
        assert not torch.isinf(result)

    @pytest.mark.parametrize("window_size", SUPPORTED_WINDOW_SIZES)
    def test_gradient_stability(self, window_size):
        """Gradients should be stable for all window sizes."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 3, 64, 64, device="cuda", requires_grad=True)
        img2 = torch.rand(1, 3, 64, 64, device="cuda")

        result = fused_ssim(img1, img2, window_size=window_size)
        result.backward()

        assert not torch.isnan(img1.grad).any()
        assert not torch.isinf(img1.grad).any()
        assert img1.grad.abs().max() < 100  # Reasonable gradient magnitude
