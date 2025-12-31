"""Tests for pytorch-msssim compatible API.

These tests verify that the ssim() function and SSIM class provide
a drop-in replacement for pytorch-msssim.
"""

import pytest
import torch

from fussim import SSIM, ssim


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSSIMFunction:
    """Test ssim() function - pytorch-msssim compatible."""

    def test_basic_usage(self):
        """Basic ssim() call with default parameters."""
        X = torch.rand(2, 3, 64, 64, device="cuda")
        Y = torch.rand(2, 3, 64, 64, device="cuda")
        result = ssim(X, Y, data_range=1.0)
        assert result.dim() == 0  # scalar
        assert 0 <= result <= 1

    def test_identical_images(self):
        """SSIM of identical images should be ~1.0."""
        X = torch.rand(1, 3, 64, 64, device="cuda")
        result = ssim(X, X, data_range=1.0)
        assert result.item() > 0.99

    def test_size_average_true(self):
        """size_average=True returns scalar."""
        X = torch.rand(4, 3, 64, 64, device="cuda")
        Y = torch.rand(4, 3, 64, 64, device="cuda")
        result = ssim(X, Y, data_range=1.0, size_average=True)
        assert result.dim() == 0

    def test_size_average_false(self):
        """size_average=False returns per-batch values."""
        X = torch.rand(4, 3, 64, 64, device="cuda")
        Y = torch.rand(4, 3, 64, 64, device="cuda")
        result = ssim(X, Y, data_range=1.0, size_average=False)
        assert result.shape == (4,)

    def test_data_range_255(self):
        """Test with data_range=255 (default)."""
        X = torch.rand(1, 3, 64, 64, device="cuda") * 255
        Y = torch.rand(1, 3, 64, 64, device="cuda") * 255
        result = ssim(X, Y, data_range=255)
        assert -1 <= result <= 1  # SSIM can be negative for dissimilar images

    def test_nonnegative_ssim(self):
        """nonnegative_ssim=True clamps negative values."""
        # Create images that might produce negative SSIM
        X = torch.rand(1, 1, 64, 64, device="cuda")
        Y = 1 - X  # Inverted image
        result = ssim(X, Y, data_range=1.0, nonnegative_ssim=True)
        assert result >= 0

    def test_gradient_flow(self):
        """Gradients flow through ssim()."""
        X = torch.rand(1, 3, 64, 64, device="cuda", requires_grad=True)
        Y = torch.rand(1, 3, 64, 64, device="cuda")
        result = ssim(X, Y, data_range=1.0)
        result.backward()
        assert X.grad is not None
        assert X.grad.shape == X.shape

    def test_invalid_win_size(self):
        """Unsupported win_size raises ValueError."""
        X = torch.rand(1, 3, 64, 64, device="cuda")
        Y = torch.rand(1, 3, 64, 64, device="cuda")
        with pytest.raises(ValueError, match="win_size"):
            ssim(X, Y, win_size=5)  # Only 7, 9, 11 are supported

    def test_invalid_win_sigma(self):
        """Non-1.5 win_sigma raises ValueError."""
        X = torch.rand(1, 3, 64, 64, device="cuda")
        Y = torch.rand(1, 3, 64, 64, device="cuda")
        with pytest.raises(ValueError, match="win_sigma"):
            ssim(X, Y, win_sigma=1.0)

    def test_custom_window_rejected(self):
        """Custom window tensor raises ValueError."""
        X = torch.rand(1, 3, 64, 64, device="cuda")
        Y = torch.rand(1, 3, 64, 64, device="cuda")
        win = torch.ones(1, 1, 11, 11, device="cuda")
        with pytest.raises(ValueError, match="win"):
            ssim(X, Y, win=win)

    def test_custom_K(self):
        """Custom K values should work."""
        X = torch.rand(1, 3, 64, 64, device="cuda")
        Y = torch.rand(1, 3, 64, 64, device="cuda")
        # Custom K values within valid range should work
        result = ssim(X, Y, K=(0.02, 0.06))
        assert -1 <= result <= 1

    def test_invalid_K(self):
        """Invalid K raises ValueError."""
        X = torch.rand(1, 3, 64, 64, device="cuda")
        Y = torch.rand(1, 3, 64, 64, device="cuda")
        # K values outside (0, 1) should raise error
        with pytest.raises(ValueError, match="K"):
            ssim(X, Y, K=(0, 0.03))  # K1=0 is invalid
        with pytest.raises(ValueError, match="K"):
            ssim(X, Y, K=(0.01, 1.5))  # K2=1.5 is invalid


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSSIMModule:
    """Test SSIM class - pytorch-msssim compatible."""

    def test_basic_usage(self):
        """Basic SSIM module usage."""
        ssim_module = SSIM(data_range=1.0)
        X = torch.rand(2, 3, 64, 64, device="cuda")
        Y = torch.rand(2, 3, 64, 64, device="cuda")
        result = ssim_module(X, Y)
        assert result.dim() == 0
        assert -1 <= result <= 1  # SSIM ranges from -1 to 1

    def test_as_loss(self):
        """SSIM module used as loss function."""
        ssim_module = SSIM(data_range=1.0)
        X = torch.rand(1, 3, 64, 64, device="cuda", requires_grad=True)
        Y = torch.rand(1, 3, 64, 64, device="cuda")
        loss = 1 - ssim_module(X, Y)
        loss.backward()
        assert X.grad is not None

    def test_size_average_false(self):
        """SSIM module with size_average=False."""
        ssim_module = SSIM(data_range=1.0, size_average=False)
        X = torch.rand(4, 3, 64, 64, device="cuda")
        Y = torch.rand(4, 3, 64, 64, device="cuda")
        result = ssim_module(X, Y)
        assert result.shape == (4,)

    def test_nonnegative(self):
        """SSIM module with nonnegative_ssim=True."""
        ssim_module = SSIM(data_range=1.0, nonnegative_ssim=True)
        X = torch.rand(1, 1, 64, 64, device="cuda")
        Y = 1 - X
        result = ssim_module(X, Y)
        assert result >= 0

    def test_channel_ignored(self):
        """channel parameter is accepted but ignored."""
        # Should not raise - channel is for compatibility only
        ssim_module = SSIM(data_range=1.0, channel=5)
        X = torch.rand(1, 3, 64, 64, device="cuda")  # 3 channels, not 5
        Y = torch.rand(1, 3, 64, 64, device="cuda")
        result = ssim_module(X, Y)
        assert -1 <= result <= 1  # SSIM ranges from -1 to 1

    def test_invalid_spatial_dims(self):
        """Non-2D spatial_dims raises ValueError."""
        with pytest.raises(ValueError, match="spatial_dims"):
            SSIM(spatial_dims=3)

    def test_invalid_win_size(self):
        """Unsupported win_size raises ValueError."""
        with pytest.raises(ValueError, match="win_size"):
            SSIM(win_size=5)  # Only 7, 9, 11 are supported


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPytorchMSSSIMCompatibility:
    """Test actual compatibility with pytorch-msssim if installed."""

    def test_matches_pytorch_msssim_values(self):
        """Values should match pytorch-msssim."""
        pytest.importorskip("pytorch_msssim")
        from pytorch_msssim import ssim as pm_ssim

        torch.manual_seed(42)
        X = torch.rand(2, 3, 128, 128, device="cuda")
        Y = torch.rand(2, 3, 128, 128, device="cuda")

        our_result = ssim(X, Y, data_range=1.0)
        pm_result = pm_ssim(X, Y, data_range=1.0)

        assert torch.isclose(our_result, pm_result, rtol=1e-4), (
            f"Ours: {our_result.item()}, pytorch-msssim: {pm_result.item()}"
        )

    def test_matches_pytorch_msssim_gradients(self):
        """Gradients should match pytorch-msssim."""
        pytest.importorskip("pytorch_msssim")
        from pytorch_msssim import ssim as pm_ssim

        torch.manual_seed(42)

        # Our implementation
        X1 = torch.rand(2, 3, 128, 128, device="cuda", requires_grad=True)
        Y = torch.rand(2, 3, 128, 128, device="cuda")
        our_result = ssim(X1, Y, data_range=1.0)
        our_result.backward()

        # pytorch-msssim
        X2 = X1.data.clone().requires_grad_(True)
        pm_result = pm_ssim(X2, Y, data_range=1.0)
        pm_result.backward()

        assert torch.allclose(X1.grad, X2.grad, rtol=1e-4, atol=1e-4), (
            "Gradients don't match pytorch-msssim"
        )

    def test_module_matches_pytorch_msssim(self):
        """SSIM module should match pytorch_msssim.SSIM."""
        pytest.importorskip("pytorch_msssim")
        from pytorch_msssim import SSIM as PM_SSIM

        torch.manual_seed(42)
        X = torch.rand(2, 3, 128, 128, device="cuda")
        Y = torch.rand(2, 3, 128, 128, device="cuda")

        our_module = SSIM(data_range=1.0)
        pm_module = PM_SSIM(data_range=1.0, channel=3)

        our_result = our_module(X, Y)
        pm_result = pm_module(X, Y)

        assert torch.isclose(our_result, pm_result, rtol=1e-4)
