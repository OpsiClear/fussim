"""Correctness tests for fused-ssim.

Tests validate that fused_ssim produces correct results compared to
reference implementations.
"""

from math import exp

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fussim import fussim

# Reference implementation from pytorch-ssim
# https://github.com/Po-Hsun-Su/pytorch-ssim


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
    """Reference SSIM implementation for validation."""
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


class TestSSIMCorrectness:
    """Test SSIM correctness against reference implementation."""

    def test_identical_images(self, identical_images):
        """SSIM of identical images should be close to 1.0."""
        img1, img2 = identical_images
        ssim_val = fused_ssim(img1, img2)
        assert ssim_val.item() > 0.99, f"Expected SSIM > 0.99, got {ssim_val.item()}"

    def test_matches_reference_same_padding(self, device):
        """fused_ssim with padding='same' should match reference."""
        torch.manual_seed(0)
        img1 = nn.Parameter(torch.rand(5, 5, 1080, 1920, device=device))
        img2 = torch.rand(5, 5, 1080, 1920, device=device)

        ref_ssim = reference_ssim(img1, img2)
        fused_ssim_val = fused_ssim(img1, img2, padding="same")

        assert torch.isclose(ref_ssim, fused_ssim_val), (
            f"Reference: {ref_ssim.item()}, Fused: {fused_ssim_val.item()}"
        )

    def test_matches_pytorch_msssim_valid_padding(self, device):
        """fused_ssim with padding='valid' should match pytorch-msssim."""
        pytest.importorskip("pytorch_msssim")
        from pytorch_msssim import SSIM

        torch.manual_seed(0)
        img1 = nn.Parameter(torch.rand(5, 5, 1080, 1920, device=device))
        img2 = torch.rand(5, 5, 1080, 1920, device=device)

        pm_ssim = SSIM(data_range=1.0, channel=5)
        pm_val = pm_ssim(img1, img2)
        fused_val = fused_ssim(img1, img2, padding="valid")

        assert torch.isclose(pm_val, fused_val), (
            f"pytorch-msssim: {pm_val.item()}, Fused: {fused_val.item()}"
        )


class TestSSIMGradients:
    """Test SSIM gradient computation."""

    def test_gradient_matches_reference(self, device):
        """Gradients should match reference implementation."""
        torch.manual_seed(0)

        # Reference
        img1_ref = nn.Parameter(torch.rand(2, 3, 256, 256, device=device))
        img2 = torch.rand(2, 3, 256, 256, device=device)
        ref_ssim = reference_ssim(img1_ref, img2)
        ref_ssim.backward()

        # Fused
        img1_fused = nn.Parameter(img1_ref.data.clone())
        fused_val = fused_ssim(img1_fused, img2, padding="same")
        fused_val.backward()

        assert torch.allclose(img1_ref.grad, img1_fused.grad, rtol=1e-4, atol=1e-4), (
            "Gradients do not match reference"
        )

    def test_gradient_matches_pytorch_msssim(self, device):
        """Gradients should match pytorch-msssim for valid padding."""
        pytest.importorskip("pytorch_msssim")
        from pytorch_msssim import SSIM

        torch.manual_seed(0)
        pm_ssim = SSIM(data_range=1.0, channel=3)

        # pytorch-msssim
        img1_pm = nn.Parameter(torch.rand(2, 3, 256, 256, device=device))
        img2 = torch.rand(2, 3, 256, 256, device=device)
        pm_val = pm_ssim(img1_pm, img2)
        pm_val.backward()

        # Fused
        img1_fused = nn.Parameter(img1_pm.data.clone())
        fused_val = fused_ssim(img1_fused, img2, padding="valid")
        fused_val.backward()

        assert torch.allclose(img1_pm.grad, img1_fused.grad, rtol=1e-4, atol=1e-4), (
            "Gradients do not match pytorch-msssim"
        )


class TestSSIMInputValidation:
    """Test input validation."""

    def test_rejects_cpu_tensors(self):
        """Should raise error for CPU tensors."""
        img1 = torch.rand(1, 3, 64, 64)
        img2 = torch.rand(1, 3, 64, 64)

        with pytest.raises(RuntimeError, match="CUDA"):
            fused_ssim(img1, img2)

    def test_rejects_mismatched_shapes(self, device):
        """Should raise error for mismatched shapes."""
        img1 = torch.rand(1, 3, 64, 64, device=device)
        img2 = torch.rand(1, 3, 128, 128, device=device)

        with pytest.raises(ValueError, match="shape"):
            fused_ssim(img1, img2)

    def test_rejects_invalid_padding(self, device):
        """Should raise error for invalid padding mode."""
        img1 = torch.rand(1, 3, 64, 64, device=device)
        img2 = torch.rand(1, 3, 64, 64, device=device)

        with pytest.raises(ValueError, match="padding"):
            fused_ssim(img1, img2, padding="invalid")

    def test_rejects_non_4d_tensors(self, device):
        """Should raise error for non-4D tensors."""
        img1 = torch.rand(3, 64, 64, device=device)
        img2 = torch.rand(3, 64, 64, device=device)

        with pytest.raises(ValueError, match="4D"):
            fused_ssim(img1, img2)


class TestSSIMInferenceMode:
    """Test inference mode (torch.no_grad())."""

    def test_inference_mode_produces_same_result(self, image_pair):
        """Inference mode should produce same SSIM value."""
        img1, img2 = image_pair

        # Training mode (with gradients)
        img1_train = img1.clone().requires_grad_(True)
        train_val = fused_ssim(img1_train, img2)

        # Inference mode (no gradients)
        with torch.no_grad():
            infer_val = fused_ssim(img1, img2)

        assert torch.isclose(train_val, infer_val), (
            f"Train: {train_val.item()}, Inference: {infer_val.item()}"
        )


class TestSSIMTrainParameter:
    """Test train parameter (matches original fused-ssim interface)."""

    def test_train_true_computes_gradients(self, device):
        """train=True should allow gradient computation."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 3, 64, 64, device=device, requires_grad=True)
        img2 = torch.rand(1, 3, 64, 64, device=device)
        ssim_val = fused_ssim(img1, img2, train=True)
        ssim_val.backward()
        assert img1.grad is not None

    def test_train_false_inference_mode(self, device):
        """train=False is for inference (use torch.no_grad() for best results)."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 3, 64, 64, device=device)
        img2 = torch.rand(1, 3, 64, 64, device=device)
        # train=False should work for inference without errors
        with torch.no_grad():
            ssim_val = fused_ssim(img1, img2, train=False)
        assert -1 <= ssim_val <= 1

    def test_returns_scalar_mean(self, device):
        """fused_ssim should always return scalar (mean) - matching original interface."""
        torch.manual_seed(42)
        img1 = torch.rand(2, 3, 64, 64, device=device)
        img2 = torch.rand(2, 3, 64, 64, device=device)
        ssim_val = fused_ssim(img1, img2)
        assert ssim_val.dim() == 0, f"Expected scalar, got shape {ssim_val.shape}"
