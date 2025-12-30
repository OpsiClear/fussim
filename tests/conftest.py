"""Shared test fixtures for fused-ssim tests."""

import pytest
import torch


@pytest.fixture
def device():
    """Get CUDA device, skip if not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def image_pair(device):
    """Create a pair of random test images."""
    torch.manual_seed(42)
    img1 = torch.rand(1, 3, 256, 256, device=device)
    img2 = torch.rand(1, 3, 256, 256, device=device)
    return img1, img2


@pytest.fixture
def identical_images(device):
    """Create identical image pair (SSIM should be 1.0)."""
    torch.manual_seed(42)
    img = torch.rand(1, 3, 256, 256, device=device)
    return img, img.clone()
