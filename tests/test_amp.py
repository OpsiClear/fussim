"""Tests for Torch AMP (Automatic Mixed Precision) compatibility.

Tests validate that fused_ssim works correctly with torch.cuda.amp.autocast()
and GradScaler for mixed precision training.
"""

import torch
import torch.nn as nn

from fused_ssim import fused_ssim


class TestAMPAutocast:
    """Test torch.cuda.amp.autocast() compatibility."""

    def test_autocast_forward(self, device):
        """Test that forward pass works under autocast."""
        torch.manual_seed(42)
        img1 = torch.rand(2, 3, 64, 64, device=device, requires_grad=True)
        img2 = torch.rand(2, 3, 64, 64, device=device)

        # Without autocast (FP32)
        ssim_fp32 = fused_ssim(img1, img2)

        # With autocast (FP16 path)
        with torch.amp.autocast(device_type="cuda"):
            ssim_fp16 = fused_ssim(img1, img2)

        # FP16 should be close to FP32
        assert torch.isclose(ssim_fp32, ssim_fp16, rtol=2e-2), (
            f"FP32: {ssim_fp32.item()}, FP16: {ssim_fp16.item()}"
        )

    def test_autocast_backward(self, device):
        """Test that backward pass works under autocast."""
        torch.manual_seed(42)
        img1 = torch.rand(2, 3, 64, 64, device=device, requires_grad=True)
        img2 = torch.rand(2, 3, 64, 64, device=device)

        with torch.amp.autocast(device_type="cuda"):
            ssim = fused_ssim(img1, img2)
            loss = 1 - ssim

        # Backward should work without issues
        loss.backward()

        assert img1.grad is not None, "Gradient should be computed"
        assert not torch.isnan(img1.grad).any(), "Gradient should not contain NaN"
        assert img1.grad.dtype == torch.float32, "Gradient should be FP32"


class TestAMPGradScaler:
    """Test torch.cuda.amp.GradScaler compatibility."""

    def test_gradscaler_training_loop(self, device):
        """Test complete training loop with GradScaler."""
        torch.manual_seed(42)

        # Simulate a simple model that outputs images
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid(),
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scaler = torch.amp.GradScaler()

        target = torch.rand(2, 3, 64, 64, device=device)
        input_img = torch.rand(2, 3, 64, 64, device=device)

        # Training loop
        for _ in range(3):
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                output = model(input_img)
                ssim = fused_ssim(output, target)
                loss = 1 - ssim

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Verify model still works
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda"):
                final_output = model(input_img)
                final_ssim = fused_ssim(final_output, target)

        assert final_ssim.item() > 0, "SSIM should be positive"
        assert not torch.isnan(final_ssim), "SSIM should not be NaN"


class TestFP16Mode:
    """Test FP16 mode via autocast."""

    def test_fp16_inference(self, device):
        """Test FP16 inference mode via autocast."""
        torch.manual_seed(42)
        img1 = torch.rand(2, 3, 64, 64, device=device)
        img2 = torch.rand(2, 3, 64, 64, device=device)

        # FP32 inference
        with torch.no_grad():
            ssim_fp32 = fused_ssim(img1, img2)

        # FP16 inference via autocast
        with torch.no_grad(), torch.autocast(device_type="cuda"):
            ssim_fp16 = fused_ssim(img1, img2)

        # FP16 precision allows for ~2% relative difference
        assert torch.isclose(ssim_fp32, ssim_fp16, rtol=2e-2), (
            f"FP32: {ssim_fp32.item()}, FP16: {ssim_fp16.item()}"
        )

    def test_fp16_training(self, device):
        """Test FP16 training mode with gradients via autocast."""
        torch.manual_seed(42)
        img1 = torch.rand(2, 3, 64, 64, device=device, requires_grad=True)
        img2 = torch.rand(2, 3, 64, 64, device=device)

        with torch.autocast(device_type="cuda"):
            ssim = fused_ssim(img1, img2)
            loss = 1 - ssim

        loss.backward()

        assert img1.grad is not None, "Gradient should be computed"
        assert not torch.isnan(img1.grad).any(), "Gradient should not contain NaN"

    def test_fp16_gradient_matches_fp32(self, device):
        """Test that FP16 gradients approximately match FP32."""
        torch.manual_seed(42)

        # FP32 gradient
        img1_fp32 = torch.rand(2, 3, 64, 64, device=device, requires_grad=True)
        img2 = torch.rand(2, 3, 64, 64, device=device)
        ssim_fp32 = fused_ssim(img1_fp32, img2)
        (1 - ssim_fp32).backward()
        grad_fp32 = img1_fp32.grad.clone()

        # FP16 gradient via autocast
        img1_fp16 = img1_fp32.data.clone().requires_grad_(True)
        with torch.autocast(device_type="cuda"):
            ssim_fp16 = fused_ssim(img1_fp16, img2)
        (1 - ssim_fp16).backward()
        grad_fp16 = img1_fp16.grad.clone()

        # Gradients should be close (allowing for FP16 precision)
        max_diff = (grad_fp32 - grad_fp16).abs().max().item()
        assert max_diff < 0.01, f"Max gradient difference: {max_diff}"


class TestRealWorldExamples:
    """Test real-world usage patterns."""

    def test_image_reconstruction_training(self, device):
        """Simulate image reconstruction training loop."""
        torch.manual_seed(42)

        # Simple autoencoder-like model
        encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        ).to(device)

        decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        ).to(device)

        params = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=0.001)
        scaler = torch.amp.GradScaler()

        # Training data
        images = torch.rand(4, 3, 64, 64, device=device)

        initial_loss = None
        for epoch in range(5):
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                encoded = encoder(images)
                reconstructed = decoder(encoded)

                # SSIM loss
                ssim = fused_ssim(reconstructed, images)
                loss = 1 - ssim

            if initial_loss is None:
                initial_loss = loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        final_loss = loss.item()

        # Loss should decrease (or at least not explode)
        assert final_loss < initial_loss + 0.1, (
            f"Loss should not explode: initial={initial_loss}, final={final_loss}"
        )
        assert not torch.isnan(torch.tensor(final_loss)), "Loss should not be NaN"

    def test_gaussian_splatting_style_usage(self, device):
        """Test usage pattern similar to 3D Gaussian Splatting."""
        torch.manual_seed(42)

        # Simulate rendered image (from differentiable renderer)
        rendered = torch.rand(1, 3, 256, 256, device=device, requires_grad=True)
        target = torch.rand(1, 3, 256, 256, device=device)

        # Compute SSIM loss (common in 3DGS)
        with torch.amp.autocast(device_type="cuda"):
            ssim = fused_ssim(rendered, target)
            l1_loss = torch.abs(rendered - target).mean()

            # Combined loss (typical in 3DGS)
            lambda_ssim = 0.2
            loss = (1 - lambda_ssim) * l1_loss + lambda_ssim * (1 - ssim)

        loss.backward()

        assert rendered.grad is not None, "Gradient should flow back to rendered image"
        assert not torch.isnan(rendered.grad).any(), "Gradients should not be NaN"

    def test_batch_processing(self, device):
        """Test processing multiple image pairs in batch."""
        torch.manual_seed(42)

        batch_sizes = [1, 2, 4, 8]

        for bs in batch_sizes:
            img1 = torch.rand(bs, 3, 128, 128, device=device, requires_grad=True)
            img2 = torch.rand(bs, 3, 128, 128, device=device)

            with torch.amp.autocast(device_type="cuda"):
                ssim = fused_ssim(img1, img2)

            assert ssim.shape == (), f"SSIM should be scalar, got shape {ssim.shape}"
            assert 0 <= ssim.item() <= 1, f"SSIM should be in [0, 1], got {ssim.item()}"

    def test_various_image_sizes(self, device):
        """Test various image sizes."""
        torch.manual_seed(42)

        sizes = [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]

        for h, w in sizes:
            img1 = torch.rand(1, 3, h, w, device=device, requires_grad=True)
            img2 = torch.rand(1, 3, h, w, device=device)

            with torch.amp.autocast(device_type="cuda"):
                ssim = fused_ssim(img1, img2)
                (1 - ssim).backward()

            assert img1.grad is not None, f"Gradient should be computed for size {h}x{w}"
            assert img1.grad.shape == img1.shape, f"Gradient shape mismatch for size {h}x{w}"
