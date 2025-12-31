"""Comprehensive tests comparing optimized fused-ssim against original implementation.

Uses subprocess isolation to avoid namespace conflicts between the two packages.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch

# Paths to implementations
OPTIMIZED_PATH = Path(__file__).parent.parent
OPTIMIZED_PYTHON = OPTIMIZED_PATH / ".venv" / "Scripts" / "python.exe"
ORIGINAL_VENV = Path("C:/temp/fused-ssim-original/.venv/Scripts/python.exe")

# Skip comparison tests if original implementation not available
skip_if_no_original = pytest.mark.skipif(
    not ORIGINAL_VENV.exists(),
    reason="Original fused-ssim not installed at C:/temp/fused-ssim-original/.venv",
)


def run_optimized(script: str, timeout: float = 60) -> dict:
    """Run script with optimized implementation."""
    # Use the project's venv python directly
    python_exe = str(OPTIMIZED_PYTHON) if OPTIMIZED_PYTHON.exists() else "python"
    result = subprocess.run(
        [python_exe, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(OPTIMIZED_PATH),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Optimized script failed:\n{result.stderr}")
    return json.loads(result.stdout)


def run_original(script: str, timeout: float = 60) -> dict:
    """Run script with original implementation."""
    result = subprocess.run(
        [str(ORIGINAL_VENV), "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd="C:/temp/fused-ssim-original",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Original script failed:\n{result.stderr}")
    return json.loads(result.stdout)


def normalize_path(path: str) -> str:
    """Convert Windows path to use forward slashes for embedding in Python scripts."""
    return path.replace("\\", "/")


class TestForwardEquivalence:
    """Test that forward pass outputs match between implementations."""

    @skip_if_no_original
    @pytest.mark.parametrize(
        "shape",
        [
            (1, 3, 256, 256),
            (1, 3, 512, 512),
            (2, 3, 256, 256),
            (1, 1, 128, 128),
            (4, 5, 128, 128),
        ],
    )
    def test_forward_random_input(self, shape):
        """Test forward pass with random inputs."""
        b, c, h, w = shape
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_normalized = normalize_path(tmpdir)
            # Save random inputs
            torch.manual_seed(42)
            img1 = torch.rand(b, c, h, w)
            img2 = torch.rand(b, c, h, w)
            torch.save({"img1": img1, "img2": img2}, f"{tmpdir}/inputs.pt")

            # Run optimized
            opt_script = f"""
import torch
import json
from fussim import fussim
data = torch.load("{tmpdir_normalized}/inputs.pt", weights_only=True)
img1, img2 = data["img1"].cuda(), data["img2"].cuda()
result = fused_ssim(img1, img2)
print(json.dumps({{"ssim": result.item()}}))
"""
            opt_result = run_optimized(opt_script)

            # Run original
            orig_script = f"""
import torch
import json
import sys
sys.path.insert(0, "C:/temp/fused-ssim-original")
from fussim import fussim
data = torch.load("{tmpdir_normalized}/inputs.pt", weights_only=True)
img1, img2 = data["img1"].cuda(), data["img2"].cuda()
result = fused_ssim(img1, img2)
print(json.dumps({{"ssim": result.item()}}))
"""
            orig_result = run_original(orig_script)

            # Compare
            assert abs(opt_result["ssim"] - orig_result["ssim"]) < 1e-5, (
                f"SSIM mismatch: optimized={opt_result['ssim']}, original={orig_result['ssim']}"
            )

    def test_forward_identical_images(self):
        """SSIM of identical images should be ~1.0."""
        script = """
import torch
import json
from fussim import fussim
torch.manual_seed(42)
img = torch.rand(1, 3, 256, 256, device="cuda")
result = fused_ssim(img, img)
print(json.dumps({"ssim": result.item()}))
"""
        opt_result = run_optimized(script)
        assert opt_result["ssim"] > 0.99, f"SSIM of identical images should be ~1.0, got {opt_result['ssim']}"

    def test_forward_inverted_images(self):
        """SSIM of inverted images should be low."""
        script = """
import torch
import json
from fussim import fussim
torch.manual_seed(42)
img1 = torch.rand(1, 3, 256, 256, device="cuda")
img2 = 1.0 - img1
result = fused_ssim(img1, img2)
print(json.dumps({"ssim": result.item()}))
"""
        opt_result = run_optimized(script)
        assert opt_result["ssim"] < 0.5, f"SSIM of inverted images should be low, got {opt_result['ssim']}"


class TestBackwardEquivalence:
    """Test that backward pass gradients match between implementations."""

    @skip_if_no_original
    @pytest.mark.parametrize(
        "shape",
        [
            (1, 3, 128, 128),
            (2, 3, 128, 128),
            (1, 5, 64, 64),
        ],
    )
    def test_backward_gradient(self, shape):
        """Test backward pass gradient computation."""
        b, c, h, w = shape
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_normalized = normalize_path(tmpdir)
            # Save random inputs
            torch.manual_seed(42)
            img1 = torch.rand(b, c, h, w)
            img2 = torch.rand(b, c, h, w)
            torch.save({"img1": img1, "img2": img2}, f"{tmpdir}/inputs.pt")

            # Run optimized
            opt_script = f"""
import torch
import json
from fussim import fussim
data = torch.load("{tmpdir_normalized}/inputs.pt", weights_only=True)
img1 = data["img1"].cuda().requires_grad_(True)
img2 = data["img2"].cuda()
result = fused_ssim(img1, img2)
result.backward()
grad_norm = img1.grad.norm().item()
grad_mean = img1.grad.mean().item()
grad_std = img1.grad.std().item()
print(json.dumps({{"grad_norm": grad_norm, "grad_mean": grad_mean, "grad_std": grad_std}}))
"""
            opt_result = run_optimized(opt_script)

            # Run original
            orig_script = f"""
import torch
import json
import sys
sys.path.insert(0, "C:/temp/fused-ssim-original")
from fussim import fussim
data = torch.load("{tmpdir_normalized}/inputs.pt", weights_only=True)
img1 = data["img1"].cuda().requires_grad_(True)
img2 = data["img2"].cuda()
result = fused_ssim(img1, img2)
result.backward()
grad_norm = img1.grad.norm().item()
grad_mean = img1.grad.mean().item()
grad_std = img1.grad.std().item()
print(json.dumps({{"grad_norm": grad_norm, "grad_mean": grad_mean, "grad_std": grad_std}}))
"""
            orig_result = run_original(orig_script)

            # Compare gradient statistics
            rtol, atol = 1e-4, 1e-5
            assert abs(opt_result["grad_norm"] - orig_result["grad_norm"]) < atol + rtol * abs(orig_result["grad_norm"]), (
                f"Gradient norm mismatch: optimized={opt_result['grad_norm']}, original={orig_result['grad_norm']}"
            )

    def test_backward_optimized_only(self):
        """Test that backward pass works correctly for optimized implementation."""
        script = """
import torch
import json
from fussim import fussim
torch.manual_seed(42)
img1 = torch.rand(1, 3, 128, 128, device="cuda", requires_grad=True)
img2 = torch.rand(1, 3, 128, 128, device="cuda")
result = fused_ssim(img1, img2)
result.backward()
grad_finite = torch.isfinite(img1.grad).all().item()
grad_nonzero = (img1.grad.abs() > 0).any().item()
print(json.dumps({"grad_finite": grad_finite, "grad_nonzero": grad_nonzero}))
"""
        result = run_optimized(script)
        assert result["grad_finite"], "Gradients should be finite"
        assert result["grad_nonzero"], "Gradients should be non-zero"


class TestDeterminism:
    """Test that both implementations are deterministic."""

    def test_optimized_determinism(self):
        """Test optimized implementation gives identical results on repeated runs."""
        script = """
import torch
import json
from fussim import fussim
results = []
for _ in range(5):
    torch.manual_seed(42)
    img1 = torch.rand(1, 3, 128, 128, device="cuda")
    img2 = torch.rand(1, 3, 128, 128, device="cuda")
    results.append(fused_ssim(img1, img2).item())
print(json.dumps({"results": results, "all_equal": len(set(results)) == 1}))
"""
        result = run_optimized(script)
        assert result["all_equal"], f"Non-deterministic results: {result['results']}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 1, 16, 16),  # Minimum size
            (1, 3, 127, 127),  # Non-power-of-2
            (1, 3, 513, 513),  # Non-power-of-2
            (1, 3, 1000, 999),  # Asymmetric non-power-of-2
        ],
    )
    def test_boundary_sizes(self, shape):
        """Test various boundary size conditions."""
        b, c, h, w = shape
        script = f"""
import torch
import json
from fussim import fussim
torch.manual_seed(42)
img1 = torch.rand({b}, {c}, {h}, {w}, device="cuda")
img2 = torch.rand({b}, {c}, {h}, {w}, device="cuda")
result = fused_ssim(img1, img2)
print(json.dumps({{"ssim": result.item(), "success": True}}))
"""
        result = run_optimized(script)
        assert result["success"]
        assert 0.0 <= result["ssim"] <= 1.0

    @pytest.mark.parametrize(
        "width",
        [255, 256, 257, 512, 513, 521, 1023, 1024],
    )
    def test_alignment_variations(self, width):
        """Test different width alignments (odd, even, prime)."""
        script = f"""
import torch
import json
from fussim import fussim
torch.manual_seed(42)
img1 = torch.rand(1, 3, 128, {width}, device="cuda")
img2 = torch.rand(1, 3, 128, {width}, device="cuda")
result = fused_ssim(img1, img2)
print(json.dumps({{"ssim": result.item(), "success": True}}))
"""
        result = run_optimized(script)
        assert result["success"]

    def test_zero_images(self):
        """Test with all-zero images."""
        script = """
import torch
import json
from fussim import fussim
img1 = torch.zeros(1, 3, 64, 64, device="cuda")
img2 = torch.zeros(1, 3, 64, 64, device="cuda")
result = fused_ssim(img1, img2)
print(json.dumps({"ssim": result.item(), "is_nan": torch.isnan(result).item()}))
"""
        result = run_optimized(script)
        # Zero images should give valid result (likely 1.0 or close to it)
        assert not result["is_nan"], "SSIM of zero images should not be NaN"

    def test_one_images(self):
        """Test with all-one images."""
        script = """
import torch
import json
from fussim import fussim
img1 = torch.ones(1, 3, 64, 64, device="cuda")
img2 = torch.ones(1, 3, 64, 64, device="cuda")
result = fused_ssim(img1, img2)
print(json.dumps({"ssim": result.item(), "is_nan": torch.isnan(result).item()}))
"""
        result = run_optimized(script)
        assert not result["is_nan"], "SSIM of one images should not be NaN"
        assert result["ssim"] > 0.99, f"SSIM of identical one images should be ~1.0, got {result['ssim']}"

    def test_constant_different_values(self):
        """Test with constant images of different values."""
        script = """
import torch
import json
from fussim import fussim
img1 = torch.full((1, 3, 64, 64), 0.3, device="cuda")
img2 = torch.full((1, 3, 64, 64), 0.7, device="cuda")
result = fused_ssim(img1, img2)
print(json.dumps({"ssim": result.item()}))
"""
        result = run_optimized(script)
        # Different constant images should have low SSIM
        assert result["ssim"] < 1.0


class TestWindowSizes:
    """Test different window sizes (optimized version only)."""

    @pytest.mark.parametrize("window_size", [7, 9, 11])
    def test_window_size_parameter(self, window_size):
        """Test that different window sizes work correctly."""
        script = f"""
import torch
import json
from fussim import fussim
torch.manual_seed(42)
img1 = torch.rand(1, 3, 128, 128, device="cuda")
img2 = torch.rand(1, 3, 128, 128, device="cuda")
result = fused_ssim(img1, img2, window_size={window_size})
print(json.dumps({{"ssim": result.item(), "window_size": {window_size}}}))
"""
        result = run_optimized(script)
        assert 0.0 <= result["ssim"] <= 1.0


class TestFP16:
    """Test FP16/AMP support (optimized version only)."""

    def test_fp16_via_autocast(self):
        """Test FP16 forward pass via autocast (recommended way)."""
        script = """
import torch
import json
from fussim import fussim
torch.manual_seed(42)
img1 = torch.rand(1, 3, 128, 128, device="cuda")
img2 = torch.rand(1, 3, 128, 128, device="cuda")
with torch.amp.autocast("cuda", dtype=torch.float16):
    result = fused_ssim(img1, img2)
print(json.dumps({"ssim": result.item()}))
"""
        result = run_optimized(script)
        assert 0.0 <= result["ssim"] <= 1.0

    def test_amp_autocast(self):
        """Test with AMP autocast."""
        script = """
import torch
import json
from fussim import fussim
torch.manual_seed(42)
img1 = torch.rand(1, 3, 128, 128, device="cuda")
img2 = torch.rand(1, 3, 128, 128, device="cuda")
with torch.amp.autocast("cuda"):
    result = fused_ssim(img1, img2)
print(json.dumps({"ssim": result.item()}))
"""
        result = run_optimized(script)
        assert 0.0 <= result["ssim"] <= 1.0

    def test_amp_with_gradscaler(self):
        """Test FP16 with gradient scaling for training."""
        script = """
import torch
import json
from fussim import fussim
torch.manual_seed(42)
img1 = torch.rand(1, 3, 128, 128, device="cuda", requires_grad=True)
img2 = torch.rand(1, 3, 128, 128, device="cuda")
scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast("cuda", dtype=torch.float16):
    result = fused_ssim(img1, img2)
scaler.scale(result).backward()
grad_finite = torch.isfinite(img1.grad).all().item()
print(json.dumps({"ssim": result.item(), "grad_finite": grad_finite}))
"""
        result = run_optimized(script)
        assert 0.0 <= result["ssim"] <= 1.0
        assert result["grad_finite"], "Gradients should be finite with GradScaler"


class TestStress:
    """Stress tests for memory and performance."""

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 3, 1080, 1920),  # HD
            (4, 3, 512, 512),  # Medium batch
            (8, 3, 256, 256),  # Large batch, small images
        ],
    )
    def test_large_inputs(self, shape):
        """Test with larger inputs."""
        b, c, h, w = shape
        script = f"""
import torch
import json
from fussim import fussim
torch.manual_seed(42)
img1 = torch.rand({b}, {c}, {h}, {w}, device="cuda")
img2 = torch.rand({b}, {c}, {h}, {w}, device="cuda")
result = fused_ssim(img1, img2)
torch.cuda.synchronize()
print(json.dumps({{"ssim": result.item(), "success": True}}))
"""
        result = run_optimized(script, timeout=120)
        assert result["success"]

    def test_consecutive_calls(self):
        """Test multiple consecutive calls without memory leaks."""
        script = """
import torch
import json
from fussim import fussim
torch.manual_seed(42)
results = []
for i in range(10):
    img1 = torch.rand(2, 3, 256, 256, device="cuda")
    img2 = torch.rand(2, 3, 256, 256, device="cuda")
    result = fused_ssim(img1, img2)
    results.append(result.item())
    torch.cuda.synchronize()
print(json.dumps({"num_calls": len(results), "success": True}))
"""
        result = run_optimized(script)
        assert result["num_calls"] == 10
        assert result["success"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
