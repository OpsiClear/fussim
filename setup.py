import os
import subprocess
import sys

from setuptools import setup

# Set DISTUTILS_USE_SDK for Windows builds to avoid VC env issues
if sys.platform == "win32":
    os.environ.setdefault("DISTUTILS_USE_SDK", "1")

    # Fix PATH to use MSVC linker instead of Git's link.exe
    # Git installs a Unix-style 'link' command that conflicts with MSVC's linker
    path = os.environ.get("PATH", "")
    path_parts = path.split(os.pathsep)
    filtered_parts = [p for p in path_parts if "Git\\usr\\bin" not in p and "Git/usr/bin" not in p]
    os.environ["PATH"] = os.pathsep.join(filtered_parts)


# Check for CUDA availability before importing torch extensions
def get_cuda_version():
    """Get CUDA version from nvcc or return None if not available."""
    try:
        output = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT)
        output = output.decode("utf-8")
        # Parse version from output like "release 12.1"
        import re

        match = re.search(r"release (\d+\.\d+)", output)
        if match:
            return match.group(1)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    return None


def find_cuda_home():
    """Find CUDA home directory from various sources."""
    # First check environment variables
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and os.path.isdir(cuda_home):
        return cuda_home

    # Common Windows paths - check these first as they're most reliable
    if sys.platform == "win32":
        for cuda_ver in ["12.8", "12.6", "12.4", "12.1", "11.8"]:
            path = f"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v{cuda_ver}"
            if os.path.isdir(path):
                return path

    # Common Linux paths
    for path in ["/usr/local/cuda", "/usr/cuda"]:
        if os.path.isdir(path):
            return path

    # Try to find from nvcc path (may not work in isolated build environments)
    try:
        import shutil

        nvcc_path = shutil.which("nvcc")
        if nvcc_path:
            # nvcc is typically at CUDA_HOME/bin/nvcc
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
            if os.path.isdir(cuda_home):
                return cuda_home
    except Exception:
        pass

    return None


def check_cuda_available():
    """Check if CUDA is available for building."""
    cuda_version = get_cuda_version()
    if cuda_version:
        return True

    cuda_home = find_cuda_home()
    if cuda_home:
        return True

    return False


# Check if we're building an sdist (source distribution) - CUDA not required for sdist
_building_sdist = "sdist" in sys.argv or "egg_info" in sys.argv

if not _building_sdist:
    # Auto-detect and set CUDA_HOME if not already set
    _cuda_home = find_cuda_home()
    if _cuda_home:
        if not os.environ.get("CUDA_HOME"):
            os.environ["CUDA_HOME"] = _cuda_home
        if not os.environ.get("CUDA_PATH"):
            os.environ["CUDA_PATH"] = _cuda_home

    if not check_cuda_available():
        raise RuntimeError(
            "CUDA is required to build fussim but was not found.\n"
            "Please ensure:\n"
            "  1. NVIDIA CUDA Toolkit is installed\n"
            "  2. nvcc is in your PATH, or\n"
            "  3. CUDA_HOME environment variable is set\n"
            "\n"
            "Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads"
        )

    import torch.utils.cpp_extension as cpp_ext  # noqa: E402
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # noqa: E402

    # If torch didn't find CUDA_HOME but we have it, patch the module
    if cpp_ext.CUDA_HOME is None and _cuda_home:
        cpp_ext.CUDA_HOME = _cuda_home
else:
    # For sdist, we don't need CUDA extensions
    _cuda_home = None

# Platform-specific compiler flags (only needed for wheel builds)
IS_WINDOWS = sys.platform == "win32"

if not _building_sdist:
    # C++ compiler flags
    if IS_WINDOWS:
        cxx_flags = ["/O2", "/std:c++17"]
    else:
        cxx_flags = ["-O3", "-std=c++17"]


def get_cuda_arch_flags():
    """
    Get CUDA architecture flags for compilation.

    Priority:
    1. TORCH_CUDA_ARCH_LIST environment variable
    2. Auto-detect from available GPUs
    3. Default to common architectures
    """
    # Check for user-specified architectures
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()

    if arch_list:
        # User specified architectures - parse and convert to nvcc flags
        return parse_cuda_arch_list(arch_list)

    # Try to auto-detect GPU architectures
    detected = detect_gpu_architectures()
    if detected:
        return detected

    # Fallback to common architectures with PTX for forward compatibility
    # This covers: Turing (7.5), Ampere (8.0, 8.6), Ada Lovelace (8.9), Hopper (9.0)
    return get_default_arch_flags()


def parse_cuda_arch_list(arch_list):
    """Parse TORCH_CUDA_ARCH_LIST format into nvcc flags."""
    flags = []

    # Handle semicolon or space separated architectures
    archs = arch_list.replace(" ", ";").split(";")
    archs = [a.strip() for a in archs if a.strip()]

    for arch in archs:
        # Handle PTX suffix
        has_ptx = "+PTX" in arch.upper()
        arch_clean = arch.replace("+PTX", "").replace("+ptx", "").strip()

        # Convert decimal format (e.g., "8.9") to nvcc format (e.g., "89")
        if "." in arch_clean:
            major, minor = arch_clean.split(".")[:2]
            arch_num = f"{major}{minor}"
        else:
            arch_num = arch_clean

        # Add gencode flags for better multi-arch support
        flags.append(f"-gencode=arch=compute_{arch_num},code=sm_{arch_num}")

        if has_ptx:
            flags.append(f"-gencode=arch=compute_{arch_num},code=compute_{arch_num}")

    return flags


def detect_gpu_architectures():
    """Detect GPU architectures from available CUDA devices."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        archs = set()
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            arch = f"{props.major}{props.minor}"
            archs.add(arch)

        if not archs:
            return None

        flags = []
        for arch in sorted(archs):
            flags.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")

        # Add PTX for newest architecture for forward compatibility
        newest = max(archs)
        flags.append(f"-gencode=arch=compute_{newest},code=compute_{newest}")

        return flags

    except Exception:
        return None


def get_supported_archs(archs):
    """Filter architectures to only those supported by the current CUDA toolkit."""
    cuda_version_num = None
    try:
        # Try to get CUDA version from torch
        import torch

        cuda_version = torch.version.cuda
        if cuda_version:
            major, minor = map(int, cuda_version.split(".")[:2])
            cuda_version_num = major * 10 + minor
    except Exception:
        pass

    if cuda_version_num is None:
        # Try to get from nvcc
        cuda_ver = get_cuda_version()
        if cuda_ver:
            try:
                major, minor = map(int, cuda_ver.split(".")[:2])
                cuda_version_num = major * 10 + minor
            except ValueError:
                cuda_version_num = None

    if cuda_version_num is None:
        # Can't determine version, return conservative subset
        return ["75", "80", "86"]

    # Map CUDA versions to max supported architecture
    # https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
    max_arch_by_cuda = {
        110: "80",  # CUDA 11.0
        111: "86",  # CUDA 11.1+
        118: "89",  # CUDA 11.8
        120: "90",  # CUDA 12.0
        128: "100",  # CUDA 12.8 (Blackwell GB100)
        129: "120",  # CUDA 12.9+ (Blackwell GB20x - RTX 5090)
    }

    # Find max supported architecture for this CUDA version
    max_arch = "80"  # Conservative default
    for cuda_ver, arch in sorted(max_arch_by_cuda.items()):
        if cuda_version_num >= cuda_ver:
            max_arch = arch

    # Filter to supported architectures
    return [a for a in archs if int(a) <= int(max_arch)]


def get_default_arch_flags():
    """Get default architecture flags covering common GPUs."""
    # Default architectures: Turing, Ampere, Ada Lovelace, Hopper, Blackwell
    # Using gencode for binary compatibility + PTX for forward compatibility
    # Note: Different CUDA toolkit versions support different max architectures:
    #   - CUDA 11.x: up to sm_86 (Ampere)
    #   - CUDA 12.0+: up to sm_90 (Hopper)
    #   - CUDA 12.8+: up to sm_120 (Blackwell)
    default_archs = ["75", "80", "86", "89", "90", "100", "120"]

    # Filter architectures based on what the current CUDA toolkit supports
    supported_archs = get_supported_archs(default_archs)

    flags = []
    for arch in supported_archs:
        flags.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")

    # Add PTX for newest for forward compatibility with future GPUs
    if supported_archs:
        newest = supported_archs[-1]
        flags.append(f"-gencode=arch=compute_{newest},code=compute_{newest}")

    return flags


if not _building_sdist:
    # NVCC compiler flags
    nvcc_flags = [
        "-O3",
        "--maxrregcount=32",
        "--use_fast_math",
        "-std=c++17",
    ]

    # Add architecture flags
    nvcc_flags.extend(get_cuda_arch_flags())

    # Windows-specific NVCC flags
    if IS_WINDOWS:
        nvcc_flags.extend(
            [
                "--allow-unsupported-compiler",  # Allow newer MSVC versions
                "-Xcompiler",
                "/wd4819",  # Suppress Unicode warning
                "-Xcompiler",
                "/wd4251",  # Suppress DLL interface warning
            ]
        )


def get_extensions():
    """Build the list of extension modules."""
    if _building_sdist:
        return []
    ext_modules = [
        CUDAExtension(
            name="fussim_cuda",
            sources=["csrc/ssim.cu", "csrc/ssim_fp16.cu", "csrc/ext.cpp"],
            include_dirs=["csrc"],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        )
    ]
    return ext_modules


if _building_sdist:
    # For sdist, no extensions or custom build commands
    setup()
else:
    setup(
        ext_modules=get_extensions(),
        cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
    )
