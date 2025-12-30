#!/bin/bash
# CUDA installation script for Windows
# Adapted from https://github.com/pyg-team/pyg-lib/

set -e

# Note: GPU driver DLLs are NOT needed for compilation.
# The CUDA Toolkit includes all required import libraries (.lib files).

case ${1} in
  cu128)
    CUDA_SHORT=12.8
    CUDA_URL=https://developer.download.nvidia.com/compute/cuda/${CUDA_SHORT}.0/local_installers
    CUDA_FILE=cuda_${CUDA_SHORT}.0_571.96_windows.exe
    ;;
  cu126)
    CUDA_SHORT=12.6
    CUDA_URL=https://developer.download.nvidia.com/compute/cuda/${CUDA_SHORT}.3/local_installers
    CUDA_FILE=cuda_${CUDA_SHORT}.3_561.17_windows.exe
    ;;
  cu124)
    CUDA_SHORT=12.4
    CUDA_URL=https://developer.download.nvidia.com/compute/cuda/${CUDA_SHORT}.1/local_installers
    CUDA_FILE=cuda_${CUDA_SHORT}.1_551.78_windows.exe
    ;;
  cu121)
    CUDA_SHORT=12.1
    CUDA_URL=https://developer.download.nvidia.com/compute/cuda/${CUDA_SHORT}.1/local_installers
    CUDA_FILE=cuda_${CUDA_SHORT}.1_531.14_windows.exe
    ;;
  cu118)
    CUDA_SHORT=11.8
    CUDA_URL=https://developer.download.nvidia.com/compute/cuda/${CUDA_SHORT}.0/local_installers
    CUDA_FILE=cuda_${CUDA_SHORT}.0_522.06_windows.exe
    ;;
  *)
    echo "Unrecognized CUDA_VERSION=${1}"
    exit 1
    ;;
esac

echo "Downloading CUDA ${CUDA_SHORT}..."
curl -L "${CUDA_URL}/${CUDA_FILE}" --output "${CUDA_FILE}"

echo "Installing CUDA ${CUDA_SHORT}..."
PowerShell -Command "Start-Process -FilePath \"${CUDA_FILE}\" -ArgumentList \"-s nvcc_${CUDA_SHORT} cuobjdump_${CUDA_SHORT} nvprune_${CUDA_SHORT} cupti_${CUDA_SHORT} cublas_dev_${CUDA_SHORT} cudart_${CUDA_SHORT} cufft_dev_${CUDA_SHORT} curand_dev_${CUDA_SHORT} cusolver_dev_${CUDA_SHORT} cusparse_dev_${CUDA_SHORT} thrust_${CUDA_SHORT} npp_dev_${CUDA_SHORT} nvrtc_dev_${CUDA_SHORT} nvml_dev_${CUDA_SHORT}\" -Wait -NoNewWindow"
echo "CUDA installation complete!"
rm -f "${CUDA_FILE}"

echo "Installing NvToolsExt..."
curl -L https://ossci-windows.s3.us-east-1.amazonaws.com/builder/NvToolsExt.7z --output /tmp/NvToolsExt.7z
7z x /tmp/NvToolsExt.7z -o"/tmp/NvToolsExt"
mkdir -p "/c/Program Files/NVIDIA Corporation/NvToolsExt/bin/x64"
mkdir -p "/c/Program Files/NVIDIA Corporation/NvToolsExt/include"
mkdir -p "/c/Program Files/NVIDIA Corporation/NvToolsExt/lib/x64"
cp -r /tmp/NvToolsExt/bin/x64/* "/c/Program Files/NVIDIA Corporation/NvToolsExt/bin/x64"
cp -r /tmp/NvToolsExt/include/* "/c/Program Files/NVIDIA Corporation/NvToolsExt/include"
cp -r /tmp/NvToolsExt/lib/x64/* "/c/Program Files/NVIDIA Corporation/NvToolsExt/lib/x64"
echo "NvToolsExt installation complete!"
