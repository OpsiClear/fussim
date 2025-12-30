#!/bin/bash
# CUDA environment setup for Windows
# Sets CUDA_HOME, CUDA_PATH, PATH, and TORCH_CUDA_ARCH_LIST

case ${1} in
  cu128)
    export CUDA_HOME="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"
    export CUDA_PATH="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8"
    export PATH="${CUDA_HOME}/bin:$PATH"
    # CUDA 12.8 supports Blackwell datacenter (sm_100) and consumer (sm_120)
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;12.0"
    ;;
  cu126)
    export CUDA_HOME="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6"
    export CUDA_PATH="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6"
    export PATH="${CUDA_HOME}/bin:$PATH"
    # CUDA 12.6 supports Blackwell consumer (sm_120, e.g., RTX 5090)
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;12.0"
    ;;
  cu124)
    export CUDA_HOME="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4"
    export CUDA_PATH="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4"
    export PATH="${CUDA_HOME}/bin:$PATH"
    # CUDA 12.4 supports up to Hopper (sm_90)
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
    ;;
  cu121)
    export CUDA_HOME="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1"
    export CUDA_PATH="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1"
    export PATH="${CUDA_HOME}/bin:$PATH"
    # CUDA 12.1 supports up to Hopper (sm_90)
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
    ;;
  cu118)
    export CUDA_HOME="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"
    export CUDA_PATH="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8"
    export PATH="${CUDA_HOME}/bin:$PATH"
    # CUDA 11.8 supports up to Hopper (sm_90)
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
    ;;
  *)
    echo "Unknown CUDA version: ${1}"
    ;;
esac
