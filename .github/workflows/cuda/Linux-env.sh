#!/bin/bash
# CUDA environment setup for Linux
# Sets CUDA_HOME, PATH, LD_LIBRARY_PATH, and TORCH_CUDA_ARCH_LIST

case ${1} in
  cu128)
    export CUDA_HOME=/usr/local/cuda-12.8
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
    # CUDA 12.8 supports Blackwell datacenter (sm_100) and consumer (sm_120)
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;12.0"
    ;;
  cu126)
    export CUDA_HOME=/usr/local/cuda-12.6
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
    # CUDA 12.6 does NOT support sm_120 (Blackwell) - only CUDA 12.8+ does
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
    ;;
  cu124)
    export CUDA_HOME=/usr/local/cuda-12.4
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
    # CUDA 12.4 supports up to Hopper (sm_90)
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
    ;;
  cu121)
    export CUDA_HOME=/usr/local/cuda-12.1
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
    # CUDA 12.1 supports up to Hopper (sm_90)
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
    ;;
  cu118)
    export CUDA_HOME=/usr/local/cuda-11.8
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
    # CUDA 11.8 supports up to Hopper (sm_90)
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
    ;;
  *)
    echo "Unknown CUDA version: ${1}"
    ;;
esac
