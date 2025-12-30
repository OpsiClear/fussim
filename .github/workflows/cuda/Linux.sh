#!/bin/bash
# CUDA installation script for Linux
# Adapted from https://github.com/pyg-team/pyg-lib/

set -e

OS=ubuntu2204

case ${1} in
  cu128)
    CUDA=12.8
    APT_KEY=${OS}-${CUDA/./-}-local
    FILENAME=cuda-repo-${APT_KEY}_${CUDA}.0-570.86.10-1_amd64.deb
    URL=https://developer.download.nvidia.com/compute/cuda/${CUDA}.0/local_installers
    ;;
  cu126)
    CUDA=12.6
    APT_KEY=${OS}-${CUDA/./-}-local
    FILENAME=cuda-repo-${APT_KEY}_${CUDA}.3-560.35.05-1_amd64.deb
    URL=https://developer.download.nvidia.com/compute/cuda/${CUDA}.3/local_installers
    ;;
  cu124)
    CUDA=12.4
    APT_KEY=${OS}-${CUDA/./-}-local
    FILENAME=cuda-repo-${APT_KEY}_${CUDA}.1-550.54.15-1_amd64.deb
    URL=https://developer.download.nvidia.com/compute/cuda/${CUDA}.1/local_installers
    ;;
  cu121)
    CUDA=12.1
    APT_KEY=${OS}-${CUDA/./-}-local
    FILENAME=cuda-repo-${APT_KEY}_${CUDA}.1-530.30.02-1_amd64.deb
    URL=https://developer.download.nvidia.com/compute/cuda/${CUDA}.1/local_installers
    ;;
  cu118)
    CUDA=11.8
    APT_KEY=${OS}-${CUDA/./-}-local
    FILENAME=cuda-repo-${APT_KEY}_${CUDA}.0-520.61.05-1_amd64.deb
    URL=https://developer.download.nvidia.com/compute/cuda/${CUDA}.0/local_installers
    ;;
  *)
    echo "Unrecognized CUDA_VERSION=${1}"
    exit 1
    ;;
esac

echo "Installing CUDA ${CUDA}..."

# Download with retry logic
MAX_RETRIES=3
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
  wget -nv https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin && break
  RETRY_COUNT=$((RETRY_COUNT + 1))
  echo "Retry $RETRY_COUNT/$MAX_RETRIES for pin file..."
  sleep 5
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
  echo "Failed to download CUDA pin file after $MAX_RETRIES retries"
  exit 1
fi

sudo mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Download CUDA installer with retry
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
  wget -nv ${URL}/${FILENAME} && break
  RETRY_COUNT=$((RETRY_COUNT + 1))
  echo "Retry $RETRY_COUNT/$MAX_RETRIES for CUDA installer..."
  sleep 5
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
  echo "Failed to download CUDA installer after $MAX_RETRIES retries"
  exit 1
fi

# Verify file was downloaded
if [ ! -f "${FILENAME}" ]; then
  echo "CUDA installer file not found: ${FILENAME}"
  exit 1
fi

sudo dpkg -i ${FILENAME}

sudo cp /var/cuda-repo-${APT_KEY}/cuda-*-keyring.gpg /usr/share/keyrings/

sudo apt-get -qq update
sudo apt install -y cuda-nvcc-${CUDA/./-} cuda-libraries-dev-${CUDA/./-} cuda-command-line-tools-${CUDA/./-}
sudo apt clean

rm -f ${FILENAME}

# Verify installation
if ! command -v /usr/local/cuda-${CUDA}/bin/nvcc &> /dev/null; then
  echo "CUDA ${CUDA} installation verification failed - nvcc not found"
  exit 1
fi

echo "CUDA ${CUDA} installation complete!"
