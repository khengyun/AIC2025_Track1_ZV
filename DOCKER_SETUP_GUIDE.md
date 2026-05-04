# AIC2025 Docker Setup Guide for NVIDIA TITAN X Pascal

This guide provides step-by-step instructions for setting up AIC2025_Track1_ZV in Docker optimized for NVIDIA TITAN X Pascal GPU.

## Overview

Based on the comprehensive setup README from `V-DETR/readme_files/setup_readme.md`, we've enhanced the Docker configuration to:

- ✅ Use stable PyTorch 1.8.1 + CUDA 10.2 stack (avoids kernel image issues on TITAN X sm_61)
- ✅ Pre-install core Python dependencies at build time
- ✅ Auto-setup library paths via entrypoint.sh
- ✅ Provide convenient build/run scripts

## Requirements

### Host System
- Ubuntu 22.04 (or similar)
- NVIDIA TITAN X Pascal GPU
- Docker installed
- NVIDIA Container Toolkit installed

### Install NVIDIA Container Toolkit

```bash
sudo apt-get update
sudo apt-get install -y curl gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Verify Docker GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

Should show NVIDIA TITAN X (Pascal).

## Quick Start

### 1. Build Docker Image

```bash
cd ~/Documents/GitHub/AIC2025_Track1_ZV
chmod +x docker-build-and-run.sh

./docker-build-and-run.sh build
```

This builds image `aic2025-me102` with:
- PyTorch 1.8.1 + CUDA 10.2
- MinkowskiEngine-compatible dependencies
- Pre-installed: mmcv-full, timm, wandb, open3d, h5py, shapely, filterpy, etc.

### 2. Run Container

```bash
./docker-build-and-run.sh run
```

Mounts repo at `/workspace/AIC2025_Track1_ZV` inside container.

### 3. Inside Container: Build MinkowskiEngine

```bash
cd /workspace/AIC2025_Track1_ZV/MinkowskiEngine

# Clean previous builds
rm -rf build dist MinkowskiEngine.egg-info ~/.cache/torch_extensions

# Setup build environment
export TORCH_CUDA_ARCH_LIST="6.1"  # For TITAN X Pascal
export MAX_JOBS=3                   # Adjust based on RAM
export OMP_NUM_THREADS=4

# Build with system OpenBLAS
python setup.py install \
  --blas_include_dirs=/usr/include/x86_64-linux-gnu \
  --blas=openblas

cd /workspace/AIC2025_Track1_ZV

# Verify MinkowskiEngine CUDA
python -c "
import torch
import MinkowskiEngine as ME
coords = torch.IntTensor([[0,0,0,0],[0,1,1,1],[0,2,2,2]]).cuda()
feats = torch.randn(3, 4, device='cuda')
x = ME.SparseTensor(features=feats, coordinates=coords)
conv = ME.MinkowskiConvolution(4, 8, kernel_size=3, dimension=3).cuda()
y = conv(x)
torch.cuda.synchronize()
print('✓ MinkowskiEngine CUDA OK:', y.F.shape, y.F.is_cuda)
"
```

If build is slow, reduce `MAX_JOBS`:
```bash
export MAX_JOBS=1
```

### 4. Inside Container: Build pytorch-metric-learning

```bash
cd /workspace/AIC2025_Track1_ZV/pytorch-metric-learning
pip install -e .
cd /workspace/AIC2025_Track1_ZV
```

### 5. Inside Container: Build PointNet2

```bash
cd /workspace/AIC2025_Track1_ZV/V-DETR/third_party/pointnet2

python setup.py install --user

# Fix RPATH for CUDA library access
export TORCH_LIB=$(python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export CUDA_HOME=/usr/local/cuda

PN2_SO=$(find /root/.local/lib/python3.7/site-packages -name "_ext*.so" | grep pointnet2 | head -1)
patchelf --set-rpath "$TORCH_LIB:$CUDA_HOME/lib64:/opt/conda/lib" "$PN2_SO"

# Verify
python -c "import pointnet2; from pointnet2 import _ext; print('✓ pointnet2 import OK')"

cd /workspace/AIC2025_Track1_ZV
```

### 6. Inside Container: Compile Cython Utils

```bash
cd /workspace/AIC2025_Track1_ZV/V-DETR/utils
python cython_compile.py build_ext --inplace

# Verify
python -c "import box_intersection; print('✓ box_intersection import OK')"

cd /workspace/AIC2025_Track1_ZV
```

### 7. Smoke Test

```bash
# Option A: Run from host
./docker-build-and-run.sh test

# Option B: Run manually inside container
python -c "
import torch
import MinkowskiEngine as ME
import pytorch_metric_learning
import mmcv
import h5py
import open3d as o3d
import shapely
import filterpy
import pointnet2
from pointnet2 import _ext

print('✓ torch:', torch.__version__, torch.version.cuda, torch.cuda.is_available())
print('✓ ME:', ME.__version__)
print('✓ mmcv:', mmcv.__version__)
print('✓ open3d:', o3d.__version__)
print('✓ pointnet2 import OK')

if torch.cuda.is_available():
    print('✓ GPU:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    coords = torch.IntTensor([[0,0,0,0],[0,1,1,1],[0,2,2,2]]).cuda()
    feats = torch.randn(3, 4, device='cuda')
    x = ME.SparseTensor(features=feats, coordinates=coords)
    conv = ME.MinkowskiConvolution(4, 8, kernel_size=3, dimension=3).cuda()
    y = conv(x)
    torch.cuda.synchronize()
    print('✓ ME CUDA op OK:', y.F.shape, y.F.is_cuda)

print('✓✓✓ All tests passed!')
"
```

## Docker Helper Commands

### Build image
```bash
./docker-build-and-run.sh build
```

### Start/attach to container
```bash
./docker-build-and-run.sh run
```

### Open new shell in running container
```bash
./docker-build-and-run.sh shell
```

### Run smoke test
```bash
./docker-build-and-run.sh test
```

### Commit container to image after setup
```bash
./docker-build-and-run.sh commit step7-ok
```

### Manual Docker commands

Start container:
```bash
docker start -ai aic2025_me102_dev
```

Stop container:
```bash
docker stop aic2025_me102_dev
```

Remove container:
```bash
docker rm aic2025_me102_dev
```

List images:
```bash
docker images | grep aic2025
```

## File Structure

```
AIC2025_Track1_ZV/
├── Dockerfile.me102         # Enhanced Dockerfile for TITAN X
├── entrypoint.sh            # Auto-setup env vars on container start
├── docker-build-and-run.sh  # Build/run helper script
└── DOCKER_SETUP_GUIDE.md    # This file
```

## Dockerfile.me102 Details

### Base Image
- `pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel`
- Includes Python 3.7.10, PyTorch 1.8.1, CUDA 10.2

### Pre-installed Packages
- Build tools: build-essential, ninja-build, patchelf, protobuf-compiler
- GPU-related: libopenblas-dev, CUDA runtime
- Python deps: setuptools, wheel, numpy, cython, ninja
- AI/ML: mmcv-full 1.6.1, timm, pytorch-metric-learning, wandb
- Data: h5py, open3d, shapely, filterpy
- Visualization: ffmpeg, libsm6, libxext6, libxrender-dev

### Environment Variables (set in Dockerfile)
```bash
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV TORCH_LIB=/opt/conda/lib/python3.7/site-packages/torch/lib
ENV OMP_NUM_THREADS=12
```

### Entrypoint
Auto-sets `LD_LIBRARY_PATH` and `OMP_NUM_THREADS` each time container starts.

## Troubleshooting

### 1. CUDA error: device kernel image is invalid

**Root cause:** MinkowskiEngine 0.5.4 built with newer CUDA (11.3+) doesn't work on TITAN X (sm_61).

**Solution:** Use Docker with CUDA 10.2 + PyTorch 1.8.1 (already in Dockerfile.me102).

### 2. ImportError: libc10.so not found

**Root cause:** PointNet2 C++ extension can't find PyTorch CUDA libs.

**Solution:** Applied in step 5 above using `patchelf --set-rpath`.

### 3. Build too slow / out of memory

**Solution:** Reduce `MAX_JOBS`:
```bash
export MAX_JOBS=1  # Instead of MAX_JOBS=3
```

### 4. numpy==1.22.4 not found

**Root cause:** Image uses Python 3.7, newer numpy requires 3.8+.

**Solution:** Dockerfile uses `numpy==1.21.6` (already fixed).

### 5. protobuf or safetensors import errors

**Solution:** Pinned in Dockerfile:
- `protobuf==4.24.4` (not 4.25.2)
- `safetensors==0.4.5` (avoids puccinialin dependency)

## After Setup: Saving Progress

After completing a major step (e.g., MinkowskiEngine built), save the container:

```bash
./docker-build-and-run.sh commit me-ok

# Later run from saved image:
docker run -it --gpus all \
  --ipc=host --shm-size=16g \
  -v ~/Documents/GitHub/AIC2025_Track1_ZV:/workspace/AIC2025_Track1_ZV \
  aic2025-me102-me-ok
```

## Checklist for Full Setup

- [ ] NVIDIA Container Toolkit installed and Docker GPU working
- [ ] Docker image `aic2025-me102` built
- [ ] Container `aic2025_me102_dev` running
- [ ] MinkowskiEngine 0.5.4 built and CUDA test passed
- [ ] pytorch-metric-learning installed
- [ ] PointNet2 built and RPATH patched
- [ ] Cython utils compiled
- [ ] Smoke test passed
- [ ] Container committed to `aic2025-me102-step7-ok`

## References

- Setup readme: `V-DETR/readme_files/setup_readme.md`
- Original Dockerfile template: `Dockerfile.me102`
- MinkowskiEngine: https://github.com/NVIDIA/MinkowskiEngine
- PyTorch: https://pytorch.org/

## Notes

- **Each container restart:** Library paths auto-set by entrypoint.sh
- **Host repo path:** `~/Documents/GitHub/AIC2025_Track1_ZV`
- **Container mount:** `/workspace/AIC2025_Track1_ZV`
- **GPU arch:** TITAN X Pascal uses CUDA sm_61 compute capability (6, 1)
- **CUDA version:** 10.2 (for MinkowskiEngine 0.5.4 compat)
- **Python:** 3.7.10 (from pytorch/pytorch:1.8.1 base image)
