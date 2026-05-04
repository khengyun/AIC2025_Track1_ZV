# AIC2025 Track 1 - Docker GPU Inference Guide

This guide provides step-by-step instructions to run V-DETR inference inside Docker with GPU support on a local machine.

## Prerequisites

### Host Requirements
- **NVIDIA GPU**: NVIDIA TITAN X Pascal or later (compute capability 6.1+)
- **NVIDIA Driver**: >= 580.x (or host driver version)
- **Docker Engine**: 20.10+ with docker-compose
- **NVIDIA Container Toolkit**: Properly installed and configured

### Verify Host GPU Setup

```bash
# Check if docker can access GPU
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

Expected output:
```
NVIDIA-SMI 580.126.09    Driver Version: 580.126.09
CUDA Version: 11.8
...
| NVIDIA TITAN X (Pascal)     | 0%      45W / 250W    |  0MiB / 12193MiB |
```

If this fails, install NVIDIA Container Toolkit:
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Building Docker Image

### Build with NVIDIA CUDA support

```bash
cd /path/to/AIC2025_Track1_ZV

# Build image (includes MinkowskiEngine CUDA, pytorch-metric-learning, V-DETR extensions)
docker compose build --no-cache

# Expected output includes:
# - MinkowskiEngine CUDA OK ✓ (smoke test pass)
# - All native extension builds succeed
```

**Build time**: 15-30 minutes (depends on CPU/internet speed)

**Image size**: ~8-10 GB (includes PyTorch 1.8.1, CUDA 10.2, compiled extensions)

### Verify Build Success

```bash
# Start container shell
docker compose run --rm aic2025 bash

# Inside container, test MinkowskiEngine CUDA:
python3 << 'PYTEST'
import torch
import MinkowskiEngine as ME
coords = torch.IntTensor([[0,0,0,0], [0,1,1,1]]).cuda()
feats = torch.randn(2, 4).cuda()
x = ME.SparseTensor(features=feats, coordinates=coords)
print(f"✓ MinkowskiEngine CUDA works. SparseTensor on {x.F.device}")
PYTEST
```

## Running Inference

### 1. Start Docker Container

```bash
docker compose run --rm aic2025 bash
```

This mounts:
- `./dataset` → `/workspace/AIC2025_Track1_ZV/dataset`
- `./weights` → `/workspace/AIC2025_Track1_ZV/weights`
- `./outputs` → `/workspace/AIC2025_Track1_ZV/outputs`
- `./V-DETR/checkpoints` → `/workspace/AIC2025_Track1_ZV/V-DETR/checkpoints`
- `./V-DETR/configs` → `/workspace/AIC2025_Track1_ZV/V-DETR/configs`

### 2. Patch Inference Configuration (Inside Container)

```bash
bash scripts/patch_inference_config.sh V-DETR/configs/best.yaml
```

This ensures:
- `test_only: true` for inference-only mode
- `merge_cls: false` for 6-class detection head
- `dataset_root_dir: /workspace/AIC2025_Track1_ZV/dataset/pcd_dataset`

### 3. Run Single-Scene Inference

```bash
cd V-DETR

python3 inference.py \
  --config configs/best.yaml \
  --test_ckpt checkpoints/best/checkpoint_best.pth \
  --output_dir "/workspace/AIC2025_Track1_ZV/outputs/detection" \
  --data_root "/workspace/AIC2025_Track1_ZV/dataset/pcd_dataset" \
  --split_name "val" \
  --scene_name Lab_000
```

**Parameters**:
- `--config`: Path to configuration YAML (already patched)
- `--test_ckpt`: Path to model checkpoint
- `--output_dir`: Where to save detection results
- `--data_root`: Path to PCD dataset root
- `--split_name`: Dataset split ("val" or "test")
- `--scene_name`: Optional scene filter (can be repeated)

### 4. Check Results

```bash
# Inside container or on host (outputs are volume-mounted)
ls -lh /workspace/AIC2025_Track1_ZV/outputs/detection/
```

Expected output files:
- `Lab_000_detections.pkl` or similar (scene predictions)
- `detection_results.json` or similar (aggregated results)

## Troubleshooting

### 1. GPU Not Detected Inside Container

```bash
# Inside container
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

If `False`:
- Verify host GPU works: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`
- Rebuild image with `--no-cache`: `docker compose build --no-cache`
- Check MinkowskiEngine build logs for "FORCE_CUDA" and "| CUDA |"

### 2. MinkowskiEngine CPU_ONLY Error

```
AssertionError: The MinkowskiEngine was compiled with CPU_ONLY flag
```

**Solution**: Rebuild image
```bash
docker compose build --no-cache
# Verify smoke test in build output shows "MinkowskiEngine CUDA OK ✓"
```

### 3. PyTorch 1.8.1 Meshgrid Error

```
TypeError: meshgrid() got an unexpected keyword argument 'indexing'
```

**Solution**: This should be handled by the backward-compatible patch in `pytorch-metric-learning`. If still occurs:
```bash
# Inside container
grep -r "indexing=" --include="*.py" | grep meshgrid
# Should find only the try/except block with fallback
```

### 4. Out of Memory (OOM)

```bash
# Increase shared memory on host before starting container
docker compose run --rm --shm-size=32gb aic2025 bash

# Adjust batch size in config if needed
# Edit V-DETR/configs/best.yaml: batchsize_per_gpu: 1 (or lower)
```

### 5. Dataset Not Found

```
FileNotFoundError: /workspace/AIC2025_Track1_ZV/dataset/pcd_dataset/...
```

**Solution**: Ensure dataset is mounted
```bash
# Host: Check dataset exists
ls -lh dataset/pcd_dataset/

# Inside container: Verify mount
ls -lh /workspace/AIC2025_Track1_ZV/dataset/pcd_dataset/

# If empty, volume may not be mounted correctly - rebuild container
docker compose down
docker compose run --rm aic2025 bash
```

## Performance Notes

### GPU Memory Usage
- Inference batch size 1: ~4-6 GB VRAM
- Inference batch size 4: ~10-12 GB VRAM
- NVIDIA TITAN X: 12 GB total

### Runtime
- Single scene (~500K points): 5-15 seconds
- Full test set (10 scenes): 1-2 minutes
- Throughput: ~30-60 scenes/hour

### Optimization Tips
1. **Larger batch size** (if VRAM permits):
   - Edit `V-DETR/configs/best.yaml`: `batchsize_per_gpu: 4`
   - Rebuild container or volume-mount the file
2. **Distributed inference** (if multiple GPUs):
   - Not yet supported in this inference script
3. **ONNX export** (for faster inference):
   - See `V-DETR/inference.py` for optional ONNX export

## Docker Compose Commands Reference

```bash
# Build image
docker compose build --no-cache

# Start interactive shell
docker compose run --rm aic2025 bash

# Run specific command
docker compose run --rm aic2025 python3 -c "import torch; print(torch.cuda.is_available())"

# Stop all containers
docker compose down

# View logs from last run
docker compose logs aic2025

# Remove image (frees ~10 GB)
docker rmi aic2025-full:latest
```

## Environment Variables Inside Container

The entrypoint script automatically exports:
- `TORCH_LIB`: Path to PyTorch C++ libraries
- `CUDA_HOME`: `/usr/local/cuda`
- `LD_LIBRARY_PATH`: Includes PyTorch, CUDA, OpenBLAS paths
- `OMP_NUM_THREADS`: `12` (adjust for your CPU)
- `PATH`: Includes CUDA bin directory

To verify:
```bash
# Inside container
echo $CUDA_HOME
echo $LD_LIBRARY_PATH
echo $OMP_NUM_THREADS
```

## Key Configuration Files

- **`docker-compose.yml`**: Service definition, volumes, GPU config, environment
- **`Dockerfile.full`**: Complete build with all dependencies, CUDA support verification
- **`.dockerignore`**: Excludes large data files from build context
- **`entrypoint.sh`**: Runtime environment setup and verification
- **`scripts/patch_inference_config.sh`**: Config patching utility
- **`V-DETR/configs/best.yaml`**: Main inference configuration

## Support / Issues

If problems persist:
1. Check host GPU: `nvidia-smi`
2. Check docker GPU: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`
3. Rebuild image: `docker compose build --no-cache`
4. Check logs: `docker compose logs aic2025`
5. Inspect container: `docker compose run --rm aic2025 bash -c "python -c 'import torch; print(torch.cuda.__dict__)'"` 

## Additional Resources

- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [PyTorch Docker Hub](https://hub.docker.com/r/pytorch/pytorch)
- [docker-compose GPU support](https://docs.docker.com/compose/gpu-support/)
- [V-DETR Repository](https://github.com/sxiaolei/V-DETR)
