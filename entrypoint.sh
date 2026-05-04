#!/bin/bash

# Setup TORCH library path
export TORCH_LIB=$(python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$TORCH_LIB:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=12
export PATH=$CUDA_HOME/bin:$PATH

# Verify GPU and PyTorch setup on container start
echo "=== AIC2025 Docker Environment ==="
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q "True"; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))')"
fi
echo "=================================="
echo ""

exec "$@"
