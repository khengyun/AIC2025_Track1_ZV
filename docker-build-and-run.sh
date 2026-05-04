#!/bin/bash
# AIC2025 TITAN X Docker Setup Script
# Usage: ./docker-build-and-run.sh [build|run|shell|test]

set -e

DOCKER_IMAGE_NAME="aic2025-me102"
DOCKER_CONTAINER_NAME="aic2025_me102_dev"
REPO_PATH="$HOME/Documents/GitHub/AIC2025_Track1_ZV"
WORKSPACE_MOUNT="/workspace/AIC2025_Track1_ZV"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Function: Check if Docker GPU is available
check_docker_gpu() {
    echo_info "Checking Docker GPU availability..."
    if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
        echo_info "Docker GPU check passed ✓"
    else
        echo_warn "Docker GPU check failed. Make sure NVIDIA Container Toolkit is installed."
        echo "Run: sudo apt-get install -y nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
    fi
}

# Function: Build Docker image
build_image() {
    echo_info "Building Docker image: $DOCKER_IMAGE_NAME"
    
    if [ ! -f "$REPO_PATH/Dockerfile.me102" ]; then
        echo_error "Dockerfile.me102 not found at $REPO_PATH/Dockerfile.me102"
        exit 1
    fi
    
    if [ ! -f "$REPO_PATH/entrypoint.sh" ]; then
        echo_error "entrypoint.sh not found at $REPO_PATH/entrypoint.sh"
        exit 1
    fi
    
    cd "$REPO_PATH"
    docker build --no-cache -f Dockerfile.me102 -t "$DOCKER_IMAGE_NAME" .
    
    echo_info "Docker image built successfully ✓"
    echo_info "Image name: $DOCKER_IMAGE_NAME"
}

# Function: Run container
run_container() {
    echo_info "Running Docker container..."
    
    if [ ! -d "$REPO_PATH" ]; then
        echo_error "Repo directory not found: $REPO_PATH"
        exit 1
    fi
    
    # Check if container already exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${DOCKER_CONTAINER_NAME}$"; then
        echo_warn "Container $DOCKER_CONTAINER_NAME already exists. Starting it..."
        docker start -ai "$DOCKER_CONTAINER_NAME"
    else
        echo_info "Creating and starting new container..."
        docker run -it --gpus all \
            --name "$DOCKER_CONTAINER_NAME" \
            --ipc=host \
            --shm-size=16g \
            -v "$REPO_PATH":"$WORKSPACE_MOUNT" \
            "$DOCKER_IMAGE_NAME"
    fi
}

# Function: Open shell in existing container
shell_container() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${DOCKER_CONTAINER_NAME}$"; then
        if docker ps --format '{{.Names}}' | grep -q "^${DOCKER_CONTAINER_NAME}$"; then
            echo_info "Attaching to running container..."
            docker exec -it "$DOCKER_CONTAINER_NAME" /bin/bash
        else
            echo_info "Starting container..."
            docker start -ai "$DOCKER_CONTAINER_NAME"
        fi
    else
        echo_error "Container $DOCKER_CONTAINER_NAME not found. Please run 'build' and 'run' first."
        exit 1
    fi
}

# Function: Run smoke test
run_smoke_test() {
    echo_info "Running smoke test..."
    
    if ! docker ps --format '{{.Names}}' | grep -q "^${DOCKER_CONTAINER_NAME}$"; then
        echo_error "Container $DOCKER_CONTAINER_NAME is not running"
        exit 1
    fi
    
    docker exec "$DOCKER_CONTAINER_NAME" bash -c '
        export TORCH_LIB=$(python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), \"lib\"))")
        export CUDA_HOME=/usr/local/cuda
        export LD_LIBRARY_PATH=$TORCH_LIB:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
        export OMP_NUM_THREADS=12
        
        python - <<PY
import torch
import mmcv

print("✓ torch:", torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("✓ mmcv:", mmcv.__version__)
if torch.cuda.is_available():
    print("✓ GPU:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    x = torch.randn(2, 2, device="cuda")
    torch.cuda.synchronize()
    print("✓ PyTorch CUDA OK")
print("Smoke test passed!")
PY
    '
}

# Function: Commit image after successful setup
commit_image() {
    local tag=$1
    echo_info "Committing container to image: $DOCKER_IMAGE_NAME-$tag"
    
    if docker ps -a --format '{{.Names}}' | grep -q "^${DOCKER_CONTAINER_NAME}$"; then
        docker commit "$DOCKER_CONTAINER_NAME" "$DOCKER_IMAGE_NAME-$tag"
        echo_info "Image committed: $DOCKER_IMAGE_NAME-$tag ✓"
    else
        echo_error "Container not found"
        exit 1
    fi
}

# Function: Show usage
show_usage() {
    cat << EOF
AIC2025 TITAN X Docker Setup Script

Usage: $0 [command]

Commands:
    build       - Build Docker image
    run         - Run/start Docker container
    shell       - Open shell in running container
    test        - Run smoke test
    commit TAG  - Commit container to image with TAG
    gpu-check   - Check Docker GPU availability
    help        - Show this help message

Examples:
    $0 build
    $0 run
    $0 shell
    $0 test
    $0 commit step7-ok

Inside container, after build steps:
    cd $WORKSPACE_MOUNT
    
    # Build MinkowskiEngine
    cd MinkowskiEngine
    export TORCH_CUDA_ARCH_LIST="6.1"
    export MAX_JOBS=3
    python setup.py install --blas_include_dirs=/usr/include/x86_64-linux-gnu --blas=openblas
    cd ..
    
    # Install pytorch-metric-learning
    cd pytorch-metric-learning
    pip install -e .
    cd ..
    
    # Compile Cython utils
    cd V-DETR/utils
    python cython_compile.py build_ext --inplace
    cd ../..
    
    # Build and patch pointnet2
    cd V-DETR/third_party/pointnet2
    python setup.py install --user
    cd ../../../
    
    # Patch pointnet2 RPATH
    PN2_SO=\$(find /root/.local/lib/python3.7/site-packages -name "_ext*.so" | grep pointnet2 | head -1)
    patchelf --set-rpath "\$TORCH_LIB:\$CUDA_HOME/lib64:/opt/conda/lib" "\$PN2_SO"

EOF
}

# Main
case "${1:-help}" in
    build)
        check_docker_gpu
        build_image
        ;;
    run)
        run_container
        ;;
    shell)
        shell_container
        ;;
    test)
        run_smoke_test
        ;;
    commit)
        if [ -z "$2" ]; then
            echo_error "Tag name required for commit"
            exit 1
        fi
        commit_image "$2"
        ;;
    gpu-check)
        check_docker_gpu
        ;;
    help)
        show_usage
        ;;
    *)
        echo_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac
