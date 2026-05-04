#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/niran/Documents/GitHub/AIC2025_Track1_ZV"
IMAGE_NAMES=("aic2025-full:latest" "aic2025-me102:latest")
CONTAINER_PATTERNS=("aic2025" "aic2025_full_dev" "aic2025_me102_dev")

cd "$REPO_DIR"

echo "=================================================="
echo "[1/8] Check host Docker GPU"
echo "=================================================="
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

echo "=================================================="
echo "[2/8] Stop docker compose services"
echo "=================================================="
docker compose down --remove-orphans || true

echo "=================================================="
echo "[3/8] Stop/remove old AIC2025 containers"
echo "=================================================="
for pattern in "${CONTAINER_PATTERNS[@]}"; do
  ids=$(docker ps -aq --filter "name=${pattern}" || true)
  if [ -n "$ids" ]; then
    echo "$ids" | xargs -r docker rm -f
  fi
done

echo "=================================================="
echo "[4/8] Remove old AIC2025 images"
echo "=================================================="
for img in "${IMAGE_NAMES[@]}"; do
  if docker image inspect "$img" >/dev/null 2>&1; then
    docker rmi -f "$img" || true
  fi
done

echo "=================================================="
echo "[5/8] Clean Docker builder cache"
echo "=================================================="
docker builder prune -af

echo "=================================================="
echo "[6/8] Clean dangling images/cache only"
echo "=================================================="
docker image prune -af

echo "=================================================="
echo "[7/8] Show disk usage after cleanup"
echo "=================================================="
docker system df

echo "=================================================="
echo "[8/8] Rebuild Docker image from scratch"
echo "=================================================="
docker compose build --no-cache --progress=plain

echo "=================================================="
echo "Build finished. Now run:"
echo "docker compose run --rm aic2025 bash"
echo "=================================================="
