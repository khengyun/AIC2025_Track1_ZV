Hướng dẫn cài đặt AIC2025_Track1_ZV trên TITAN X Pascal bằng Docker
Tài liệu này tổng hợp lại toàn bộ quá trình setup repo ZIOVISION/AIC2025_Track1_ZV trên máy có GPU NVIDIA TITAN X Pascal, bao gồm các lỗi đã gặp và cách fix.

Kết luận chính: trên host Ubuntu 22.04, stack PyTorch 1.10/1.12 + CUDA 11.3 + MinkowskiEngine 0.5.4 build được nhưng runtime lỗi CUDA error: device kernel image is invalid. Hướng ổn định đã test pass là chạy bằng Docker legacy: PyTorch 1.8.1 + CUDA 10.2 + MinkowskiEngine 0.5.4.

0. Trạng thái cuối cùng đã pass
Các test đã pass trong Docker:

torch: 1.8.1
torch cuda: 10.2
cuda available: True
GPU: NVIDIA TITAN X (Pascal) (6, 1)
MinkowskiEngine: 0.5.4
ME CUDA op OK: torch.Size([3, 8]) True
pointnet2 import OK
Step 7 smoke test OK
1. Vì sao không dùng setup native trên Ubuntu 22.04?
Đã thử native với các env:

aic2025_clean: PyTorch 1.12.1 + CUDA 11.3 + MinkowskiEngine 0.5.4
aic2025_me110: PyTorch 1.10.2 + CUDA 11.3 + MinkowskiEngine 0.5.4
Hiện tượng:

PyTorch CUDA trước khi import MinkowskiEngine: OK
MinkowskiEngine import: OK
PyTorch CUDA sau khi import MinkowskiEngine: FAIL
RuntimeError: CUDA error: device kernel image is invalid
Đã thử các hướng sau nhưng vẫn lỗi:

- Build lại MinkowskiEngine với sm_61
- Build lại với 6.1+PTX
- Patch RPATH
- Dùng system OpenBLAS
- Dùng conda CUDA libs / system CUDA libs
- Hạ PyTorch từ 1.12.1 xuống 1.10.2
Kết luận: lỗi không phải do GPU hỏng hay PyTorch CUDA hỏng. Lỗi nằm ở compatibility giữa MinkowskiEngine 0.5.4 + CUDA 11.3 + TITAN X Pascal sm_61.

Docker legacy giải quyết được vì dùng stack cũ hơn:

PyTorch 1.8.1
CUDA 10.2
Python 3.7.10
MinkowskiEngine 0.5.4
2. Kiểm tra Docker GPU
Trên host:

docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
Kết quả mong muốn: container nhìn thấy NVIDIA TITAN X (Pascal).

Nếu gặp lỗi kiểu:

failed to discover GPU vendor from CDI: no known GPU vendor found
cần cài NVIDIA Container Toolkit:

sudo apt-get update
sudo apt-get install -y curl gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
Sau đó test lại docker run --rm --gpus all ... nvidia-smi.

3. Tạo Docker image legacy
Tạo thư mục Docker:

mkdir -p ~/Documents/AIC2026/docker_aic2025
cd ~/Documents/AIC2026/docker_aic2025
Tạo Dockerfile.me102:

cat > Dockerfile.me102 <<'DOCKER'
FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN rm -f /etc/apt/sources.list.d/cuda.list \
          /etc/apt/sources.list.d/nvidia-ml.list \
          /etc/apt/sources.list.d/cuda*.list \
          /etc/apt/sources.list.d/nvidia*.list || true && \
    apt-get update && apt-get install -y \
    git curl ca-certificates \
    build-essential libopenblas-dev ninja-build patchelf \
    libprotobuf-dev protobuf-compiler \
    ffmpeg libsm6 libxext6 libxrender-dev libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "setuptools==59.5.0" \
    "wheel" \
    "numpy==1.21.6" \
    "typing_extensions" \
    "ninja" \
    "cython"

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH

WORKDIR /workspace
CMD ["/bin/bash"]
DOCKER
Build image:

docker build --no-cache -f Dockerfile.me102 -t aic2025-me102 .
Lỗi đã gặp: NVIDIA apt key trong image cũ
Ban đầu build lỗi:

NO_PUBKEY A4B469963BF863CC
The repository 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 InRelease' is not signed.
Cách fix: trong Dockerfile đã xóa các file apt source NVIDIA cũ trước khi chạy apt-get update:

rm -f /etc/apt/sources.list.d/cuda.list \
      /etc/apt/sources.list.d/nvidia-ml.list \
      /etc/apt/sources.list.d/cuda*.list \
      /etc/apt/sources.list.d/nvidia*.list
Lỗi đã gặp: numpy==1.22.4 không hỗ trợ Python 3.7
Image pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel dùng Python 3.7.10. numpy==1.22.4 không phù hợp, nên đổi thành:

numpy==1.21.6
4. Chạy container
Repo trên host:

~/Documents/AIC2026/AIC2025_Track1_ZV
Chạy container:

docker run -it --gpus all \
  --name aic2025_me102_dev \
  --ipc=host \
  --shm-size=16g \
  -v ~/Documents/AIC2026/AIC2025_Track1_ZV:/workspace/AIC2025_Track1_ZV \
  aic2025-me102
Sau khi vào container:

cd /workspace/AIC2025_Track1_ZV
Kiểm tra PyTorch/CUDA/nvcc:

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
x = torch.randn(2, 2, device="cuda")
torch.cuda.synchronize()
print("PyTorch CUDA OK")
PY

which nvcc
nvcc --version
python --version
Kết quả mong muốn:

torch: 1.8.1
torch cuda: 10.2
cuda available: True
gpu: NVIDIA TITAN X (Pascal) (6, 1)
PyTorch CUDA OK
nvcc release 10.2
Python 3.7.10
5. Build MinkowskiEngine
Trong container:

cd /workspace/AIC2025_Track1_ZV/MinkowskiEngine

rm -rf build dist MinkowskiEngine.egg-info ~/.cache/torch_extensions

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_LIB=$(python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$TORCH_LIB:$CONDA_PREFIX/lib:$CUDA_HOME/lib64
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:$CUDA_HOME/lib64
export CPATH=/usr/include/x86_64-linux-gnu
export TORCH_CUDA_ARCH_LIST="6.1"
export MAX_JOBS=3
export OMP_NUM_THREADS=4

python setup.py install --blas_include_dirs=/usr/include/x86_64-linux-gnu --blas=openblas
Nếu máy lag, giảm:

export MAX_JOBS=1
Test MinkowskiEngine import + CUDA
cd /workspace/AIC2025_Track1_ZV

export OMP_NUM_THREADS=12

python -c "import torch; import MinkowskiEngine as ME; print('torch:', torch.__version__, torch.version.cuda); print('ME:', ME.__version__); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); x=torch.randn(2,2,device='cuda'); torch.cuda.synchronize(); print('torch CUDA after ME import OK')"
Test convolution:

python -c "import torch; import MinkowskiEngine as ME; coords=torch.IntTensor([[0,0,0,0],[0,1,1,1],[0,2,2,2]]).cuda(); feats=torch.randn(3,4,device='cuda'); x=ME.SparseTensor(features=feats, coordinates=coords); conv=ME.MinkowskiConvolution(4,8,kernel_size=3,dimension=3).cuda(); y=conv(x); torch.cuda.synchronize(); print('ME CUDA op OK:', y.F.shape, y.F.is_cuda)"
Kết quả mong muốn:

torch CUDA after ME import OK
ME CUDA op OK: torch.Size([3, 8]) True
6. Cài requirements root
Không chạy trực tiếp full requirements nếu có package không hợp Python 3.7. Cách đã dùng:

cd /workspace/AIC2025_Track1_ZV
Cài riêng các package nhạy cảm:

pip install --no-cache-dir \
  "protobuf==4.24.4" \
  "safetensors==0.4.5" \
  "huggingface-hub==0.16.4" \
  "timm==0.6.13" \
  "wandb==0.13.11"
Tạo requirements đã lọc:

grep -viE "^(torch|torchvision|torchaudio|cudatoolkit|mmcv|mmcv-full|open3d|Minkowski|protobuf|timm|wandb|safetensors|huggingface-hub)" requirements.txt > /tmp/root_requirements_safe.txt
Cài phần còn lại:

pip install --no-cache-dir -r /tmp/root_requirements_safe.txt
Cài Open3D riêng:

pip install --no-cache-dir "open3d==0.17.0"
Lỗi đã gặp: pip install requirements.txt
Sai:

pip install requirements.txt
Đúng:

pip install -r requirements.txt
Lỗi đã gặp: protobuf==4.25.2
Python 3.7 không tìm được protobuf==4.25.2, nên dùng:

pip install "protobuf==4.24.4"
Lỗi đã gặp: safetensors kéo puccinialin
safetensors bản mới cố build từ source và kéo dependency không tồn tại puccinialin. Fix bằng pin:

pip install "safetensors==0.4.5"
7. Cài pytorch-metric-learning
cd /workspace/AIC2025_Track1_ZV/pytorch-metric-learning
pip install -e .
cd /workspace/AIC2025_Track1_ZV
Test:

python - <<'PY'
import torch
import MinkowskiEngine as ME
import pytorch_metric_learning

print("torch:", torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("ME:", ME.__version__)
print("pytorch_metric_learning import OK")
print("GPU:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
8. Cài Step 7 third-party dependencies
8.1 openmim
pip install openmim==0.3.9
8.2 mmcv-full
Vì container dùng CUDA 10.2 + PyTorch 1.8.1, dùng wheel đúng stack:

pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html
Test:

python -c "import mmcv; print('mmcv:', mmcv.__version__)"
8.3 pointnet2
cd /workspace/AIC2025_Track1_ZV/V-DETR/third_party/pointnet2
python setup.py install --user
cd /workspace/AIC2025_Track1_ZV
Build thành công khi thấy:

Installed /root/.local/lib/python3.7/site-packages/pointnet2-0.0.0-py3.7-linux-x86_64.egg
Finished processing dependencies for pointnet2==0.0.0
Lỗi đã gặp: ImportError: libc10.so
Khi import:

python -c "import pointnet2; from pointnet2 import _ext"
lỗi:

ImportError: libc10.so: cannot open shared object file: No such file or directory
Fix tạm thời bằng environment:

export TORCH_LIB=$(python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$TORCH_LIB:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=12
Patch RPATH cố định cho _ext.so:

PN2_SO=$(find /root/.local/lib/python3.7/site-packages -name "_ext*.so" | grep pointnet2 | head -1)
patchelf --set-rpath "$TORCH_LIB:$CUDA_HOME/lib64:/opt/conda/lib" "$PN2_SO"
Test lại:

python -c "import pointnet2; from pointnet2 import _ext; print('pointnet2 import OK after RPATH patch')"
8.4 h5py/open3d/shapely/filterpy
Đã cài trong requirements, nhưng có thể chạy lại:

pip install h5py open3d==0.17.0 shapely filterpy
8.5 Compile Cython utils
cd /workspace/AIC2025_Track1_ZV/V-DETR/utils
python cython_compile.py build_ext --inplace
cd /workspace/AIC2025_Track1_ZV
Warning Cython/NumPy deprecated API có thể bỏ qua nếu không có ERROR hoặc Traceback.

Test:

cd /workspace/AIC2025_Track1_ZV/V-DETR/utils
python -c "import box_intersection; print('box_intersection import OK')"
cd /workspace/AIC2025_Track1_ZV
9. Smoke test tổng sau khi cài xong
Trước khi test, export env:

export TORCH_LIB=$(python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$TORCH_LIB:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=12
Chạy test 1 dòng:

python -c "import torch; import MinkowskiEngine as ME; import pytorch_metric_learning; import mmcv; import h5py; import open3d as o3d; import shapely; import filterpy; import pointnet2; from pointnet2 import _ext; print('torch:', torch.__version__, torch.version.cuda, torch.cuda.is_available()); print('ME:', ME.__version__); print('mmcv:', mmcv.__version__); print('open3d:', o3d.__version__); print('pointnet2 import OK'); print('GPU:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); coords=torch.IntTensor([[0,0,0,0],[0,1,1,1],[0,2,2,2]]).cuda(); feats=torch.randn(3,4,device='cuda'); x=ME.SparseTensor(features=feats, coordinates=coords); conv=ME.MinkowskiConvolution(4,8,kernel_size=3,dimension=3).cuda(); y=conv(x); torch.cuda.synchronize(); print('ME CUDA op OK:', y.F.shape, y.F.is_cuda); print('Step 7 smoke test OK')"
Kết quả mong muốn:

torch: 1.8.1 10.2 True
ME: 0.5.4
mmcv: 1.6.1
open3d: 0.17.0
pointnet2 import OK
GPU: NVIDIA TITAN X (Pascal) (6, 1)
ME CUDA op OK: torch.Size([3, 8]) True
Step 7 smoke test OK
10. Lưu container thành image
Thoát container:

exit
Trên host:

docker commit aic2025_me102_dev aic2025-me102-step7-ok
Sau này vào lại container cũ:

docker start -ai aic2025_me102_dev
Hoặc chạy container mới từ image đã commit:

docker run -it --gpus all \
  --ipc=host \
  --shm-size=16g \
  -v ~/Documents/AIC2026/AIC2025_Track1_ZV:/workspace/AIC2025_Track1_ZV \
  aic2025-me102-step7-ok
11. Các lệnh one-line hữu ích
Start container:

docker start -ai aic2025_me102_dev
Kiểm tra torch + GPU:

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
Test MinkowskiEngine CUDA:

python -c "import torch; import MinkowskiEngine as ME; coords=torch.IntTensor([[0,0,0,0],[0,1,1,1],[0,2,2,2]]).cuda(); feats=torch.randn(3,4,device='cuda'); x=ME.SparseTensor(features=feats, coordinates=coords); conv=ME.MinkowskiConvolution(4,8,kernel_size=3,dimension=3).cuda(); y=conv(x); torch.cuda.synchronize(); print('ME CUDA op OK:', y.F.shape, y.F.is_cuda)"
Fix env runtime libs:

export TORCH_LIB=$(python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"); export CUDA_HOME=/usr/local/cuda; export LD_LIBRARY_PATH=$TORCH_LIB:$CUDA_HOME/lib64:$LD_LIBRARY_PATH; export OMP_NUM_THREADS=12
Test pointnet2:

python -c "import pointnet2; from pointnet2 import _ext; print('pointnet2 import OK')"
12. Ghi chú quan trọng
Không chạy lại Step 3 README trong Docker:
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
Lệnh này sẽ phá stack đang chạy được.

Không chạy lại Step 5 README y nguyên:
conda install -c conda-forge openblas -y
MinkowskiEngine đã build ổn với system OpenBLAS trong Docker.

Mỗi khi restart/vào lại container, nên export lại:
export TORCH_LIB=$(python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$TORCH_LIB:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=12
Nên commit image sau mỗi mốc lớn:
docker commit aic2025_me102_dev aic2025-me102-me-ok
docker commit aic2025_me102_dev aic2025-me102-step6-ok
docker commit aic2025_me102_dev aic2025-me102-step7-ok
13. Checklist hoàn tất environment
[x] Docker thấy GPU
[x] PyTorch 1.8.1 CUDA 10.2 chạy được
[x] nvcc 10.2 có trong container
[x] MinkowskiEngine 0.5.4 build được
[x] MinkowskiEngine CUDA convolution pass
[x] requirements root đã cài bằng cách lọc package không phù hợp Python 3.7
[x] protobuf fix về 4.24.4
[x] safetensors fix về 0.4.5
[x] pytorch-metric-learning cài được
[x] mmcv-full 1.6.1 cài đúng CUDA 10.2 / torch 1.8
[x] pointnet2 build được
[x] pointnet2 import fix bằng LD_LIBRARY_PATH + patchelf
[x] Cython utils compile được
[ ] Chạy inference/training sample nhỏ của repo