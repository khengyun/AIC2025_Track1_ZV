# Build & Installation Report
**Date:** May 4, 2026  
**Project:** AIC2025_Track1_ZV  
**Status:** ❌ **BUILD FAILED - Multiple Critical Issues**

---

## Executive Summary

All 5 build steps have **FAILED**. The primary issues are:
1. **Python 3.13.12 incompatibility** - distutils removed in Python 3.12+
2. **Missing dependencies** - Cython, OpenBLAS development headers, patchelf
3. **CUDA not accessible** - PyTorch reports CUDA unavailable despite nvcc present
4. **Network issues** - pip unable to fetch setuptools (403 Forbidden)

**Priority:** CRITICAL - All components must be fixed before proceeding

---

## Detailed Issue Analysis

### [1] ✗ MinkowskiEngine - FAILED

**Error Type:** C++ Compilation Error  
**Root Cause:** Missing OpenBLAS development headers and/or BLAS configuration  

**Error Message:**
```
broadcast_kernel.hpp:28: error: expected declaration or statement
```

**Missing Dependencies:**
- `libopenblas-dev` - OpenBLAS development headers
- `liblapack-dev` - LAPACK development headers  
- `libblas-dev` - BLAS development headers

**Fix:**
```bash
sudo apt-get install -y libopenblas-dev liblapack-dev libblas-dev
cd MinkowskiEngine
export TORCH_CUDA_ARCH_LIST="6.1"
export MAX_JOBS=3
python setup.py install --blas_include_dirs=/usr/include/x86_64-linux-gnu --blas=openblas
```

**Blocking:** YES - Required for point cloud processing

---

### [2] ✗ pytorch-metric-learning - FAILED

**Error Type:** Network Error (pip)  
**Root Cause:** 403 Forbidden response from PyPI (likely firewall/network issue)  

**Error Message:**
```
SError('Tunnel connection failed: 403 Forbidden'): /simple/setuptools/
ERROR: Could not find a version that satisfies the requirement setuptools>=40.8.0
```

**Possible Causes:**
- Corporate firewall blocking PyPI
- pip index server not reachable
- Stale pip cache

**Fix Options:**

**Option A - Update pip and retry:**
```bash
pip install --upgrade pip
pip install -e pytorch-metric-learning/
```

**Option B - Use alternative PyPI mirror:**
```bash
pip install -i https://mirrors.aliyun.com/pypi/simple/ -e pytorch-metric-learning/
```

**Option C - Install from requirements.txt offline:**
```bash
cd pytorch-metric-learning
pip install --no-index -f wheels/ -e .
```

**Blocking:** PARTIAL - Required for ReID model training

---

### [3] ✗ V-DETR/utils (Cython) - FAILED

**Error Type:** Missing Module  
**Root Cause:** Cython not installed  

**Error Message:**
```
ModuleNotFoundError: No module named 'Cython'
```

**Missing Dependencies:**
- `cython` - Python C extension compiler

**Fix:**
```bash
pip install cython
cd V-DETR/utils
python cython_compile.py build_ext --inplace
```

**Blocking:** YES - V-DETR utils cannot be compiled without Cython

---

### [4] ✗ PointNet2 - FAILED

**Error Type:** Python Version Incompatibility  
**Root Cause:** `setup.py` uses deprecated `distutils` (removed in Python 3.12+)  

**Current Python:** 3.13.12 ❌  
**Required Python:** ≤3.10 or modern build system (PEP 517/518)  

**Error Message:**
```
This deprecation is overdue, please update your project and remove deprecated calls to avoid build errors
```

**Fix Options:**

**Option A - Downgrade Python to 3.10 (Recommended):**
```bash
# Using conda
conda create -n py310 python=3.10
conda activate py310

# Then rebuild in this environment
cd V-DETR/third_party/pointnet2
python setup.py install --user
```

**Option B - Patch setup.py to use modern build system:**
Edit `V-DETR/third_party/pointnet2/setup.py`:
- Replace `from distutils.core import setup` with proper setuptools
- Add `pyproject.toml` with PEP 517 metadata

**Blocking:** YES - PointNet2 is core 3D feature extractor

---

### [5] ✗ PointNet2 Shared Library Patch - BLOCKED

**Status:** Blocked (depends on PointNet2 build success)  
**Missing Tool:** `patchelf` not installed  

**Fix:**
```bash
sudo apt-get install -y patchelf
# Then retry after PointNet2 builds successfully
```

**Blocking:** YES - Affects runtime library paths

---

## Environment Issues

### CUDA Accessibility Issue ⚠️

**Current Status:**
- ✓ nvcc available: `NVIDIA (R) Cuda compiler driver, release ...`
- ✓ PyTorch built with CUDA 13.0: `torch 2.11.0+cu130`
- ✗ **PyTorch cannot access CUDA:** `torch.cuda.is_available() = False`
- ✗ **Device count:** 0

**Library Path Issues:**
- ✗ `/usr/local/cuda/lib64` NOT FOUND
- ✗ `/opt/conda/lib` NOT FOUND  
- ✓ `/usr/lib/x86_64-linux-gnu` EXISTS

**Root Cause:** CUDA library paths not properly configured in Python environment

**Fixes Needed:**
```bash
# Option 1: Set LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Option 2: Link CUDA libraries to system path
sudo ln -s /usr/local/cuda/lib64/*.so* /usr/local/lib/
sudo ldconfig

# Verify:
python3 -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

---

## Recommended Fix Sequence

### Phase 1: System Dependencies (5-10 minutes)
```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  libopenblas-dev \
  liblapack-dev \
  libblas-dev \
  python3-dev \
  cython \
  patchelf
```

### Phase 2: Python Build Tools (2-5 minutes)
```bash
pip install --upgrade pip setuptools wheel build
pip install cython
```

### Phase 3: CUDA Configuration (2-5 minutes)
```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
```

### Phase 4: Component Builds (15-45 minutes)

**4a. MinkowskiEngine:**
```bash
cd MinkowskiEngine
export TORCH_CUDA_ARCH_LIST="6.1"
export MAX_JOBS=3
python setup.py install --blas_include_dirs=/usr/include/x86_64-linux-gnu --blas=openblas
cd ..
```

**4b. pytorch-metric-learning:**
```bash
cd pytorch-metric-learning
pip install -e .
cd ..
```

**4c. V-DETR utils:**
```bash
cd V-DETR/utils
python cython_compile.py build_ext --inplace
cd ../..
```

**4d. PointNet2 (choose one):**

*Option 1: Create Python 3.10 environment*
```bash
conda create -n py310 python=3.10 pytorch::pytorch pytorch::pytorch-cuda=12.1 -c pytorch -c nvidia
conda activate py310
cd V-DETR/third_party/pointnet2
python setup.py install --user
cd ../../../
```

*Option 2: Patch setup.py (Advanced)*
```bash
# Modify V-DETR/third_party/pointnet2/setup.py to use setuptools
# Create pyproject.toml with PEP 517 configuration
cd V-DETR/third_party/pointnet2
python -m build
pip install dist/*.whl
cd ../../../
```

### Phase 5: Library Patching (1-2 minutes)
```bash
export TORCH_LIB=$(python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$TORCH_LIB:$CUDA_HOME/lib64:/opt/conda/lib:$LD_LIBRARY_PATH

PN2_SO=$(find ~/.local/lib -name "_ext*.so" | grep pointnet2 | head -1)
if [ ! -z "$PN2_SO" ]; then
  patchelf --set-rpath "$TORCH_LIB:$CUDA_HOME/lib64:/opt/conda/lib" "$PN2_SO"
else
  echo "Warning: PointNet2 .so not found"
fi
```

---

## Estimated Total Time

| Phase | Component | Est. Time |
|-------|-----------|-----------|
| 1 | System Dependencies | 5-10 min |
| 2 | Python Tools | 2-5 min |
| 3 | CUDA Config | 2-5 min |
| 4a | MinkowskiEngine | 5-15 min |
| 4b | pytorch-metric-learning | 2-5 min |
| 4c | V-DETR utils | 1-3 min |
| 4d | PointNet2 | 10-30 min |
| 5 | Library Patching | 1-2 min |
| **TOTAL** | | **28-75 min** |

---

## Verification Steps

After each phase, verify:

```bash
# After Phase 3 (CUDA):
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# After Phase 4a (MinkowskiEngine):
python3 -c "import MinkowskiEngine as ME; print(f'MinkowskiEngine: {ME.__version__}')"

# After Phase 4b (pytorch-metric-learning):
python3 -c "import pytorch_metric_learning; print('pytorch-metric-learning OK')"

# After Phase 4c (V-DETR):
python3 -c "from V_DETR import utils; print('V-DETR utils OK')"

# After Phase 4d (PointNet2):
python3 -c "from pointnet2_ops import pointnet2_utils; print('PointNet2 OK')"
```

---

## Critical Notes

⚠️ **Python Version:**  
Current 3.13.12 will NOT work with PointNet2's setup.py. Either:
- Use Python 3.10 environment (simplest)
- OR significantly patch setup.py files

⚠️ **CUDA Access:**  
PyTorch shows CUDA is unavailable. This MUST be fixed before training, as all models require GPU access.

⚠️ **TITAN X Pascal:**  
Compute capability 6.1 is set correctly, but verify with:
```bash
python3 -c "import torch; print(torch.cuda.get_device_capability())"
```

---

## Support for Issues

If any step fails, check:

1. **Check detailed error:** Re-run command to capture full error
2. **Check dependencies:** `apt-cache search <package>`
3. **Check PATH:** `echo $PATH`, `which python3`
4. **Check CUDA:** `nvcc --version`, `nvidia-smi` (if GPU available)
5. **Check network:** `pip list`, `pip index versions setuptools`

---

**Report Generated:** 2026-05-04  
**Next Step:** Execute Phase 1-3 system setup, then proceed with builds
