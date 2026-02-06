# MMPose Docker Setup – Lessons Learned

> **Tested on:** Dell Pro Max Tower T2 | RTX 5090 (32GB) | Ubuntu 22.04 LTS | Driver 590.48.01
> **Date:** February 2026

---

## Working Configuration

| Component | Version | Notes |
|-----------|---------|-------|
| CUDA Base Image | 12.8.0-cudnn-devel-ubuntu22.04 | Must use `devel` for mmcv compilation |
| PyTorch | 2.9.1+cu128 | First fully stable Blackwell support |
| mmcv | 2.1.0 | Built from source with CUDA ops |
| mmdet | 3.2.x (< 3.3.0) | 3.3.0 has mmcv version check issues |
| mmengine | 0.10.7 | Installed via mim |
| mmpose | 1.3.2 | Installed from source (editable) |
| pyrealsense2 | 2.56.x | For Intel RealSense D435 camera |
| Python | 3.10 | Ships with Ubuntu 22.04 base image |

---

## Issues Encountered & Solutions

### 1. RTX 5090 (Blackwell) requires CUDA 12.8+

**Problem:** The official MMPose Dockerfile uses CUDA 10.2 / PyTorch 1.8 — far too old for RTX 5090 (Blackwell, compute capability sm_120).

**Solution:** Use `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04` base image and PyTorch 2.9.1 with cu128:
```dockerfile
RUN pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128
```

Set the CUDA architecture list to include Blackwell:
```dockerfile
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 12.0+PTX"
```

---

### 2. chumpy fails to build (ModuleNotFoundError: No module named 'pip')

**Problem:** `chumpy` (an MMPose runtime dependency for SMPL body meshes) has a broken `pyproject.toml` build system. When pip creates an isolated build environment, chumpy's setup.py tries to import pip, which isn't available in the isolated env.

**Solution:** Install with `--no-build-isolation` so it uses the system pip/numpy:
```dockerfile
RUN pip install --no-build-isolation chumpy
```

---

### 3. xtcocotools numpy ABI mismatch (dtype size changed)

**Problem:** The prebuilt `xtcocotools` wheel from PyPI was compiled against a different numpy version, causing:
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
```

**Solution:** Force rebuild from source against the installed numpy:
```dockerfile
RUN pip install --no-cache-dir --no-binary xtcocotools --no-build-isolation xtcocotools
```

Key flags:
- `--no-binary xtcocotools`: forces source build instead of using prebuilt wheel
- `--no-build-isolation`: uses system numpy for compilation
- `--no-cache-dir`: prevents pip from reusing a cached incompatible wheel

---

### 4. mmcv installs without CUDA ops (No module named 'mmcv._ext')

**Problem:** No prebuilt mmcv wheel with CUDA ops exists for the cu128/torch2.9 combination. Both `pip install` and `mim install` grab the pure-Python wheel (`mmcv-2.1.0-py2.py3-none-any.whl`), which lacks compiled CUDA extensions like `MultiScaleDeformableAttention`.

**Solution:** Force source build with CUDA ops enabled:
```dockerfile
ENV MMCV_WITH_OPS="1"
ENV FORCE_CUDA="1"

RUN pip install mmcv==2.1.0 --no-cache-dir --no-binary mmcv --no-build-isolation
```

**Build time:** ~10-15 minutes (compiles C++/CUDA extensions).

The devel base image is required for this step (provides nvcc and CUDA headers).

---

### 5. mmdet 3.3.0 incompatible with mmcv 2.2.0

**Problem:** `mim install "mmcv>=2.0.1"` installs mmcv 2.2.0 by default. However, mmdet 3.3.0 has a strict version check requiring `mmcv>=2.0.0rc4,<2.2.0`, causing:
```
AssertionError: MMCV==2.2.0 is used but incompatible. Please install mmcv>=2.0.0rc4, <2.2.0.
```

**Solution:** Pin mmcv to 2.1.0 and mmdet to <3.3.0:
```dockerfile
RUN pip install mmcv==2.1.0 --no-cache-dir --no-binary mmcv --no-build-isolation && \
    mim install "mmdet>=3.1.0,<3.3.0"
```

---

### 6. Editable install breaks .mim config resolution

**Problem:** `pip install -e .` (PEP 660 editable install) creates a path finder redirect but doesn't create the `.mim` directory in site-packages. MMPose/mmdet look for config files at:
```
/usr/local/lib/python3.10/dist-packages/mmpose/.mim/demo/mmdetection_cfg/...
```
This path doesn't exist, causing:
```
ValueError: Cannot find model: .../rtmdet_m_640-8xb32_coco-person.py in mmdet
```

**Solution:** Manually create symlinks from site-packages to the source:
```dockerfile
RUN mkdir -p /usr/local/lib/python3.10/dist-packages/mmpose/.mim && \
    ln -sf /mmpose/demo /usr/local/lib/python3.10/dist-packages/mmpose/.mim/demo && \
    ln -sf /mmpose/configs /usr/local/lib/python3.10/dist-packages/mmpose/.mim/configs && \
    ln -sf /mmpose/tools /usr/local/lib/python3.10/dist-packages/mmpose/.mim/tools && \
    ln -sf /mmpose/model-index.yml /usr/local/lib/python3.10/dist-packages/mmpose/.mim/model-index.yml && \
    ln -sf /mmpose/dataset-index.yml /usr/local/lib/python3.10/dist-packages/mmpose/.mim/dataset-index.yml
```

---

### 7. Docker needs special flags for RealSense camera + display

**Problem:** The container needs USB access for the RealSense camera and X11 forwarding for the OpenCV display window.

**Solution:** Run with these flags:
```bash
# On host first:
xhost +local:docker

# Then run container:
docker run --gpus all --shm-size=8g -it \
    --privileged \
    -v /dev/bus/usb:/dev/bus/usb \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/workspace/mmpose_alb/demo/realsense_pose.py:/mmpose/demo/realsense_pose.py \
    -v ~/mmpose-output:/mmpose/output \
    mmpose:latest
```

---

### 8. NVIDIA Container Toolkit required for GPU passthrough

**Problem:** `docker run --gpus all` fails with:
```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

**Solution:** Install NVIDIA Container Toolkit on the host:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

### 9. Docker permission denied without sudo

**Problem:** Docker commands fail with `permission denied` without sudo.

**Solution:** Add user to docker group:
```bash
sudo usermod -aG docker $USER
newgrp docker
```

---

## Complete Dockerfile

See: `docker/Dockerfile.rtx5090`

## Build & Run

```bash
# Build (takes ~20-30 minutes due to mmcv CUDA compilation)
docker build -t mmpose:latest -f docker/Dockerfile.rtx5090 .

# Run with RealSense + display
xhost +local:docker
docker run --gpus all --shm-size=8g -it \
    --privileged \
    -v /dev/bus/usb:/dev/bus/usb \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/mmpose-output:/mmpose/output \
    mmpose:latest

# Inside container — run skeleton tracker
python demo/realsense_pose.py
```

---

## Key Takeaway

The main pain point is that **MMPose's dependency ecosystem (mmcv, mmdet, mmengine) does not have prebuilt wheels for cutting-edge GPU architectures**. For RTX 5090 / Blackwell, you must build mmcv from source with CUDA ops, pin specific version combinations, and fix broken third-party packages (chumpy, xtcocotools) manually. Once the Dockerfile has all these fixes baked in, subsequent builds work reliably.
