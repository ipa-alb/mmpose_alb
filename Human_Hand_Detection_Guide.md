# Human & Hand Detection with MMPose

> Run human body pose estimation and hand keypoint detection using MMPose v1.3.x.
> Stand: February 2026

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Human Body Pose Estimation](#3-human-body-pose-estimation)
4. [Hand Pose Estimation](#4-hand-pose-estimation)
5. [Whole-Body Detection (Body + Hands + Face)](#5-whole-body-detection-body--hands--face)
6. [Python API Usage](#6-python-api-usage)
7. [Model Overview](#7-model-overview)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

| Requirement        | Minimum        | Recommended          |
|--------------------|----------------|----------------------|
| OS                 | Linux (x86_64) | Ubuntu 20.04 / 22.04 |
| Python             | 3.7+           | 3.8 - 3.10           |
| PyTorch            | 1.8+           | 2.0+                 |
| CUDA               | 9.2+           | 11.7+ / 12.x         |
| GPU VRAM           | 4 GB           | 8 GB+                |

---

## 2. Installation

### Option A: Docker (recommended)

```bash
# Build the image (uses the RTX 5090 Dockerfile or the standard one)
docker build -t mmpose:latest -f docker/Dockerfile.rtx5090 .

# Start the container
docker run --gpus all --shm-size=8g -it \
    -v $(pwd)/output:/mmpose/output \
    mmpose:latest
```

### Option B: Local install

```bash
# 1. Create a conda environment
conda create --name mmpose python=3.8 -y
conda activate mmpose

# 2. Install PyTorch (adjust CUDA version to match your system)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 3. Install OpenMMLab dependencies
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"

# 4. Install MMPose from source
cd /path/to/mmpose
pip install -r requirements.txt
pip install -v -e .
```

### Verify the installation

```bash
python -c "
import torch, mmpose, mmcv, mmengine
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.cuda.is_available()} ({torch.cuda.get_device_name(0)})')
print(f'MMPose:   {mmpose.__version__}')
print(f'MMCV:     {mmcv.__version__}')
print(f'MMEngine: {mmengine.__version__}')
"
```

---

## 3. Human Body Pose Estimation

Human body pose estimation detects **17 keypoints** (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) per person.

### 3.1 Top-Down with RTMPose (recommended)

This approach first detects people with a bounding-box detector, then estimates keypoints inside each box.

**Models needed:**

| Role            | Config                                                                | Checkpoint URL |
|-----------------|-----------------------------------------------------------------------|----------------|
| Person detector | `demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py`             | [rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth) |
| Pose estimator  | `configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py` | [rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth) |

**Run on an image:**

```bash
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
    configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth \
    --input tests/data/coco/000000197388.jpg \
    --show --output-root output/
```

**Run on a video:**

```bash
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
    configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth \
    --input your_video.mp4 \
    --show --output-root output/
```

**Available RTMPose body sizes** (configs in `configs/body_2d_keypoint/rtmpose/body8/`):

| Model      | Input Size | COCO AP | Use case               |
|------------|------------|---------|------------------------|
| RTMPose-t  | 256x192    | 65.9    | Edge / mobile devices  |
| RTMPose-s  | 256x192    | 69.7    | Real-time on CPU       |
| RTMPose-m  | 256x192    | 74.9    | Best speed/accuracy    |
| RTMPose-l  | 256x192    | 76.7    | High accuracy          |

### 3.2 One-Stage with RTMO (no detector needed)

RTMO performs detection and pose estimation in a single forward pass. No MMDetection dependency for inference. Best when there are 4+ people in the frame.

```bash
python demo/inferencer_demo.py tests/data/coco/000000197388.jpg \
    --pose2d rtmo \
    --vis-out-dir output/
```

**Available RTMO sizes** (configs in `configs/body_2d_keypoint/rtmo/`):

| Model  | Input Size | COCO AP | Latency (V100) |
|--------|------------|---------|-----------------|
| RTMO-s | 640x640    | 67.7    | 8.9 ms          |
| RTMO-m | 640x640    | 70.9    | 12.4 ms         |
| RTMO-l | 640x640    | 72.4    | 19.1 ms         |

### 3.3 Quickest Way (Inferencer alias)

```bash
python demo/inferencer_demo.py your_image.jpg \
    --pose2d human \
    --vis-out-dir output/
```

---

## 4. Hand Pose Estimation

Hand pose estimation detects **21 keypoints** per hand (wrist + 4 keypoints per finger).

### 4.1 Hand Detection + Hand Pose (top-down)

**Models needed:**

| Role          | Config                                                           | Checkpoint URL |
|---------------|------------------------------------------------------------------|----------------|
| Hand detector | `demo/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py`            | [rtmdet_nano_8xb32-300e_hand-267f9c8f.pth](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth) |
| Hand pose     | `configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py` | [rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth) |

**Run on an image:**

```bash
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth \
    configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth \
    --input tests/data/onehand10k/9.jpg \
    --show --output-root output/
```

### 4.2 Quickest Way (Inferencer alias)

```bash
python demo/inferencer_demo.py tests/data/onehand10k \
    --pose2d hand \
    --vis-out-dir output/ \
    --bbox-thr 0.5 --kpt-thr 0.05
```

### 4.3 3D Hand Pose

For 3D hand keypoint estimation using InterNet:

```bash
python demo/hand3d_internet_demo.py \
    configs/hand_3d_keypoint/internet/interhand3d/internet_res50_4xb16-20e_interhand3d-256x256.py \
    https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3dv1.0_all_256x256-42b7f2ac_20210702.pth \
    --input tests/data/interhand2.6m/image/val/ \
    --output-root output/
```

See `demo/docs/en/3d_hand_demo.md` for full options.

---

## 5. Whole-Body Detection (Body + Hands + Face)

Whole-body models estimate **133 keypoints** in a single pass:
- 17 body + 6 feet + 68 face + 42 hands (21 per hand)

This is ideal when you need body **and** hand keypoints together.

### 5.1 RTMW (state-of-the-art)

Trained on 14 combined datasets. Configs in `configs/wholebody_2d_keypoint/rtmpose/cocktail14/`.

| Model  | Input Size | Whole AP | Hand AP | Checkpoint |
|--------|------------|----------|---------|------------|
| RTMW-m | 256x192    | 58.2     | 49.1    | [rtmw-dw-l-m_simcc-cocktail14_270e-256x192-20231122.pth](https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-l-m_simcc-cocktail14_270e-256x192-20231122.pth) |
| RTMW-l | 384x288    | 70.1     | 66.3    | [rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth](https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth) |
| RTMW-x | 384x288    | 70.2     | 66.4    | [rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth](https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth) |

**Run RTMW-l (best balance of accuracy and speed):**

```bash
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
    configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth \
    --input your_image.jpg \
    --show --output-root output/
```

### 5.2 DWPose (distilled, lighter)

Configs in `configs/wholebody_2d_keypoint/dwpose/ubody/`. Uses knowledge distillation for faster inference with competitive accuracy.

---

## 6. Python API Usage

### 6.1 Human body

```python
from mmpose.apis import MMPoseInferencer

inferencer = MMPoseInferencer('human')
result = next(inferencer('image.jpg', show=True, out_dir='output/'))

# Access keypoints
for person in result['predictions'][0]:
    keypoints = person['keypoints']        # shape: (17, 2)
    scores = person['keypoint_scores']     # shape: (17,)
```

### 6.2 Hand

```python
from mmpose.apis import MMPoseInferencer

inferencer = MMPoseInferencer('hand')
result = next(inferencer('hand_image.jpg', show=True, out_dir='output/'))

for hand in result['predictions'][0]:
    keypoints = hand['keypoints']          # shape: (21, 2)
    scores = hand['keypoint_scores']       # shape: (21,)
```

### 6.3 Whole-body (custom config)

```python
from mmpose.apis import MMPoseInferencer

inferencer = MMPoseInferencer(
    pose2d='configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py',
    pose2d_weights='https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth',
    det_model='demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py',
    det_weights='https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
)
result = next(inferencer('image.jpg', show=True, out_dir='output/'))

for person in result['predictions'][0]:
    all_keypoints = person['keypoints']    # shape: (133, 2)
    # Indices:  0-16  body
    #          17-22  feet
    #          23-90  face
    #          91-111 left hand (21 kpts)
    #         112-132 right hand (21 kpts)
```

### 6.4 Video / Webcam

```python
from mmpose.apis import MMPoseInferencer

inferencer = MMPoseInferencer('human')

# Video file
for result in inferencer('video.mp4', show=True, out_dir='output/'):
    pass

# Webcam (device index)
for result in inferencer(0, show=True):
    pass
```

---

## 7. Model Overview

### Which approach to choose?

| Goal                          | Approach                  | Detector needed? | Keypoints |
|-------------------------------|---------------------------|------------------|-----------|
| Body only (fast)              | RTMPose body8             | Yes (person)     | 17        |
| Body only (single-stage)      | RTMO                      | No               | 17        |
| Hands only                    | RTMPose hand5             | Yes (hand)       | 21        |
| Body + Hands + Face combined  | RTMW cocktail14           | Yes (person)     | 133       |
| Body + Hands + Face (lighter) | DWPose                    | Yes (person)     | 133       |

### Person detectors

| Detector      | Config                                                        | Speed   | Use case        |
|---------------|---------------------------------------------------------------|---------|-----------------|
| RTMDet-nano   | `demo/mmdetection_cfg/rtmdet_nano_320-8xb32_coco-person.py`  | Fastest | Real-time / CPU |
| RTMDet-m      | `demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py`     | Fast    | Best balance    |

### Hand detector

| Detector      | Config                                                  |
|---------------|---------------------------------------------------------|
| RTMDet-nano   | `demo/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py`   |

---

## 8. Troubleshooting

### `ModuleNotFoundError: No module named 'mmdet'`

MMDetection is required for all top-down approaches. Install it:

```bash
mim install "mmdet>=3.1.0"
```

Alternatively, use RTMO which does not need a separate detector.

### `RuntimeError: CUDA out of memory`

- Use a smaller model variant (e.g. RTMPose-t instead of RTMPose-l)
- Reduce input resolution
- If using Docker, increase shared memory: `--shm-size=16g`

### `FileNotFoundError` for checkpoint files

Checkpoints are downloaded automatically when you pass a URL. To download manually:

```bash
mim download mmpose --config rtmpose-m_8xb256-420e_body8-256x192 --dest checkpoints/
```

### Inference runs but no detections appear

- Lower the bounding-box threshold: `--bbox-thr 0.3`
- Lower the keypoint threshold: `--kpt-thr 0.1`
- Ensure the image actually contains the target (people for body, hands for hand models)

### No display / `cannot open display`

Running headless (SSH, Docker without X11)? Drop `--show` and use `--output-root output/` to save results to disk instead.

---

## Further Reading

- [2D Human Pose Demo docs](demo/docs/en/2d_human_pose_demo.md)
- [2D Hand Demo docs](demo/docs/en/2d_hand_demo.md)
- [2D Whole-Body Demo docs](demo/docs/en/2d_wholebody_pose_demo.md)
- [RTMPose project README](projects/rtmpose/README.md)
- [RTMO project README](projects/rtmo/README.md)
- [Inference user guide](docs/en/user_guides/inference.md)
