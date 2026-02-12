# MMPose – Running Guide

> **Tested on:** Dell Pro Max Tower T2 | RTX 5090 (32GB) | Ubuntu 22.04 | Driver 590.48.01
> **Date:** February 2026
> **Docker Image:** `mmpose:latest` (built from `docker/Dockerfile.rtx5090`)

---

## Start the Container

```bash
# Allow X11 display access (run on host)
xhost +local:docker

# Start container with GPU, USB (RealSense), and display
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

## Body Skeleton Tracking (WORKING)

Detects **17 keypoints** per person (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles).

### Live RealSense feed

```bash
python demo/realsense_pose.py --pose2d human
```

### On a test image

```bash
python demo/inferencer_demo.py tests/data/coco/000000000785.jpg \
    --pose2d human \
    --vis-out-dir output/
```

### With RTMPose-m (faster)

```bash
python demo/realsense_pose.py --pose2d rtmpose-m
```

### Status: WORKING

- Model auto-downloads on first run
- Live skeleton overlay with FPS counter
- Press ESC or q to quit

---

## Hand Tracking (WORKING)

Detects **21 keypoints** per hand (wrist + 4 keypoints per finger).

### Live RealSense feed

```bash
python demo/realsense_pose.py --pose2d hand
```

### On a test image

```bash
python demo/inferencer_demo.py tests/data/onehand10k/9.jpg \
    --pose2d hand \
    --vis-out-dir output/
```

### Status: WORKING

- Uses RTMDet-nano hand detector + RTMPose hand model
- Model auto-downloads on first run

---

## Whole-Body Tracking (WORKING)

Detects **133 keypoints** in one pass: 17 body + 6 feet + 68 face + 21 left hand + 21 right hand.

### Live RealSense feed

```bash
python demo/realsense_pose.py --pose2d wholebody
```

### On a test image

```bash
python demo/inferencer_demo.py tests/data/coco/000000000785.jpg \
    --pose2d wholebody \
    --vis-out-dir output/
```

### Status: WORKING

- Uses RTMW model (body + hands + face combined)
- Model auto-downloads on first run

---

## RealSense Camera Info

| Property | Value |
|----------|-------|
| Camera | Intel RealSense D435 |
| Serial | 838212073332 |
| Resolution | 640x480 @ 30fps (default) |

### Custom resolution / serial

```bash
python demo/realsense_pose.py --pose2d human --width 1280 --height 720 --fps 15
python demo/realsense_pose.py --pose2d hand --serial 838212073332
```

### Save output video

```bash
python demo/realsense_pose.py --pose2d human --out-dir output/
```

Output saved to `output/realsense_pose.mp4` (mounted at `~/mmpose-output/` on host).

---

## Quick Reference

| Mode | Command | Keypoints | Status |
|------|---------|-----------|--------|
| Body | `--pose2d human` | 17 | WORKING |
| Body (fast) | `--pose2d rtmpose-m` | 17 | WORKING |
| Hand | `--pose2d hand` | 21 | WORKING |
| Whole-body | `--pose2d wholebody` | 133 | WORKING |
