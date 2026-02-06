#!/bin/bash
# ============================================================
# Run MMPose container on Lenovo Legion Pro 7i (RTX 4080)
# with RealSense camera + X11 display forwarding
# ============================================================

set -e

# Allow Docker to access the X11 display
xhost +local:docker 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

docker run --gpus all --shm-size=8g -it \
    --privileged \
    -v /dev/bus/usb:/dev/bus/usb \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$REPO_DIR/demo/realsense_pose.py":/mmpose/demo/realsense_pose.py \
    -v ~/mmpose-output:/mmpose/output \
    mmpose-legion:latest
