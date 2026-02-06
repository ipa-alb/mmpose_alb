"""RealSense D435 + MMPose live skeleton tracking.

Usage (inside Docker container):
    python demo/realsense_pose.py
    python demo/realsense_pose.py --pose2d rtmpose-m   # faster model
    python demo/realsense_pose.py --serial 838212073332 # specific camera

Press ESC or 'q' to quit.
"""

import argparse
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from mmpose.apis import MMPoseInferencer


def parse_args():
    parser = argparse.ArgumentParser(
        description='RealSense + MMPose skeleton tracker')
    parser.add_argument(
        '--serial', type=str, default='838212073332',
        help='RealSense serial number')
    parser.add_argument(
        '--pose2d', type=str, default='human',
        help='Pose model alias (human, rtmpose-m, rtmpose-l, etc.)')
    parser.add_argument(
        '--width', type=int, default=640,
        help='Stream width')
    parser.add_argument(
        '--height', type=int, default=480,
        help='Stream height')
    parser.add_argument(
        '--fps', type=int, default=30,
        help='Stream FPS')
    parser.add_argument(
        '--device', type=str, default=None,
        help='Inference device (cuda:0, cpu)')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3,
        help='Keypoint score threshold')
    parser.add_argument(
        '--radius', type=int, default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness', type=int, default=2,
        help='Skeleton link thickness')
    parser.add_argument(
        '--out-dir', type=str, default='',
        help='Save output video to this directory')
    return parser.parse_args()


def main():
    args = parse_args()

    # --- RealSense setup ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    print(f'Starting RealSense {args.serial} '
          f'({args.width}x{args.height} @ {args.fps}fps)...')
    pipeline.start(config)

    # --- MMPose setup ---
    print(f'Loading pose model: {args.pose2d}...')
    inferencer = MMPoseInferencer(
        args.pose2d, device=args.device)
    print('Ready. Press ESC or q to quit.')

    # --- Video writer (optional) ---
    writer = None
    if args.out_dir:
        import os
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, 'realsense_pose.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            out_path, fourcc, args.fps, (args.width, args.height))
        print(f'Recording to {out_path}')

    frame_count = 0
    t_start = time.time()

    try:
        while True:
            # Capture frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())

            # Run pose estimation
            results = next(inferencer(
                frame,
                show=False,
                return_vis=True,
                radius=args.radius,
                thickness=args.thickness,
                kpt_thr=args.kpt_thr))

            # Get visualized frame
            vis_frame = results['visualization'][0]
            # Convert RGB back to BGR for OpenCV display
            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)

            # FPS overlay
            frame_count += 1
            elapsed = time.time() - t_start
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(vis_frame, f'FPS: {fps:.1f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display
            cv2.imshow('MMPose - RealSense Skeleton Tracker', vis_frame)

            # Save frame
            if writer:
                writer.write(vis_frame)

            # Quit on ESC or 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:
        pipeline.stop()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f'\nDone. {frame_count} frames in {elapsed:.1f}s '
              f'({fps:.1f} FPS avg)')


if __name__ == '__main__':
    main()
