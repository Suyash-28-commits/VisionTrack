"""
video_processor.py
------------------
Handles reading a video into frames and writing an annotated video back out.
Keeping OpenCV's low-level VideoCapture/VideoWriter details isolated here
keeps the rest of the codebase free of raw OpenCV plumbing.
"""

from typing import List, Tuple

import cv2
import numpy as np

from .utils import get_logger

logger = get_logger(__name__)


def read_video(path: str) -> Tuple[List[np.ndarray], float, Tuple[int, int]]:
    """
    Returns (frames, fps, (width, height)).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    logger.info(f"Read {len(frames)} frames at {fps:.2f} fps ({width}x{height})")
    return frames, fps, (width, height)


def write_video(frames: List[np.ndarray], path: str, fps: float,
                 size: Tuple[int, int]) -> str:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    for frame in frames:
        writer.write(frame)
    writer.release()
    logger.info(f"Wrote annotated video to {path}")
    return path