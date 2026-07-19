"""
utils.py
--------
Small shared helpers used by multiple modules.
"""

import os
import logging
import numpy as np


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def build_time_array(n_frames: int, fps: float) -> np.ndarray:
    """Return the time (seconds) of each frame, given frame count and fps."""
    return np.arange(n_frames) / fps


def find_landing_index(y: np.ndarray, launch_idx: int = 0) -> int:
    """
    Find the frame index where the projectile returns to (approximately)
    its launch height, searching AFTER the peak height.
    y must be in world coordinates with "up" positive.
    """
    peak_idx = int(np.argmax(y[launch_idx:])) + launch_idx
    launch_height = y[launch_idx]

    for i in range(peak_idx, len(y)):
        if y[i] <= launch_height:
            return i
    return len(y) - 1  # fallback: last available frame