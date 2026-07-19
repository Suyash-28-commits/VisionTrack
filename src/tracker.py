"""
tracker.py
----------
Builds a full per-frame trajectory from YOLO's per-frame detections.

Handles two real-world problems:
1. Missed detections (motion blur, low contrast, occlusion) -> linear
   interpolation across small gaps.
2. Pixel-level jitter -> Savitzky-Golay smoothing before any physics is
   computed downstream (physics.py differentiates this trajectory, and
   differentiation amplifies noise).
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .config import MAX_INTERP_GAP_FRAMES, SMOOTHING_WINDOW, SMOOTHING_POLYORDER
from .detector import BallDetector, Detection
from .utils import get_logger

logger = get_logger(__name__)


class BallTracker:
    def __init__(self, detector: BallDetector):
        self.detector = detector

    def track_video(self, frames: List[np.ndarray]) -> pd.DataFrame:
        """
        Run detection on every frame and assemble a raw trajectory table.
        Returns a DataFrame with columns: frame, cx_px, cy_px, w_px, h_px,
        confidence, detected (bool).
        """
        rows = []
        for i, frame in enumerate(frames):
            det: Optional[Detection] = self.detector.detect(frame)
            if det is not None:
                rows.append({
                    "frame": i, "cx_px": det.cx, "cy_px": det.cy,
                    "w_px": det.w, "h_px": det.h,
                    "confidence": det.confidence, "detected": True,
                })
            else:
                rows.append({
                    "frame": i, "cx_px": np.nan, "cy_px": np.nan,
                    "w_px": np.nan, "h_px": np.nan,
                    "confidence": 0.0, "detected": False,
                })

        df = pd.DataFrame(rows)
        n_missed = (~df["detected"]).sum()
        logger.info(f"Detected ball in {len(df) - n_missed}/{len(df)} frames")
        return df

    def interpolate_gaps(self, df: pd.DataFrame,
                          max_gap: int = MAX_INTERP_GAP_FRAMES) -> pd.DataFrame:
        """
        Fill in short gaps of missed detections using linear interpolation.
        Gaps longer than max_gap are left as NaN (too unreliable to guess)
        and are dropped before physics calculations.
        """
        df = df.copy()
        for col in ["cx_px", "cy_px", "w_px", "h_px"]:
            df[col] = df[col].interpolate(method="linear", limit=max_gap, limit_area="inside")
        df = df.dropna(subset=["cx_px", "cy_px"]).reset_index(drop=True)
        return df

    def smooth_trajectory(self, df: pd.DataFrame,
                           window: int = SMOOTHING_WINDOW,
                           polyorder: int = SMOOTHING_POLYORDER) -> pd.DataFrame:
        """
        Apply a Savitzky-Golay filter to cx_px / cy_px. This preserves the
        overall shape of the parabola while removing high-frequency pixel
        jitter that would otherwise blow up when differentiated for
        velocity/acceleration.
        """
        df = df.copy()
        n = len(df)
        w = min(window, n if n % 2 == 1 else n - 1)
        if w <= polyorder:
            logger.warning("Not enough points to smooth reliably; skipping smoothing.")
            return df

        df["cx_px_smooth"] = savgol_filter(df["cx_px"].values, w, polyorder)
        df["cy_px_smooth"] = savgol_filter(df["cy_px"].values, w, polyorder)
        return df

    def process(self, frames: List[np.ndarray]) -> pd.DataFrame:
        """Full pipeline: detect -> interpolate -> smooth."""
        raw = self.track_video(frames)
        filled = self.interpolate_gaps(raw)
        smoothed = self.smooth_trajectory(filled)
        return smoothed