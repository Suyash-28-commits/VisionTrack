"""
config.py
---------
Central place for constants and default settings used across VisionTrack.
Keeping these here means no module hardcodes "magic numbers".
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# YOLO detection
# ---------------------------------------------------------------------------
# COCO class index for "sports ball". Fixed by the COCO dataset definition
# that YOLOv8's pretrained checkpoints were trained on -- no training needed.
COCO_SPORTS_BALL_CLASS_ID = 32

# yolov8n (nano) = smallest/fastest variant. Good enough for a single small
# round object, and fast enough for near real-time use in a Streamlit demo.
DEFAULT_YOLO_WEIGHTS = "yolov8n.pt"

DEFAULT_CONF_THRESHOLD = 0.25

# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------
# If YOLO misses the ball for up to this many consecutive frames, interpolate
# instead of dropping the point (motion blur / low contrast are common near
# the peak of the trajectory).
MAX_INTERP_GAP_FRAMES = 8

# Savitzky-Golay smoothing applied to the raw pixel trajectory before
# differentiating into velocity/acceleration. Differentiation amplifies
# pixel jitter, so smoothing first is essential.
SMOOTHING_WINDOW = 7       # must be odd, > polyorder
SMOOTHING_POLYORDER = 2

# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
GRAVITY = 9.81  # m/s^2


@dataclass
class CalibrationConfig:
    """Result of the pixel -> real-world calibration step."""
    meters_per_pixel: float
    origin_x_px: float     # pixel x of world origin (launch point)
    origin_y_px: float     # pixel y of world origin
    frame_height_px: int


@dataclass
class AppPaths:
    uploads_dir: str = "data/uploads"
    outputs_dir: str = "outputs"