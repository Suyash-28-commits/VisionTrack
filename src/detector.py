"""
detector.py
-----------
Thin wrapper around a PRETRAINED YOLOv8 model, used strictly for object
DETECTION (finding the ball's bounding box in a single frame).

IMPORTANT: No custom training happens here. We rely entirely on the
COCO-pretrained weights shipped by Ultralytics, using the "sports ball"
class that is already part of COCO's 80 classes.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from ultralytics import YOLO

from .config import COCO_SPORTS_BALL_CLASS_ID, DEFAULT_YOLO_WEIGHTS, DEFAULT_CONF_THRESHOLD
from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    cx: float          # bounding box center x (pixels)
    cy: float          # bounding box center y (pixels)
    w: float           # bounding box width (pixels)
    h: float           # bounding box height (pixels)
    confidence: float


class BallDetector:
    def __init__(
        self,
        weights: str = DEFAULT_YOLO_WEIGHTS,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        class_id: int = COCO_SPORTS_BALL_CLASS_ID,
    ):
        logger.info(f"Loading pretrained YOLOv8 weights: {weights}")
        self.model = YOLO(weights)   # downloads pretrained COCO weights on first run
        self.conf_threshold = conf_threshold
        self.class_id = class_id

    def detect(self, frame: np.ndarray) -> Optional[Detection]:
        """
        Run detection on a single BGR frame (as read by OpenCV).
        Returns the highest-confidence "sports ball" detection, or None.
        """
        results = self.model(
            frame,
            classes=[self.class_id],
            conf=self.conf_threshold,
            verbose=False,
        )

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        # If multiple balls are detected, pick the highest-confidence one.
        confidences = boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confidences))

        xyxy = boxes.xyxy.cpu().numpy()[best_idx]
        x1, y1, x2, y2 = xyxy
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = (x2 - x1), (y2 - y1)

        return Detection(cx=cx, cy=cy, w=w, h=h, confidence=float(confidences[best_idx]))