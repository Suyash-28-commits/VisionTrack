"""
calibration.py
--------------
Converts pixel coordinates into real-world units (meters).

Method: the user selects two points in the first frame that correspond to a
KNOWN real-world distance (e.g. two marks on a ruler, or a checkerboard
square edge). This gives a single scale factor:

    meters_per_pixel = known_real_distance_m / pixel_distance

We also need to define a world origin (usually the launch point) so that
position, height, and range are all measured relative to where the
projectile was released, not relative to the image's top-left corner
(which is OpenCV/image convention, with y increasing DOWNWARD).
"""

import numpy as np

from .config import CalibrationConfig


def compute_scale(
    point_a_px: tuple,
    point_b_px: tuple,
    known_distance_m: float,
) -> float:
    """meters_per_pixel from two reference pixel points and their real distance."""
    ax, ay = point_a_px
    bx, by = point_b_px
    pixel_distance = np.hypot(bx - ax, by - ay)
    if pixel_distance == 0:
        raise ValueError("Reference points cannot be identical.")
    return known_distance_m / pixel_distance


def build_calibration(
    point_a_px: tuple,
    point_b_px: tuple,
    known_distance_m: float,
    launch_point_px: tuple,
    frame_height_px: int,
) -> CalibrationConfig:
    scale = compute_scale(point_a_px, point_b_px, known_distance_m)
    return CalibrationConfig(
        meters_per_pixel=scale,
        origin_x_px=launch_point_px[0],
        origin_y_px=launch_point_px[1],
        frame_height_px=frame_height_px,
    )


def pixels_to_world(x_px: np.ndarray, y_px: np.ndarray,
                     calib: CalibrationConfig) -> tuple:
    """
    Convert an array of pixel coordinates into world coordinates (meters),
    with the origin at the launch point and "up" as positive y.

    Image/pixel convention: y grows downward, origin top-left.
    World convention: y grows upward, origin at launch point.
    """
    x_world = (x_px - calib.origin_x_px) * calib.meters_per_pixel
    # Flip y: pixel y grows down, world y should grow up.
    y_world = (calib.origin_y_px - y_px) * calib.meters_per_pixel
    return x_world, y_world