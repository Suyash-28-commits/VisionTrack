"""
physics.py
----------
ALL kinematic quantities are derived here using classical mechanics
formulas applied to the tracked (and calibrated) trajectory. No machine
learning is used anywhere in this module -- by design.

Derivations are summarized in each function's docstring; the full
step-by-step math lives in docs/math_derivations.md.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import GRAVITY
from .utils import find_landing_index


@dataclass
class PhysicsResults:
    time: np.ndarray
    x: np.ndarray                 # world x position (m)
    y: np.ndarray                 # world y position (m)
    vx: np.ndarray                # horizontal velocity (m/s)
    vy: np.ndarray                # vertical velocity (m/s)
    v_resultant: np.ndarray       # resultant speed (m/s)
    ax: np.ndarray                # horizontal acceleration (m/s^2)
    ay: np.ndarray                # vertical acceleration (m/s^2)
    launch_angle_deg: float
    launch_speed: float
    max_height_m: float
    horizontal_range_m: float
    time_of_flight_s: float
    landing_index: int
    theoretical: dict = field(default_factory=dict)


def compute_kinematics(time: np.ndarray, x: np.ndarray, y: np.ndarray) -> PhysicsResults:
    """
    Given calibrated world-coordinate trajectory (x(t), y(t)), compute the
    full kinematic profile.

    Velocity: central-difference derivative of position w.r.t. time.
        v_i = (x[i+1] - x[i-1]) / (t[i+1] - t[i-1])     (interior points)
    np.gradient implements this automatically, using forward/backward
    differences at the endpoints.

    Acceleration: derivative of velocity w.r.t. time (same method).

    Resultant velocity: v = sqrt(vx^2 + vy^2)   (Pythagorean combination
    of orthogonal velocity components).

    Launch angle: angle of the initial velocity vector above horizontal.
        theta_0 = atan2(vy[0], vx[0])

    Max height: peak of y(t) above the launch height (y is already
    zeroed at the launch point during calibration).

    Landing index: first frame after the peak where y returns to <= the
    launch height (i.e., the projectile has landed / returned to
    release height).

    Horizontal range: x displacement between launch and landing.

    Time of flight: elapsed time between launch and landing.
    """
    vx = np.gradient(x, time)
    vy = np.gradient(y, time)
    v_resultant = np.sqrt(vx**2 + vy**2)

    ax = np.gradient(vx, time)
    ay = np.gradient(vy, time)

    launch_angle_rad = np.arctan2(vy[0], vx[0])
    launch_angle_deg = np.degrees(launch_angle_rad)
    launch_speed = v_resultant[0]

    landing_idx = find_landing_index(y, launch_idx=0)

    max_height_m = float(np.max(y) - y[0])
    horizontal_range_m = float(x[landing_idx] - x[0])
    time_of_flight_s = float(time[landing_idx] - time[0])

    results = PhysicsResults(
        time=time, x=x, y=y, vx=vx, vy=vy, v_resultant=v_resultant,
        ax=ax, ay=ay,
        launch_angle_deg=float(launch_angle_deg),
        launch_speed=float(launch_speed),
        max_height_m=max_height_m,
        horizontal_range_m=horizontal_range_m,
        time_of_flight_s=time_of_flight_s,
        landing_index=landing_idx,
    )
    results.theoretical = compute_theoretical(launch_speed, launch_angle_deg)
    return results


def compute_theoretical(v0: float, theta_deg: float, g: float = GRAVITY) -> dict:
    """
    Ideal (no air resistance) projectile motion formulas, evaluated using
    the MEASURED launch speed/angle from tracking. This is used purely as
    a comparison/error-analysis baseline against the measured trajectory
    -- it is still 100% classical physics, not a learned model.

        Time of flight:  T = 2 * v0 * sin(theta) / g
        Max height:      H = (v0 * sin(theta))^2 / (2 * g)
        Range:           R = v0^2 * sin(2 * theta) / g
    """
    theta_rad = np.radians(theta_deg)
    T = 2 * v0 * np.sin(theta_rad) / g
    H = (v0 * np.sin(theta_rad)) ** 2 / (2 * g)
    R = (v0 ** 2) * np.sin(2 * theta_rad) / g
    return {"time_of_flight_s": T, "max_height_m": H, "horizontal_range_m": R}


def results_to_dataframe(results: PhysicsResults) -> pd.DataFrame:
    """Flatten the per-frame arrays into a tidy DataFrame for CSV export."""
    return pd.DataFrame({
        "time_s": results.time,
        "x_m": results.x,
        "y_m": results.y,
        "vx_mps": results.vx,
        "vy_mps": results.vy,
        "v_resultant_mps": results.v_resultant,
        "ax_mps2": results.ax,
        "ay_mps2": results.ay,
    })


def summary_dict(results: PhysicsResults) -> dict:
    """Scalar summary values for display in the dashboard / PDF report."""
    return {
        "Launch Speed (m/s)": round(results.launch_speed, 3),
        "Launch Angle (deg)": round(results.launch_angle_deg, 2),
        "Max Height (m)": round(results.max_height_m, 3),
        "Horizontal Range (m)": round(results.horizontal_range_m, 3),
        "Time of Flight (s)": round(results.time_of_flight_s, 3),
        "Theoretical Max Height (m)": round(results.theoretical["max_height_m"], 3),
        "Theoretical Range (m)": round(results.theoretical["horizontal_range_m"], 3),
        "Theoretical Time of Flight (s)": round(results.theoretical["time_of_flight_s"], 3),
    }