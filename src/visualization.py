"""
visualization.py
----------------
Builds all graphs (position, velocity, acceleration, trajectory) and
overlays the tracked trajectory onto the original video frames.
"""

from typing import List

import cv2
import numpy as np
import plotly.graph_objects as go

from .physics import PhysicsResults


def plot_position_vs_time(results: PhysicsResults) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results.time, y=results.x, name="x (horizontal)"))
    fig.add_trace(go.Scatter(x=results.time, y=results.y, name="y (vertical)"))
    fig.update_layout(title="Position vs Time", xaxis_title="Time (s)",
                       yaxis_title="Position (m)")
    return fig


def plot_velocity_vs_time(results: PhysicsResults) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results.time, y=results.vx, name="Vx (horizontal)"))
    fig.add_trace(go.Scatter(x=results.time, y=results.vy, name="Vy (vertical)"))
    fig.add_trace(go.Scatter(x=results.time, y=results.v_resultant, name="V (resultant)"))
    fig.update_layout(title="Velocity vs Time", xaxis_title="Time (s)",
                       yaxis_title="Velocity (m/s)")
    return fig


def plot_acceleration_vs_time(results: PhysicsResults) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results.time, y=results.ax, name="ax (horizontal)"))
    fig.add_trace(go.Scatter(x=results.time, y=results.ay, name="ay (vertical)"))
    fig.add_hline(y=-9.81, line_dash="dash",
                  annotation_text="Expected ay = -g (9.81 m/s²)")
    fig.update_layout(title="Acceleration vs Time", xaxis_title="Time (s)",
                       yaxis_title="Acceleration (m/s²)")
    return fig


def plot_trajectory(results: PhysicsResults) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results.x, y=results.y, mode="lines+markers",
                              name="Measured trajectory"))
    fig.update_layout(title="Projectile Trajectory", xaxis_title="Horizontal distance (m)",
                       yaxis_title="Height (m)")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)  # equal aspect ratio
    return fig


def overlay_trajectory_on_frames(
    frames: List[np.ndarray],
    cx_series: np.ndarray,
    cy_series: np.ndarray,
) -> List[np.ndarray]:
    """
    Draws the ball's bounding-box center and a fading trail of previous
    positions onto each frame, returning a new list of annotated frames.
    """
    annotated = []
    trail = []
    for i, frame in enumerate(frames):
        out = frame.copy()
        if i < len(cx_series):
            trail.append((int(cx_series[i]), int(cy_series[i])))

        for j in range(1, len(trail)):
            cv2.line(out, trail[j - 1], trail[j], (0, 255, 0), 2)
        if trail:
            cv2.circle(out, trail[-1], 6, (0, 0, 255), -1)

        annotated.append(out)
    return annotated