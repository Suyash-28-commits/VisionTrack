"""
app.py
------
VisionTrack: AI-Powered Kinematics & Projectile Analysis
Streamlit dashboard entry point.

Pipeline: upload video -> YOLOv8 detects+tracks ball -> user calibrates
pixels-to-meters -> classical physics engine computes kinematics ->
graphs + tracked-video overlay + CSV/PDF export.
"""

import os
import tempfile

import streamlit as st
import numpy as np

from src.detector import BallDetector
from src.tracker import BallTracker
from src.calibration import build_calibration, pixels_to_world
from src.physics import compute_kinematics, results_to_dataframe, summary_dict
from src.visualization import (
    plot_position_vs_time, plot_velocity_vs_time,
    plot_acceleration_vs_time, plot_trajectory, overlay_trajectory_on_frames,
)
from src.video_processor import read_video, write_video
from src.export import export_csv_bytes, export_pdf_report
from src.utils import ensure_dir, build_time_array

st.set_page_config(page_title="VisionTrack", layout="wide")
st.title("🎯 VisionTrack — AI-Powered Kinematics & Projectile Analysis")
st.caption(
    "YOLOv8 (pretrained) handles detection & tracking only. "
    "All velocity, acceleration, height, range, and time-of-flight values "
    "are computed with classical physics — no ML is used for these."
)

ensure_dir("data/uploads")
ensure_dir("outputs")

# ---------------------------------------------------------------------------
# 1. Video input
# ---------------------------------------------------------------------------
st.header("1. Upload Video")
uploaded_file = st.file_uploader("Upload a projectile motion video", type=["mp4", "mov", "avi"])

if uploaded_file:
    video_path = os.path.join("data/uploads", uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(video_path)
    frames, fps, (width, height) = read_video(video_path)
    st.success(f"Loaded {len(frames)} frames @ {fps:.2f} FPS, resolution {width}x{height}")

    # -----------------------------------------------------------------------
    # 2. Calibration
    # -----------------------------------------------------------------------
    st.header("2. Camera Calibration")
    st.markdown(
        "Enter the pixel coordinates of two points a **known real-world "
        "distance apart** (e.g. two marks on a ruler, or a checkerboard "
        "edge), plus the launch point of the projectile in the first frame. "
        "Tip: open the first frame in any image viewer to read off pixel "
        "coordinates."
    )
    st.image(frames[0][:, :, ::-1], caption="First frame (for reading pixel coordinates)")

    col1, col2, col3 = st.columns(3)
    with col1:
        ax_px = st.number_input("Reference point A — x (px)", value=0)
        ay_px = st.number_input("Reference point A — y (px)", value=0)
    with col2:
        bx_px = st.number_input("Reference point B — x (px)", value=100)
        by_px = st.number_input("Reference point B — y (px)", value=0)
    with col3:
        known_distance = st.number_input("Real-world distance A→B (meters)", value=1.0, min_value=0.001)

    lx_px = st.number_input("Launch point — x (px)", value=0)
    ly_px = st.number_input("Launch point — y (px)", value=float(height))

    # -----------------------------------------------------------------------
    # 3. Run detection + tracking + physics
    # -----------------------------------------------------------------------
    st.header("3. Run Analysis")
    if st.button("▶ Run VisionTrack Analysis"):
        with st.spinner("Loading pretrained YOLOv8 and detecting the ball frame-by-frame..."):
            detector = BallDetector()
            tracker = BallTracker(detector)
            traj_df = tracker.process(frames)

        st.success(f"Tracked {len(traj_df)} usable frames after interpolation/smoothing.")

        calib = build_calibration(
            point_a_px=(ax_px, ay_px),
            point_b_px=(bx_px, by_px),
            known_distance_m=known_distance,
            launch_point_px=(lx_px, ly_px),
            frame_height_px=height,
        )

        cx = traj_df["cx_px_smooth"].values
        cy = traj_df["cy_px_smooth"].values
        x_world, y_world = pixels_to_world(cx, cy, calib)
        time = build_time_array(len(traj_df), fps)

        results = compute_kinematics(time, x_world, y_world)

        # ---------------------------------------------------------------
        # 4. Results dashboard
        # ---------------------------------------------------------------
        st.header("4. Results")
        tab_video, tab_graphs, tab_data = st.tabs(["Tracked Video", "Graphs", "Data & Export"])

        with tab_video:
            with st.spinner("Rendering tracked video overlay..."):
                annotated_frames = overlay_trajectory_on_frames(
                    [frames[i] for i in traj_df["frame"].values], cx, cy
                )
                out_path = os.path.join("outputs", "tracked_" + uploaded_file.name)
                write_video(annotated_frames, out_path, fps, (width, height))
            st.video(out_path)

        with tab_graphs:
            st.plotly_chart(plot_trajectory(results), use_container_width=True)
            st.plotly_chart(plot_position_vs_time(results), use_container_width=True)
            st.plotly_chart(plot_velocity_vs_time(results), use_container_width=True)
            st.plotly_chart(plot_acceleration_vs_time(results), use_container_width=True)

        with tab_data:
            st.subheader("Summary (Measured vs Theoretical)")
            st.table(summary_dict(results))

            df_out = results_to_dataframe(results)
            st.dataframe(df_out, use_container_width=True)

            csv_bytes = export_csv_bytes(df_out)
            st.download_button("⬇ Download CSV", csv_bytes, file_name="visiontrack_data.csv")

            pdf_bytes = export_pdf_report(results, video_name=uploaded_file.name)
            st.download_button("⬇ Download PDF Report", pdf_bytes,
                                file_name="visiontrack_report.pdf")
else:
    st.info("Upload a video to begin.")