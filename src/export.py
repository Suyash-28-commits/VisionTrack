"""
export.py
---------
CSV export of the tracked/physics data, and a PDF summary report.
"""

import io

import pandas as pd
from fpdf import FPDF

from .physics import PhysicsResults, summary_dict


def export_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def export_pdf_report(results: PhysicsResults, video_name: str = "uploaded_video") -> bytes:
    summary = summary_dict(results)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "VisionTrack: Projectile Motion Analysis Report", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Source video: {video_name}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Measured vs Theoretical Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for key, value in summary.items():
        pdf.cell(0, 7, f"{key}: {value}", ln=True)

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Methodology", ln=True)
    pdf.set_font("Helvetica", "", 10)
    method_text = (
        "Detection: pretrained YOLOv8 (COCO 'sports ball' class), no custom "
        "training. Tracking: per-frame centroid extraction with linear "
        "interpolation across short detection gaps and Savitzky-Golay "
        "smoothing. Calibration: pixel-to-meter scale derived from a "
        "user-marked reference distance. Physics: classical kinematics "
        "(central-difference velocity/acceleration, standard projectile "
        "motion equations) -- no machine learning is used to compute any "
        "physical quantity."
    )
    pdf.multi_cell(0, 6, method_text)

    return bytes(pdf.output(dest="S"))