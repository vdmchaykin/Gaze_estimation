import cv2
import csv
import pandas as pd
import numpy as np

from pupil_pl_detector import PupilLabsLikeDetector

VIDEO_PATH = "eye_video.mp4"
TIMESTAMPS_PATH = "world_timestamps.csv"
OUT_CSV = "pupil_by_timestamp.csv"

def pick_col(df, options):
    for c in options:
        if c in df.columns:
            return c
    return None

def split_eyes(frame):
    """
    Minimal deterministic split (no face landmarks):
      - crop vertically to reduce noise
      - left eye: left half
      - right eye: right half
    Adjust y0/y1 if needed for your video.
    """
    H, W = frame.shape[:2]
    y0 = int(0.25 * H)
    y1 = int(0.90 * H)
    left = frame[y0:y1, :W//2]
    right = frame[y0:y1, W//2:]
    return left, right

def to_fields(p):
    if p is None:
        return None, None, None
    diameter_px = float(2.0 * np.sqrt(p.a * p.b))
    return float(p.cx), float(p.cy), diameter_px

def main():
    ts = pd.read_csv(TIMESTAMPS_PATH)

    col_ts = pick_col(ts, ["timestamp [ns]", "timestamp_ns", "timestamp"])
    if col_ts is None:
        raise RuntimeError(f"Timestamp column not found in {TIMESTAMPS_PATH}. Columns: {list(ts.columns)}")

    col_section = pick_col(ts, ["section id", "section_id"])
    col_recording = pick_col(ts, ["recording id", "recording_id"])

    timestamps = ts[col_ts].tolist()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    det = PupilLabsLikeDetector(
        dark_percentile=15.0,
        dark_offset=3,
        dark_dilate_ksize=15,
        highlight_percentile=99.7,
        edge_support_min=0.20,
        min_contour_points=20,
        canny_sigma=0.45,
        max_aspect=5.0,
        min_radius=4,
    )


    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "section_id",
            "recording_id",
            "timestamp_ns",
            "frame_idx",
            "left_cx", "left_cy", "left_diameter_px",
            "right_cx", "right_cy", "right_diameter_px",
        ])

        for i, ts_ns in enumerate(timestamps):
            ok, frame = cap.read()

            # if video ends early: keep identical timestamps file length, fill None
            if not ok:
                w.writerow([
                    ts.loc[i, col_section] if col_section else None,
                    ts.loc[i, col_recording] if col_recording else None,
                    int(ts_ns),
                    i,
                    None, None, None,
                    None, None, None,
                ])
                continue

            left_eye, right_eye = split_eyes(frame)

            pL = det.detect_one_eye(left_eye)
            pR = det.detect_one_eye(right_eye)

            lcx, lcy, ldia = to_fields(pL)
            rcx, rcy, rdia = to_fields(pR)

            w.writerow([
                ts.loc[i, col_section] if col_section else None,
                ts.loc[i, col_recording] if col_recording else None,
                int(ts_ns),
                i,
                lcx, lcy, ldia,
                rcx, rcy, rdia,
            ])

    cap.release()
    print(f"[OK] wrote {OUT_CSV} with {len(timestamps)} rows (exactly timestamps).")

if __name__ == "__main__":
    main()
