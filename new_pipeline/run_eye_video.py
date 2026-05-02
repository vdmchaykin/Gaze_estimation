"""
Pupil detection on a single Neon Sensor Module eye video.

Each frame is split horizontally into left / right eye halves.
HeatmapNet runs independently on each half.

Outputs (written to --out directory):
  pupils.csv   – timestamp [ns], xL, yL, xR, yR   (pixel coords in each half crop)
  vis.mp4      – full-frame video with green cross markers on each pupil
"""

import os
import sys
import argparse
import csv

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.video_io import open_video, split_eyes
from utils.heatmap_infer import load_heatmapnet, infer_pupil_from_heatmapnet, draw_cross

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_VIDEO  = "data/2026-04-20_13-58-02-dffaeb67/Neon Sensor Module v1 ps1.mp4"
DEFAULT_CKPT   = "checkpoints_openeds/openeds_finetuned_lpw_validated.pth"
DEFAULT_OUT    = "new_pipeline/outputs"
DEFAULT_STATES = "data/2026-04-20_13-58-02-dffaeb67/3d_eye_states.csv"

INPUT_SIZE = 256
KP_INDEX   = 1      # 0 = iris heatmap, 1 = pupil heatmap
SIGMA      = 1.2
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pupil detection on Neon eye video")
    p.add_argument("--video",  default=DEFAULT_VIDEO,
                   help="Path to Neon Sensor Module eye video")
    p.add_argument("--ckpt",   default=DEFAULT_CKPT,
                   help="HeatmapNet checkpoint (.pth)")
    p.add_argument("--out",    default=DEFAULT_OUT,
                   help="Output directory for CSV and video")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--kp_index", type=int, default=KP_INDEX,
                   help="Heatmap channel: 0=iris, 1=pupil")
    p.add_argument("--states", default=DEFAULT_STATES,
                   help="Path to 3d_eye_states.csv (provides timestamps)")
    p.add_argument("--no_video", action="store_true",
                   help="Skip writing the visualisation video (faster)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"Device : {args.device}")
    print(f"Video  : {args.video}")
    print(f"Ckpt   : {args.ckpt}")
    print(f"States : {args.states}")

    with open(args.states, newline="") as f:
        reader = csv.DictReader(f)
        timestamps = [int(row["timestamp [ns]"]) for row in reader]

    model = load_heatmapnet(args.ckpt, args.device)

    cap, fps, nframes, w, h = open_video(args.video)
    print(f"Frames : {nframes}  |  {w}x{h} @ {fps:.1f} fps")

    if len(timestamps) != nframes:
        raise RuntimeError(
            f"Timestamp count ({len(timestamps)}) does not match video frame count ({nframes}). "
            f"Check that --states matches the correct recording."
        )

    csv_path = os.path.join(args.out, "pupils.csv")
    vid_path = os.path.join(args.out, "vis.mp4")

    # VideoWriter: reconstruct full frame (both halves side by side)
    vid_writer = None
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vid_writer = cv2.VideoWriter(vid_path, fourcc, fps, (w, h))

    with open(csv_path, "w", newline="") as csv_file:
        cw = csv.writer(csv_file)
        cw.writerow(["timestamp [ns]", "xL", "yL", "xR", "yR"])

        for idx in range(nframes):
            ok, frame = cap.read()
            if not ok:
                break

            left_gray, right_gray = split_eyes(frame)

            pred_l = infer_pupil_from_heatmapnet(
                model, left_gray, args.device, INPUT_SIZE, args.kp_index, SIGMA
            )
            pred_r = infer_pupil_from_heatmapnet(
                model, right_gray, args.device, INPUT_SIZE, args.kp_index, SIGMA
            )

            cw.writerow([timestamps[idx],
                         round(pred_l.x, 3), round(pred_l.y, 3),
                         round(pred_r.x, 3), round(pred_r.y, 3)])

            if vid_writer is not None:
                vis_l = draw_cross(left_gray,  pred_l.x, pred_l.y)
                vis_r = draw_cross(right_gray, pred_r.x, pred_r.y)
                # pad right half to match left if frame width is odd
                if vis_l.shape[1] != vis_r.shape[1]:
                    vis_r = vis_r[:, : vis_l.shape[1]]
                vis_full = np.concatenate([vis_l, vis_r], axis=1)
                vid_writer.write(vis_full)

            if idx % 500 == 0:
                print(f"  [{idx:>6}/{nframes}]  "
                      f"L=({pred_l.x:6.1f}, {pred_l.y:5.1f})  "
                      f"R=({pred_r.x:6.1f}, {pred_r.y:5.1f})")

    cap.release()
    if vid_writer is not None:
        vid_writer.release()

    print(f"\nDone.")
    print(f"  CSV   : {csv_path}")
    if not args.no_video:
        print(f"  Video : {vid_path}")


if __name__ == "__main__":
    main()
