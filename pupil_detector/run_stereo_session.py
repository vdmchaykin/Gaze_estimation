import os
import csv
import argparse
from typing import Optional, Tuple

import cv2
import numpy as np

import pupil_detector_refactored as pd


def read_eye_states_timestamps_ns(path: str) -> list[Optional[int]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return []

    header = [c.strip().lower() for c in rows[0]]
    has_header = any("timestamp" in c for c in header)
    start = 1 if has_header else 0

    col = 0
    if has_header:
        col = None
        for i, name in enumerate(header):
            if "timestamp" in name and "ns" in name:
                col = i
                break
        if col is None:
            for i, name in enumerate(header):
                if "timestamp" in name:
                    col = i
                    break
        if col is None:
            col = 0

    ts = []
    for r in rows[start:]:
        if col >= len(r):
            ts.append(None)
            continue
        s = r[col].strip()
        if not s:
            ts.append(None)
            continue
        if s.isdigit():
            ts.append(int(s))
        else:
            try:
                ts.append(int(float(s)))
            except Exception:
                ts.append(None)

    return ts


def make_writer_mjpg_avi(path, fps, w, h):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
    if vw.isOpened():
        return vw
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vw = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
    if vw.isOpened():
        return vw
    return None


def detect_pupil_in_eye_frame(eye_bgr: np.ndarray, cfg: dict) -> Optional[Tuple[float, float, float, float, float]]:
    gray = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2GRAY)

    masked = eye_bgr
    if cfg["edge_mask"] != "off":
        side = cfg["edge_mask"]
        if side == "auto":
            side = pd.pick_side_auto(gray)

        width_px, _thr_val, _ = pd.find_dark_edge_width(
            gray,
            side=side,
            max_scan_frac=cfg["edge_max_frac"],
            thr_mode=cfg["edge_thr_mode"],
            rel_offset=cfg["edge_rel_offset"],
            smooth_k=cfg["edge_smooth_k"],
            min_width_px=cfg["edge_min_width_px"],
        )
        masked, _bbox = pd.apply_edge_mask(masked, side, int(width_px), fill_value=cfg["edge_fill"])

    gray_m = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    x0, y0, x1, y1 = cfg["roi"]
    roi = gray_m[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    blur_k = cfg["blur_k"] if cfg["blur_k"] % 2 == 1 else (cfg["blur_k"] + 1)
    roi_blur = cv2.GaussianBlur(roi, (blur_k, blur_k), 0)
    edges = cv2.Canny(roi_blur, cfg["canny_lo"], cfg["canny_hi"])

    dark_mask, _peak, _thr_dark = pd.compute_dark_region_mask(roi, offset=cfg["dark_offset"])
    spec_mask = pd.compute_specular_mask(roi, thr=cfg["spec_thr"], dilate=cfg["spec_dilate"])
    edges_final = cv2.bitwise_and(cv2.bitwise_and(edges, dark_mask), cv2.bitwise_not(spec_mask))

    ys, xs = np.where(edges_final > 0)
    edge_pts = np.stack([xs, ys], axis=1).astype(np.float32) if len(xs) else np.zeros((0, 2), np.float32)

    contours, _ = cv2.findContours(edges_final, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    roi_h, roi_w = roi.shape
    roi_cx, roi_cy = roi_w / 2.0, roi_h / 2.0
    roi_rad = 0.5 * min(roi_w, roi_h)

    best = None  # (score, support, dist, asp, ell)
    for cnt in contours:
        if cnt is None or len(cnt) < 5:
            continue
        pts = cnt.reshape(-1, 2).astype(np.float32)

        segments = pd.split_contour_by_curvature(
            pts,
            min_len=cfg["split_min_len"],
            max_jump=cfg["split_max_jump"],
            max_corner_deg=cfg["split_corner_deg"],
        )

        for seg in segments:
            if len(seg) < 5:
                continue
            try:
                ell = cv2.fitEllipse(seg.reshape(-1, 1, 2).astype(np.float32))
            except cv2.error:
                continue

            area = pd.ellipse_area(ell)
            if area < cfg["min_ellipse_area"]:
                continue

            asp = pd.ellipse_aspect_ratio(ell)
            if asp > cfg["max_aspect"]:
                continue

            (ecx, ecy), (A, B), _ang = ell
            dist = float(np.sqrt((ecx - roi_cx) ** 2 + (ecy - roi_cy) ** 2) / (roi_rad + 1e-9))
            if dist > cfg["max_center_dist"]:
                continue

            support, _ = pd.ellipse_support_score(ell, edge_pts, dist_px=cfg["support_dist_px"])
            support = float(support)
            if support < cfg["support_min_frac"]:
                continue

            score = support - 0.25 * dist - 0.05 * max(0.0, asp - 1.0)
            if (best is None) or (score > best[0]):
                best = (float(score), support, dist, float(asp), ell)

    if best is None:
        return None

    score, support, _dist, _asp, ell = best
    (cx, cy), (A, B), _ang = ell

    cx_eye = float(cx + x0)
    cy_eye = float(cy + y0)
    diam_px = float(min(A, B))
    return (cx_eye, cy_eye, diam_px, float(support), float(score))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root_dir", required=True, help="e.g. data/GenData")
    ap.add_argument("--session", required=True, help="session folder name (e.g. 2024-11-20_15-20-28-4ff00407)")

    ap.add_argument("--video_name", default="Neon Sensor Module v1 ps1.mp4")
    ap.add_argument("--eye_states_name", default="3d_eye_states.csv")

    ap.add_argument("--out_csv", default="pupil_LR_by_timestamp.csv")
    ap.add_argument("--out_video", default="pupil_LR_debug.avi")
    ap.add_argument("--no_debug", action="store_true")
    ap.add_argument("--preview", action="store_true")

    # ROI inside each half: x0,y0,x1,y1
    ap.add_argument("--roi", default="",
                    help="ROI inside EACH HALF: x0,y0,x1,y1. Empty -> default centered 60x50.")

    # detector params
    ap.add_argument("--edge_mask", choices=["off", "left", "right", "auto"], default="auto")
    ap.add_argument("--edge_max_frac", type=float, default=0.45)
    ap.add_argument("--edge_thr_mode", choices=["otsu", "relative"], default="otsu")
    ap.add_argument("--edge_rel_offset", type=int, default=15)
    ap.add_argument("--edge_smooth_k", type=int, default=15)
    ap.add_argument("--edge_min_width_px", type=int, default=10)
    ap.add_argument("--edge_fill", type=int, default=255)

    ap.add_argument("--dark_offset", type=int, default=18)
    ap.add_argument("--spec_thr", type=int, default=220)
    ap.add_argument("--spec_dilate", type=int, default=3)
    ap.add_argument("--canny_lo", type=int, default=30)
    ap.add_argument("--canny_hi", type=int, default=90)
    ap.add_argument("--blur_k", type=int, default=5)

    ap.add_argument("--split_min_len", type=int, default=18)
    ap.add_argument("--split_max_jump", type=float, default=6.0)
    ap.add_argument("--split_corner_deg", type=float, default=75.0)

    ap.add_argument("--min_ellipse_area", type=float, default=250.0)
    ap.add_argument("--max_aspect", type=float, default=3.5)
    ap.add_argument("--max_center_dist", type=float, default=0.9)
    ap.add_argument("--support_dist_px", type=float, default=2.0)
    ap.add_argument("--support_min_frac", type=float, default=0.20)

    args = ap.parse_args()

    sess_dir = os.path.join(args.root_dir, args.session)
    if not os.path.isdir(sess_dir):
        raise NotADirectoryError(f"Session dir not found: {sess_dir}")

    video_path = os.path.join(sess_dir, args.video_name)
    ts_path = os.path.join(sess_dir, args.eye_states_name)

    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    if not os.path.exists(ts_path):
        raise FileNotFoundError(ts_path)

    timestamps = read_eye_states_timestamps_ns(ts_path)
    if not timestamps:
        raise RuntimeError("No timestamps in eye states file")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    ret, first = cap.read()
    if not ret:
        raise RuntimeError("Empty video / cannot read first frame")

    H, W = first.shape[:2]
    half = W // 2

    roi_eye = pd.parse_roi(args.roi, half, H)

    cfg = {
        "roi": roi_eye,
        "edge_mask": args.edge_mask,
        "edge_max_frac": args.edge_max_frac,
        "edge_thr_mode": args.edge_thr_mode,
        "edge_rel_offset": args.edge_rel_offset,
        "edge_smooth_k": args.edge_smooth_k,
        "edge_min_width_px": args.edge_min_width_px,
        "edge_fill": args.edge_fill,
        "dark_offset": args.dark_offset,
        "spec_thr": args.spec_thr,
        "spec_dilate": args.spec_dilate,
        "canny_lo": args.canny_lo,
        "canny_hi": args.canny_hi,
        "blur_k": args.blur_k,
        "split_min_len": args.split_min_len,
        "split_max_jump": args.split_max_jump,
        "split_corner_deg": args.split_corner_deg,
        "min_ellipse_area": args.min_ellipse_area,
        "max_aspect": args.max_aspect,
        "max_center_dist": args.max_center_dist,
        "support_dist_px": args.support_dist_px,
        "support_min_frac": args.support_min_frac,
    }

    n_ts = len(timestamps)
    n = min(frame_count, n_ts) if frame_count > 0 else n_ts

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out_csv_path = os.path.join(sess_dir, args.out_csv)
    out_video_path = os.path.join(sess_dir, args.out_video)

    writer = None
    if not args.no_debug:
        writer = make_writer_mjpg_avi(out_video_path, fps, W, H)
        if writer is None:
            print("[WARN] Could not open debug video writer -> will not save debug video")

    if args.preview:
        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)

    with open(out_csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([
            "frame_idx", "timestamp_ns",
            "L_cx_px", "L_cy_px", "L_diam_px", "L_support", "L_score",
            "R_cx_px", "R_cy_px", "R_diam_px", "R_support", "R_score",
        ])

        for i in range(n):
            ret, frame = cap.read()
            if not ret:
                print(f"[WARN] video ended at frame {i} (expected {n})")
                break

            ts = timestamps[i]

            left_eye = frame[:, :half]
            right_eye = frame[:, half:half + half]

            cfg_L = dict(cfg)
            cfg_R = dict(cfg)
            if cfg["edge_mask"] in ("left", "right"):
                cfg_R["edge_mask"] = "left" if cfg["edge_mask"] == "right" else "right"

            L = detect_pupil_in_eye_frame(left_eye, cfg_L)
            R = detect_pupil_in_eye_frame(right_eye, cfg_R)

            def fmt(det, x_offset=0):
                if det is None:
                    return ["", "", "", "", "", ""]
                cx, cy, d, sup, sc = det
                return [f"{(cx + x_offset):.3f}", f"{cy:.3f}", f"{d:.3f}", f"{sup:.4f}", f"{sc:.4f}"]

            wr.writerow(
                [i, int(ts) if ts is not None else ""] +
                fmt(L, x_offset=0) +
                fmt(R, x_offset=half)
            )

            if writer is not None or args.preview:
                vis = frame.copy()
                x0, y0, x1, y1 = roi_eye
                cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 255), 1)
                cv2.rectangle(vis, (half + x0, y0), (half + x1, y1), (255, 255, 255), 1)

                if L is not None:
                    cv2.circle(vis, (int(L[0]), int(L[1])), 3, (0, 0, 255), -1)
                if R is not None:
                    cv2.circle(vis, (int(R[0] + half), int(R[1])), 3, (0, 0, 255), -1)

                if writer is not None:
                    writer.write(vis)
                if args.preview:
                    cv2.imshow("preview", vis)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break

    cap.release()
    if writer is not None:
        writer.release()
    if args.preview:
        cv2.destroyAllWindows()

    print(f"Saved CSV: {out_csv_path}")
    if writer is not None:
        print(f"Saved debug video: {out_video_path}")
    print(f"ROI (per half): {roi_eye}")
    print(f"Frames used: {min(n, i+1)} / timestamps: {len(timestamps)}")


if __name__ == "__main__":
    main()
