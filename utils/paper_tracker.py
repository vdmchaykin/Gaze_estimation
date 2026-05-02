from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import cv2
import numpy as np

from utils.apriltag_paper import (
    create_detector,
    detect_tags,
    PaperSpec,
    draw_tag_debug,
    _paper_corners,  # internal helper; ok to reuse
)

# --------- config ---------

@dataclass
class TrackConfig:
    world_video: str
    out_video: str = "outputs/paper_track_debug.mp4"
    out_npz: str = "outputs/paper_homographies.npz"

    # process every Nth frame (speed)
    frame_step: int = 1

    # paper rectified size (only for defining target corner coords)
    paper_w: int = 1000
    paper_h: int = 700

    # tag ids at corners of the sheet
    tag_to_corner: Dict[int, str] = None  # e.g., {2:'TL',4:'TR',6:'BR',5:'BL'}

    # consider last H "stale" after this many seconds (still shown, but flagged)
    stale_after_s: float = 0.8

    # if no valid H ever found, just render tag overlay and status
    codec: str = "mp4v"


# --------- helpers ---------

_CORNER_ORDER = ["TL", "TR", "BR", "BL"]

def _corner_idx(name: str) -> int:
    return _CORNER_ORDER.index(name)

def _compute_h_from_centers(tag_centers: Dict[int, np.ndarray], spec: PaperSpec) -> Optional[np.ndarray]:
    """
    Uses 4 tag centers as anchors for 4 paper corners.
    Requires all 4 tags specified in spec.tag_to_corner to be present.
    """
    world_pts = []
    paper_pts = []
    paper_c = _paper_corners(spec)  # TL,TR,BR,BL

    for tid, corner_name in spec.tag_to_corner.items():
        if tid not in tag_centers:
            return None
        world_pts.append(tag_centers[tid])
        paper_pts.append(paper_c[_corner_idx(corner_name)])

    world_pts = np.array(world_pts, dtype=np.float32)
    paper_pts = np.array(paper_pts, dtype=np.float32)

    H, _ = cv2.findHomography(world_pts, paper_pts, method=0)
    if H is None:
        return None
    return H.astype(np.float64)

def _overlay_status(img: np.ndarray, text_lines: List[str]) -> np.ndarray:
    out = img.copy()
    x, y = 15, 30
    for i, line in enumerate(text_lines):
        yy = y + i * 28
        cv2.putText(out, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(out, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(out, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return out

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# --------- main ---------

def track_paper_homography(cfg: TrackConfig):
    if cfg.tag_to_corner is None or len(cfg.tag_to_corner) != 4:
        raise ValueError("cfg.tag_to_corner must map exactly 4 tag ids to corners, e.g. {2:'TL',4:'TR',6:'BR',5:'BL'}")

    _ensure_dir(cfg.out_video)
    _ensure_dir(cfg.out_npz)

    cap = cv2.VideoCapture(cfg.world_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open world video: {cfg.world_video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*cfg.codec)
    vw = cv2.VideoWriter(cfg.out_video, fourcc, fps / cfg.frame_step, (W, Hh))
    if not vw.isOpened():
        raise RuntimeError(f"Could not open VideoWriter: {cfg.out_video}")

    det = create_detector()

    spec = PaperSpec(
        paper_w=cfg.paper_w,
        paper_h=cfg.paper_h,
        tag_to_corner=cfg.tag_to_corner
    )

    # storage per processed frame
    frame_idxs = []
    H_list = []          # (M, 3, 3)
    H_valid = []         # (M,)
    H_age_s = []         # (M,)
    detected_ids_list = []

    last_H = None
    last_update_frame = None

    required_ids = sorted(list(cfg.tag_to_corner.keys()))

    i = 0
    processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if i % cfg.frame_step != 0:
            i += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = detect_tags(gray, det)

        tag_corners = {}
        tag_centers = {}
        for d in dets:
            tid = int(d.tag_id)
            poly = np.array(d.corners, dtype=np.float32)
            tag_corners[tid] = poly
            tag_centers[tid] = poly.mean(axis=0)

        detected_ids = sorted(tag_centers.keys())
        detected_ids_list.append(detected_ids)

        # update H if all required tags present
        have_all = all(tid in tag_centers for tid in required_ids)
        updated = False

        if have_all:
            Hwp = _compute_h_from_centers(tag_centers, spec)
            if Hwp is not None:
                last_H = Hwp
                last_update_frame = i
                updated = True

        # compute age
        if last_H is None or last_update_frame is None:
            age_s = float("inf")
            valid_now = False
        else:
            age_s = (i - last_update_frame) / max(1e-6, fps)
            valid_now = True

        stale = valid_now and (age_s > cfg.stale_after_s)

        # visualize
        vis = draw_tag_debug(frame, tag_corners)

        status_lines = [
            f"frame {i}/{nframes-1} | detected {len(detected_ids)} tags: {detected_ids}",
            f"required ids: {required_ids}",
        ]

        if updated:
            status_lines.append("H: UPDATED (4/4 tags)")
        else:
            if have_all:
                status_lines.append("H: FAILED (findHomography returned None)")
            else:
                status_lines.append(f"H: NOT UPDATED ({sum(tid in tag_centers for tid in required_ids)}/4 tags)")

        if valid_now:
            status_lines.append(f"H usable: YES | age={age_s:.3f}s" + (" | STALE" if stale else ""))
        else:
            status_lines.append("H usable: NO (no valid H found yet)")

        vis = _overlay_status(vis, status_lines)

        vw.write(vis)

        # record
        frame_idxs.append(i)
        if last_H is None:
            H_list.append(np.zeros((3, 3), dtype=np.float64))
            H_valid.append(False)
        else:
            H_list.append(last_H.copy())
            H_valid.append(True)
        H_age_s.append(age_s)

        processed += 1
        if processed % 200 == 0:
            print(f"[paper_tracker] processed {processed} frames (step={cfg.frame_step})", flush=True)

        i += 1

    cap.release()
    vw.release()

    H_arr = np.stack(H_list, axis=0) if len(H_list) else np.zeros((0, 3, 3), dtype=np.float64)
    np.savez(
        cfg.out_npz,
        frame_idx=np.array(frame_idxs, dtype=np.int32),
        H_world_to_paper=H_arr,
        H_valid=np.array(H_valid, dtype=np.bool_),
        H_age_s=np.array(H_age_s, dtype=np.float32),
        fps=np.float32(fps),
        world_w=np.int32(W),
        world_h=np.int32(Hh),
        required_ids=np.array(required_ids, dtype=np.int32),
        paper_w=np.int32(cfg.paper_w),
        paper_h=np.int32(cfg.paper_h),
    )

    print(f"[paper_tracker] saved video: {cfg.out_video}")
    print(f"[paper_tracker] saved homographies: {cfg.out_npz}")
