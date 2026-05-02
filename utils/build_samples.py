from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import cv2
import os

from utils.video_io import open_video, read_frame, split_eyes
from utils.neon_io import (
    read_gaze_csv, read_events_csv, read_eye_states_csv,
    extract_test_intervals, filter_df_by_intervals, filter_gaze_quality
)
from utils.heatmap_infer import infer_pupil_from_heatmapnet, draw_cross
from utils.paper_gaze import (
    read_world_timestamps_csv,
    load_paper_homographies,
    build_paper_gaze_targets,
    PaperGazeConfig,
)


@dataclass
class BuildParams:
    device: str
    input_size: int
    kp_index: int
    sigma: float
    frame_step: int
    conf_thresh: float
    tolerance_ms: float
    vis_n: int
    max_process: int  # -1 => no limit


def _merge_asof_nearest(a: pd.DataFrame, b: pd.DataFrame, tol_ms: float) -> pd.DataFrame:
    """
    a: must have 'timestamp [ns]'
    b: must have 'timestamp [ns]'
    """
    a = a.sort_values("timestamp [ns]").reset_index(drop=True)
    b = b.sort_values("timestamp [ns]").reset_index(drop=True)
    tol_ns = int(tol_ms * 1e6)  # ms -> ns

    out = pd.merge_asof(
        a, b,
        on="timestamp [ns]",
        direction="nearest",
        tolerance=tol_ns
    )
    return out


def build_samples_for_session(
    session_id: str,
    session_dir: str,
    eye_video_path: str,
    model,
    out_dir: str,
    p: BuildParams,
) -> pd.DataFrame:
    """
    Build supervised samples for ONE session.
    Returns df with columns:
      timestamp [ns], frame_idx,
      xL,yL,confL,xR,yR,confR,
      gaze x [px], gaze y [px]
    """
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, "vis_eye", session_id)
    os.makedirs(vis_dir, exist_ok=True)

    # --- load CSVs ---
    gaze = read_gaze_csv(os.path.join(session_dir, "gaze.csv"))
    events = read_events_csv(os.path.join(session_dir, "events.csv"))
    eye_states = read_eye_states_csv(os.path.join(session_dir, "3d_eye_states.csv"))

    # --- keep only test intervals ---
    intervals = extract_test_intervals(events)

    gaze = filter_df_by_intervals(gaze, "timestamp [ns]", intervals)
    gaze = filter_gaze_quality(gaze)

    # IMPORTANT: eye_states has 1 row per eye frame index (same length as video frames)
    eye_states_test = filter_df_by_intervals(eye_states, "timestamp [ns]", intervals)

    # --- open video and sanity-check length ---
    cap, fps, nframes, W, H = open_video(eye_video_path)
    if len(eye_states) != nframes:
        print(f"[WARN][{session_id}] eye_states rows ({len(eye_states)}) != video frames ({nframes}). "
              f"Still proceeding, but frame/timestamp alignment may be off.")

    # indices in ORIGINAL eye_states indexing (frame_idx)
    # (we filtered rows, but keep their original index as frame index if available)
    # if eye_states has no explicit frame idx column, we assume row order = frame index
    idx_all = eye_states_test.index.values.astype(int)

    idx_all = idx_all[::max(1, p.frame_step)]
    if p.max_process and p.max_process > 0:
        idx_all = idx_all[:p.max_process]

    rows = []
    saved_vis = 0

    for j, frame_idx in enumerate(idx_all):
        frame = read_frame(cap, int(frame_idx))
        if frame is None:
            continue

        left, right = split_eyes(frame)

        predL = infer_pupil_from_heatmapnet(model, left, p.device, p.input_size, p.kp_index, p.sigma)
        predR = infer_pupil_from_heatmapnet(model, right, p.device, p.input_size, p.kp_index, p.sigma)

        if predL.conf < p.conf_thresh and predR.conf < p.conf_thresh:
            continue

        ts_ns = int(eye_states.loc[frame_idx, "timestamp [ns]"]) if frame_idx in eye_states.index else int(eye_states["timestamp [ns]"].iloc[frame_idx])

        rows.append({
            "session_id": session_id,
            "frame_idx": int(frame_idx),
            "timestamp [ns]": ts_ns,
            "xL": predL.x, "yL": predL.y, "confL": predL.conf,
            "xR": predR.x, "yR": predR.y, "confR": predR.conf,
        })

        if saved_vis < p.vis_n:
            imgL = draw_cross(left, predL.x, predL.y)
            imgR = draw_cross(right, predR.x, predR.y)
            vis = cv2.hconcat([imgL, imgR])
            cv2.imwrite(os.path.join(vis_dir, f"eye_{saved_vis:04d}_frame{frame_idx}.png"), vis)
            saved_vis += 1

        if (j + 1) % 200 == 0:
            print(f"[{session_id}] processed {j+1}/{len(idx_all)} selected frames", flush=True)

    cap.release()

    pupil_df = pd.DataFrame(rows)
    if len(pupil_df) == 0:
        raise RuntimeError(f"[{session_id}] No pupil predictions produced (maybe conf_thresh too high?).")

    # ---- build paper_u,v targets from gaze + world_timestamps + homographies ----
    world_ts_path = os.path.join(session_dir, "world_timestamps.csv")
    if not os.path.exists(world_ts_path):
        raise FileNotFoundError(f"Missing world_timestamps.csv: {world_ts_path}")

    # NOTE: предполагаем, что ты уже сгенерил homographies npz для этой сессии
    # Лучше хранить по сессиям: outputs/<session_id>/paper_homographies.npz
    # Пока используем общий путь, если ты так делал.
    H_npz_path = os.path.join(out_dir, f"{session_id}_paper_homographies.npz")

    if not os.path.exists(H_npz_path):
        raise FileNotFoundError(
            f"Missing paper homographies npz: {H_npz_path}\n"
            f"Run paper tracker first and save npz there."
        )

    Hpack = load_paper_homographies(H_npz_path)
    world_ts = read_world_timestamps_csv(world_ts_path)

    pg_cfg = PaperGazeConfig(
        tolerance_ms=p.tolerance_ms,
        max_age_s=2.0,      # можно потом вынести в args
        uv_margin=0.15,
    )

    paper_gaze = build_paper_gaze_targets(gaze, world_ts, Hpack, pg_cfg)

    # now match pupil timestamps -> paper targets by nearest timestamp
    paper_keep = paper_gaze[["timestamp [ns]", "paper_u", "paper_v"]].copy()
    merged = _merge_asof_nearest(pupil_df, paper_keep, p.tolerance_ms)
    merged = merged.dropna(subset=["paper_u", "paper_v"]).reset_index(drop=True)

    # Save session samples (paper)
    merged.to_csv(os.path.join(out_dir, f"samples_paper_{session_id}.csv"), index=False)

    # also store paper size for downstream (once is enough)
    with open(os.path.join(out_dir, "paper_spec_used.txt"), "w") as f:
        f.write(f"paper_w={Hpack['paper_w']}\n")
        f.write(f"paper_h={Hpack['paper_h']}\n")


    # Save per-session artifacts
    pupil_df.to_csv(os.path.join(out_dir, f"pupil_predictions_{session_id}.csv"), index=False)
    merged.to_csv(os.path.join(out_dir, f"samples_{session_id}.csv"), index=False)

    print(f"[{session_id}] samples: {len(merged)} (from pupil frames {len(pupil_df)})")
    return merged


def concat_samples(samples_list: List[pd.DataFrame]) -> pd.DataFrame:
    if len(samples_list) == 0:
        raise ValueError("No samples to concatenate.")
    df = pd.concat(samples_list, axis=0, ignore_index=True)
    return df
