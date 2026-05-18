# pupil_detector

Pupil detection pipeline for dual-eye videos (left eye = left half of frame, right eye = right half).
Outputs per-frame pupil position, diameter and confidence as CSV (confidence and diameter not always presented, because if there is no detected elipse, then output of the heatmap network is taken (only coordinates))

---

## Quick start

```bash
python pipeline/pupil_detector.py \
  --video "data/Vadim/Neon Sensor Module v1 ps1.mp4" \
  --states data/Vadim/csv/timestamps.csv \
  --out pipeline/outputs/pupils \
  --heatmap_ckpt checkpoints_openeds/openeds_finetuned_lpw_validated.pth \
  --heatmap_roi_size 30 \
  --step 0 \
  --preview --preview_eye both --preview_scale 2.0
```

Run from the project root (`Gaze_estimation/`).

---

## Outputs

| File | Contents |
|------|----------|
| `<out>.avi` | Annotated video |
| `<out>_pupils.csv` | `timestamp [ns], xL, yL, diameter_L, confidence_L, xR, yR, diameter_R, confidence_R` | -- classical + heatmap predictions
| `<out>_heatmap.csv` | `timestamp [ns], heatmap_xL, heatmap_yL, heatmap_xR, heatmap_yR` | -- pure heatmap predictions

Empty `xL/yL` means no pupil detected for that frame.
If a heatmap model is loaded and detection fails, heatmap coordinates are written instead.

---

## Key flags

### Input

| Flag       | Default | Description |
|------------|---------|-------------|
| `--video`  | —       | Path to input video (**required**) |
| `--states` | `""`    | CSV with a `timestamp_ns` or `timestamp [ns]` column; if omitted, frame index is used as timestamp |
| `--out`    | `debug_step_<N>.avi` | Output path prefix (no extension) |

### Heatmap-based ROI

| Flag                     | Default| Description |
|--------------------------|--------|-------------|
| `--heatmap_ckpt`         | `""`   | Path to HeatmapNet `.pth` checkpoint; if omitted, the full half-frame is used as ROI |
| `--heatmap_roi_size`     | `30`   | Side length (px) of the square ROI centered on the heatmap prediction |
| `--heatmap_kp_index`     | `1`    | Keypoint channel index in the heatmap output (`0` for single-channel checkpoints) |
| `--heatmap_prior_weight` | `0.3`  | Scoring penalty for distance from heatmap prediction (`0` = disabled) |

### Detection quality

| Flag                    | Default | Description |
|-------------------------|---------|-------------|
| `--canny_low`           | `50`    | Canny lower hysteresis threshold |
| `--canny_high`          | `100`   | Canny upper hysteresis threshold |
| `--min_ellipse_area`    | `250`   | Minimum ellipse area in px² — filters out tiny detections |
| `--max_aspect`          | `3.5`   | Maximum axis ratio — rejects elongated ellipses |
| `--support_min_frac`    | `0.20`  | Minimum fraction of edge points that must lie on the ellipse (0–1) |
| `--max_seg_straightness`| `0.92`  | Chord/arc ratio above which a segment is rejected as an eyelash (1.0 = straight line) |
| `--gap_bridge_max_dist` | `0`     | Bridge arc endpoint gaps up to this many pixels; `0` = disabled |

### Visualization / debug

| Flag              | Default | Description |
|-------------------|---------|-------------|
| `--step`          | `0`     | `0` = 4×2 mosaic of all pipeline steps; `1–8` = single step |
| `--preview`       | `false` | Show a live preview window during processing |
| `--preview_eye`   | `both`  | Which eye to show in preview: `both`, `left`, `right` |
| `--preview_scale` | `2.0`   | Upscale factor for the preview window (does not affect saved video) |

---

## Full pipeline

Run all three steps from the project root (`Gaze_estimation/`) for a given participant (e.g. `Melissa`):

```bash
# 1. Pupil detection (~200 fps)
python pipeline/pupil_detector.py \
  --video "data/Melissa/Neon Sensor Module v1 ps1.mp4" \
  --states data/Melissa/csv/timestamps.csv \
  --out pipeline/outputs/pupils \
  --heatmap_ckpt checkpoints_openeds/openeds_finetuned_lpw_validated.pth \
  --heatmap_roi_size 30

# 2. Downsample to 30 fps (timestamp-aligned, confidence-weighted average)
python pipeline/downsample.py --data_dir data/Melissa

# 3. Gaze mapping (polynomial Ridge regression + IMU quaternions)
python pipeline/gaze_mapping.py --data_dir data/Melissa
```

**Outputs** (written to `pipeline/`):

| File | Contents |
|------|----------|
| `30_fps_data.csv` | Downsampled pupil coordinates aligned to scene camera timestamps |
| `gaze_predictions.csv` | Per-frame gaze coordinates `(pred_gaze_x, pred_gaze_y)` |
| `gaze_overlay.mp4` | Scene video with predicted gaze point overlaid |

**Prerequisites** — files expected under `data/<participant>/`:

| File | Description |
|------|-------------|
| `Neon Scene Camera v1 ps1.mp4` | Scene camera video (30 fps) |
| `Neon Sensor Module v1 ps1.mp4` | Dual-eye video (~200 fps) |
| `csv/timestamps.csv` | Eye sensor timestamps (`timestamp_ns` column) |
| `csv/imu.csv` | IMU data with quaternion columns (`quat_w/x/y/z`) |
| `calibration_points.csv` | Ground-truth gaze points from `calibration_tool.py` |

---

## Using as a library

```python
from pipeline.pupil_detector import DetectorConfig, PupilDetector, FrameContext

cfg = DetectorConfig(
    video="",                  # not used when calling process() directly
    heatmap_ckpt="checkpoints_openeds/openeds_finetuned_lpw_validated.pth",
    heatmap_roi_size=30,
    support_min_frac=0.20,
    step=8,
)

detector = PupilDetector(cfg)

# frame_bgr : np.ndarray (H, W, 3) — one eye half
# roi_rect  : (x0, y0, x1, y1) — ignored when heatmap_ckpt is set
ctx = FrameContext(frame_idx=0, roi_rect=(0, 0, frame_bgr.shape[1], frame_bgr.shape[0]))
vis = detector.process(frame_bgr, ctx)

# Results are written into ctx:
print(ctx.pupil_cx, ctx.pupil_cy)      # center in full-frame coords
print(ctx.pupil_diameter)              # mean of ellipse axes
print(ctx.pupil_confidence)            # fraction of edge points on ellipse (0–1)
```

`FrameContext` fields after `process()`:

| Field | Type | Meaning |
|-------|------|---------|
| `pupil_cx`, `pupil_cy` | `float` | Pupil center (full-frame px); `NaN` if not detected |
| `pupil_diameter` | `float` | Mean ellipse axis length (px); `NaN` if not detected |
| `pupil_confidence` | `float` | Edge support fraction 0–1; `NaN` if not detected |
| `heatmap_cx`, `heatmap_cy` | `float` | Heatmap prediction (full-frame px); `NaN` if no model |
