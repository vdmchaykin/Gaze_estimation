#!/usr/bin/env python3
"""
Calibration tool for gaze mapping.

Plays the scene video with a small eye-camera overlay in the corner.
You click on a numbered calibration point in the scene at the moment
the person is looking at it. Each click records (point_id, frame_idx,
timestamp_ns, gaze_x, gaze_y).  Collect 9 points, then press 's' to
save a calibration CSV that feeds directly into the gaze mapping model.

Usage:
    python calibration_tool.py \\
        --scene  "Neon Scene Camera v1 ps1.mp4" \\
        --eye    "Neon Sensor Module v1 ps1.mp4" \\
        --timestamps world_timestamps.csv \\
        --out    calibration.csv \\
        --scale  0.6

Controls:
    Space        pause / resume
    ← / →        step one frame (while paused)
    Click        mark calibration point at cursor position
    1 – 9        set the ID of the NEXT point to mark
    u            undo last marked point
    s / Enter    save CSV and exit
    q / Esc      quit without saving
"""

import argparse
import csv
import os

# Force X11 (xcb) before cv2 initializes Qt — fixes NULL window handle on Wayland
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# Point Qt to cv2's own plugin directory so libqxcb.so is found
import sysconfig as _sc
_site = _sc.get_path("purelib")
_qt_plugins = os.path.join(_site, "cv2", "qt", "plugins")
if os.path.isdir(_qt_plugins):
    os.environ.setdefault("QT_PLUGIN_PATH", _qt_plugins)

# Point Qt to system fonts to suppress font warnings
for _font_dir in ("/usr/share/fonts", "/usr/share/fonts/truetype"):
    if os.path.isdir(_font_dir):
        os.environ.setdefault("QT_QPA_FONTDIR", _font_dir)
        break

import cv2
import numpy as np
import pandas as pd


# One distinct colour per calibration point
_POINT_COLORS = [
    (0,   0,   255),   # P1  red
    (0,  120,  255),   # P2  orange
    (0,  220,  220),   # P3  yellow
    (0,  200,   80),   # P4  green
    (200, 200,   0),   # P5  cyan
    (255,  80,   0),   # P6  blue
    (255,   0,  180),  # P7  violet
    (120,   0,  255),  # P8  purple
    (0,  180,  255),   # P9  gold
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_scene_timestamps(path: str) -> list:
    df = pd.read_csv(path)
    col = next((c for c in df.columns if "timestamp" in c.lower()), None)
    if col is None:
        raise ValueError(f"No timestamp column found in {path}")
    ts = pd.to_numeric(df[col], errors="coerce").dropna().astype(np.int64).tolist()
    return ts


def _seek_eye(eye_cap, scene_frame_idx, eye_fps, scene_fps):
    """Seek eye video to the frame matching scene_frame_idx and return it."""
    if eye_cap is None:
        return None
    target = int(scene_frame_idx * eye_fps / scene_fps)
    eye_cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = eye_cap.read()
    return frame if ok else None


SAVE_BTN_RECT = None  # set at draw time in original-res coords: (x0,y0,x1,y1)

def _draw_hud(frame, points, next_id, paused, frame_idx, total):
    """Draw status bar, help bar, SAVE button, and all placed calibration dots."""
    global SAVE_BTN_RECT
    h, w = frame.shape[:2]

    # ── top status bar ──────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 38), (25, 25, 25), -1)
    status  = "PAUSED" if paused else "PLAYING"
    s_color = (0, 200, 255) if paused else (60, 230, 60)
    info = (f"{status}   Frame {frame_idx}/{total}   "
            f"Points {len(points)}/9   Next: P{next_id}")
    cv2.putText(frame, info, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, s_color, 1, cv2.LINE_AA)

    # ── SAVE button (top-right) ──────────────────────────────────────────────
    btn_w, btn_h = 160, 30
    bx0 = w - btn_w - 8
    by0 = 4
    bx1 = w - 8
    by1 = by0 + btn_h
    SAVE_BTN_RECT = (bx0, by0, bx1, by1)
    cv2.rectangle(frame, (bx0, by0), (bx1, by1), (0, 180, 0), -1)
    cv2.rectangle(frame, (bx0, by0), (bx1, by1), (255, 255, 255), 1)
    cv2.putText(frame, "SAVE & FINISH", (bx0 + 8, by1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # ── bottom help bar ──────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, h - 28), (w, h), (25, 25, 25), -1)
    cv2.putText(frame,
                "Space=pause  Click=mark  1-9=set ID  u=undo  s=save  q=quit",
                (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (170, 170, 170), 1, cv2.LINE_AA)

    # ── calibration dots ────────────────────────────────────────────────────
    for pid, _fidx, _ts, x, y in points:
        col = _POINT_COLORS[(pid - 1) % len(_POINT_COLORS)]
        cv2.circle(frame, (x, y), 16, col, -1)
        cv2.circle(frame, (x, y), 16, (255, 255, 255), 2)
        cv2.putText(frame, str(pid), (x - 7, y + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def _embed_eye(scene_frame, eye_frame, frac=0.22, margin=12):
    """Paste a scaled-down eye frame into the bottom-right corner."""
    sh, sw = scene_frame.shape[:2]
    eh, ew = eye_frame.shape[:2]
    tw = max(80, int(sw * frac))
    th = max(1,  int(tw * eh / max(ew, 1)))

    small = cv2.resize(eye_frame, (tw, th), interpolation=cv2.INTER_AREA)

    x0 = sw - tw - margin
    y0 = sh - th - margin - 28          # 28 = bottom help bar height
    # clamp
    x0 = max(0, x0);  y0 = max(38, y0) # 38 = top status bar height

    # border
    cv2.rectangle(scene_frame,
                  (x0 - 2, y0 - 18), (x0 + tw + 2, y0 + th + 2),
                  (200, 200, 200), 1)
    cv2.putText(scene_frame, "Eye cam",
                (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (200, 200, 200), 1, cv2.LINE_AA)
    scene_frame[y0:y0 + th, x0:x0 + tw] = small


# ---------------------------------------------------------------------------
# Main tool
# ---------------------------------------------------------------------------

class CalibrationTool:
    def __init__(self, scene_path, eye_path, ts_path, output_csv, scale):
        self.scene_path  = scene_path
        self.eye_path    = eye_path
        self.ts_path     = ts_path
        self.output_csv  = output_csv
        self.scale       = scale

        self.points   = []   # list of (point_id, frame_idx, timestamp_ns, x, y)
        self.next_id  = 1
        self.paused   = True
        self._click   = None  # pending click in original-res scene coords
        self._save_clicked = False

    # ── mouse callback ───────────────────────────────────────────────────────
    def _on_mouse(self, event, mx, my, flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # convert display coords → original-res coords
        ox = int(mx / self.scale)
        oy = int(my / self.scale)
        # check SAVE button first
        if SAVE_BTN_RECT is not None:
            bx0, by0, bx1, by1 = SAVE_BTN_RECT
            if bx0 <= ox <= bx1 and by0 <= oy <= by1:
                self._save_clicked = True
                return
        # ignore clicks on top / bottom bars
        if oy < 38 or oy > self._scene_h - 28:
            return
        self._click = (ox, oy)

    # ── run loop ─────────────────────────────────────────────────────────────
    def run(self):
        scene_cap = cv2.VideoCapture(self.scene_path)
        if not scene_cap.isOpened():
            raise RuntimeError(f"Cannot open: {self.scene_path}")

        eye_cap = None
        if self.eye_path and os.path.exists(self.eye_path):
            eye_cap = cv2.VideoCapture(self.eye_path)
            if not eye_cap.isOpened():
                print(f"[warn] Could not open eye video: {self.eye_path}")
                eye_cap = None

        scene_fps   = scene_cap.get(cv2.CAP_PROP_FPS) or 30.0
        scene_total = int(scene_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._scene_w = int(scene_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._scene_h = int(scene_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        eye_fps = (eye_cap.get(cv2.CAP_PROP_FPS) or 200.0) if eye_cap else 0.0

        timestamps = []
        if self.ts_path and os.path.exists(self.ts_path):
            timestamps = _load_scene_timestamps(self.ts_path)
            if len(timestamps) < scene_total:
                print(f"[warn] timestamp count {len(timestamps)} < frame count {scene_total} — last frames will use frame index as timestamp")
            elif len(timestamps) > scene_total:
                # Neon exports extra leading timestamps; trim to video length
                timestamps = timestamps[-scene_total:]

        win = "Calibration Tool"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        dw = int(self._scene_w * self.scale)
        dh = int(self._scene_h * self.scale)
        cv2.resizeWindow(win, dw, dh)

        print(f"\nScene  : {self._scene_w}×{self._scene_h}  {scene_fps:.1f} fps  {scene_total} frames")
        if eye_cap:
            print(f"Eye    : {int(eye_cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
                  f"{int(eye_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}  {eye_fps:.1f} fps")
        print(f"Output : {self.output_csv}\n")

        frame_idx  = 0
        eye_frame  = None
        delay_ms   = max(1, int(1000 / scene_fps))

        # Read first frame
        ok, cur_frame = scene_cap.read()
        if not ok:
            raise RuntimeError("Cannot read first frame.")

        # imshow must be called before setMouseCallback — it is what actually
        # creates the Qt window handle that setMouseCallback requires
        first_disp = cv2.resize(cur_frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(win, first_disp)
        cv2.waitKey(1)
        cv2.setMouseCallback(win, self._on_mouse)

        while True:
            # ── advance eye video to match scene position ─────────────────
            if eye_cap and not self.paused:
                eye_target = int(frame_idx * eye_fps / scene_fps)
                while int(eye_cap.get(cv2.CAP_PROP_POS_FRAMES)) < eye_target:
                    ok_e, ef = eye_cap.read()
                    if ok_e:
                        eye_frame = ef
                    else:
                        break

            # ── build display frame ───────────────────────────────────────
            vis = cur_frame.copy()
            _draw_hud(vis, self.points, self.next_id, self.paused,
                      frame_idx, scene_total)
            if eye_frame is not None:
                _embed_eye(vis, eye_frame)

            disp = cv2.resize(vis, (dw, dh), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(win, disp)

            # ── process pending click ─────────────────────────────────────
            if self._click is not None:
                sx, sy = self._click
                self._click = None
                ts = timestamps[frame_idx] if frame_idx < len(timestamps) else frame_idx
                # overwrite existing point with same ID, or append
                self.points = [p for p in self.points if p[0] != self.next_id]
                self.points.append((self.next_id, frame_idx, ts, sx, sy))
                self.points.sort(key=lambda p: p[0])
                print(f"  P{self.next_id}  ({sx:4d}, {sy:4d})  frame={frame_idx}  ts={ts}")
                if self.next_id < 9:
                    self.next_id += 1

            # ── SAVE button click ─────────────────────────────────────────
            if self._save_clicked:
                self._save_clicked = False
                self._save()
                break

            # ── keyboard ─────────────────────────────────────────────────
            wait = delay_ms if not self.paused else 30
            key  = cv2.waitKey(wait) & 0xFF

            if key in (ord('q'), 27):          # q / Esc → quit
                print("Quit without saving.")
                break

            elif key in (ord('s'), 13):        # s / Enter → save
                self._save()
                break

            elif key == ord(' '):              # Space → pause / resume
                self.paused = not self.paused

            elif key == ord('u'):              # u → undo
                if self.points:
                    removed = self.points.pop()
                    self.next_id = removed[0]
                    print(f"  Undid P{removed[0]}")

            elif ord('1') <= key <= ord('9'):  # 1-9 → set next ID
                self.next_id = key - ord('0')
                print(f"  Next point ID → {self.next_id}")

            # Arrow keys: Linux OpenCV returns 81/83 after & 0xFF
            elif key in (81, 2):               # ← step back
                if self.paused and frame_idx > 0:
                    frame_idx -= 1
                    scene_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ok, cur_frame = scene_cap.read()
                    if not ok:
                        frame_idx += 1
                    else:
                        eye_frame = _seek_eye(eye_cap, frame_idx, eye_fps, scene_fps)

            elif key in (83, 3):               # → step forward
                if self.paused:
                    ok, nf = scene_cap.read()
                    if ok:
                        frame_idx += 1
                        cur_frame = nf
                        eye_frame = _seek_eye(eye_cap, frame_idx, eye_fps, scene_fps)

            # ── advance scene if playing ──────────────────────────────────
            if not self.paused:
                ok, nf = scene_cap.read()
                if ok:
                    frame_idx += 1
                    cur_frame = nf
                else:
                    self.paused = True   # reached end

        scene_cap.release()
        if eye_cap:
            eye_cap.release()
        cv2.destroyAllWindows()

    # ── save ─────────────────────────────────────────────────────────────────
    def _save(self):
        if not self.points:
            print("Nothing to save.")
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.output_csv)), exist_ok=True)
        with open(self.output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["point_id", "frame_idx", "timestamp_ns", "gaze_x", "gaze_y"])
            for pid, fidx, ts, x, y in self.points:
                w.writerow([pid, fidx, ts, x, y])
        print(f"\nSaved {len(self.points)} calibration points → {self.output_csv}")
        print(f"{'ID':>4}  {'  X':>6}  {'  Y':>6}  {'frame':>7}")
        for pid, fidx, _ts, x, y in self.points:
            print(f"  P{pid}  {x:6d}  {y:6d}  {fidx:7d}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    data_dir = os.path.join(os.path.dirname(__file__),
                            "../data/2026-04-20_13-58-02-dffaeb67")

    ap = argparse.ArgumentParser(
        description="Manual calibration: click 9 points on the scene video.")
    ap.add_argument("--scene",
                    default=os.path.join(data_dir, "Neon Scene Camera v1 ps1.mp4"),
                    help="Scene video path")
    ap.add_argument("--eye",
                    default=os.path.join(data_dir, "Neon Sensor Module v1 ps1.mp4"),
                    help="Eye camera video path (optional, shown as overlay)")
    ap.add_argument("--timestamps",
                    default=os.path.join(data_dir, "world_timestamps.csv"),
                    help="Scene timestamp CSV (world_timestamps.csv)")
    ap.add_argument("--out",
                    default=os.path.join(data_dir, "calibration_points.csv"),
                    help="Output CSV for calibration points")
    ap.add_argument("--scale", type=float, default=0.6,
                    help="Display scale factor (default 0.6 → 960×720 window)")
    args = ap.parse_args()

    tool = CalibrationTool(
        scene_path=args.scene,
        eye_path=args.eye,
        ts_path=args.timestamps,
        output_csv=args.out,
        scale=args.scale,
    )
    tool.run()


if __name__ == "__main__":
    main()
