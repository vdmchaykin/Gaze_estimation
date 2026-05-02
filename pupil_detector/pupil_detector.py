import cv2
import numpy as np
import argparse
import os


# -------------------------
# Pre-step: mask dark edge band (left/right)
# -------------------------
def smooth_1d(x, k=9):
    if k <= 1:
        return x
    k = int(k)
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(x.astype(np.float32), kernel, mode="same")


def find_dark_edge_width(gray, side="right", max_scan_frac=0.45, thr_mode="otsu",
                         rel_offset=15, smooth_k=15, min_width_px=10):
    """
    Returns width (pixels) of dark band from chosen edge until it gets "skin-bright".

    side: 'right' or 'left'
    max_scan_frac: scan up to this fraction of image width
    thr_mode:
      - 'otsu': threshold computed via Otsu on the scanned strip
      - 'relative': threshold = median(strip) + rel_offset
    """
    h, w = gray.shape
    max_scan_w = int(w * max_scan_frac)
    max_scan_w = max(max_scan_w, min_width_px)

    if side == "right":
        strip = gray[:, w - max_scan_w : w]
        # column means from edge inward (right->left)
        col_mean = strip.mean(axis=0)          # left->right within strip
        col_mean = col_mean[::-1]              # now index 0 is rightmost edge
    else:
        strip = gray[:, 0 : max_scan_w]
        col_mean = strip.mean(axis=0)          # already from left edge inward

    col_mean_s = smooth_1d(col_mean, smooth_k)

    # decide threshold (brightness where "skin starts")
    if thr_mode == "otsu":
        # cv2.threshold returns (retval, dst)
        thr_val, _ = cv2.threshold(strip, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # make it a bit more conservative (optional)
        thr_val = float(min(255.0, thr_val + 10.0))
    else:
        thr_val = float(np.median(strip) + rel_offset)


    # find first column where brightness crosses threshold
    # (i.e., leaving dark band)
    idx = None
    for i, m in enumerate(col_mean_s):
        if m >= thr_val:
            idx = i
            break

    if idx is None:
        idx = max_scan_w  # whole scanned area is dark

    width = int(max(idx, min_width_px))
    width = min(width, max_scan_w)
    return width, thr_val, col_mean_s


def apply_edge_mask(frame_bgr, side, width_px, fill_value=255):
    """
    Paint the dark edge band as white (fill_value).
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    if side == "right":
        x0 = max(0, w - width_px)
        out[:, x0:w] = fill_value
        bbox = (x0, 0, w, h)
    else:
        x1 = min(w, width_px)
        out[:, 0:x1] = fill_value
        bbox = (0, 0, x1, h)

    return out, bbox


def pick_side_auto(gray):
    """
    Auto: choose side with darker edge region.
    Compare mean brightness in left/right 15% strips.
    """
    h, w = gray.shape
    sw = int(w * 0.15)
    left = gray[:, :sw].mean()
    right = gray[:, w - sw:].mean()
    return "left" if left < right else "right"


# -------------------------
# Step 1: initial estimation of pupil region
# -------------------------
def initial_guess_darkest_patch(gray, search_rect, patch=24, step=8):
    x0, y0, x1, y1 = search_rect
    h, w = gray.shape

    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(w, x1); y1 = min(h, y1)

    best_val = 1e18
    best = (int((x0 + x1) / 2), int((y0 + y1) / 2))

    for y in range(y0, y1 - patch, step):
        for x in range(x0, x1 - patch, step):
            roi = gray[y:y + patch, x:x + patch]
            val = float(np.mean(roi))
            if val < best_val:
                best_val = val
                best = (x + patch // 2, y + patch // 2)

    return best


def draw_dashed_rect(img, x0, y0, x1, y1, color=(255, 255, 255), thickness=1, dash=8, gap=6):
    def dashed_line(p1, p2):
        xA, yA = p1
        xB, yB = p2
        length = int(np.hypot(xB - xA, yB - yA))
        if length <= 0:
            return
        vx = (xB - xA) / length
        vy = (yB - yA) / length
        t = 0
        while t < length:
            t2 = min(t + dash, length)
            xa, ya = int(xA + vx * t), int(yA + vy * t)
            xb, yb = int(xA + vx * t2), int(yA + vy * t2)
            cv2.line(img, (xa, ya), (xb, yb), color, thickness)
            t += dash + gap

    dashed_line((x0, y0), (x1, y0))
    dashed_line((x1, y0), (x1, y1))
    dashed_line((x1, y1), (x0, y1))
    dashed_line((x0, y1), (x0, y0))


def draw_step1_debug(frame_bgr, roi_rect, guess_center, pupil_square_size=140):
    out = frame_bgr.copy()
    x0, y0, x1, y1 = roi_rect

    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)

    cx, cy = guess_center
    hs = pupil_square_size // 2
    px0, py0 = cx - hs, cy - hs
    px1, py1 = cx + hs, cy + hs
    cv2.rectangle(out, (px0, py0), (px1, py1), (255, 255, 255), 2)

    inner = int(pupil_square_size * 0.6)
    hs2 = inner // 2
    qx0, qy0 = cx - hs2, cy - hs2
    qx1, qy1 = cx + hs2, cy + hs2
    draw_dashed_rect(out, qx0, qy0, qx1, qy1, color=(255, 255, 255), thickness=1)

    cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)
    return out


def parse_roi(roi_str, w, h):
    if roi_str.strip():
        x0, y0, x1, y1 = map(int, roi_str.split(","))
        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(1, min(w, x1))
        y1 = max(1, min(h, y1))
        if x1 <= x0 + 10 or y1 <= y0 + 10:
            raise ValueError("ROI too small or invalid.")
        return (x0, y0, x1, y1)

    x0 = int(w * 0.15); x1 = int(w * 0.85)
    y0 = int(h * 0.15); y1 = int(h * 0.85)
    return (x0, y0, x1, y1)


def make_writer(output_path, fps, frame_size):
    w, h = frame_size
    fps = float(fps) if fps and fps > 0 else 30.0

    for codec in ("XVID", "MJPG"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if writer.isOpened():
            return writer, codec, fps

    raise RuntimeError("Could not open VideoWriter (XVID/MJPG).")

def draw_step2_canny(frame_bgr, roi_rect):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    x0, y0, x1, y1 = roi_rect

    roi = gray[y0:y1, x0:x1]

    # сглаживание — критично
    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)

    # Canny
    edges = cv2.Canny(roi_blur, 30, 90)

    # визуализация
    out = frame_bgr.copy()

    # рисуем ROI
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)

    # накладываем edges (зелёные)
    ys, xs = np.where(edges > 0)
    out[y0 + ys, x0 + xs] = (0, 255, 0)

    return out

def compute_dark_region_mask(roi_gray, offset=18, smooth_hist=True):
    """
    STEP 3A:
    Build dark-region mask from ROI histogram.
    """
    # histogram
    hist = cv2.calcHist([roi_gray], [0], None, [256], [0, 256]).flatten()

    if smooth_hist:
        hist = cv2.GaussianBlur(hist.reshape(-1, 1), (1, 9), 0).flatten()

    # darkest peak (left side of histogram)
    search_end = 80
    peak_idx = int(np.argmax(hist[:search_end]))

    thr = min(255, peak_idx + offset)

    dark_mask = (roi_gray <= thr).astype(np.uint8) * 255
    return dark_mask, peak_idx, thr

def compute_specular_mask(roi_gray, thr=220, dilate=3):
    """
    STEP 4:
    specular highlights mask (bright pixels) + dilation to remove halo edges.
    """
    spec = (roi_gray >= thr).astype(np.uint8) * 255
    if dilate and dilate > 0:
        k = 2 * dilate + 1
        spec = cv2.dilate(spec, np.ones((k, k), np.uint8), iterations=1)
    return spec

def _angle_deg(v1, v2):
    n1 = np.linalg.norm(v1) + 1e-9
    n2 = np.linalg.norm(v2) + 1e-9
    v1 = v1 / n1
    v2 = v2 / n2
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def split_contour_by_curvature(points, min_len=40, max_jump=6.0, max_corner_deg=75.0):
    """
    points: (N,2) float/int
    Returns list of segments (arrays) split where:
      - consecutive points jump too much
      - corner angle is too sharp
    """
    if points is None or len(points) < 3:
        return []

    pts = points.astype(np.float32)
    segments = []
    seg = [pts[0]]

    for i in range(1, len(pts) - 1):
        p_prev = pts[i - 1]
        p = pts[i]
        p_next = pts[i + 1]

        # split by big jump
        if np.linalg.norm(p - p_prev) > max_jump:
            if len(seg) >= min_len:
                segments.append(np.array(seg, dtype=np.float32))
            seg = [p]
            continue

        # split by sharp corner
        v1 = p - p_prev
        v2 = p_next - p
        ang = _angle_deg(v1, v2)
        if ang > max_corner_deg:
            seg.append(p)
            if len(seg) >= min_len:
                segments.append(np.array(seg, dtype=np.float32))
            seg = [p]
            continue

        seg.append(p)

    seg.append(pts[-1])
    if len(seg) >= min_len:
        segments.append(np.array(seg, dtype=np.float32))

    return segments


def draw_polyline(img, pts, color, thickness=1):
    pts_i = pts.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(img, [pts_i], isClosed=False, color=color, thickness=thickness)


def ellipse_area(ell):
    (_, _), (A, B), _ = ell
    return float(np.pi * (A / 2.0) * (B / 2.0))


def ellipse_aspect_ratio(ell):
    (_, _), (A, B), _ = ell
    if A <= 0 or B <= 0:
        return 1e9
    return float(max(A / B, B / A))

def ellipse_support_score(ell, edge_points_xy, dist_px=2.0):
    """
    edge_points_xy: (N,2) points in ROI coords (x,y), float32
    Returns (support_frac, score)
    support_frac = fraction of edge points close to ellipse boundary
    """
    if ell is None or edge_points_xy is None or len(edge_points_xy) == 0:
        return 0.0, 0.0

    (cx, cy), (A, B), ang = ell
    a = (A / 2.0) + 1e-9
    b = (B / 2.0) + 1e-9
    theta = np.deg2rad(ang)

    # shift to ellipse center
    pts = edge_points_xy.astype(np.float32)
    x = pts[:, 0] - cx
    y = pts[:, 1] - cy

    # rotate into ellipse axis frame
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    xr = x * cos_t - y * sin_t
    yr = x * sin_t + y * cos_t

    # implicit ellipse value: =1 on boundary
    val = (xr / a) ** 2 + (yr / b) ** 2

    # tolerance band around 1.0
    # approximate: convert pixel tolerance to normalized tolerance
    tol = dist_px / min(a, b)
    supported = np.abs(val - 1.0) <= tol

    support_frac = float(np.mean(supported))
    score = support_frac
    return support_frac, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="path to eye video (.avi/.mp4/...)")
    ap.add_argument("--step", type=int, default=0, help="step: 0=edge mask only, 1=edge mask + ROI/guess")
    ap.add_argument("--roi", type=str, default="", help="ROI as x0,y0,x1,y1. If empty -> central ROI")
    ap.add_argument("--preview", action="store_true", help="show preview window while writing video")
    ap.add_argument("--out", type=str, default="", help="output video path (.avi). Default: debug_step_{step}.avi")
    ap.add_argument("--spec_thr", type=int, default=220, help="specular threshold in ROI (bright pixels)")
    ap.add_argument("--spec_dilate", type=int, default=3, help="dilation radius for specular mask (px)")
    ap.add_argument("--split_min_len", type=int, default=18, help="min points per sub-contour")
    ap.add_argument("--split_max_jump", type=float, default=6.0, help="split if point jump > this")
    ap.add_argument("--split_corner_deg", type=float, default=75.0, help="split if corner angle > this")
    ap.add_argument("--min_ellipse_area", type=float, default=250.0, help="min ellipse area to accept candidate")
    ap.add_argument("--max_aspect", type=float, default=3.5, help="max ellipse aspect ratio")
    ap.add_argument("--max_center_dist", type=float, default=0.9,
                    help="max normalized distance of ellipse center to ROI center (0..~1)")
    ap.add_argument("--draw_all_ellipses", action="store_true", help="draw all ellipse candidates (otherwise top-N)")
    ap.add_argument("--topN", type=int, default=10, help="draw only top N largest-area ellipses if not draw_all_ellipses")
    ap.add_argument("--support_dist_px", type=float, default=2.0,
                help="distance (px) for edge support band around ellipse")
    ap.add_argument("--support_min_frac", type=float, default=0.20,
                help="minimum support fraction to accept final ellipse")


    # edge mask params
    ap.add_argument("--edge_mask", type=str, default="auto", choices=["auto", "left", "right", "off"],
                    help="mask dark edge band on this side")
    ap.add_argument("--edge_max_frac", type=float, default=0.45, help="max fraction of width to scan for dark band")
    ap.add_argument("--edge_thr_mode", type=str, default="otsu", choices=["otsu", "relative"],
                    help="how to find threshold between dark band and skin")
    ap.add_argument("--edge_rel_offset", type=int, default=15, help="for relative mode: median + offset")
    ap.add_argument("--edge_fill", type=int, default=255, help="fill value (255=white) for masked band")
    ap.add_argument("--show_mask_box", action="store_true", help="draw rectangle around masked band")

    args = ap.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Input video not found: {args.video}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, first = cap.read()
    if not ret:
        raise RuntimeError("Empty video or could not read first frame.")

    h, w = first.shape[:2]
    roi_rect = parse_roi(args.roi, w, h)

    output_path = args.out.strip() if args.out.strip() else f"debug_step_{args.step}.avi"
    writer, used_codec, used_fps = make_writer(output_path, fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if args.preview:
        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)

    frame_idx = 0
    side_fixed = None  # for auto: decide once from first frame (more stable)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- pre-step: edge dark mask ----
        masked_frame = frame
        mask_bbox = None
        thr_val = None
        width_px = 0

        if args.edge_mask != "off":
            if args.edge_mask == "auto":
                if side_fixed is None:
                    side_fixed = pick_side_auto(gray)
                side = side_fixed
            else:
                side = args.edge_mask

            width_px, thr_val, _profile = find_dark_edge_width(
                gray, side=side,
                max_scan_frac=args.edge_max_frac,
                thr_mode=args.edge_thr_mode,
                rel_offset=args.edge_rel_offset
            )
            masked_frame, mask_bbox = apply_edge_mask(masked_frame, side, width_px, fill_value=args.edge_fill)

        # ---- Step output ----
        if args.step == 0:
            out = masked_frame.copy()
            if args.show_mask_box and mask_bbox is not None:
                x0, y0, x1, y1 = mask_bbox
                cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)
                cv2.putText(out, f"edge={side_fixed or args.edge_mask} w={width_px}px thr={thr_val:.1f}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)

        elif args.step == 1:
            gray_m = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
            guess = initial_guess_darkest_patch(gray_m, roi_rect)
            out = draw_step1_debug(masked_frame, roi_rect, guess)

            if args.show_mask_box and mask_bbox is not None:
                x0, y0, x1, y1 = mask_bbox
                cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)
        elif args.step == 2:
            out = draw_step2_canny(masked_frame, roi_rect)
        elif args.step == 3:
            # STEP 3A: show dark region mask
            gray_m = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

            x0, y0, x1, y1 = roi_rect
            roi = gray_m[y0:y1, x0:x1]

            dark_mask, peak, thr = compute_dark_region_mask(roi, offset=18)

            out = masked_frame.copy()

            # overlay dark region (blue)
            mask_col = np.zeros_like(out)
            mask_col[y0:y1, x0:x1, 0] = dark_mask  # blue channel
            out = cv2.addWeighted(out, 1.0, mask_col, 0.6, 0)

            cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)
            cv2.putText(out, f"dark peak={peak}, thr={thr}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
        elif args.step == 4:
            # STEP 3B: edges filtered by dark region
            gray_m = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

            x0, y0, x1, y1 = roi_rect
            roi = gray_m[y0:y1, x0:x1]

            # edges
            roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
            edges = cv2.Canny(roi_blur, 30, 90)

            # dark region
            dark_mask, peak, thr = compute_dark_region_mask(roi, offset=14)

            # filter edges
            edges_dark = cv2.bitwise_and(edges, dark_mask)

            out = masked_frame.copy()
            cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)

            ys, xs = np.where(edges_dark > 0)
            out[y0 + ys, x0 + xs] = (0, 255, 0)

            cv2.putText(out, f"edges ∩ dark (peak={peak}, thr={thr})",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
        elif args.step == 5:
            # STEP 4: edges filtered by dark region AND NOT specular
            gray_m = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

            x0, y0, x1, y1 = roi_rect
            roi = gray_m[y0:y1, x0:x1]

            # edges
            roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
            edges = cv2.Canny(roi_blur, 30, 90)

            # dark region (step 3)
            dark_mask, peak, thr = compute_dark_region_mask(roi, offset=18)

            # specular mask (step 4)
            spec_mask = compute_specular_mask(roi, thr=args.spec_thr, dilate=args.spec_dilate)

            # filter edges: keep only dark and not specular
            edges_dark = cv2.bitwise_and(edges, dark_mask)
            edges_final = cv2.bitwise_and(edges_dark, cv2.bitwise_not(spec_mask))

            out = masked_frame.copy()
            cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)

            # draw edges_final in green
            ys, xs = np.where(edges_final > 0)
            out[y0 + ys, x0 + xs] = (0, 255, 0)

            # optional: visualize specular mask as red overlay (helps debug)
            # comment out if you don't want
            red = np.zeros_like(out)
            red[y0:y1, x0:x1, 2] = spec_mask  # red channel
            out = cv2.addWeighted(out, 1.0, red, 0.35, 0)

            cv2.putText(out, f"edges ∩ dark ∩ !spec (peak={peak}, thr={thr}, spec={args.spec_thr})",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
        elif args.step == 6:
            # STEP 5: edges -> contours -> split into sub-contours (curvature continuity approx)
            gray_m = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
            x0, y0, x1, y1 = roi_rect
            roi = gray_m[y0:y1, x0:x1]

            # edges (same as step 4)
            roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
            edges = cv2.Canny(roi_blur, 30, 90)

            # dark region (step 3)
            dark_mask, peak, thr = compute_dark_region_mask(roi, offset=18)

            # specular mask (step 4)
            spec_mask = compute_specular_mask(roi, thr=args.spec_thr, dilate=args.spec_dilate)

            edges_dark = cv2.bitwise_and(edges, dark_mask)
            edges_final = cv2.bitwise_and(edges_dark, cv2.bitwise_not(spec_mask))

            # contours from edges_final
            contours, _ = cv2.findContours(edges_final, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            out = masked_frame.copy()
            cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)

            # draw split segments in different colors (repeat palette)
            palette = [
                (0, 255, 0),
                (0, 200, 255),
                (255, 200, 0),
                (255, 0, 255),
                (255, 255, 0),
                (0, 255, 255),
                (200, 200, 255),
            ]

            seg_count = 0
            for cnt in contours:
                if cnt is None or len(cnt) < 5:
                    continue
                pts = cnt.reshape(-1, 2).astype(np.float32)

                segments = split_contour_by_curvature(
                    pts,
                    min_len=args.split_min_len,
                    max_jump=args.split_max_jump,
                    max_corner_deg=args.split_corner_deg
                )

                for seg in segments:
                    c = palette[seg_count % len(palette)]
                    # draw in ROI coordinates -> shift by (x0,y0)
                    seg_shift = seg.copy()
                    seg_shift[:, 0] += x0
                    seg_shift[:, 1] += y0
                    draw_polyline(out, seg_shift, c, thickness=2)
                    seg_count += 1

            cv2.putText(out, f"step5: segments={seg_count} (peak={peak} thr={thr} spec={args.spec_thr})",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)

        elif args.step == 7:
            # STEP 6: ellipse fitting for candidate segments
            gray_m = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
            x0, y0, x1, y1 = roi_rect
            roi = gray_m[y0:y1, x0:x1]

            # edges
            roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
            edges = cv2.Canny(roi_blur, 30, 90)

            # dark + !spec
            dark_mask, peak, thr = compute_dark_region_mask(roi, offset=18)
            spec_mask = compute_specular_mask(roi, thr=args.spec_thr, dilate=args.spec_dilate)
            edges_dark = cv2.bitwise_and(edges, dark_mask)
            edges_final = cv2.bitwise_and(edges_dark, cv2.bitwise_not(spec_mask))

            # contours
            contours, _ = cv2.findContours(edges_final, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            # collect ellipse candidates
            candidates = []
            roi_h, roi_w = roi.shape
            roi_cx, roi_cy = roi_w / 2.0, roi_h / 2.0
            roi_rad = 0.5 * min(roi_w, roi_h)

            for cnt in contours:
                if cnt is None or len(cnt) < 5:
                    continue
                pts = cnt.reshape(-1, 2).astype(np.float32)

                segments = split_contour_by_curvature(
                    pts,
                    min_len=args.split_min_len,
                    max_jump=args.split_max_jump,
                    max_corner_deg=args.split_corner_deg
                )

                for seg in segments:
                    if len(seg) < 5:
                        continue
                    seg_cv = seg.reshape(-1, 1, 2).astype(np.float32)

                    try:
                        ell = cv2.fitEllipse(seg_cv)
                    except cv2.error:
                        continue

                    # filters
                    area = ellipse_area(ell)
                    if area < args.min_ellipse_area:
                        continue

                    asp = ellipse_aspect_ratio(ell)
                    if asp > args.max_aspect:
                        continue

                    (ecx, ecy), (A, B), ang = ell
                    dist = np.sqrt((ecx - roi_cx) ** 2 + (ecy - roi_cy) ** 2) / (roi_rad + 1e-9)
                    if dist > args.max_center_dist:
                        continue

                    # store (score for now = area; selection happens in step 7)
                    candidates.append((area, dist, asp, ell))

            # sort candidates by area (bigger first)
            candidates.sort(key=lambda t: t[0], reverse=True)

            out = masked_frame.copy()
            cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)

            # draw filtered edges in green (optional, helps)
            ys, xs = np.where(edges_final > 0)
            out[y0 + ys, x0 + xs] = (0, 255, 0)

            # decide which ellipses to draw
            to_draw = candidates if args.draw_all_ellipses else candidates[:args.topN]

            # draw candidate ellipses in blue
            for (area, dist, asp, ell) in to_draw:
                (ecx, ecy), (A, B), ang = ell
                ell_full = ((ecx + x0, ecy + y0), (A, B), ang)
                cv2.ellipse(out, ell_full, (255, 0, 0), 2)

            cv2.putText(out, f"step6: ellipses={len(candidates)} drawn={len(to_draw)}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
        elif args.step == 8:
            # STEP 7: select final ellipse by edge support
            gray_m = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
            x0, y0, x1, y1 = roi_rect
            roi = gray_m[y0:y1, x0:x1]

            # edges
            roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
            edges = cv2.Canny(roi_blur, 30, 90)

            # dark + !spec (same pipeline)
            dark_mask, peak, thr = compute_dark_region_mask(roi, offset=18)
            spec_mask = compute_specular_mask(roi, thr=args.spec_thr, dilate=args.spec_dilate)
            edges_dark = cv2.bitwise_and(edges, dark_mask)
            edges_final = cv2.bitwise_and(edges_dark, cv2.bitwise_not(spec_mask))

            # edge points (ROI coords)
            ys, xs = np.where(edges_final > 0)
            edge_pts = np.stack([xs, ys], axis=1).astype(np.float32) if len(xs) else np.zeros((0, 2), np.float32)

            # contours -> segments -> ellipse candidates (как в step 6)
            contours, _ = cv2.findContours(edges_final, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            roi_h, roi_w = roi.shape
            roi_cx, roi_cy = roi_w / 2.0, roi_h / 2.0
            roi_rad = 0.5 * min(roi_w, roi_h)

            candidates = []
            for cnt in contours:
                if cnt is None or len(cnt) < 5:
                    continue
                pts = cnt.reshape(-1, 2).astype(np.float32)

                segments = split_contour_by_curvature(
                    pts,
                    min_len=args.split_min_len,
                    max_jump=args.split_max_jump,
                    max_corner_deg=args.split_corner_deg
                )

                for seg in segments:
                    if len(seg) < 5:
                        continue
                    seg_cv = seg.reshape(-1, 1, 2).astype(np.float32)
                    try:
                        ell = cv2.fitEllipse(seg_cv)
                    except cv2.error:
                        continue

                    area = ellipse_area(ell)
                    if area < args.min_ellipse_area:
                        continue

                    asp = ellipse_aspect_ratio(ell)
                    if asp > args.max_aspect:
                        continue

                    (ecx, ecy), (A, B), ang = ell
                    dist = np.sqrt((ecx - roi_cx) ** 2 + (ecy - roi_cy) ** 2) / (roi_rad + 1e-9)
                    if dist > args.max_center_dist:
                        continue

                    # support score
                    support_frac, support_score = ellipse_support_score(
                        ell, edge_pts, dist_px=args.support_dist_px
                    )

                    # “augmented” scoring: support + small penalties
                    # support is main thing; penalize far center + extreme aspect a bit
                    final_score = support_score - 0.25 * dist - 0.05 * max(0.0, asp - 1.0)

                    candidates.append((final_score, support_frac, dist, asp, ell))

            # render
            out = masked_frame.copy()
            cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)

            # draw edges_final (optional) in green
            out[y0 + ys, x0 + xs] = (0, 255, 0)

            if not candidates:
                cv2.putText(out, "step7: no ellipse candidates",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2, cv2.LINE_AA)
            else:
                candidates.sort(key=lambda t: t[0], reverse=True)
                best_score, best_support, best_dist, best_asp, best_ell = candidates[0]

                if best_support < args.support_min_frac:
                    cv2.putText(out, f"step7: best support too low ({best_support:.2f})",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    (cx, cy), (A, B), ang = best_ell
                    cx_full = cx + x0
                    cy_full = cy + y0

                    # final ellipse: red
                    cv2.ellipse(out, ((cx_full, cy_full), (A, B), ang), (0, 0, 255), 2)
                    # center: red dot
                    cv2.circle(out, (int(cx_full), int(cy_full)), 3, (0, 0, 255), -1)

                    cv2.putText(out, f"step7: score={best_score:.3f} supp={best_support:.2f} dist={best_dist:.2f} asp={best_asp:.2f}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (255, 255, 255), 2, cv2.LINE_AA)


        else:
            out = masked_frame.copy()
            cv2.putText(out, f"Step {args.step} not implemented yet",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255), 2, cv2.LINE_AA)

        writer.write(out)

        if args.preview:
            cv2.imshow("preview", out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                cv2.waitKey(0)

        frame_idx += 1

    cap.release()
    writer.release()
    if args.preview:
        cv2.destroyAllWindows()

    print(f"Saved: {output_path}")
    print(f"Codec: {used_codec}, FPS: {used_fps:.2f}, Frames: {frame_idx}")
    print(f"ROI: {roi_rect}")
    print(f"edge_mask={args.edge_mask} (auto picked: {side_fixed})")


if __name__ == "__main__":
    main()
