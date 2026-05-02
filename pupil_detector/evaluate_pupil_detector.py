"""
Evaluation script for pupil detector.

Evaluates pupil detection on LPW (Labelled Pupils in the Wild) dataset.

Metrics computed:
- Detection Rate: % of frames where pupil was detected
- Localization Error: Euclidean distance between predicted and ground truth
- Accuracy at thresholds: % within N pixels of ground truth
- Normalized Error: Error normalized by eye image dimensions

Usage:
    python evaluate_pupil_detector.py
    python evaluate_pupil_detector.py --subjects 1 2 3 --videos 1 4 9
    python evaluate_pupil_detector.py --visualize --max_frames 100
"""

import argparse
import os
import sys
import csv
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pupil_detector_refactored as pd


# ==================== DATA CLASSES ====================
@dataclass
class DetectionResult:
    """Single frame detection result"""
    frame_idx: int
    gt_x: float
    gt_y: float
    pred_x: Optional[float] = None
    pred_y: Optional[float] = None
    pred_diam: Optional[float] = None
    detected: bool = False
    error_px: Optional[float] = None


@dataclass 
class VideoResults:
    """Results for a single video"""
    subject: str
    video: str
    results: List[DetectionResult] = field(default_factory=list)
    
    @property
    def n_frames(self) -> int:
        return len(self.results)
    
    @property
    def n_detected(self) -> int:
        return sum(1 for r in self.results if r.detected)
    
    @property
    def detection_rate(self) -> float:
        if self.n_frames == 0:
            return 0.0
        return 100.0 * self.n_detected / self.n_frames
    
    @property
    def errors(self) -> np.ndarray:
        return np.array([r.error_px for r in self.results if r.detected and r.error_px is not None])


# ==================== DETECTOR CONFIG ====================
def get_default_detector_config(img_w: int, img_h: int) -> dict:
    """Default configuration for pupil detector"""
    # ROI: full screen
    x0, y0 = 0, 0
    x1, y1 = img_w, img_h
    
    return {
        # Edge masking
        "edge_mask": "auto",
        "edge_max_frac": 0.45,
        "edge_thr_mode": "otsu",
        "edge_rel_offset": 15,
        "edge_smooth_k": 15,
        "edge_min_width_px": 10,
        "edge_fill": 255,
        
        # ROI
        "roi": (x0, y0, x1, y1),
        
        # Preprocessing
        "blur_k": 5,
        "canny_lo": 30,
        "canny_hi": 80,
        
        # Dark region & specular
        "dark_offset": 30,
        "spec_thr": 245,
        "spec_dilate": 3,
        
        # Contour splitting
        "split_min_len": 8,
        "split_max_jump": 15,
        "split_corner_deg": 60,
        
        # Ellipse filtering
        "min_ellipse_area": 100,
        "max_aspect": 2.5,
        "max_center_dist": 0.8,
        
        # Support calculation
        "support_dist_px": 3.0,
        
        # Scoring weights
        "support_weight": 0.5,
        "dist_weight": 0.3,
        "asp_weight": 0.2,
    }


# ==================== DETECTION FUNCTION ====================
def detect_pupil_in_frame(frame_bgr: np.ndarray, cfg: dict) -> Optional[Tuple[float, float, float]]:
    """
    Detect pupil in a single frame.
    
    Returns:
        (cx, cy, diameter) or None if not detected
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # Edge masking
    masked = frame_bgr
    if cfg["edge_mask"] != "off":
        side = cfg["edge_mask"]
        if side == "auto":
            side = pd.pick_side_auto(gray)
        
        width_px, _, _ = pd.find_dark_edge_width(
            gray, side=side,
            max_scan_frac=cfg["edge_max_frac"],
            thr_mode=cfg["edge_thr_mode"],
            rel_offset=cfg["edge_rel_offset"],
            smooth_k=cfg["edge_smooth_k"],
            min_width_px=cfg["edge_min_width_px"],
        )
        masked, _ = pd.apply_edge_mask(masked, side, int(width_px), fill_value=cfg["edge_fill"])
    
    gray_m = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    
    # ROI
    x0, y0, x1, y1 = cfg["roi"]
    roi = gray_m[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    
    # Edge detection
    blur_k = cfg["blur_k"] if cfg["blur_k"] % 2 == 1 else cfg["blur_k"] + 1
    roi_blur = cv2.GaussianBlur(roi, (blur_k, blur_k), 0)
    edges = cv2.Canny(roi_blur, cfg["canny_lo"], cfg["canny_hi"])
    
    # Masks
    dark_mask, _, _ = pd.compute_dark_region_mask(roi, offset=cfg["dark_offset"])
    spec_mask = pd.compute_specular_mask(roi, thr=cfg["spec_thr"], dilate=cfg["spec_dilate"])
    edges_final = cv2.bitwise_and(cv2.bitwise_and(edges, dark_mask), cv2.bitwise_not(spec_mask))
    
    # Get all edge points for support calculation
    ys, xs = np.where(edges_final > 0)
    edge_pts = np.stack([xs, ys], axis=1).astype(np.float32) if len(xs) > 0 else np.zeros((0, 2), np.float32)
    
    # Find contours
    contours, _ = cv2.findContours(edges_final, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    roi_h, roi_w = roi.shape
    roi_cx, roi_cy = roi_w / 2.0, roi_h / 2.0
    roi_rad = 0.5 * min(roi_w, roi_h)
    
    best = None
    for cnt in contours:
        if cnt is None or len(cnt) < 5:
            continue
        
        pts = cnt.reshape(-1, 2).astype(np.float32)
        
        # Split contour
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
            
            (ecx, ecy), (A, B), _ = ell
            dist = float(np.sqrt((ecx - roi_cx)**2 + (ecy - roi_cy)**2) / (roi_rad + 1e-9))
            if dist > cfg["max_center_dist"]:
                continue
            
            # Support: fraction of edge points near ellipse
            support, _ = pd.ellipse_support_score(ell, edge_pts, dist_px=cfg.get("support_dist_px", 3.0))
            
            # Score
            score = (cfg["support_weight"] * support 
                     - cfg["dist_weight"] * dist 
                     - cfg["asp_weight"] * (asp - 1.0))
            
            if best is None or score > best[0]:
                best = (score, support, dist, asp, ell)
    
    if best is None:
        return None
    
    _, _, _, _, ell = best
    (ecx, ecy), (A, B), _ = ell
    
    # Convert to full image coordinates
    cx = ecx + x0
    cy = ecy + y0
    diam = (A + B) / 2.0
    
    return (cx, cy, diam)


# ==================== LOAD LPW DATA ====================
def load_lpw_ground_truth(txt_path: str) -> List[Tuple[float, float]]:
    """Load ground truth pupil positions from LPW .txt file"""
    gt_positions = []
    with open(txt_path, 'r') as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) >= 2:
                try:
                    px, py = float(vals[0]), float(vals[1])
                    gt_positions.append((px, py))
                except ValueError:
                    gt_positions.append((-1, -1))  # Invalid
            else:
                gt_positions.append((-1, -1))
    return gt_positions


def get_lpw_videos(lpw_root: str, subjects: List[str] = None, videos: List[str] = None):
    """Get list of (subject, video_path, txt_path) tuples"""
    video_list = []
    
    subject_dirs = sorted(d for d in os.listdir(lpw_root) 
                          if os.path.isdir(os.path.join(lpw_root, d)))
    
    for subj in subject_dirs:
        if subjects and subj not in subjects:
            continue
            
        subj_dir = os.path.join(lpw_root, subj)
        
        for fname in sorted(os.listdir(subj_dir)):
            if not fname.endswith('.avi'):
                continue
            
            video_name = fname.replace('.avi', '')
            if videos and video_name not in videos:
                continue
            
            video_path = os.path.join(subj_dir, fname)
            txt_path = video_path.replace('.avi', '.txt')
            
            if os.path.exists(txt_path):
                video_list.append((subj, video_path, txt_path))
    
    return video_list


# ==================== EVALUATION FUNCTIONS ====================
def evaluate_video(video_path: str, txt_path: str, 
                   frame_step: int = 1,
                   max_frames: int = None,
                   visualize: bool = False,
                   vis_output_dir: str = None,
                   use_refactored: bool = False,
                   vis_step: int = 8) -> VideoResults:
    """Evaluate detector on a single video
    
    Args:
        use_refactored: If True, use full PupilDetector from pupil_detector_refactored.py
        vis_step: Visualization step (0-8) when use_refactored=True
    """
    
    subject = os.path.basename(os.path.dirname(video_path))
    video_name = os.path.basename(video_path)
    
    results = VideoResults(subject=subject, video=video_name)
    
    # Load ground truth
    gt_positions = load_lpw_ground_truth(txt_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Error: Cannot open {video_path}")
        return results
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Get detector config
    cfg = get_default_detector_config(w, h)
    
    # Setup refactored detector if requested
    detector_refactored = None
    frame_ctx = None
    if use_refactored:
        # Create DetectorConfig for refactored detector
        det_cfg = pd.DetectorConfig(
            video=video_path,
            step=vis_step,
            roi="",  # Will use default from parse_roi
            preview=False,
            out="",
            edge_mask="off",  # LPW doesn't have dark edges like Neon
        )
        detector_refactored = pd.PupilDetector(det_cfg)
        roi_rect = pd.parse_roi("", w, h)
        frame_ctx = pd.FrameContext(frame_idx=0, roi_rect=roi_rect, side_fixed=None)
    
    # Visualization setup
    vis_writer = None
    if visualize and vis_output_dir:
        os.makedirs(vis_output_dir, exist_ok=True)
        vis_path = os.path.join(vis_output_dir, f"{subject}_{video_name.replace('.avi', '')}_eval.avi")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        vis_writer = cv2.VideoWriter(vis_path, fourcc, 30.0, (w, h))
    
    frame_idx = 0
    processed = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx >= len(gt_positions):
            break
        
        if max_frames and processed >= max_frames:
            break
        
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue
        
        gt_x, gt_y = gt_positions[frame_idx]
        
        # Skip invalid GT
        if gt_x < 0 or gt_y < 0:
            frame_idx += 1
            continue
        
        # Detect pupil
        detection = detect_pupil_in_frame(frame, cfg)
        
        result = DetectionResult(
            frame_idx=frame_idx,
            gt_x=gt_x,
            gt_y=gt_y
        )
        
        if detection is not None:
            pred_x, pred_y, pred_diam = detection
            result.pred_x = pred_x
            result.pred_y = pred_y
            result.pred_diam = pred_diam
            result.detected = True
            result.error_px = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
        
        results.results.append(result)
        
        # Visualization
        if vis_writer:
            if use_refactored and detector_refactored is not None:
                # Use refactored detector visualization
                frame_ctx.frame_idx = frame_idx
                prep = detector_refactored.preprocess(frame, frame_ctx)
                vis_frame = detector_refactored.render(frame, prep, frame_ctx)
            else:
                vis_frame = frame.copy()
                
                # Draw prediction (red/blue)
                if detection:
                    color = (0, 255, 0) if result.error_px < 10 else (0, 165, 255) if result.error_px < 20 else (0, 0, 255)
                    cv2.circle(vis_frame, (int(pred_x), int(pred_y)), 5, color, -1)
                    cv2.circle(vis_frame, (int(pred_x), int(pred_y)), int(pred_diam/2), color, 1)
                    cv2.putText(vis_frame, f"Err: {result.error_px:.1f}px", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    cv2.putText(vis_frame, "NOT DETECTED", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Always draw GT (green circle with crosshair)
            cv2.circle(vis_frame, (int(gt_x), int(gt_y)), 8, (0, 255, 0), 2)
            cv2.line(vis_frame, (int(gt_x)-12, int(gt_y)), (int(gt_x)+12, int(gt_y)), (0, 255, 0), 1)
            cv2.line(vis_frame, (int(gt_x), int(gt_y)-12), (int(gt_x), int(gt_y)+12), (0, 255, 0), 1)
            cv2.putText(vis_frame, "GT", (int(gt_x)+10, int(gt_y)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show error if detected
            if detection:
                err_color = (0, 255, 0) if result.error_px < 10 else (0, 165, 255) if result.error_px < 20 else (0, 0, 255)
                cv2.putText(vis_frame, f"Err: {result.error_px:.1f}px", (10, h-30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, err_color, 2)
            
            cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, h-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            vis_writer.write(vis_frame)
        
        frame_idx += 1
        processed += 1
    
    cap.release()
    if vis_writer:
        vis_writer.release()
    
    return results


def compute_metrics(all_results: List[VideoResults], 
                    thresholds: List[float] = [5, 10, 15, 20, 25, 30]) -> Dict:
    """Compute aggregate metrics from all video results"""
    
    all_errors = []
    total_frames = 0
    total_detected = 0
    
    for vr in all_results:
        total_frames += vr.n_frames
        total_detected += vr.n_detected
        all_errors.extend(vr.errors.tolist())
    
    all_errors = np.array(all_errors)
    
    metrics = {
        "n_videos": len(all_results),
        "n_frames": total_frames,
        "n_detected": total_detected,
        "detection_rate": 100.0 * total_detected / total_frames if total_frames > 0 else 0,
        "n_with_error": len(all_errors),
    }
    
    if len(all_errors) > 0:
        metrics.update({
            "error_mean": np.mean(all_errors),
            "error_median": np.median(all_errors),
            "error_std": np.std(all_errors),
            "error_min": np.min(all_errors),
            "error_max": np.max(all_errors),
            "error_q25": np.percentile(all_errors, 25),
            "error_q75": np.percentile(all_errors, 75),
            "error_q90": np.percentile(all_errors, 90),
            "error_q95": np.percentile(all_errors, 95),
        })
        
        # Accuracy at thresholds
        for t in thresholds:
            metrics[f"acc_{t}px"] = 100.0 * np.mean(all_errors <= t)
    
    return metrics, all_errors


def print_metrics(metrics: Dict, title: str = "Pupil Detection Evaluation"):
    """Pretty print metrics"""
    print("\n" + "=" * 65)
    print(f" {title}")
    print("=" * 65)
    
    print(f"\n Dataset Statistics:")
    print(f"   Videos evaluated:  {metrics['n_videos']}")
    print(f"   Total frames:      {metrics['n_frames']}")
    print(f"   Detected frames:   {metrics['n_detected']}")
    print(f"   Detection Rate:    {metrics['detection_rate']:.1f}%")
    
    if metrics.get('n_with_error', 0) > 0:
        print(f"\n── Localization Error (pixels) ──")
        print(f"   Mean:      {metrics['error_mean']:.2f} px")
        print(f"   Median:    {metrics['error_median']:.2f} px")
        print(f"   Std:       {metrics['error_std']:.2f} px")
        print(f"   Min/Max:   {metrics['error_min']:.2f} / {metrics['error_max']:.2f} px")
        print(f"   Q25/Q75:   {metrics['error_q25']:.2f} / {metrics['error_q75']:.2f} px")
        print(f"   Q90/Q95:   {metrics['error_q90']:.2f} / {metrics['error_q95']:.2f} px")
        
        print(f"\n── Accuracy at Thresholds ──")
        thresholds = sorted([k for k in metrics.keys() if k.startswith("acc_")],
                            key=lambda x: float(x.split("_")[1].replace("px", "")))
        for t in thresholds:
            threshold = t.split("_")[1]
            print(f"   Within {threshold:>4}:  {metrics[t]:.1f}%")
    
    print("=" * 65)


def print_per_video_results(all_results: List[VideoResults]):
    """Print per-video breakdown"""
    print("\n── Per-Video Results ──")
    print(f"{'Subject':<10} {'Video':<15} {'Frames':<8} {'Detected':<10} {'Rate':<8} {'Mean Err':<10} {'Med Err':<10}")
    print("-" * 80)
    
    for vr in all_results:
        errors = vr.errors
        mean_err = f"{np.mean(errors):.1f}px" if len(errors) > 0 else "N/A"
        med_err = f"{np.median(errors):.1f}px" if len(errors) > 0 else "N/A"
        
        print(f"{vr.subject:<10} {vr.video:<15} {vr.n_frames:<8} {vr.n_detected:<10} "
              f"{vr.detection_rate:.1f}%   {mean_err:<10} {med_err:<10}")


# ==================== VISUALIZATION ====================
def plot_error_distribution(errors: np.ndarray, save_path: str = None):
    """Plot error distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(errors):.1f}px')
    ax1.axvline(np.median(errors), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(errors):.1f}px')
    ax1.set_xlabel('Localization Error (pixels)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Error Distribution', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # CDF
    ax2 = axes[1]
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    ax2.plot(sorted_errors, cdf, color='steelblue', linewidth=2)
    ax2.set_xlabel('Localization Error (pixels)', fontsize=12)
    ax2.set_ylabel('Cumulative %', fontsize=12)
    ax2.set_title('Cumulative Distribution Function', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Threshold markers
    for threshold in [5, 10, 15, 20]:
        pct = 100 * np.mean(errors <= threshold)
        ax2.axvline(threshold, color='gray', linestyle=':', alpha=0.5)
        ax2.axhline(pct, color='gray', linestyle=':', alpha=0.5)
        ax2.annotate(f'{pct:.0f}%@{threshold}px', xy=(threshold, pct),
                     xytext=(threshold+1, pct+3), fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(description="Evaluate pupil detector on LPW dataset")
    parser.add_argument("--lpw_root", type=str, default="../data/LPW",
                        help="Path to LPW dataset root")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Specific subjects to evaluate (e.g., 1 2 3)")
    parser.add_argument("--videos", nargs="+", default=None,
                        help="Specific videos to evaluate (e.g., 1 4 9)")
    parser.add_argument("--frame_step", type=int, default=1,
                        help="Process every N-th frame")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Max frames per video")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization videos")
    parser.add_argument("--output", type=str, default="pupil_eval_output",
                        help="Output directory")
    parser.add_argument("--use_refactored", action="store_true",
                        help="Use full PupilDetector from pupil_detector_refactored.py for visualization")
    parser.add_argument("--vis_step", type=int, default=8, choices=range(0, 9),
                        help="Visualization step (0-8) when --use_refactored is set. "
                             "0=edge_mask, 1=roi_guess, 2=canny, 3=dark_mask, 4=edges_in_dark, "
                             "5=edges_dark_no_spec, 6=segments, 7=ellipse_candidates, 8=best_ellipse")
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lpw_root = os.path.join(script_dir, args.lpw_root) if not os.path.isabs(args.lpw_root) else args.lpw_root
    
    if not os.path.exists(lpw_root):
        print(f"Error: LPW dataset not found at {lpw_root}")
        return
    
    # Get videos
    video_list = get_lpw_videos(lpw_root, args.subjects, args.videos)
    
    if not video_list:
        print("No videos found to evaluate")
        return
    
    print(f"Found {len(video_list)} videos to evaluate")
    
    # Evaluate each video
    all_results = []
    
    for subj, video_path, txt_path in tqdm(video_list, desc="Evaluating videos"):
        vis_dir = os.path.join(args.output, "visualizations") if args.visualize else None
        
        results = evaluate_video(
            video_path, txt_path,
            frame_step=args.frame_step,
            max_frames=args.max_frames,
            visualize=args.visualize,
            vis_output_dir=vis_dir,
            use_refactored=args.use_refactored,
            vis_step=args.vis_step
        )
        
        all_results.append(results)
    
    # Compute metrics
    metrics, all_errors = compute_metrics(all_results)
    
    # Print results
    print_metrics(metrics)
    print_per_video_results(all_results)
    
    # Save plots
    if len(all_errors) > 0:
        os.makedirs(args.output, exist_ok=True)
        plot_error_distribution(all_errors, os.path.join(args.output, "error_distribution.png"))
    
    # Save metrics to CSV
    csv_path = os.path.join(args.output, "metrics.csv")
    os.makedirs(args.output, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in sorted(metrics.items()):
            writer.writerow([k, v])
    print(f"\nMetrics saved to: {csv_path}")


if __name__ == "__main__":
    main()
