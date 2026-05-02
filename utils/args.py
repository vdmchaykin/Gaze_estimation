import argparse
import torch

def parse_args():
    ap = argparse.ArgumentParser(
        description="Multi-session eye-video → pupil → gaze mapping pipeline"
    )

    # ========= DATA ROOTS =========
    ap.add_argument(
        "--gendata_root",
        required=True,
        help="Path to GenData root containing session folders"
    )


    ap.add_argument(
        "--eye_video_name",
        default="Neon Sensor Module v1 ps1.mp4",
        help="Filename of eye video inside each eye_videos/<session>/ folder"
    )
    ap.add_argument(
        "--world_video_name",
        default="scene_video.mp4",
        help="Filename of world video inside each GenData session folder"
    )

    # ========= SPLIT (3 train / 1 val) =========
    ap.add_argument(
        "--val_session",
        required=True,
        help="Name of the session folder to use for validation "
             "(e.g. 2024-11-20_15-20-28-4ff00407)"
    )

    # ========= MODEL =========
    ap.add_argument("--checkpoint", required=True, help="HeatmapNet checkpoint (.pth)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # ========= HEATMAP / PUPIL =========
    ap.add_argument("--input_size", type=int, default=256)
    ap.add_argument(
        "--kp_index",
        type=int,
        default=0,
        help="Which heatmap channel corresponds to pupil (0 or 1)"
    )
    ap.add_argument("--sigma", type=float, default=1.2)
    ap.add_argument("--conf_thresh", type=float, default=0.0)

    # ========= EYE VIDEO SAMPLING =========
    ap.add_argument(
        "--frame_step",
        type=int,
        default=5,
        help="Process every N-th eye frame (~200Hz). 5 ≈ 40Hz"
    )
    ap.add_argument(
        "--max_process",
        type=int,
        default=-1,
        help="Process at most this many eye frames per session (-1 = no limit)"
    )

    # ========= TIMESTAMP MATCHING =========
    ap.add_argument(
        "--tolerance_ms",
        type=float,
        default=10.0,
        help="Max allowed timestamp difference when matching gaze to eye frames"
    )

    # ========= GAZE MAPPING =========
    ap.add_argument("--alpha", type=float, default=1e-2)
    ap.add_argument("--poly_degree", type=int, default=2)

    # ========= OUTPUT / VIS =========
    ap.add_argument("--out_dir", default="out_eye_to_gaze")
    ap.add_argument(
        "--vis_n",
        type=int,
        default=60,
        help="Number of eye-frame visualizations per session"
    )
    ap.add_argument(
        "--demo_seconds",
        type=float,
        default=15.0,
        help="Length of demo video (seconds) for the validation session"
    )

    return ap.parse_args()
