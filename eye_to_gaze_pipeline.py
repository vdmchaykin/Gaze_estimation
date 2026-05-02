import os
import json

from utils.args import parse_args
from utils.sessions import discover_sessions, split_train_val
from utils.heatmap_infer import load_heatmapnet
from utils.build_samples import build_samples_for_session, concat_samples, BuildParams
from utils.video_io import open_video

from train.train_mapping import train_and_validate, MappingConfig

def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "args_used.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


    paper_w = 1000
    paper_h = 700
    spec_txt = os.path.join(args.out_dir, "paper_spec_used.txt")
    if os.path.exists(spec_txt):
        kv = {}
        for line in open(spec_txt, "r"):
            if "=" in line:
                k,v = line.strip().split("=")
                kv[k]=int(v)
        paper_w = kv.get("paper_w", paper_w)
        paper_h = kv.get("paper_h", paper_h)
    print("Paper canvas:", paper_w, paper_h)


    # ---- discover sessions ----
    sessions = discover_sessions(
        gendata_root=args.gendata_root,
        world_video_name=args.world_video_name,
        eye_video_name=args.eye_video_name,
    )
    train_sessions, val_session = split_train_val(sessions, args.val_session)

    print("Found sessions:", len(sessions))
    print("Train sessions:", [s.session_id for s in train_sessions])
    print("Val session:", val_session.session_id)

    # ---- load HeatmapNet ----
    model = load_heatmapnet(args.checkpoint, args.device)

    # ---- determine eye crop size from val eye video ----
    # Eye video frame: [left | right] => each eye width is half.
    cap, fps, nframes, Wfull, Hfull = open_video(str(val_session.eye_video))
    cap.release()
    eye_w, eye_h = Wfull // 2, Hfull
    print(f"Eye frame: {Wfull}x{Hfull} => per-eye: {eye_w}x{eye_h}, fps={fps}, frames={nframes}")

    # ---- build params ----
    bp = BuildParams(
        device=args.device,
        input_size=args.input_size,
        kp_index=args.kp_index,
        sigma=args.sigma,
        frame_step=args.frame_step,
        conf_thresh=args.conf_thresh,
        tolerance_ms=args.tolerance_ms,
        vis_n=args.vis_n,
        max_process=args.max_process,
    )

    # ---- build samples for train sessions ----
    train_samples = []
    for s in train_sessions:
        df = build_samples_for_session(
            session_id=s.session_id,
            session_dir=str(s.gendata_dir),
            eye_video_path=str(s.eye_video),
            model=model,
            out_dir=args.out_dir,
            p=bp,
        )
        train_samples.append(df)

    train_df = concat_samples(train_samples)
    train_df.to_csv(os.path.join(args.out_dir, "samples_train_all.csv"), index=False)
    print("Train samples total:", len(train_df))

    # ---- build samples for val session ----
    val_df = build_samples_for_session(
        session_id=val_session.session_id,
        session_dir=str(val_session.gendata_dir),
        eye_video_path=str(val_session.eye_video),
        model=model,
        out_dir=args.out_dir,
        p=bp,
    )
    val_df.to_csv(os.path.join(args.out_dir, "samples_val.csv"), index=False)
    print("Val samples total:", len(val_df))

    # ---- train mapping + validate ----
    cfg = MappingConfig(alpha=args.alpha, poly_degree=args.poly_degree)
    metrics = train_and_validate(train_df, val_df, eye_w, eye_h, paper_w, paper_h, cfg, args.out_dir)


    print("VAL metrics:", metrics)
    print("Saved model:", os.path.join(args.out_dir, "gaze_mapping_model.npz"))
    print("Saved val predictions:", os.path.join(args.out_dir, "val_predictions.csv"))
    print("Saved metrics:", os.path.join(args.out_dir, "metrics_val.json"))
    print("Saved visualizations in:", os.path.join(args.out_dir, "vis_eye"))


if __name__ == "__main__":
    main()
