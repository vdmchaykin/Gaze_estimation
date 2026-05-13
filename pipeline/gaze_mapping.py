import cv2
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

DATA_DIR = "../data/2026-04-20_13-58-02-dffaeb67"
PUPIL_CSV = "30_fps_data.csv"
CALIB_CSV = f"{DATA_DIR}/calibration_points.csv"
WORLD_TS_CSV = f"{DATA_DIR}/world_timestamps.csv"
SCENE_VIDEO = f"{DATA_DIR}/Neon Scene Camera v1 ps1.mp4"
OUTPUT_VIDEO = "gaze_overlay.mp4"

FEATURES = ["xL", "yL", "xR", "yR"]
EXTRA = ["xm", "ym", "dx", "dy"]
POLY_DEGREE = 2


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["xm"] = (df["xL"] + df["xR"]) / 2.0
    df["ym"] = (df["yL"] + df["yR"]) / 2.0
    df["dx"] = df["xR"] - df["xL"]
    df["dy"] = df["yR"] - df["yL"]
    return df


def load_pupil_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"timestamp [ns]": "timestamp"})
    df["timestamp"] = df["timestamp"].round().astype(np.int64)
    return add_features(df)


def load_calibration(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"timestamp_ns": "timestamp"})
    df["timestamp"] = df["timestamp"].astype(np.int64)
    return df


def load_world_timestamps(path: str) -> pd.DataFrame:
    world = pd.read_csv(path)
    world = world.rename(columns={"timestamp [ns]": "timestamp"})
    world["timestamp"] = world["timestamp"].astype(np.int64)
    world = world.sort_values("timestamp").reset_index(drop=True)
    world["frame_idx_world"] = np.arange(len(world), dtype=np.int64)
    return world[["timestamp", "frame_idx_world"]]


def main():
    # 1) Load data
    pupil = load_pupil_data(PUPIL_CSV)
    calib = load_calibration(CALIB_CSV)
    print(f"[OK] pupil rows: {len(pupil)}, calibration points: {len(calib)}")

    # 2) Attach pupil features to each calibration point
    train = pd.merge_asof(
        calib.sort_values("timestamp"),
        pupil.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=50_000_000,  # 50 ms
    )
    print(f"[OK] training rows after merge: {len(train)}")

    # 3) Train polynomial Ridge model
    X_train = train[FEATURES + EXTRA].to_numpy(np.float64)
    Y_train = train[["gaze_x", "gaze_y"]].to_numpy(np.float64)

    model = Pipeline([
        ("poly", PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),
        ("ridge", Ridge(alpha=10.0, random_state=0)),
    ])
    model.fit(X_train, Y_train)

    residuals = np.linalg.norm(model.predict(X_train) - Y_train, axis=1)
    for pid, res in zip(train["point_id"].tolist(), residuals.tolist()):
        print(f"  calib point {int(pid):2d}: {res:.1f} px")
    print(f"[OK] model trained | mean residual: {residuals.mean():.1f} px")

    # 4) Predict gaze for all pupil frames
    pred = model.predict(pupil[FEATURES + EXTRA].to_numpy(np.float64))
    pupil["pred_gaze_x"] = pred[:, 0]
    pupil["pred_gaze_y"] = pred[:, 1]

    # 5) Match pupil timestamps → scene video frames
    world = load_world_timestamps(WORLD_TS_CSV)
    matched = pd.merge_asof(
        pupil.sort_values("timestamp"),
        world.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=5_000_000,  # 5 ms
    ).dropna(subset=["frame_idx_world"])
    matched["frame_idx_world"] = matched["frame_idx_world"].astype(int)
    print(f"[OK] matched to world frames: {len(matched)}")

    draw = {
        int(r.frame_idx_world): (int(round(r.pred_gaze_x)), int(round(r.pred_gaze_y)))
        for r in matched.itertuples(index=False)
    }

    # 6) Write overlay video
    cap = cv2.VideoCapture(SCENE_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {SCENE_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in draw:
            x, y = draw[i]
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x, y), 20, (0, 0, 255), -1)
                cv2.circle(frame, (x, y), 22, (255, 255, 255), 2)
        out.write(frame)
        i += 1

    cap.release()
    out.release()
    print(f"[OK] overlay saved: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
