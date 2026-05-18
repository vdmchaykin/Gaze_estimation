import argparse
import cv2
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

PUPIL_CSV    = "30_fps_data.csv"
OUTPUT_VIDEO = "gaze_overlay.mp4"
OUTPUT_CSV   = "gaze_predictions.csv"

FEATURES     = ["xL", "yL", "xR", "yR"]
EXTRA        = ["xm", "ym", "dx", "dy"]
IMU_FEATURES = ["imu_qw", "imu_qx", "imu_qy", "imu_qz"]
POLY_DEGREE  = 2


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


def load_imu(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"timestamp_ns": "timestamp"})
    df["timestamp"] = df["timestamp"].astype(np.int64)
    return df[["timestamp", "quat_w", "quat_x", "quat_y", "quat_z"]]


def merge_imu(pupil: pd.DataFrame, imu: pd.DataFrame) -> pd.DataFrame:
    """Interpolate IMU quaternions to pupil timestamps using NLERP."""
    pupil = pupil.copy()
    t_pupil = pupil["timestamp"].to_numpy(np.float64)
    t_imu   = imu["timestamp"].to_numpy(np.float64)
    q = np.column_stack([
        np.interp(t_pupil, t_imu, imu[col].to_numpy(np.float64))
        for col in ["quat_w", "quat_x", "quat_y", "quat_z"]
    ])
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    pupil[["imu_qw", "imu_qx", "imu_qy", "imu_qz"]] = q
    return pupil


def load_calibration(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"timestamp_ns": "timestamp"})
    df["timestamp"] = df["timestamp"].astype(np.int64)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/Vadim")
    args = parser.parse_args()

    data_dir  = args.data_dir
    calib_csv = f"{data_dir}/calibration_points.csv"
    imu_csv   = f"{data_dir}/csv/imu.csv"
    scene_video = f"{data_dir}/Neon Scene Camera v1 ps1.mp4"

    # 1) Load data
    pupil = load_pupil_data(PUPIL_CSV)
    calib = load_calibration(calib_csv)
    imu   = load_imu(imu_csv)
    pupil = merge_imu(pupil, imu)
    print(f"[OK] pupil rows: {len(pupil)}, calibration points: {len(calib)}, IMU rows: {len(imu)}")

    # 2) Attach pupil features to each calibration point by frame index
    pupil_indexed = pupil.reset_index(drop=True)
    pupil_indexed["frame_idx"] = pupil_indexed.index
    train = calib.merge(pupil_indexed, on="frame_idx", how="inner")
    print(f"[OK] training rows after merge: {len(train)}")

    # 3) Train models: baseline (no IMU) vs full (with IMU)
    Y_train = train[["gaze_x", "gaze_y"]].to_numpy(np.float64)

    def make_model():
        return Pipeline([
            ("poly", PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),
            ("ridge", Ridge(alpha=10.0, random_state=0)),
        ])

    model_base = make_model()
    model_base.fit(train[FEATURES + EXTRA].to_numpy(np.float64), Y_train)
    res_base = np.linalg.norm(model_base.predict(train[FEATURES + EXTRA].to_numpy(np.float64)) - Y_train, axis=1)

    model = make_model()
    model.fit(train[FEATURES + EXTRA + IMU_FEATURES].to_numpy(np.float64), Y_train)
    res_imu = np.linalg.norm(model.predict(train[FEATURES + EXTRA + IMU_FEATURES].to_numpy(np.float64)) - Y_train, axis=1)

    print(f"\n{'pt':>4}  {'no IMU':>8}  {'+ IMU':>8}  {'delta':>8}")
    print("-" * 36)
    for pid, rb, ri in zip(train["point_id"].tolist(), res_base.tolist(), res_imu.tolist()):
        marker = " <" if ri < rb else ""
        print(f"  {int(pid):2d}  {rb:8.1f}  {ri:8.1f}  {ri - rb:+8.1f}{marker}")
    print("-" * 36)
    print(f"{'mean':>4}  {res_base.mean():8.1f}  {res_imu.mean():8.1f}  {res_imu.mean() - res_base.mean():+8.1f}")
    print()

    # 4) Predict gaze for all pupil frames
    # Interpolate NaN values (blinks / missed detections) before prediction
    pupil_filled = pupil.copy()
    nan_cols = [c for c in FEATURES + EXTRA if pupil_filled[c].isna().any()]
    if nan_cols:
        pupil_filled[nan_cols] = pupil_filled[nan_cols].interpolate(method="linear").ffill().bfill()
        print(f"[OK] interpolated NaN in: {nan_cols}")

    pred = model.predict(pupil_filled[FEATURES + EXTRA + IMU_FEATURES].to_numpy(np.float64))
    pupil = pupil.reset_index(drop=True)
    pupil["pred_gaze_x"] = pred[:, 0]
    pupil["pred_gaze_y"] = pred[:, 1]

    # 5) Write overlay video
    cap = cv2.VideoCapture(scene_video)
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

        if i < len(pupil):
            row = pupil.iloc[i]
            x, y = int(round(row.pred_gaze_x)), int(round(row.pred_gaze_y))
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x, y), 20, (0, 255, 0), -1)
                cv2.circle(frame, (x, y), 22, (255, 255, 255), 2)

        out.write(frame)
        i += 1

    cap.release()
    out.release()
    print(f"[OK] overlay saved: {OUTPUT_VIDEO}")

    # 6) Save predictions CSV
    pupil[["timestamp", "pred_gaze_x", "pred_gaze_y"]].to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] predictions saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
