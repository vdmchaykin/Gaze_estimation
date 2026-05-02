import cv2
from pathlib import Path


def split_eyes_video(
    input_video,
    left_out="left_eye.avi",
    right_out="right_eye.avi"
):
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    fps = int(round(fps_in)) if fps_in and fps_in > 1 else 30

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mid_x = width // 2

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    out_left = cv2.VideoWriter(left_out, fourcc, fps, (mid_x, height))
    out_right = cv2.VideoWriter(right_out, fourcc, fps, (width - mid_x, height))

    if not out_left.isOpened() or not out_right.isOpened():
        raise RuntimeError("VideoWriter failed to open")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        left_eye  = frame[:, :mid_x]
        right_eye = frame[:, mid_x:]

        out_left.write(left_eye)
        out_right.write(right_eye)

    cap.release()
    out_left.release()
    out_right.release()

    print("Saved:")
    print(" ", left_out)
    print(" ", right_out)


if __name__ == "__main__":
    input_path = "eye_video.mp4"   # <-- путь к твоему видео

    if not Path(input_path).exists():
        raise FileNotFoundError(input_path)

    split_eyes_video(input_path)
