
# from utils.video_io import open_video, read_frame, split_eyes
# from utils.heatmap_infer import load_heatmapnet, infer_pupil_from_heatmapnet, draw_cross
# import cv2, os

# SESSION_ID = '2024-11-20_15-16-53-ef3474a6'
# eye_video = f'data/GenData/{SESSION_ID}/Neon Sensor Module v1 ps1.mp4'
# ckpt = 'checkpoints_openeds/openeds_finetuned_lpw_validated.pth'

# os.makedirs('outputs', exist_ok=True)

# cap, fps, nframes, w, h = open_video(eye_video)
# frame = read_frame(cap, 0)
# cap.release()

# left, right = split_eyes(frame)

# model = load_heatmapnet(ckpt, device='cuda')

# predL = infer_pupil_from_heatmapnet(model, left, device='cuda', input_size=256, kp_index=0, sigma=1.2)
# predR = infer_pupil_from_heatmapnet(model, right, device='cuda', input_size=256, kp_index=0, sigma=1.2)

# imgL = draw_cross(left, predL.x, predL.y)
# imgR = draw_cross(right, predR.x, predR.y)
# vis = cv2.hconcat([imgL, imgR])
# cv2.imwrite('outputs/debug_pupil_frame0.png', vis)

# print('Left:', predL)
# print('Right:', predR)
# print('Saved outputs/debug_pupil_frame0.png')


# import os
# from utils.heatmap_infer import load_heatmapnet
# from utils.build_samples import build_samples_for_session, BuildParams

# SESSION_ID = '2024-11-20_15-16-53-ef3474a6'
# SESSION_DIR = f'data/GenData/{SESSION_ID}'
# EYE_VIDEO = f'{SESSION_DIR}/Neon Sensor Module v1 ps1.mp4'

# CKPT = 'checkpoints_openeds/openeds_finetuned_lpw_validated.pth'  # <-- поправь
# OUT = 'outputs/eye_to_gaze'

# os.makedirs(OUT, exist_ok=True)

# model = load_heatmapnet(CKPT, device='cuda')

# p = BuildParams(
#   device='cuda',
#   input_size=256,
#   kp_index=0,
#   sigma=1.2,
#   frame_step=5,
#   conf_thresh=0.0,
#   tolerance_ms=10.0,
#   vis_n=30,
#   max_process=2000
# )

# df = build_samples_for_session(SESSION_ID, SESSION_DIR, EYE_VIDEO, model, OUT, p)
# print(df.head())
# print('Saved:', f'{OUT}/samples_{SESSION_ID}.csv')
