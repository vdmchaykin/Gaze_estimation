#!/bin/sh
#SBATCH --partition=A100devel
#SBATCH --time=0:59:59
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -o ./outputs/output_%j.out # STDOUT

# python -u train/pretrain_unityeyes.py
# python -u tools/visualize_unityeyes_pretrain.py
# python -u train/finetune_openeds.py
# python -u tools/visualize_openeds_finetune.py
# python -u train/finetune_openeds_val_LPW.py
# python -u tools/visualize_lpw_predictions.py
# python -u tools/eval_lpw_pixel_error.py

python eye_to_gaze_pipeline.py \
  --gendata_root data/GenData \
  --checkpoint checkpoints_openeds/openeds_finetuned_lpw_validated.pth \
  --val_session 2024-11-20_15-20-28-4ff00407 \
  --world_video_name scene_video.mp4 \
  --eye_video_name "Neon Sensor Module v1 ps1.mp4" \
  --out_dir outputs/eye_to_gaze \
  --kp_index 0 \
  --frame_step 5 \
  --sigma 1.2 \
  --tolerance_ms 10
