import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

# -------------------------------------------------
# Add project root
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.heatmap_net import HeatmapNet
from datasets.lpw_dataset import LPWDataset
from utils.peak import extract_single_peak

# -------------------------------------------------
# Config
# -------------------------------------------------
LPW_ROOT = "data/LPW"
CKPT_PATH = "checkpoints_openeds/openeds_finetuned_lpw_validated.pth"

IMG_SIZE = 256
HEATMAP_SIZE = 64

BATCH_SIZE = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# Utils
# -------------------------------------------------
def heatmap_argmax(hm):
    """hm: (H, W)"""
    idx = np.argmax(hm)
    y, x = np.unravel_index(idx, hm.shape)
    return np.array([x, y], dtype=np.float32)

# -------------------------------------------------
# Load model
# -------------------------------------------------
model = HeatmapNet(num_keypoints=2).to(DEVICE)
state = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

print("✔ Loaded finetuned model", flush=True)

# -------------------------------------------------
# Dataset
# -------------------------------------------------
dataset = LPWDataset(
    LPW_ROOT,
    img_size=IMG_SIZE,
    heatmap_size=HEATMAP_SIZE,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

# -------------------------------------------------
# Evaluation
# -------------------------------------------------
errors = []

with torch.no_grad():
    for imgs, gt_heatmaps in loader:
        imgs = imgs.to(DEVICE)

        preds = model(imgs).cpu().numpy()
        gt = gt_heatmaps.numpy()

        for i in range(len(imgs)):
            pred_hm = preds[i, 0]   # pupil channel
            gt_hm   = gt[i, 0]

            pred_xy, conf = extract_single_peak(pred_hm, sigma=1.0)
            gt_xy, _      = extract_single_peak(gt_hm, sigma=0.0)  # GT без сглаживания

            #! optional: confidence filtering
            if conf < 0.2:
                continue
        

            err = np.linalg.norm(pred_xy - gt_xy)
            errors.append(err)

errors = np.array(errors)

# -------------------------------------------------
# Report
# -------------------------------------------------
print("\n========== LPW Pixel Error ==========", flush=True)
print(f"Samples evaluated : {len(errors)}")
print(f"Mean error (px)   : {errors.mean():.2f}")
print(f"Median error (px) : {np.median(errors):.2f}")
print(f"Std error (px)    : {errors.std():.2f}")
print(f"90% percentile   : {np.percentile(errors, 90):.2f}")
print(f"95% percentile   : {np.percentile(errors, 95):.2f}")
print(f"Max error (px)   : {errors.max():.2f}")
print("=====================================\n", flush=True)
