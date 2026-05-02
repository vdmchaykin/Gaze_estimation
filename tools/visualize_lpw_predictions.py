import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

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

OUT_DIR = "vis_lpw"
NUM_SAMPLES = 10

IMG_SIZE = 256
HEATMAP_SIZE = 64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)

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
    max_samples=NUM_SAMPLES
)

# -------------------------------------------------
# Visualization
# -------------------------------------------------
with torch.no_grad():
    for i in range(len(dataset)):
        img, gt_heatmaps = dataset[i]

        img_in = img.unsqueeze(0).to(DEVICE)
        pred_heatmaps = model(img_in)[0].cpu().numpy()

        # pupil channel only
        gt = gt_heatmaps[0].numpy()
        pred = pred_heatmaps[0]

        img_np = img.permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Input
        axes[0].imshow(img_np, cmap="gray")
        axes[0].set_title("Input image")
        axes[0].axis("off")

        # GT
        axes[1].imshow(img_np, cmap="gray")
        axes[1].imshow(gt, alpha=0.6, cmap="jet")
        axes[1].set_title("GT pupil heatmap")
        axes[1].axis("off")

        # Prediction
        axes[2].imshow(img_np, cmap="gray")
        axes[2].imshow(pred, alpha=0.6, cmap="jet")
        axes[2].set_title("Predicted pupil heatmap")
        axes[2].axis("off")

        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"sample_{i}.png")
        plt.savefig(out_path)
        plt.close()

        print(f"Saved {out_path}", flush=True)
