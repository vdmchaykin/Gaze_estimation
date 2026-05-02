import os
import sys
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Add project root
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.heatmap_net import HeatmapNet
from datasets.openeds_dataset import OpenEDSDataset

# -------------------------------------------------
# Config
# -------------------------------------------------
DATA_ROOT = "data/OpenEDS"
CKPT_PATH = "checkpoints_openeds/openeds_finetuned.pth"
OUT_DIR = "vis_openeds"

IMG_SIZE = 256
HEATMAP_SIZE = 64
NUM_SAMPLES = 10        # <-- HERE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------
# Load model
# -------------------------------------------------
model = HeatmapNet(num_keypoints=2).to(DEVICE)
state = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

print("✔ Loaded finetuned OpenEDS model")

# -------------------------------------------------
# Dataset
# -------------------------------------------------
dataset = OpenEDSDataset(
    DATA_ROOT,
    img_size=IMG_SIZE,
    heatmap_size=HEATMAP_SIZE,
    verbose=False
)

# -------------------------------------------------
# Pick RANDOM indices (IMPORTANT)
# -------------------------------------------------
indices = random.sample(range(len(dataset)), NUM_SAMPLES)

# -------------------------------------------------
# Visualization
# -------------------------------------------------
with torch.no_grad():
    for vis_id, idx in enumerate(indices):
        img, gt_heatmaps = dataset[idx]

        img_in = img.unsqueeze(0).to(DEVICE)
        pred_heatmaps = model(img_in)[0].cpu().numpy()
        gt_heatmaps = gt_heatmaps.numpy()

        img_np = img.permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        # -------------------------
        # Input image
        # -------------------------
        axes[0, 0].imshow(img_np[..., 0], cmap="gray")
        axes[0, 0].set_title("Input image")
        axes[0, 0].axis("off")

        # -------------------------
        # GT iris
        # -------------------------
        axes[0, 1].imshow(img_np[..., 0], cmap="gray")
        axes[0, 1].imshow(gt_heatmaps[0], alpha=0.5, cmap="jet")
        axes[0, 1].set_title("GT iris heatmap")
        axes[0, 1].axis("off")

        # -------------------------
        # GT pupil
        # -------------------------
        axes[0, 2].imshow(img_np[..., 0], cmap="gray")
        axes[0, 2].imshow(gt_heatmaps[1], alpha=0.5, cmap="jet")
        axes[0, 2].set_title("GT pupil heatmap")
        axes[0, 2].axis("off")

        # -------------------------
        # Pred iris
        # -------------------------
        axes[1, 1].imshow(img_np[..., 0], cmap="gray")
        axes[1, 1].imshow(pred_heatmaps[0], alpha=0.5, cmap="jet")
        axes[1, 1].set_title("Pred iris heatmap")
        axes[1, 1].axis("off")

        # -------------------------
        # Pred pupil
        # -------------------------
        axes[1, 2].imshow(img_np[..., 0], cmap="gray")
        axes[1, 2].imshow(pred_heatmaps[1], alpha=0.5, cmap="jet")
        axes[1, 2].set_title("Pred pupil heatmap")
        axes[1, 2].axis("off")

        axes[1, 0].axis("off")

        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"sample_{vis_id}.png")
        plt.savefig(out_path)
        plt.close()

        print(f"Saved {out_path}", flush=True)
