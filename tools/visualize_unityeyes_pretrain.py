import os
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -------------------------------------------------
# Add project root to PYTHONPATH
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.heatmap_net import HeatmapNet
from datasets.unityeyes_dataset import UnityEyesDataset

# -------------------------------------------------
# Config
# -------------------------------------------------
DATA_ROOT = "data/UnityEyes/imgs"
CKPT_PATH = "checkpoints/unityeyes_epoch_015_loss_2.90e-06.pth"  # iris-only checkpoint
OUT_DIR = "vis_unityeyes"
NUM_SAMPLES = 8

IMG_SIZE = 256
HEATMAP_SIZE = 64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------
# Helper: upsample heatmap to image size
# -------------------------------------------------
def upsample_heatmap(hm, size):
    """
    hm: torch.Tensor [H, W]
    returns: np.ndarray [size, size]
    """
    hm = hm.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    hm = F.interpolate(
        hm,
        size=(size, size),
        mode="bilinear",
        align_corners=False
    )
    return hm.squeeze().cpu().numpy()

# -------------------------------------------------
# Load model (IRIS ONLY)
# -------------------------------------------------
model = HeatmapNet(num_keypoints=1).to(DEVICE)
state = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# -------------------------------------------------
# Dataset (IRIS ONLY)
# -------------------------------------------------
dataset = UnityEyesDataset(
    DATA_ROOT,
    img_size=IMG_SIZE,
    heatmap_size=HEATMAP_SIZE,
    max_samples=NUM_SAMPLES,
    verbose=False
)

# -------------------------------------------------
# Visualization
# -------------------------------------------------
with torch.no_grad():
    for i in range(len(dataset)):
        # dataset returns:
        # img:        [3, IMG_SIZE, IMG_SIZE]
        # gt_heatmap: [1, HEATMAP_SIZE, HEATMAP_SIZE]
        img, gt_heatmaps = dataset[i]

        img_in = img.unsqueeze(0).to(DEVICE)

        # model output: [1, 1, h, w] -> take [0]
        pred_heatmaps = model(img_in)[0].cpu()
        gt_heatmaps = gt_heatmaps.cpu()

        # ------------------------------------------
        # Debug: compare GT vs Pred numerically
        # ------------------------------------------
        def argmax_xy(hm):
            idx = torch.argmax(hm)
            y = idx // hm.shape[1]
            x = idx % hm.shape[1]
            return x.item(), y.item()

        gt_xy = argmax_xy(gt_heatmaps[0])
        pred_xy = argmax_xy(pred_heatmaps[0])

        print(
            f"[sample {i}] Δ heatmap pixels:",
            abs(gt_xy[0] - pred_xy[0]),
            abs(gt_xy[1] - pred_xy[1]),
            flush=True
        )


        img_np = img.permute(1, 2, 0).numpy()

        # Upsample heatmaps to image resolution
        gt_hm_up = upsample_heatmap(gt_heatmaps[0], IMG_SIZE)
        pred_hm_up = upsample_heatmap(pred_heatmaps[0], IMG_SIZE)

        # -------------------------------------------------
        # Plot: Input | GT iris | Pred iris
        # -------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Input image
        axes[0].imshow(img_np)
        axes[0].set_title("Input image")
        axes[0].axis("off")

        # GT iris heatmap
        axes[1].imshow(img_np)
        axes[1].imshow(gt_hm_up, alpha=0.5, cmap="jet")
        axes[1].set_title("GT iris heatmap")
        axes[1].axis("off")

        # Predicted iris heatmap
        axes[2].imshow(img_np)
        axes[2].imshow(pred_hm_up, alpha=0.5, cmap="jet")
        axes[2].set_title("Pred iris heatmap")
        axes[2].axis("off")

        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"sample_{i}.png")

        

        plt.savefig(out_path)
        plt.close()

        print(f"Saved {out_path}", flush=True)
