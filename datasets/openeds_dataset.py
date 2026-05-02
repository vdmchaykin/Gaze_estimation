import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

from utils.heatmap import generate_heatmap


# -------------------------------------------------
# Helper: center of binary mask (CORRECT VERSION)
# -------------------------------------------------
def mask_center(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


# -------------------------------------------------
# Dataset
# -------------------------------------------------
class OpenEDSDataset(Dataset):
    def __init__(
        self,
        root,
        img_size=256,
        heatmap_size=64,
        sigma=2,
        max_samples=None,
        verbose=True
    ):
        self.root = root
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        self.samples = []

        data_root = os.path.join(root, "openEDS", "openEDS")
        subjects = sorted([
            d for d in os.listdir(data_root)
            if d.startswith("S_")
        ])

        if verbose:
            print(f"[OpenEDS] Found subjects: {subjects}")

        for subj in subjects:
            subj_dir = os.path.join(data_root, subj)
            for f in os.listdir(subj_dir):
                if f.endswith(".png"):
                    name = os.path.splitext(f)[0]
                    self.samples.append({
                        "subject": subj,
                        "name": name
                    })

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        if verbose:
            print(f"[OpenEDS] Total samples: {len(self.samples)}")

        self.img_tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),   # -> [1, H, W]
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        subj = sample["subject"]
        name = sample["name"]

        # -------------------------------------------------
        # IMAGE
        # -------------------------------------------------
        img_path = os.path.join(
            self.root, "openEDS", "openEDS", subj, f"{name}.png"
        )
        img = Image.open(img_path).convert("L")
        img = self.img_tf(img)          # [1, H, W]
        img = img.repeat(3, 1, 1)       # [3, H, W]

        # -------------------------------------------------
        # MASK (IMPORTANT: resize!)
        # -------------------------------------------------
        mask_path = os.path.join(
            self.root, "openEDS", "openEDS", subj, f"{name}.npy"
        )
        mask = np.load(mask_path)

        # resize mask to match image
        mask = cv2.resize(
            mask,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST
        )

        # -------------------------------------------------
        # VALIDITY CHECK
        # -------------------------------------------------
        iris_center = mask_center(mask == 2)
        pupil_center = mask_center(mask == 3)

        if iris_center is None or pupil_center is None:
            # skip invalid / closed eye frame
            return self.__getitem__((idx + 1) % len(self))

        # -------------------------------------------------
        # HEATMAPS
        # -------------------------------------------------
        scale = self.heatmap_size / self.img_size

        iris_hm = generate_heatmap(
            (self.heatmap_size, self.heatmap_size),
            (iris_center[0] * scale, iris_center[1] * scale),
            sigma=self.sigma
        )

        pupil_hm = generate_heatmap(
            (self.heatmap_size, self.heatmap_size),
            (pupil_center[0] * scale, pupil_center[1] * scale),
            sigma=self.sigma
        )

        heatmaps = torch.stack([iris_hm, pupil_hm])  # [2, Hm, Wm]

        return img, heatmaps
