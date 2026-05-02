import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

from utils.heatmap import generate_heatmap
from utils.unityeyes import mean_point


class UnityEyesDataset(Dataset):
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

        if verbose:
            print(f"[Dataset] Scanning directory: {root}")

        all_files = sorted([
            f for f in os.listdir(root)
            if f.endswith(".jpg")
        ])

        if max_samples is not None:
            all_files = all_files[:max_samples]
            if verbose:
                print(f"[Dataset] Limiting dataset to {len(all_files)} samples")

        self.files = all_files

        if verbose:
            print(f"[Dataset] Total samples: {len(self.files)}")

        self.img_tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]

        # -------------------------
        # Load image
        # -------------------------
        img_path = os.path.join(self.root, img_name)
        img_pil = Image.open(img_path).convert("RGB")

        orig_w, orig_h = img_pil.size  # IMPORTANT

        img = self.img_tf(img_pil)    # resized to img_size × img_size

        # -------------------------
        # Load JSON
        # -------------------------
        json_path = os.path.join(
            self.root, img_name.replace(".jpg", ".json")
        )
        with open(json_path) as f:
            data = json.load(f)

        # -------------------------
        # Iris center (ONLY target)
        # -------------------------
        iris_center = mean_point(data["iris_2d"])  # (x, y) in original image coords

        # -------------------------
        # Scale: original → resized
        # -------------------------
        sx = self.img_size / orig_w
        sy = self.img_size / orig_h

        iris_resized = (
            iris_center[0] * sx,
            iris_center[1] * sy
        )

        # -------------------------
        # Scale: resized → heatmap
        # -------------------------
        hm_scale = self.heatmap_size / self.img_size

        iris_hm = generate_heatmap(
            (self.heatmap_size, self.heatmap_size),
            (
                iris_resized[0] * hm_scale,
                iris_resized[1] * hm_scale
            ),
            sigma=self.sigma
        )

        # shape: [1, H, W]
        heatmaps = iris_hm.unsqueeze(0).float()


        return img, heatmaps
