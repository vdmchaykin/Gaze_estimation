import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

from utils.heatmap import generate_heatmap


class LPWDataset(Dataset):
    def __init__(
        self,
        root,
        img_size=256,
        heatmap_size=64,
        sigma=2,
        frame_step=10,
        max_samples=None,
        verbose=True,
    ):
        self.root = root
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.frame_step = frame_step

        self.samples = []

        if verbose:
            print(f"[LPW] Scanning dataset at {root}")

        subject_dirs = sorted(
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        )

        for subj in subject_dirs:
            subj_dir = os.path.join(root, subj)

            for fname in os.listdir(subj_dir):
                if not fname.endswith(".avi"):
                    continue

                video_path = os.path.join(subj_dir, fname)
                txt_path = video_path.replace(".avi", ".txt")

                if not os.path.exists(txt_path):
                    continue

                with open(txt_path) as f:
                    lines = f.readlines()

                for frame_idx, line in enumerate(lines):
                    if frame_idx % self.frame_step != 0:
                        continue

                    vals = line.strip().split()
                    if len(vals) < 2:
                        continue

                    try:
                        px = float(vals[0])
                        py = float(vals[1])
                    except ValueError:
                        continue

                    if px < 0 or py < 0:
                        continue

                    self.samples.append({
                        "video": video_path,
                        "frame_idx": frame_idx,
                        "pupil": (px, py),
                    })
        
        if verbose:
            print(f"[LPW] frame_step = {frame_step}")
            print(f"[LPW] Total samples: {len(self.samples)}")

        self.img_tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        video_path = sample["video"]
        frame_idx = sample["frame_idx"]
        px, py = sample["pupil"]

        # --- read frame ---
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")

        # original frame size
        orig_h, orig_w = frame.shape[:2]

        # --- image preprocessing ---
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(frame_gray)
        img = self.img_tf(img)          # [1, img_size, img_size]
        img = img.repeat(3, 1, 1)       # [3, img_size, img_size]

        # --- CORRECT SCALING ---
        # original → resized image
        px_img = px * self.img_size / orig_w
        py_img = py * self.img_size / orig_h

        # resized image → heatmap
        scale = self.heatmap_size / self.img_size
        px_hm = px_img * scale
        py_hm = py_img * scale

        pupil_hm = generate_heatmap(
            (self.heatmap_size, self.heatmap_size),
            (px_hm, py_hm),
            sigma=self.sigma,
        )

        heatmaps = pupil_hm.unsqueeze(0)  # [1, H, W]

        return img, heatmaps
