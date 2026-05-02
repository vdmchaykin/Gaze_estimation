import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

UNITY_CKPT = "checkpoints/unityeyes_epoch_015_loss_2.90e-06.pth"
OUT_DIR = "checkpoints_openeds"

IMG_SIZE = 256
HEATMAP_SIZE = 64

BATCH_SIZE = 16
EPOCHS_HEAD = 7        # train head only
EPOCHS_ALL = 10         # finetune full model

LR_HEAD = 1e-3
LR_ALL = 1e-4

LOG_EVERY = 50          # batches

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------
# Dataset
# -------------------------------------------------
dataset = OpenEDSDataset(
    DATA_ROOT,
    img_size=IMG_SIZE,
    heatmap_size=HEATMAP_SIZE,
    verbose=True
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

# -------------------------------------------------
# Model
# -------------------------------------------------
model = HeatmapNet(num_keypoints=2).to(DEVICE)

# ---- load pretrained encoder (UnityEyes) ----
state = torch.load(UNITY_CKPT, map_location="cpu")

# remove old head (1 keypoint)
state = {k: v for k, v in state.items() if not k.startswith("head.")}

model.load_state_dict(state, strict=False)
print("✔ Loaded pretrained encoder from UnityEyes", flush=True)

# -------------------------------------------------
# Loss
# -------------------------------------------------
criterion = nn.MSELoss()

# =================================================
# PHASE 1 — TRAIN HEAD ONLY
# =================================================
for p in model.encoder.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(
    model.head.parameters(),
    lr=LR_HEAD
)

print("\n▶ Phase 1: training head only", flush=True)

for epoch in range(EPOCHS_HEAD):
    total_loss = 0.0
    t0 = time.time()

    for batch_idx, (imgs, heatmaps) in enumerate(loader):
        imgs = imgs.to(DEVICE)
        heatmaps = heatmaps.to(DEVICE)

        preds = model(imgs)
        loss = criterion(preds, heatmaps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_so_far = total_loss / (batch_idx + 1)

        if batch_idx % LOG_EVERY == 0:
            elapsed = (time.time() - t0) / 60
            print(
                f"[Head E{epoch+1}/{EPOCHS_HEAD}] "
                f"Batch {batch_idx}/{len(loader)} | "
                f"loss={loss.item():.6f} | "
                f"avg={avg_so_far:.6f} | "
                f"time={elapsed:.1f}min",
                flush=True
            )

    avg_epoch = total_loss / len(loader)
    epoch_time = (time.time() - t0) / 60
    print(
        f"[Head E{epoch+1}] finished | "
        f"avg loss={avg_epoch:.6f} | "
        f"epoch time={epoch_time:.1f}min\n",
        flush=True
    )

# =================================================
# PHASE 2 — FINETUNE FULL MODEL
# =================================================
for p in model.encoder.parameters():
    p.requires_grad = True

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR_ALL
)

print("\n▶ Phase 2: finetuning full model", flush=True)

for epoch in range(EPOCHS_ALL):
    total_loss = 0.0
    t0 = time.time()

    for batch_idx, (imgs, heatmaps) in enumerate(loader):
        imgs = imgs.to(DEVICE)
        heatmaps = heatmaps.to(DEVICE)

        preds = model(imgs)
        loss = criterion(preds, heatmaps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_so_far = total_loss / (batch_idx + 1)

        if batch_idx % LOG_EVERY == 0:
            elapsed = (time.time() - t0) / 60
            print(
                f"[Full E{epoch+1}/{EPOCHS_ALL}] "
                f"Batch {batch_idx}/{len(loader)} | "
                f"loss={loss.item():.6f} | "
                f"avg={avg_so_far:.6f} | "
                f"time={elapsed:.1f}min",
                flush=True
            )

    avg_epoch = total_loss / len(loader)
    epoch_time = (time.time() - t0) / 60
    print(
        f"[Full E{epoch+1}] finished | "
        f"avg loss={avg_epoch:.6f} | "
        f"epoch time={epoch_time:.1f}min\n",
        flush=True
    )

# -------------------------------------------------
# Save final model
# -------------------------------------------------
out_path = os.path.join(OUT_DIR, "openeds_finetuned.pth")
torch.save(model.state_dict(), out_path)
print(f"✔ Saved finetuned model to {out_path}", flush=True)
