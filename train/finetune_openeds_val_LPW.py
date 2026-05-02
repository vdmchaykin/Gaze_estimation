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
from datasets.lpw_dataset import LPWDataset

# -------------------------------------------------
# Config
# -------------------------------------------------
OPENEDS_ROOT = "data/OpenEDS"
LPW_ROOT     = "data/LPW"

UNITY_CKPT = "checkpoints/unityeyes_epoch_015_loss_2.90e-06.pth"
OUT_DIR = "checkpoints_openeds"

IMG_SIZE = 256
HEATMAP_SIZE = 64

BATCH_SIZE = 16
VAL_BATCH_SIZE = 8

EPOCHS_HEAD = 10
EPOCHS_ALL  = 20

LR_HEAD = 1e-3
LR_ALL  = 1e-4

LOG_EVERY = 50
VAL_LOG_EVERY = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------
# Datasets
# -------------------------------------------------
print("▶ Loading OpenEDS (train)", flush=True)
train_dataset = OpenEDSDataset(
    OPENEDS_ROOT,
    img_size=IMG_SIZE,
    heatmap_size=HEATMAP_SIZE,
    verbose=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

print("\n▶ Loading LPW (validation)", flush=True)
val_dataset = LPWDataset(
    LPW_ROOT,
    img_size=IMG_SIZE,
    heatmap_size=HEATMAP_SIZE,
    max_samples=300   # можно убрать
)

val_loader = DataLoader(
    val_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# -------------------------------------------------
# Model
# -------------------------------------------------
model = HeatmapNet(num_keypoints=2).to(DEVICE)

state = torch.load(UNITY_CKPT, map_location="cpu")
state = {k: v for k, v in state.items() if not k.startswith("head.")}
model.load_state_dict(state, strict=False)

print("✔ Loaded pretrained encoder from UnityEyes\n", flush=True)

# -------------------------------------------------
# Loss
# -------------------------------------------------
criterion = nn.MSELoss()

# -------------------------------------------------
# Validation (LPW – pupil only)
# -------------------------------------------------
def validate_lpw(model, epoch_tag=""):
    model.eval()
    total_loss = 0.0
    t0 = time.time()

    print(f"   ▶ Validation on LPW {epoch_tag}", flush=True)

    with torch.no_grad():
        for batch_idx, (imgs, heatmaps) in enumerate(val_loader):
            imgs = imgs.to(DEVICE)
            heatmaps = heatmaps.to(DEVICE)

            preds = model(imgs)
            loss = criterion(preds[:, 1], heatmaps[:, 1])  # pupil only

            total_loss += loss.item()

            if batch_idx % VAL_LOG_EVERY == 0:
                elapsed = (time.time() - t0) / 60
                print(
                    f"      [VAL] Batch {batch_idx}/{len(val_loader)} | "
                    f"loss={loss.item():.6f} | "
                    f"time={elapsed:.1f}min",
                    flush=True
                )

    avg_loss = total_loss / len(val_loader)
    total_time = (time.time() - t0) / 60

    print(
        f"   ✔ Validation finished | "
        f"avg val loss={avg_loss:.6f} | "
        f"time={total_time:.1f}min\n",
        flush=True
    )

    model.train()
    return avg_loss

# =================================================
# PHASE 1 — TRAIN HEAD ONLY
# =================================================
for p in model.encoder.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(model.head.parameters(), lr=LR_HEAD)

print("▶ Phase 1: training head only\n", flush=True)

for epoch in range(EPOCHS_HEAD):
    total_loss = 0.0
    t0 = time.time()

    for batch_idx, (imgs, heatmaps) in enumerate(train_loader):
        imgs = imgs.to(DEVICE)
        heatmaps = heatmaps.to(DEVICE)

        preds = model(imgs)
        loss = criterion(preds, heatmaps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % LOG_EVERY == 0:
            elapsed = (time.time() - t0) / 60
            print(
                f"[Head E{epoch+1}/{EPOCHS_HEAD}] "
                f"Batch {batch_idx}/{len(train_loader)} | "
                f"loss={loss.item():.6f} | "
                f"time={elapsed:.1f}min",
                flush=True
            )

    train_avg = total_loss / len(train_loader)
    val_avg = validate_lpw(model, epoch_tag=f"(Head E{epoch+1})")
    epoch_time = (time.time() - t0) / 60

    print(
        f"[Head E{epoch+1}] "
        f"train={train_avg:.6f} | "
        f"val(LPW)={val_avg:.6f} | "
        f"time={epoch_time:.1f}min\n",
        flush=True
    )

# =================================================
# PHASE 2 — FINETUNE FULL MODEL
# =================================================
for p in model.encoder.parameters():
    p.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=LR_ALL)

print("▶ Phase 2: finetuning full model\n", flush=True)

for epoch in range(EPOCHS_ALL):
    total_loss = 0.0
    t0 = time.time()

    for batch_idx, (imgs, heatmaps) in enumerate(train_loader):
        imgs = imgs.to(DEVICE)
        heatmaps = heatmaps.to(DEVICE)

        preds = model(imgs)
        loss = criterion(preds, heatmaps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % LOG_EVERY == 0:
            elapsed = (time.time() - t0) / 60
            print(
                f"[Full E{epoch+1}/{EPOCHS_ALL}] "
                f"Batch {batch_idx}/{len(train_loader)} | "
                f"loss={loss.item():.6f} | "
                f"time={elapsed:.1f}min",
                flush=True
            )

    train_avg = total_loss / len(train_loader)
    val_avg = validate_lpw(model, epoch_tag=f"(Full E{epoch+1})")
    epoch_time = (time.time() - t0) / 60

    print(
        f"[Full E{epoch+1}] "
        f"train={train_avg:.6f} | "
        f"val(LPW)={val_avg:.6f} | "
        f"time={epoch_time:.1f}min\n",
        flush=True
    )

# -------------------------------------------------
# Save final model
# -------------------------------------------------
out_path = os.path.join(OUT_DIR, "openeds_finetuned_lpw_validated.pth")
torch.save(model.state_dict(), out_path)
print(f"✔ Saved finetuned model to {out_path}", flush=True)
