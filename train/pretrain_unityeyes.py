print(">>> pretrain_unityeyes.py started")

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

print(">>> Project root added to sys.path:", PROJECT_ROOT, flush=True)


import torch
from torch.utils.data import DataLoader
import time

from models.heatmap_net import HeatmapNet
from datasets.unityeyes_dataset import UnityEyesDataset

# -------------------------
# Config
# -------------------------
DATA_ROOT = "data/UnityEyes/imgs"
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4
NUM_KEYPOINTS = 1
NUM_WORKERS = 4

LOG_EVERY = 50   # batches

SAVE_EVERY = 5
CKPT_DIR = "checkpoints"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Dataset / Loader
# -------------------------

print(">>> Initializing UnityEyes dataset")

dataset = UnityEyesDataset(
    DATA_ROOT,
    verbose=True
)

print(">>> Dataset initialized")

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda")
)
print(">>> Creating DataLoader")

print(f"Dataset size: {len(dataset)} samples")
print(f"Using device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print("-" * 50)

# -------------------------
# Model / Optim
# -------------------------
model = HeatmapNet(num_keypoints=NUM_KEYPOINTS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.MSELoss()

# -------------------------
# Training loop
# -------------------------
global_step = 0
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    epoch_start = time.time()

    for batch_idx, (imgs, heatmaps) in enumerate(loader):
        imgs = imgs.to(DEVICE)
        heatmaps = heatmaps.to(DEVICE)

        preds = model(imgs)
        loss = criterion(preds, heatmaps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        epoch_loss += loss_val
        global_step += 1

        if batch_idx % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            print(
                f"[Epoch {epoch+1}/{EPOCHS}] "
                f"[Batch {batch_idx}/{len(loader)}] "
                f"loss={loss_val:.6f} "
                f"avg_epoch_loss={epoch_loss / (batch_idx+1):.6f} "
                f"time={elapsed/60:.1f}min"
            )

    epoch_time = time.time() - epoch_start
    print(
        f"Epoch {epoch+1} finished | "
        f"avg loss = {epoch_loss / len(loader):.6f} | "
        f"epoch time = {epoch_time/60:.1f} min"
    )
    # --- periodic checkpointing ---
    if (epoch + 1) % SAVE_EVERY == 0:
        ckpt_name = (
            f"unityeyes_epoch_{epoch+1:03d}_"
            f"loss_{(epoch_loss / len(loader)):.2e}.pth"
        )
        ckpt_path = os.path.join(CKPT_DIR, ckpt_name)

        torch.save(model.state_dict(), ckpt_path)

        print(
            f"✔ Checkpoint saved: {ckpt_path}",
            flush=True
        )
    print("-" * 50)

# -------------------------
# Save weights
# -------------------------
torch.save(model.state_dict(), "unityeyes_final.pth")
print("✔ Pretraining finished")
print(f"Total time: {(time.time() - start_time)/60:.1f} min")
