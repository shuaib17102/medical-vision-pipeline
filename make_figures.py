"""Generate README figures: training curves + qualitative test results."""
import glob
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch

from model import IMG_SIZE, load_model
from train import ISICDataset

IMAGE_DIR = "data/ISIC2018_Task1-2_Training_Input"
MASK_DIR = "data/ISIC2018_Task1_Training_GroundTruth"
SEED = 42
os.makedirs("docs", exist_ok=True)

# ---------- Figure 1: training curves from MLflow ----------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.MlflowClient()
exp = client.get_experiment_by_name("isic2018-segmentation")
runs = [
    r for r in client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"])
    if r.data.params.get("epochs") == "30" and "test_dice" in r.data.metrics
]
if not runs:
    raise SystemExit("No finished 30-epoch run found in mlflow.db")
run = runs[0]


def hist(name):
    h = sorted(client.get_metric_history(run.info.run_id, name), key=lambda m: m.step)
    return [m.step for m in h], [m.value for m in h]


fig, ax = plt.subplots(1, 2, figsize=(11, 4))
s, v = hist("train_loss")
ax[0].plot(s, v)
ax[0].set(title="Training loss (Dice + CE)", xlabel="Epoch", ylabel="Loss")
s, d = hist("val_dice")
_, i = hist("val_iou")
ax[1].plot(s, d, label="Val Dice")
ax[1].plot(s, i, label="Val IoU")
best = int(np.argmax(d))
ax[1].axvline(s[best], ls="--", c="gray", label=f"Best epoch ({s[best]})")
ax[1].set(title="Validation metrics", xlabel="Epoch", ylim=(0.5, 1.0))
ax[1].legend()
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("docs/training_curves.png", dpi=200)
plt.close(fig)

# ---------- Rebuild the exact test split used in train.py ----------
split_file = "docs/test_split.txt"
if os.path.exists(split_file):
    names = open(split_file).read().split()
    test_p = [os.path.join(IMAGE_DIR, nm) for nm in names]
else:
    paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    random.Random(SEED).shuffle(paths)
    n = len(paths)
    test_p = paths[int(0.8 * n) + int(0.1 * n):]
print(f"Test images: {len(test_p)}")

ds = ISICDataset(test_p, MASK_DIR, IMG_SIZE)
model = load_model("models/best_model.pth")
imgs, gts, preds, dices = [], [], [], []
with torch.no_grad():
    for k in range(len(ds)):
        sample = ds[k]
        p = (torch.sigmoid(model(sample["image"].unsqueeze(0)))[0, 0] > 0.5).numpy()
        g = sample["label"][0].numpy() > 0.5
        dices.append(2.0 * (p & g).sum() / (p.sum() + g.sum() + 1e-8))
        imgs.append(sample["image"].permute(1, 2, 0).numpy())
        gts.append(g)
        preds.append(p)
dices = np.array(dices)
print(f"Mean per-image Dice on test set: {dices.mean():.4f}  (should be close to test_dice in log)")

# ---------- Figure 2: Dice distribution ----------
fig, a = plt.subplots(figsize=(6, 4))
a.hist(dices, bins=25)
a.axvline(dices.mean(), ls="--", c="red", label=f"Mean {dices.mean():.3f}")
a.axvline(np.median(dices), ls=":", c="black", label=f"Median {np.median(dices):.3f}")
a.set(title="Per-image Dice on the test set", xlabel="Dice", ylabel="Images")
a.legend()
fig.tight_layout()
fig.savefig("docs/dice_distribution.png", dpi=200)
plt.close(fig)

# ---------- Figure 3: qualitative results ----------
order = np.argsort(dices)
picks = [
    ("Good", order[int(0.9 * len(order))]),
    ("Median", order[len(order) // 2]),
    ("Hard", order[2]),
    ("Worst", order[0]),
]
cols = ["Image", "Ground truth", "Prediction", "Overlay (green=GT, red=pred)"]
fig, axes = plt.subplots(len(picks), 4, figsize=(12, 3 * len(picks)))
for r, (label, idx) in enumerate(picks):
    axes[r, 0].imshow(imgs[idx])
    axes[r, 1].imshow(gts[idx], cmap="gray")
    axes[r, 2].imshow(preds[idx], cmap="gray")
    axes[r, 3].imshow(imgs[idx])
    axes[r, 3].contour(gts[idx].astype(float), levels=[0.5], colors="lime", linewidths=1.5)
    axes[r, 3].contour(preds[idx].astype(float), levels=[0.5], colors="red", linewidths=1.5)
    axes[r, 0].set_ylabel(f"{label}\nDice {dices[idx]:.2f}", fontsize=11)
    for c in range(4):
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])
        if r == 0:
            axes[r, c].set_title(cols[c], fontsize=11)
fig.tight_layout()
fig.savefig("docs/qualitative_results.png", dpi=200)
plt.close(fig)
print("Saved figures to docs/")