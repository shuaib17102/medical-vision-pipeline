"""Train Attention U-Net on ISIC 2018 Task 1 and track everything with MLflow."""
import argparse
import glob
import os
import random

import mlflow
import numpy as np
import torch
from monai.data import DataLoader
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import Compose, RandFlipd, RandRotate90d
from monai.utils import set_determinism
from PIL import Image
from torch.utils.data import Dataset

from model import IMG_SIZE, build_model


class ISICDataset(Dataset):
    """Loads an image + its mask with PIL, resizes, applies MONAI transforms."""

    def __init__(self, image_paths, mask_dir, size, transform=None):
        self.image_paths = image_paths
        self.mask_dir = mask_dir
        self.size = size
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.mask_dir, f"{stem}_segmentation.png")

        img = Image.open(img_path).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((self.size, self.size), Image.NEAREST)

        img = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        mask = torch.from_numpy((np.asarray(mask) > 127).astype(np.float32)).unsqueeze(0)

        sample = {"image": img, "label": mask}
        if self.transform:
            sample = self.transform(sample)
        return sample


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    dice = DiceMetric(include_background=True, reduction="mean")
    iou = MeanIoU(include_background=True, reduction="mean")
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        pred = (torch.sigmoid(model(x)) > 0.5).float()
        dice(y_pred=pred, y=y)
        iou(y_pred=pred, y=y)
    return dice.aggregate().item(), iou.aggregate().item()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", default="data/ISIC2018_Task1-2_Training_Input")
    p.add_argument("--mask_dir", default="data/ISIC2018_Task1_Training_GroundTruth")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img_size", type=int, default=IMG_SIZE)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="use only N images (smoke test)")
    return p.parse_args()


def main():
    args = parse_args()
    set_determinism(seed=args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    paths = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))
    if not paths:
        raise SystemExit(f"No .jpg files found in {args.image_dir}. Check your data folder.")
    random.Random(args.seed).shuffle(paths)
    if args.limit:
        paths = paths[: args.limit]

    n = len(paths)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    train_p = paths[:n_train]
    val_p = paths[n_train : n_train + n_val]
    test_p = paths[n_train + n_val :]
    print(f"Split -> train {len(train_p)}, val {len(val_p)}, test {len(test_p)}")

    # NEW: Save the exact test split for reproducibility
    os.makedirs("docs", exist_ok=True)
    with open("docs/test_split.txt", "w") as f:
        f.write("\n".join(os.path.basename(p) for p in test_p))

    keys = ["image", "label"]
    train_tf = Compose([
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandRotate90d(keys=keys, prob=0.5, max_k=3),
    ])

    def make_loader(p, tf, shuffle):
        ds = ISICDataset(p, args.mask_dir, args.img_size, tf)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers)

    train_loader = make_loader(train_p, train_tf, True)
    val_loader = make_loader(val_p, None, False)
    test_loader = make_loader(test_p, None, False)

    model = build_model().to(device)
    loss_fn = DiceCELoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs("models", exist_ok=True)
    best_path = os.path.join("models", "best_model.pth")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("isic2018-segmentation")

    with mlflow.start_run():
        mlflow.log_params(vars(args))
        mlflow.log_param("n_images", n)
        mlflow.log_param("architecture", "monai.AttentionUnet")
        mlflow.log_param("device", device)

        best_dice = -1.0
        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            for batch in train_loader:
                x = batch["image"].to(device)
                y = batch["label"].to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
                running += loss.item()
            train_loss = running / max(len(train_loader), 1)

            val_dice, val_iou = evaluate(model, val_loader, device)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_dice", val_dice, step=epoch)
            mlflow.log_metric("val_iou", val_iou, step=epoch)
            print(f"Epoch {epoch}/{args.epochs} loss={train_loss:.4f} "
                  f"val_dice={val_dice:.4f} val_iou={val_iou:.4f}")

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(model.state_dict(), best_path)

        # Final held-out test evaluation with the best checkpoint
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        test_dice, test_iou = evaluate(model, test_loader, device)
        mlflow.log_metric("best_val_dice", best_dice)
        mlflow.log_metric("test_dice", test_dice)
        mlflow.log_metric("test_iou", test_iou)
        mlflow.log_artifact(best_path)
        mlflow.log_artifact("docs/test_split.txt")
        print(f"TEST  dice={test_dice:.4f}  iou={test_iou:.4f}")


if __name__ == "__main__":
    main()