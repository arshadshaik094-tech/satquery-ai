"""
Fine-tune Siamese MobileNet-V3 change detection on LEVIR-CD dataset.

LEVIR-CD: 637 bitemporal 1024x1024 pairs, 0.5m/px Google Earth imagery.
Labels: binary change mask (buildings changed / not changed).

Dataset download: https://justchenhao.github.io/LEVIR/
Expected layout:
    <data-dir>/train/A/<id>.png   (T1 image)
    <data-dir>/train/B/<id>.png   (T2 image)
    <data-dir>/train/label/<id>.png (binary mask, white=change)
    <data-dir>/val/A/, B/, label/

Usage:
    python train_levir_change.py --data-dir ./data/LEVIR-CD --epochs 50 --out ../weights/change_levir.pt

Set in .env:
    CHANGE_HEAD_WEIGHTS=weights/change_levir.pt
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm


class LevirCDDataset(Dataset):
    def __init__(self, root: str, split: str = "train", size: int = 256):
        self.root = Path(root) / split
        self.ids  = sorted(p.stem for p in (self.root / "A").glob("*.png"))
        self.size = size
        self.img_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        i = self.ids[idx]
        a    = self.img_tf(Image.open(self.root / "A"     / f"{i}.png").convert("RGB"))
        b    = self.img_tf(Image.open(self.root / "B"     / f"{i}.png").convert("RGB"))
        mask = self.mask_tf(Image.open(self.root / "label" / f"{i}.png").convert("L"))
        mask = (mask > 0.5).float()
        return a, b, mask


class SiameseChangeNet(nn.Module):
    """
    Siamese MobileNet-V3-Large feature extractor + lightweight decoder.
    Outputs a per-pixel change probability map.
    """
    def __init__(self):
        super().__init__()
        import timm
        backbone = timm.create_model(
            "mobilenetv3_large_100",
            pretrained=True,
            features_only=True,
        )
        self.backbone = backbone
        feat_ch = backbone.feature_info.channels()[-1]   # 960 for MobileNetV3-Large

        self.decoder = nn.Sequential(
            nn.Conv2d(feat_ch, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, a, b):
        fa = self.backbone(a)[-1]
        fb = self.backbone(b)[-1]
        diff = torch.abs(fa - fb)
        out  = self.decoder(diff)
        return out  # raw logits, (B, 1, H', W')


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        bce  = F.binary_cross_entropy_with_logits(logits, targets)
        prob = torch.sigmoid(logits)
        num  = (2 * prob * targets).sum(dim=(2, 3)) + self.smooth
        den  = prob.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth
        dice = 1 - (num / den).mean()
        return bce + dice


def iou_score(pred_mask, gt_mask, thresh=0.5):
    pred = (pred_mask > thresh).float()
    inter = (pred * gt_mask).sum()
    union = ((pred + gt_mask) > 0).float().sum()
    return (inter / (union + 1e-6)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",   default="./data/LEVIR-CD")
    ap.add_argument("--epochs",     type=int,   default=50)
    ap.add_argument("--batch-size", type=int,   default=8)
    ap.add_argument("--lr",         type=float, default=1e-4)
    ap.add_argument("--workers",    type=int,   default=4)
    ap.add_argument("--size",       type=int,   default=256)
    ap.add_argument("--out",        default="../weights/change_levir.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    train_ds = LevirCDDataset(args.data_dir, "train", args.size)
    val_ds   = LevirCDDataset(args.data_dir, "val",   args.size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=(device=="cuda"))

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model     = SiameseChangeNet().to(device)
    scaler    = torch.amp.GradScaler("cuda") if device == "cuda" else None
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_iou = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for a, b, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            a, b, mask = a.to(device), b.to(device), mask.to(device)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast("cuda"):
                    logits = model(a, b)
                    logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear")
                    loss   = criterion(logits, mask)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(a, b)
                logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear")
                loss   = criterion(logits, mask)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        # Validation IoU
        model.eval()
        ious = []
        with torch.no_grad():
            for a, b, mask in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]  "):
                a, b, mask = a.to(device), b.to(device), mask.to(device)
                if scaler:
                    with torch.amp.autocast("cuda"):
                        logits = model(a, b)
                else:
                    logits = model(a, b)
                logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear")
                prob   = torch.sigmoid(logits)
                ious.append(iou_score(prob, mask))

        val_iou = np.mean(ious)
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"loss={total_loss/len(train_loader):.4f} | "
              f"val_IoU={val_iou:.4f} | "
              f"time={time.time()-t0:.1f}s")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "val_iou": val_iou,
                "arch": "SiameseMobileNetV3",
            }, args.out)
            print(f"  ✓ Saved best model (IoU={val_iou:.4f})")

    print(f"\nDone. Best val IoU: {best_iou:.4f} | Saved: {args.out}")


if __name__ == "__main__":
    main()
