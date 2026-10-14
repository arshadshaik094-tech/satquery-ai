"""
Fine-tune dual ResNet-18 SAR+Optical fusion model on SEN12MS dataset.

SEN12MS: 180,662 patch triplets of Sentinel-1 (SAR), Sentinel-2 (MSI), and MODIS land cover.
We use the RGB subset + SAR VV/VH bands.

Dataset: https://mediatum.ub.tum.de/1474000
Expected layout (after extraction):
    <data-dir>/ROIs<N>_<season>/
        s1_<season>/
            s1_<season>_p<patch>/<id>.tif   (SAR: 2 bands VV, VH)
        s2_<season>/
            s2_<season>_p<patch>/<id>.tif   (Optical: 13 bands, use B4/B3/B2 for RGB)
        lc_<season>/
            lc_<season>_p<patch>/<id>.tif   (MODIS LC labels)

Usage:
    python train_sar_fusion.py --data-dir ./data/SEN12MS --epochs 30 --out ../weights/sar_fusion.pt

Set in .env:
    SAR_FUSION_WEIGHTS=weights/sar_fusion.pt
"""
import argparse
import time
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np


MODIS_LC_CLASSES = {
    1: "Evergreen Needleleaf Forest",
    2: "Evergreen Broadleaf Forest",
    3: "Deciduous Needleleaf Forest",
    4: "Deciduous Broadleaf Forest",
    5: "Mixed Forest",
    6: "Closed Shrublands",
    7: "Open Shrublands",
    8: "Woody Savannas",
    9: "Savannas",
    10: "Grasslands",
    11: "Permanent Wetlands",
    12: "Croplands",
    13: "Urban and Built-up",
    14: "Cropland/Natural Vegetation Mosaic",
    15: "Snow and Ice",
    16: "Barren or Sparsely Vegetated",
    17: "Water",
}
NUM_CLASSES = 17


class SEN12MSDataset(Dataset):
    """Loads SAR + Optical patch pairs from SEN12MS."""

    def __init__(self, data_dir: str, size: int = 256, max_samples: int = None):
        self.data_dir = Path(data_dir)
        self.size     = size
        self.samples  = []

        # Scan for all SAR tif files → infer optical and label path
        for sar_file in sorted(self.data_dir.rglob("**/s1_*/**/*.tif")):
            # Derive optical path by replacing s1 → s2
            opt_file = Path(str(sar_file).replace("/s1_", "/s2_").replace("\\s1_", "\\s2_"))
            lc_file  = Path(str(sar_file).replace("/s1_", "/lc_").replace("\\s1_", "\\lc_"))

            if opt_file.exists() and lc_file.exists():
                self.samples.append((sar_file, opt_file, lc_file))

        if max_samples:
            self.samples = self.samples[:max_samples]

        self.opt_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.sar_tf = transforms.Compose([
            transforms.Resize((size, size)),
        ])

        print(f"SEN12MS: {len(self.samples)} patch pairs found")

    def __len__(self): return len(self.samples)

    def _load_tif(self, path: Path, bands=None) -> torch.Tensor:
        try:
            import rasterio
            with rasterio.open(path) as src:
                data = src.read()  # (C, H, W)
            if bands is not None:
                data = data[bands]
            data = data.astype(np.float32)
            # Normalize to 0-1
            for i in range(data.shape[0]):
                mn, mx = data[i].min(), data[i].max()
                if mx > mn:
                    data[i] = (data[i] - mn) / (mx - mn)
            return torch.from_numpy(data)
        except Exception:
            # Fallback: blank tensor
            c = len(bands) if bands is not None else 2
            return torch.zeros(c, 256, 256)

    def __getitem__(self, idx):
        sar_path, opt_path, lc_path = self.samples[idx]

        # SAR: VV=0, VH=1 → 2 bands
        sar = self._load_tif(sar_path, bands=[0, 1])
        sar = F.interpolate(sar.unsqueeze(0), size=(self.size, self.size),
                            mode="bilinear", align_corners=False).squeeze(0)

        # Optical: B4=3, B3=2, B2=1 (RGB) → 3 bands
        opt = self._load_tif(opt_path, bands=[3, 2, 1])
        opt = F.interpolate(opt.unsqueeze(0), size=(self.size, self.size),
                            mode="bilinear", align_corners=False).squeeze(0)
        opt = self.opt_tf(opt)

        # Label: MODIS LC (1-17) → 0-indexed
        lc_raw = self._load_tif(lc_path, bands=[0])
        label_arr = lc_raw[0]
        label_arr = F.interpolate(label_arr.unsqueeze(0).unsqueeze(0),
                                  size=(self.size, self.size),
                                  mode="nearest").squeeze().long()
        label_arr = (label_arr - 1).clamp(0, NUM_CLASSES - 1)

        # Use majority label for patch-level classification
        label = torch.mode(label_arr.view(-1)).values.long()
        return sar, opt, label


class DualResNet18Fusion(nn.Module):
    """
    Dual-branch ResNet-18: one for SAR, one for Optical.
    Features are concatenated → fusion head → classification.
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        import timm

        # SAR branch (2-channel input)
        sar_backbone = timm.create_model("resnet18", pretrained=True, num_classes=0)
        # Adapt first conv for 2-channel SAR
        orig_conv = sar_backbone.conv1
        sar_backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2,
                                        padding=3, bias=False)
        # Average ImageNet weights across channels as initialization
        with torch.no_grad():
            sar_backbone.conv1.weight[:, :, :, :] = \
                orig_conv.weight[:, :2, :, :].mean(dim=1, keepdim=True).expand_as(
                    sar_backbone.conv1.weight)
        self.sar_branch = sar_backbone

        # Optical branch (3-channel input)
        self.opt_branch = timm.create_model("resnet18", pretrained=True, num_classes=0)

        feat_dim = self.sar_branch.num_features + self.opt_branch.num_features  # 512+512

        self.fusion_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, sar, opt):
        f_sar = self.sar_branch(sar)
        f_opt = self.opt_branch(opt)
        fused = torch.cat([f_sar, f_opt], dim=1)
        return self.fusion_head(fused)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",    default="./data/SEN12MS")
    ap.add_argument("--epochs",      type=int,   default=30)
    ap.add_argument("--batch-size",  type=int,   default=32)
    ap.add_argument("--lr",          type=float, default=1e-4)
    ap.add_argument("--workers",     type=int,   default=4)
    ap.add_argument("--size",        type=int,   default=256)
    ap.add_argument("--max-samples", type=int,   default=50000)
    ap.add_argument("--out",         default="../weights/sar_fusion.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    dataset = SEN12MSDataset(args.data_dir, args.size, args.max_samples)
    n_val   = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=(device=="cuda"))

    model     = DualResNet18Fusion(NUM_CLASSES).to(device)
    scaler    = torch.amp.GradScaler("cuda") if device == "cuda" else None
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = correct = total = 0
        t0 = time.time()

        for sar, opt, labels in tqdm(train_loader,
                                     desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            sar, opt, labels = sar.to(device), opt.to(device), labels.to(device)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast("cuda"):
                    logits = model(sar, opt)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(sar, opt)
                loss   = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds    = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        scheduler.step()
        train_acc = correct / total * 100

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for sar, opt, labels in tqdm(val_loader,
                                         desc=f"Epoch {epoch+1}/{args.epochs} [val]  "):
                sar, opt, labels = sar.to(device), opt.to(device), labels.to(device)
                if scaler:
                    with torch.amp.autocast("cuda"):
                        preds = model(sar, opt).argmax(dim=1)
                else:
                    preds = model(sar, opt).argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total * 100
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"loss={total_loss/len(train_loader):.4f} | "
              f"train_acc={train_acc:.2f}% | "
              f"val_acc={val_acc:.2f}% | "
              f"time={time.time()-t0:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch":       epoch + 1,
                "state_dict":  model.state_dict(),
                "val_acc":     val_acc,
                "num_classes": NUM_CLASSES,
                "classes":     MODIS_LC_CLASSES,
            }, args.out)
            print(f"  ✓ Saved best model (val_acc={val_acc:.2f}%)")

    print(f"\nDone. Best val acc: {best_acc:.2f}% | Saved: {args.out}")


if __name__ == "__main__":
    main()
