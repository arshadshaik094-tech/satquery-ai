"""
Fine-tune a dual-branch ResNet-18 fusion head on the real SEN12MS dataset
(co-registered Sentinel-1 SAR / Sentinel-2 optical patches with MODIS-
derived land-cover labels — used here as a proxy built-up/water target).

Dataset: https://mediatum.ub.tum.de/1474000 (SEN12MS, ~48GB, requires TUM
account + download agreement).
Expected layout after SEN12MS's own unpack script:
    <data-dir>/ROIs/.../*_s1_*.tif   (SAR, 2-channel VV/VH)
    <data-dir>/ROIs/.../*_s2_*.tif   (optical, 13-band Sentinel-2)
    <data-dir>/ROIs/.../*_lc_*.tif   (IGBP land-cover label raster)

This script binarizes the IGBP labels into {built-up, water, other} as the
supervision target for the fusion head described in the app.

Usage:
    python train_sar_fusion_head.py --data-dir ./data/SEN12MS --epochs 15 --out ./sar_fusion_head.pt

Requires `pip install rasterio` (not in the main API requirements.txt
since it's only needed for training, to keep the inference server lean).
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

try:
    import rasterio
except ImportError as e:  # pragma: no cover
    raise SystemExit("Run: pip install rasterio") from e

# IGBP classes 13 (Urban/Built-up) and 17 (Water) per SEN12MS documentation
BUILTUP_IGBP = {13}
WATER_IGBP = {17}


class Sen12msFusionDataset(Dataset):
    def __init__(self, s1_paths, s2_paths, lc_paths, size=224):
        self.s1_paths, self.s2_paths, self.lc_paths = s1_paths, s2_paths, lc_paths
        self.size = size

    def __len__(self):
        return len(self.s1_paths)

    def _read_resize(self, path, bands):
        with rasterio.open(path) as src:
            arr = src.read(bands).astype(np.float32)
        arr = np.transpose(arr, (1, 2, 0))
        return arr

    def __getitem__(self, idx):
        s1 = self._read_resize(self.s1_paths[idx], [1, 2])          # VV, VH
        s2 = self._read_resize(self.s2_paths[idx], [4, 3, 2])       # RGB bands
        with rasterio.open(self.lc_paths[idx]) as src:
            lc = src.read(1)
        builtup_frac = np.isin(lc, list(BUILTUP_IGBP)).mean()
        water_frac = np.isin(lc, list(WATER_IGBP)).mean()
        target = torch.tensor([builtup_frac, water_frac], dtype=torch.float32)

        s1_t = torch.from_numpy(s1).permute(2, 0, 1)
        s1_t = torch.cat([s1_t, s1_t[:1]], dim=0)  # pad 2ch SAR -> 3ch for ResNet-18
        s2_t = torch.from_numpy(s2).permute(2, 0, 1)

        s1_t = nn.functional.interpolate(s1_t.unsqueeze(0), size=(self.size, self.size)).squeeze(0)
        s2_t = nn.functional.interpolate(s2_t.unsqueeze(0), size=(self.size, self.size)).squeeze(0)
        return s1_t / (s1_t.abs().max() + 1e-6), s2_t / (s2_t.max() + 1e-6), target


class DualBranchFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.sar_branch = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.optical_branch = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.sar_branch.fc = nn.Identity()
        self.optical_branch.fc = nn.Identity()
        self.head = nn.Sequential(nn.Linear(1024, 128), nn.ReLU(), nn.Linear(128, 2), nn.Sigmoid())

    def forward(self, sar, optical):
        f_sar = self.sar_branch(sar)
        f_opt = self.optical_branch(optical)
        return self.head(torch.cat([f_sar, f_opt], dim=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="./sar_fusion_head.pt")
    args = ap.parse_args()

    root = Path(args.data_dir)
    s1_paths = sorted(root.rglob("*_s1_*.tif"))
    s2_paths = sorted(root.rglob("*_s2_*.tif"))
    lc_paths = sorted(root.rglob("*_lc_*.tif"))
    assert s1_paths and len(s1_paths) == len(s2_paths) == len(lc_paths), (
        "SEN12MS triplets not found/aligned under --data-dir — check the SEN12MS unpack layout."
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = Sen12msFusionDataset(s1_paths, s2_paths, lc_paths)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = DualBranchFusion().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for sar, opt_img, target in tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}"):
            sar, opt_img, target = sar.to(device), opt_img.to(device), target.to(device)
            opt.zero_grad()
            pred = model(sar, opt_img)
            loss = criterion(pred, target)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"Saved fine-tuned SAR fusion head to {args.out}")


if __name__ == "__main__":
    main()
