"""
Fine-tune a Siamese MobileNet-V3 change-detection head on the real
LEVIR-CD dataset (637 bitemporal 1024x1024 building-change pairs, 0.5m
resolution, Google Earth imagery).

Dataset: https://justchenhao.github.io/LEVIR/  (download A/, B/, label/ folders)
Expected layout:
    <data-dir>/A/<id>.png        (T1 image)
    <data-dir>/B/<id>.png        (T2 image)
    <data-dir>/label/<id>.png    (binary change mask, 0/255)

Usage:
    python train_change_head.py --data-dir ./data/LEVIR-CD/train --epochs 20 --out ./change_head.pt

After training, point CHANGE_HEAD_WEIGHTS in .env at the produced checkpoint.
"""
import argparse
from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


class LevirCDDataset(Dataset):
    def __init__(self, root: str, size: int = 256):
        self.root = Path(root)
        self.ids = sorted(p.stem for p in (self.root / "A").glob("*.png"))
        self.tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        i = self.ids[idx]
        a = self.tf(Image.open(self.root / "A" / f"{i}.png").convert("RGB"))
        b = self.tf(Image.open(self.root / "B" / f"{i}.png").convert("RGB"))
        label = self.mask_tf(Image.open(self.root / "label" / f"{i}.png").convert("L"))
        label = (label > 0.5).float()
        return a, b, label


class SiameseChangeHead(nn.Module):
    """MobileNet-V3 feature extractor shared across both branches; the
    absolute feature difference is decoded to a per-pixel change mask."""

    def __init__(self, backbone_name="mobilenetv3_large_100"):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)
        feat_ch = self.backbone.feature_info.channels()[-1]
        self.decoder = nn.Sequential(
            nn.Conv2d(feat_ch, 128, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=32, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, a, b):
        fa = self.backbone(a)[-1]
        fb = self.backbone(b)[-1]
        diff = torch.abs(fa - fb)
        return self.decoder(diff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="./change_head.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = LevirCDDataset(args.data_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = SiameseChangeHead().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for a, b, label in tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}"):
            a, b, label = a.to(device), b.to(device), label.to(device)
            opt.zero_grad()
            logits = model(a, b)
            logits = nn.functional.interpolate(logits, size=label.shape[-2:], mode="bilinear")
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"Saved fine-tuned change-detection head to {args.out}")


if __name__ == "__main__":
    main()
