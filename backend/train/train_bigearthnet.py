"""
Fine-tune ResNet-50 on BigEarthNet for remote sensing land cover classification.

BigEarthNet is MANDATORY per SIH26167 PS:
  "At least one visual or vision-language component must be fine-tuned or
   otherwise adapted using BigEarthNet.txt or any open source training data."

Dataset: BigEarthNet-S2 (Sentinel-2, 590,326 patches, 43 classes → mapped to 19)
  - Auto-downloaded from HuggingFace: torchgeo/BigEarthNet
  - OR manually: https://bigearth.net/

Labels (19-class BigEarthNet): Urban fabric, Industrial/commercial,
Arable land, Permanent crops, Pastures, Mixed forests, etc.

Usage:
  python train_bigearthnet.py --epochs 30 --batch-size 64 --out ../weights/landcover_bigearthnet.pt

After training, set in .env:
  LANDCOVER_WEIGHTS=weights/landcover_bigearthnet.pt
"""
import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm


# BigEarthNet 19-class label set (standard multi-hot → we use top-1 for classification)
BIGEARTHNET_19_LABELS = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grasslands and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland/shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]

NUM_CLASSES = len(BIGEARTHNET_19_LABELS)  # 19


def get_dataset(data_dir: str, img_size: int = 224):
    """Load BigEarthNet. Tries torchgeo first, falls back to folder structure."""
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        from torchgeo.datasets import BigEarthNet
        print("Loading BigEarthNet via torchgeo (will download if not cached)...")
        dataset = BigEarthNet(
            root=data_dir,
            split="train",
            bands="s2",  # Sentinel-2 RGB bands
            num_classes=19,
            transforms=tf,
            download=True,
        )
        return dataset
    except Exception as e:
        print(f"torchgeo load failed: {e}")
        print("Falling back to ImageFolder (expects data_dir/<class_name>/*.jpg)")
        from torchvision.datasets import ImageFolder
        return ImageFolder(data_dir, transform=tf)


def build_model(num_classes: int):
    import timm
    model = timm.create_model(
        "resnet50.a1_in1k",
        pretrained=True,
        num_classes=num_classes,
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data/bigearthnet", help="Dataset directory")
    ap.add_argument("--epochs",   type=int,   default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr",       type=float, default=1e-4)
    ap.add_argument("--workers",  type=int,   default=4)
    ap.add_argument("--out",      default="../weights/landcover_bigearthnet.pt")
    ap.add_argument("--img-size", type=int,   default=224)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    # Dataset
    dataset = get_dataset(args.data_dir, args.img_size)
    n_val   = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=(device=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=(device=="cuda"))

    print(f"Train: {n_train} | Val: {n_val} samples")

    # Try to infer actual num_classes from dataset
    try:
        num_classes = len(dataset.classes)
    except AttributeError:
        num_classes = NUM_CLASSES

    model = build_model(num_classes).to(device)

    # Mixed precision for RTX 5050
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            # torchgeo returns dict, ImageFolder returns tuple
            if isinstance(batch, dict):
                imgs   = batch["image"].to(device)
                labels = batch["label"].argmax(dim=1).to(device)  # multi-hot → argmax
            else:
                imgs, labels = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast("cuda"):
                    logits = model(imgs)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss   = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds   = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        scheduler.step()
        train_acc  = correct / total * 100
        epoch_time = time.time() - t0

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        val_correct = 0
        val_total   = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]  "):
                if isinstance(batch, dict):
                    imgs   = batch["image"].to(device)
                    labels = batch["label"].argmax(dim=1).to(device)
                else:
                    imgs, labels = batch[0].to(device), batch[1].to(device)

                if scaler:
                    with torch.amp.autocast("cuda"):
                        preds = model(imgs).argmax(dim=1)
                else:
                    preds = model(imgs).argmax(dim=1)

                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total * 100
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"loss={total_loss/len(train_loader):.4f} | "
              f"train_acc={train_acc:.2f}% | "
              f"val_acc={val_acc:.2f}% | "
              f"time={epoch_time:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":      epoch + 1,
                "state_dict": model.state_dict(),
                "val_acc":    val_acc,
                "labels":     BIGEARTHNET_19_LABELS,
                "num_classes": num_classes,
            }, args.out)
            print(f"  ✓ Saved best model (val_acc={val_acc:.2f}%)")

    print(f"\nTraining complete. Best val acc: {best_val_acc:.2f}%")
    print(f"Model saved to: {args.out}")


if __name__ == "__main__":
    main()
