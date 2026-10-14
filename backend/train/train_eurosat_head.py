"""
Fine-tune a ResNet-50 (timm) classification head on the real EuroSAT
dataset (RGB version, 10 classes, 27000 images, ~90MB download).

Usage:
    python train_eurosat_head.py --data-dir ./data/EuroSAT --epochs 8 --out ./eurosat_head.pt

Dataset download (run once, requires internet):
    wget https://madm.dfki.de/files/sentinel/EuroSAT.zip -O EuroSAT.zip
    unzip EuroSAT.zip -d ./data/

After training, point EUROSAT_HEAD_WEIGHTS in your .env at the produced
--out checkpoint so app/models/landcover_resnet.py loads the real
fine-tuned head instead of the classical-CV fallback.
"""
import argparse
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Path to EuroSAT ImageFolder root")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="./eurosat_head.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageFolder(args.data_dir, transform=transform)
    assert dataset.classes == sorted(EUROSAT_CLASSES), (
        f"Dataset classes {dataset.classes} don't match expected EuroSAT classes"
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=len(EUROSAT_CLASSES))
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for x, y in tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
        print(f"epoch {epoch+1}: loss={total_loss/n:.4f} acc={correct/n:.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"Saved fine-tuned EuroSAT head to {args.out}")


if __name__ == "__main__":
    main()
