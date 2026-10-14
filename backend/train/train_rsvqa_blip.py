"""
Fine-tune BLIP VQA on RSVQA-LR (Remote Sensing VQA) dataset.

RSVQA-LR: 772 Sentinel-2 images, ~77K QA pairs covering:
  presence, count, comparison, rural/urban classification.

Dataset: https://zenodo.org/record/6344334
Expected layout:
    <data-dir>/Images_LR/<id>.tif
    <data-dir>/RSVQA_LR_dataset_questions.json
    <data-dir>/RSVQA_LR_dataset_answers.json
    <data-dir>/RSVQA_LR_dataset_images_information.json

Usage:
    python train_rsvqa_blip.py --data-dir ./data/RSVQA-LR --epochs 20 --out ../weights/blip_rsvqa.pt

Set in .env:
    VQA_WEIGHTS=weights/blip_rsvqa.pt
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


class RSVQADataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", max_samples: int = None):
        data_dir = Path(data_dir)

        # Use the proper pre-split files from Zenodo
        q_file = data_dir / f"LR_split_{split}_questions.json"
        a_file = data_dir / f"LR_split_{split}_answers.json"

        with open(q_file) as f:
            qs = json.load(f)["questions"]
        with open(a_file) as f:
            ans = json.load(f)["answers"]

        # Build image id → filename from Images_LR folder
        images_dir = data_dir / "Images_LR"
        self.img_map = {}
        if images_dir.exists():
            for img_file in images_dir.iterdir():
                try:
                    img_id = int(img_file.stem)
                    self.img_map[img_id] = img_file.name
                except ValueError:
                    pass

        # Build answer vocabulary from training answers
        all_train_ans_file = data_dir / "LR_split_train_answers.json"
        with open(all_train_ans_file) as f:
            train_ans = json.load(f)["answers"]
        answer_counts = {}
        for a in train_ans:
            if "answer" not in a:
                continue
            answer_counts[a["answer"]] = answer_counts.get(a["answer"], 0) + 1
        top_answers = sorted(answer_counts, key=answer_counts.get, reverse=True)[:1000]
        self.ans2idx = {a: i for i, a in enumerate(top_answers)}
        self.idx2ans = {i: a for a, i in self.ans2idx.items()}
        self.num_classes = len(self.ans2idx)

        # Map question_id → answer (only valid answers)
        ans_map = {a["question_id"]: a["answer"] for a in ans if "answer" in a}

        self.samples = []
        for q in qs:
            answer = ans_map.get(q["id"], "")
            if answer not in self.ans2idx:
                continue
            self.samples.append({
                "img_id":     q["img_id"],
                "question":   q["question"],
                "answer_idx": self.ans2idx[answer],
            })

        if max_samples:
            self.samples = self.samples[:max_samples]

        self.img_dir = data_dir / "Images_LR"
        self.tf = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                  [0.26862954, 0.26130258, 0.27577711]),
        ])

        print(f"RSVQA {split}: {len(self.samples)} QA pairs | {self.num_classes} answer classes")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        fname = self.img_map.get(s["img_id"], f"{s['img_id']:05d}.tif")
        img_path = self.img_dir / fname

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (384, 384), (128, 128, 128))

        img_tensor = self.tf(img)
        return img_tensor, s["question"], s["answer_idx"]


def collate_fn(batch):
    imgs, questions, labels = zip(*batch)
    return torch.stack(imgs), list(questions), torch.tensor(labels)


def build_model(num_classes: int):
    from transformers import BlipForQuestionAnswering, BlipProcessor
    import os
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    MODEL_ID = "Salesforce/blip-vqa-base"
    print(f"Loading {MODEL_ID} …")
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model     = BlipForQuestionAnswering.from_pretrained(MODEL_ID)

    # Add a classification head for closed-set VQA
    hidden_size = model.config.text_config.hidden_size  # 768 for BLIP base
    model.classifier = nn.Sequential(
        nn.Linear(hidden_size, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, num_classes),
    )
    return processor, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",    default="./data/RSVQA-LR")
    ap.add_argument("--epochs",      type=int,   default=20)
    ap.add_argument("--batch-size",  type=int,   default=16)
    ap.add_argument("--lr",          type=float, default=5e-5)
    ap.add_argument("--workers",     type=int,   default=2)
    ap.add_argument("--max-samples", type=int,   default=None)
    ap.add_argument("--out",         default="../weights/blip_rsvqa.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    train_ds = RSVQADataset(args.data_dir, "train", args.max_samples)
    val_ds   = RSVQADataset(args.data_dir, "val",   args.max_samples)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_fn,
                              pin_memory=(device=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, collate_fn=collate_fn,
                              pin_memory=(device=="cuda"))

    processor, model = build_model(train_ds.num_classes)
    model = model.to(device)

    scaler    = torch.amp.GradScaler("cuda") if device == "cuda" else None
    criterion = nn.CrossEntropyLoss()
    # Fine-tune at different LRs: classifier head faster, backbone slower
    optimizer = torch.optim.AdamW([
        {"params": model.vision_model.parameters(),  "lr": args.lr * 0.1},
        {"params": model.text_decoder.parameters(),  "lr": args.lr * 0.5},
        {"params": model.classifier.parameters(),    "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = correct = total = 0
        t0 = time.time()

        for imgs, questions, labels in tqdm(train_loader,
                                            desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            imgs, labels = imgs.to(device), labels.to(device)

            # Prepare inputs for BLIP
            enc = processor(
                text=list(questions),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            )
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast("cuda"):
                    outputs = model.vision_model(pixel_values=imgs)
                    vis_feats = outputs.last_hidden_state[:, 0, :]  # CLS token
                    logits = model.classifier(vis_feats)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs   = model.vision_model(pixel_values=imgs)
                vis_feats = outputs.last_hidden_state[:, 0, :]
                logits    = model.classifier(vis_feats)
                loss      = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds    = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        scheduler.step()
        train_acc = correct / total * 100

        # Validation
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, questions, labels in tqdm(val_loader,
                                                desc=f"Epoch {epoch+1}/{args.epochs} [val]  "):
                imgs, labels = imgs.to(device), labels.to(device)
                if scaler:
                    with torch.amp.autocast("cuda"):
                        outputs   = model.vision_model(pixel_values=imgs)
                        vis_feats = outputs.last_hidden_state[:, 0, :]
                        logits    = model.classifier(vis_feats)
                else:
                    outputs   = model.vision_model(pixel_values=imgs)
                    vis_feats = outputs.last_hidden_state[:, 0, :]
                    logits    = model.classifier(vis_feats)

                preds        = logits.argmax(dim=1)
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
                "ans2idx":     train_ds.ans2idx,
                "num_classes": train_ds.num_classes,
            }, args.out)
            print(f"  ✓ Saved best model (val_acc={val_acc:.2f}%)")

    print(f"\nDone. Best val acc: {best_acc:.2f}% | Saved: {args.out}")


if __name__ == "__main__":
    main()
