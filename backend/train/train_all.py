"""
Master training runner — trains all SIH26167 models sequentially.
Run this overnight on your RTX 5050. Put laptop on performance mode.

Training order (shortest → longest):
  1. LEVIR-CD change detection (~25 mins)
  2. BigEarthNet land cover (~2-3 hours)
  3. RSVQA BLIP VQA (~3-4 hours)
  4. SEN12MS SAR fusion (~1-2 hours)

Total estimated time: 7-10 hours on RTX 5050 (8GB VRAM)

Usage:
    python train_all.py

After completion, add to backend/.env:
    LANDCOVER_WEIGHTS=weights/landcover_bigearthnet.pt
    CHANGE_HEAD_WEIGHTS=weights/change_levir.pt
    VQA_WEIGHTS=weights/blip_rsvqa.pt
    SAR_FUSION_WEIGHTS=weights/sar_fusion.pt
"""
import subprocess
import sys
import time
from pathlib import Path


STEPS = [
    {
        "name": "LEVIR-CD Change Detection",
        "script": "train_levir_change.py",
        "args": [
            "--data-dir", "./data/LEVIR-CD",
            "--epochs",   "50",
            "--batch-size", "8",
            "--out",      "../weights/change_levir.pt",
        ],
        "requires": "./data/LEVIR-CD/train/A",
        "note": "Download from https://justchenhao.github.io/LEVIR/ first",
    },
    {
        "name": "BigEarthNet Land Cover (ResNet-50)",
        "script": "train_bigearthnet.py",
        "args": [
            "--data-dir",   "./data/bigearthnet",
            "--epochs",     "30",
            "--batch-size", "64",
            "--out",        "../weights/landcover_bigearthnet.pt",
        ],
        "requires": None,  # torchgeo auto-downloads
        "note": "Auto-downloads ~12GB via torchgeo if not present",
    },
    {
        "name": "RSVQA BLIP VQA",
        "script": "train_rsvqa_blip.py",
        "args": [
            "--data-dir",   "./data/RSVQA-LR",
            "--epochs",     "20",
            "--batch-size", "16",
            "--out",        "../weights/blip_rsvqa.pt",
        ],
        "requires": "./data/RSVQA-LR/RSVQA_LR_dataset_questions.json",
        "note": "Download from https://zenodo.org/record/6344334 first",
    },
    {
        "name": "SEN12MS SAR Fusion",
        "script": "train_sar_fusion.py",
        "args": [
            "--data-dir",    "./data/SEN12MS",
            "--epochs",      "30",
            "--batch-size",  "32",
            "--max-samples", "50000",
            "--out",         "../weights/sar_fusion.pt",
        ],
        "requires": "./data/SEN12MS",
        "note": "Download from https://mediatum.ub.tum.de/1474000 first",
    },
]


def check_gpu():
    import torch
    if not torch.cuda.is_available():
        print("⚠ WARNING: CUDA not available. Training will be very slow on CPU.")
        print("Make sure you installed: pip install torch==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128")
        return False
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    return True


def main():
    print("=" * 60)
    print("  SatQuery AI — Full Model Training Pipeline")
    print("  SIH26167 — ISRO Remote Sensing AI")
    print("=" * 60)

    check_gpu()

    # Create weights dir
    Path("../weights").mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    results = []

    for i, step in enumerate(STEPS):
        print(f"\n{'='*60}")
        print(f"  Step {i+1}/{len(STEPS)}: {step['name']}")
        print(f"{'='*60}")

        # Check if data exists
        if step["requires"] and not Path(step["requires"]).exists():
            print(f"⚠ SKIPPING: Data not found at {step['requires']}")
            print(f"  {step['note']}")
            results.append((step["name"], "SKIPPED - data missing"))
            continue

        start = time.time()
        cmd = [sys.executable, step["script"]] + step["args"]
        print(f"Running: {' '.join(cmd)}\n")

        result = subprocess.run(cmd, cwd=Path(__file__).parent)

        elapsed = time.time() - start
        status = "SUCCESS" if result.returncode == 0 else f"FAILED (code {result.returncode})"
        results.append((step["name"], f"{status} in {elapsed/60:.1f} min"))

    total_time = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"  Training Summary ({total_time/3600:.1f} hours total)")
    print(f"{'='*60}")
    for name, status in results:
        icon = "✓" if "SUCCESS" in status else "✗" if "FAILED" in status else "⚠"
        print(f"  {icon} {name}: {status}")

    print(f"\n{'='*60}")
    print("  Next: Update backend/.env with weight paths:")
    print("    LANDCOVER_WEIGHTS=weights/landcover_bigearthnet.pt")
    print("    CHANGE_HEAD_WEIGHTS=weights/change_levir.pt")
    print("    VQA_WEIGHTS=weights/blip_rsvqa.pt")
    print("    SAR_FUSION_WEIGHTS=weights/sar_fusion.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
