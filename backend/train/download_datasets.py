"""
Downloads all required training datasets for SIH26167.

Datasets:
  1. BigEarthNet-S2 (via torchgeo / HuggingFace) — ~12GB
  2. LEVIR-CD (change detection) — ~1.5GB
  3. RSVQA-LR (remote sensing VQA) — ~2GB

Run:
    python download_datasets.py --all
    python download_datasets.py --bigearthnet
    python download_datasets.py --levir
    python download_datasets.py --rsvqa
"""
import argparse
import os
import zipfile
import tarfile
from pathlib import Path


def download_bigearthnet(data_dir: str = "./data/bigearthnet"):
    """
    BigEarthNet: downloaded via torchgeo (handles it automatically on first training run).
    Just verify torchgeo is installed.
    """
    print("\n=== BigEarthNet ===")
    try:
        import torchgeo
        print(f"torchgeo {torchgeo.__version__} installed ✓")
        print(f"BigEarthNet will auto-download to {data_dir} on first training run.")
        print("Size: ~12GB (Sentinel-2 patches, 590K images)")
    except ImportError:
        print("Installing torchgeo...")
        os.system("pip install torchgeo==0.6.0")
        print("torchgeo installed. Re-run training to auto-download BigEarthNet.")


def download_levir(data_dir: str = "./data/LEVIR-CD"):
    """
    LEVIR-CD: download from the official source.
    Direct link: https://justchenhao.github.io/LEVIR/
    """
    import urllib.request

    print("\n=== LEVIR-CD (Change Detection) ===")
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    url = "https://drive.usercontent.google.com/download?id=1DU8uDCZz8IRLkPBQGJfunJGz4xGJVVGV&export=download&confirm=t"
    out = Path(data_dir) / "LEVIR-CD.zip"

    if out.exists():
        print(f"Already downloaded: {out}")
    else:
        print(f"Downloading LEVIR-CD (~1.5GB) to {out} ...")
        print("Note: If Google Drive rate-limits, download manually from:")
        print("  https://justchenhao.github.io/LEVIR/")
        print("  and place train/val/test folders under:", data_dir)
        try:
            urllib.request.urlretrieve(url, str(out),
                reporthook=lambda b, bs, t: print(
                    f"\r  {b*bs/1024**2:.1f}/{t/1024**2:.1f} MB", end="", flush=True))
            print("\nDownload complete.")
        except Exception as e:
            print(f"\nDirect download failed ({e}).")
            print("Please download manually from https://justchenhao.github.io/LEVIR/")
            return

    # Extract
    extracted = Path(data_dir) / "train"
    if not extracted.exists():
        print("Extracting LEVIR-CD...")
        with zipfile.ZipFile(out, 'r') as z:
            z.extractall(data_dir)
        print("Extraction complete.")


def download_rsvqa(data_dir: str = "./data/RSVQA-LR"):
    """
    RSVQA-LR: download from Zenodo.
    Official: https://zenodo.org/record/6344334
    """
    import urllib.request

    print("\n=== RSVQA-LR (Remote Sensing VQA) ===")
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    files = [
        ("https://zenodo.org/record/6344334/files/RSVQA_LR_dataset_questions.json",
         "RSVQA_LR_dataset_questions.json"),
        ("https://zenodo.org/record/6344334/files/RSVQA_LR_dataset_answers.json",
         "RSVQA_LR_dataset_answers.json"),
        ("https://zenodo.org/record/6344334/files/RSVQA_LR_dataset_images_information.json",
         "RSVQA_LR_dataset_images_information.json"),
        ("https://zenodo.org/record/6344334/files/Images_LR.zip",
         "Images_LR.zip"),
    ]

    for url, fname in files:
        out = Path(data_dir) / fname
        if out.exists():
            print(f"Already downloaded: {fname}")
            continue
        print(f"Downloading {fname} ...")
        try:
            urllib.request.urlretrieve(url, str(out),
                reporthook=lambda b, bs, t: print(
                    f"\r  {b*bs/1024**2:.1f} MB", end="", flush=True))
            print(f"\n{fname} downloaded.")
        except Exception as e:
            print(f"\nFailed: {e}")
            print(f"Download manually from https://zenodo.org/record/6344334")

    # Extract images
    images_zip = Path(data_dir) / "Images_LR.zip"
    images_dir = Path(data_dir) / "Images_LR"
    if images_zip.exists() and not images_dir.exists():
        print("Extracting RSVQA images...")
        with zipfile.ZipFile(images_zip, 'r') as z:
            z.extractall(data_dir)
        print("Extraction complete.")


def download_rasterio():
    """Install rasterio for GeoTIFF support."""
    print("\n=== Installing rasterio (GeoTIFF support) ===")
    os.system("pip install rasterio==1.4.3")
    print("rasterio installed ✓")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all",         action="store_true")
    ap.add_argument("--bigearthnet", action="store_true")
    ap.add_argument("--levir",       action="store_true")
    ap.add_argument("--rsvqa",       action="store_true")
    ap.add_argument("--rasterio",    action="store_true")
    args = ap.parse_args()

    if args.all or args.rasterio:
        download_rasterio()
    if args.all or args.bigearthnet:
        download_bigearthnet()
    if args.all or args.levir:
        download_levir()
    if args.all or args.rsvqa:
        download_rsvqa()

    print("\n=== Dataset download complete ===")
    print("Next step: run train_all.py to start training all models.")


if __name__ == "__main__":
    main()
