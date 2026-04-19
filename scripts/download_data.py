#!/usr/bin/env python3
"""Download ImageNet-100 from Hugging Face and organize into train/val folders.

Saves to /home/toposae/toposae/data/imagenet-100/train/<class>/ and .../val/<class>/
"""
import os
import sys
from pathlib import Path

DATA_ROOT = Path("/home/toposae/toposae/data/imagenet-100")


def check_already_done():
    train_dir = DATA_ROOT / "train"
    val_dir = DATA_ROOT / "val"
    if train_dir.exists() and val_dir.exists():
        n_train = sum(1 for _ in train_dir.rglob("*.JPEG")) + sum(1 for _ in train_dir.rglob("*.jpg"))
        n_val = sum(1 for _ in val_dir.rglob("*.JPEG")) + sum(1 for _ in val_dir.rglob("*.jpg"))
        if n_train > 100000 and n_val > 4000:
            print(f"Data already present: {n_train} train, {n_val} val images. Skipping download.")
            return True
    return False


def download_and_organize():
    from datasets import load_dataset
    from PIL import Image

    print("Downloading ImageNet-100 from Hugging Face (clane9/imagenet-100)...")
    print("This may take 30-60 minutes depending on connection speed.")

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    for split_name, hf_split in [("train", "train"), ("val", "validation")]:
        split_dir = DATA_ROOT / split_name
        if split_dir.exists():
            count = sum(1 for _ in split_dir.rglob("*") if Path(_).is_file())
            if count > 1000:
                print(f"  {split_name} already has {count} files, skipping.")
                continue

        print(f"\nDownloading {split_name} split...")
        try:
            ds = load_dataset("clane9/imagenet-100", split=hf_split, trust_remote_code=True)
        except Exception as e:
            print(f"  ERROR loading {split_name}: {e}")
            sys.exit(1)

        print(f"  {len(ds)} images. Saving to {split_dir}...")
        saved = 0
        for i, sample in enumerate(ds):
            img = sample["image"]
            label = sample["label"]
            # Get class name from dataset features
            if hasattr(ds.features["label"], "int2str"):
                class_name = ds.features["label"].int2str(label)
            else:
                class_name = f"class_{label:04d}"

            class_dir = split_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            img_path = class_dir / f"{i:08d}.JPEG"
            if not img_path.exists():
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(img_path, "JPEG", quality=95)
            saved += 1

            if (i + 1) % 5000 == 0:
                print(f"  Saved {i+1}/{len(ds)} images...")

        print(f"  Done: {saved} images saved to {split_dir}")


def main():
    if check_already_done():
        return

    download_and_organize()

    # Final check
    train_count = sum(1 for _ in (DATA_ROOT / "train").rglob("*.JPEG"))
    val_count = sum(1 for _ in (DATA_ROOT / "val").rglob("*.JPEG"))
    print(f"\nDownload complete: {train_count} train, {val_count} val images")
    print(f"Data path: {DATA_ROOT}")


if __name__ == "__main__":
    main()
