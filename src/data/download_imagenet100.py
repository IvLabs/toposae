#!/usr/bin/env python
"""Download and prepare ImageNet-100 dataset from Hugging Face.

Uses the pre-made clane9/imagenet-100 dataset (~14 GB) and converts it
to ImageFolder directory structure for use with torchvision.

Creates:
    <output_dir>/
    ├── train/
    │   ├── n01558993/
    │   ├── n01601694/
    │   └── ...
    └── val/
        ├── n01558993/
        ├── n01601694/
        └── ...

Usage:
    source venv/bin/activate
    export HF_TOKEN=your_token
    python src/data/download_imagenet100.py --output-dir data/imagenet-100

Or with existing ImageNet:
    python src/data/download_imagenet100.py --imagenet-dir /path/to/imagenet --output-dir data/imagenet-100
"""
import argparse
import os
import sys
from pathlib import Path
import torch.utils.data


# Standard ImageNet-100 class IDs
IMAGENET100_CLASSES = [
    'n01558993', 'n01601694', 'n01675722', 'n01749939', 'n01819313',
    'n01820546', 'n01917289', 'n01983481', 'n02085620', 'n02099601',
    'n02106550', 'n02111889', 'n02119789', 'n02120997', 'n02123045',
    'n02123159', 'n02123394', 'n02124075', 'n02128757', 'n02129165',
    'n02130308', 'n02219486', 'n02226429', 'n02229544', 'n02268443',
    'n02277742', 'n02279972', 'n02317335', 'n02342885', 'n02356798',
    'n02364673', 'n02403003', 'n02437312', 'n02445715', 'n02480495',
    'n02480855', 'n02481823', 'n02483362', 'n02484975', 'n02486261',
    'n02486410', 'n02487347', 'n02488291', 'n02488702', 'n02489166',
    'n02490219', 'n02492035', 'n02492660', 'n02493509', 'n02493793',
    'n02494079', 'n02497673', 'n02500267', 'n02607072', 'n02655020',
    'n02666196', 'n02708093', 'n02747177', 'n02788148', 'n02804414',
    'n02843684', 'n02894605', 'n03000247', 'n03100240', 'n03345487',
    'n03417042', 'n03425413', 'n03444034', 'n03445924', 'n03459775',
    'n03529860', 'n03594945', 'n03637318', 'n03657121', 'n03786901',
    'n03937543', 'n03956157', 'n04005630', 'n04081281', 'n04118776',
    'n04201297', 'n04252225', 'n04263257', 'n04285008', 'n04311174',
    'n04332243', 'n04335435', 'n04336792', 'n04346328', 'n04347754',
    'n04371774', 'n04443257', 'n04456115', 'n04501370', 'n04554684',
    'n04579145', 'n04591713', 'n07614500', 'n07693725', 'n07720875',
    'n07745940', 'n07753113', 'n07836838', 'n07871810', 'n09246464',
    'n09472597',
]


def download_subset_hf(output_dir: str, hf_token: str = None):
    """Download ImageNet-100 from clane9/imagenet-100 on Hugging Face.
    
    Downloads ~14 GB and saves in ImageFolder format.
    
    Args:
        output_dir: Where to create ImageNet-100.
        hf_token: Hugging Face token (or set HF_TOKEN env var).
    """
    from datasets import load_dataset
    from huggingface_hub import login
    
    token = hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
        print("Authenticated with Hugging Face")
    
    output_dir = Path(output_dir)
    
    # Load the dataset (downloads ~14 GB)
    print("Loading clane9/imagenet-100 from Hugging Face...")
    print("This will download ~14 GB of images.")
    print()
    
    ds_train = load_dataset("clane9/imagenet-100", split="train")
    ds_val = load_dataset("clane9/imagenet-100", split="validation")
    
    # Get class names from features
    label_names = ds_train.features["label"].names
    print(f"Classes in dataset: {len(label_names)}")
    print(f"Train samples: {len(ds_train)}, Val samples: {len(ds_val)}")
    print()
    
    # Save to ImageFolder format
    for split_name, ds in [("train", ds_train), ("val", ds_val)]:
        print(f"Saving {split_name} split...")
        total = len(ds)
        saved = 0
        class_counts = {}
        
        for i in range(total):
            item = ds[i]
            label = item["label"]
            wnid = label_names[label]
            image = item["image"]
            
            class_dir = output_dir / split_name / wnid
            class_dir.mkdir(parents=True, exist_ok=True)
            
            class_counts[wnid] = class_counts.get(wnid, 0) + 1
            ext = "png" if hasattr(image, 'format') and image.format == 'PNG' else "JPEG"
            img_path = class_dir / f"{wnid}_{class_counts[wnid]:04d}.{ext}"
            image.save(img_path)
            saved += 1
            
            if saved % 1000 == 0:
                print(f"  {saved}/{total} images saved...")
        
        print(f"  {split_name}: {saved} images from {len(class_counts)} classes")
        print()
    
    print(f"ImageNet-100 ready at: {output_dir}")
    # Show disk usage
    total_size = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file())
    print(f"Total size: {total_size / (1024**3):.1f} GB")


def subset_from_full_imagenet(imagenet_dir: str, output_dir: str):
    """Create ImageNet-100 from existing full ImageNet (symlinks, ~0 extra space)."""
    imagenet_dir = Path(imagenet_dir)
    output_dir = Path(output_dir)
    
    if not imagenet_dir.exists():
        raise FileNotFoundError(f"ImageNet not found: {imagenet_dir}")
    
    for split in ['train', 'val']:
        src = imagenet_dir / split
        dst = output_dir / split
        if not src.exists():
            raise FileNotFoundError(f"ImageNet {split} not found: {src}")
        dst.mkdir(parents=True, exist_ok=True)
        for cid in IMAGENET100_CLASSES:
            if (src / cid).exists():
                (dst / cid).symlink_to((src / cid).resolve())
    
    print(f"ImageNet-100 created at: {output_dir} (symlinks)")


def main():
    parser = argparse.ArgumentParser(description='Download/prepare ImageNet-100')
    parser.add_argument('--imagenet-dir', type=str, default=None,
                       help='Path to existing full ImageNet directory')
    parser.add_argument('--output-dir', type=str, default='data/imagenet-100',
                       help='Output directory for ImageNet-100')
    parser.add_argument('--hf-token', type=str, default=None,
                       help='Hugging Face token (or set HF_TOKEN env var)')
    args = parser.parse_args()
    
    if args.imagenet_dir:
        subset_from_full_imagenet(args.imagenet_dir, args.output_dir)
    else:
        download_subset_hf(args.output_dir, args.hf_token)


if __name__ == '__main__':
    main()
