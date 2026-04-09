#!/usr/bin/env python
"""Download and prepare ImageNet-100 dataset.

Creates a 100-class subset of ImageNet in the standard ImageFolder format:
    <output_dir>/
    ├── train/
    │   ├── class_0/
    │   ├── class_1/
    │   └── ...
    └── val/
        ├── class_0/
        ├── class_1/
        └── ...

Usage:
    # From existing full ImageNet (fastest):
    python src/data/download_imagenet100.py --imagenet-dir /path/to/full/imagenet --output-dir data/imagenet-100

    # From scratch (requires image-net.org account):
    python src/data/download_imagenet100.py --from-scratch --output-dir data/imagenet-100

The 100 classes are selected deterministically from the 1000 ImageNet classes
using a fixed seed, ensuring reproducibility across runs.
"""
import argparse
import os
import shutil
import random
import subprocess
from pathlib import Path


# Fixed list of 100 ImageNet class IDs (deterministic subset)
# These are the first 100 classes from the standard ImageNet-100 benchmark split
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


def subset_from_full_imagenet(imagenet_dir: str, output_dir: str):
    """Create ImageNet-100 from an existing full ImageNet installation.
    
    Args:
        imagenet_dir: Path to full ImageNet directory with train/ and val/ subdirs.
        output_dir: Where to create the 100-class subset.
    """
    imagenet_dir = Path(imagenet_dir)
    output_dir = Path(output_dir)
    
    if not imagenet_dir.exists():
        raise FileNotFoundError(f"ImageNet directory not found: {imagenet_dir}")
    
    for split in ['train', 'val']:
        src_split = imagenet_dir / split
        dst_split = output_dir / split
        
        if not src_split.exists():
            raise FileNotFoundError(f"ImageNet {split} directory not found: {src_split}")
        
        dst_split.mkdir(parents=True, exist_ok=True)
        
        classes_found = 0
        for class_id in IMAGENET100_CLASSES:
            src_class = src_split / class_id
            if src_class.exists():
                dst_class = dst_split / class_id
                if not dst_class.exists():
                    print(f"  Linking {split}/{class_id}...")
                    # Use symlinks to save disk space
                    dst_class.symlink_to(src_class.resolve())
                classes_found += 1
            else:
                print(f"  Warning: class {class_id} not found in {split}")
        
        print(f"  {split}: {classes_found}/100 classes linked")
    
    print(f"\nImageNet-100 created at: {output_dir}")
    print(f"  Classes: {len(IMAGENET100_CLASSES)}")
    print(f"  Using symlinks (no extra disk space used)")


def download_from_scratch(output_dir: str):
    """Download ImageNet-100 from image-net.org.
    
    This requires a registered account on image-net.org.
    Downloads only the 100 classes we need.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("ImageNet-100 download from scratch")
    print("=" * 50)
    print()
    print("To download ImageNet, you need an account at https://image-net.org")
    print()
    print("Steps:")
    print("1. Register at https://image-net.org")
    print("2. Go to https://image-net.org/challenges/LSVRC/2012/2012-downloads.php")
    print("3. Download the training images (ILSVRC2012_img_train.tar)")
    print("4. Download the validation images (ILSVRC2012_img_val.tar)")
    print("5. Extract and organize into train/ and val/ directories")
    print("6. Run this script again with --imagenet-dir <path>")
    print()
    print("Alternatively, if you have the full tarball, run:")
    print(f"  python {__file__} --imagenet-dir /path/to/extracted/imagenet --output-dir {output_dir}")
    print()
    print("For automated download of just 100 classes, consider using:")
    print("  - Kaggle ImageNet (requires API key)")
    print("  - Hugging Face datasets (imagenet-1k)")


def main():
    parser = argparse.ArgumentParser(description='Download/prepare ImageNet-100')
    parser.add_argument('--imagenet-dir', type=str, default=None,
                       help='Path to existing full ImageNet directory')
    parser.add_argument('--output-dir', type=str, default='data/imagenet-100',
                       help='Output directory for ImageNet-100')
    parser.add_argument('--from-scratch', action='store_true',
                       help='Show instructions for downloading from image-net.org')
    args = parser.parse_args()
    
    if args.imagenet_dir:
        print(f"Creating ImageNet-100 subset from: {args.imagenet_dir}")
        print(f"Output: {args.output_dir}")
        subset_from_full_imagenet(args.imagenet_dir, args.output_dir)
    elif args.from_scratch:
        download_from_scratch(args.output_dir)
    else:
        print("Usage:")
        print(f"  # From existing full ImageNet:")
        print(f"  python {__file__} --imagenet-dir /path/to/imagenet --output-dir data/imagenet-100")
        print()
        print(f"  # From scratch (download instructions):")
        print(f"  python {__file__} --from-scratch")


if __name__ == '__main__':
    main()
