"""ImageNet data loading utilities."""
import os
from typing import Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch


def get_imagenet_transforms(image_size=128, is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def get_class_subset(dataset, num_classes):
    indices = []
    for i, (path, target) in enumerate(dataset.samples):
        if target < num_classes:
            indices.append(i)
    return indices


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=1000, num_classes=100, image_size=128):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.data = torch.randn(num_samples, 3, image_size, image_size)
        self.targets = torch.randint(0, num_classes, (num_samples,))
        self.classes = list(range(num_classes))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def get_dataloaders(data_dir, num_classes=100, image_size=128, batch_size=8, num_workers=2, use_synthetic=True):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    try:
        train_dataset = datasets.ImageFolder(train_dir, transform=get_imagenet_transforms(image_size, True))
        val_dataset = datasets.ImageFolder(val_dir, transform=get_imagenet_transforms(image_size, False))
    except FileNotFoundError:
        if use_synthetic:
            print(f"ImageNet not found at {data_dir}, using synthetic data for testing")
            train_dataset = SyntheticDataset(num_samples=1000, num_classes=num_classes, image_size=image_size)
            val_dataset = SyntheticDataset(num_samples=200, num_classes=num_classes, image_size=image_size)
        else:
            raise
    
    if num_classes < len(train_dataset.classes):
        class_indices = get_class_subset(train_dataset, num_classes)
        train_dataset = Subset(train_dataset, class_indices)
        val_dataset = Subset(val_dataset, get_class_subset(val_dataset, num_classes))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader
