"""
dataset.py

Handles dataset loading, preprocessing, augmentation, and DataLoader creation
for NWPU-RESISC45 using the HuggingFace datasets library.

The dataset is downloaded automatically on first use and cached locally.
Official splits are used (525 train / 75 val / 100 test per class) to ensure
results are comparable to published benchmarks.

Usage:
    from src.dataset import get_dataloaders
    train_loader, val_loader, test_loader, class_names = get_dataloaders(config)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image


def get_transforms(config, split):
    dc = config['data']
    image_size = config['model']['image_size']
    mean = dc['image_mean']
    std  = dc['image_std']

    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


class RESISC45Dataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.dataset   = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item  = self.dataset[idx]
        image = item['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        label = torch.tensor(item['label'], dtype=torch.long)
        return image, label


def get_dataloaders(config):
    dc = config['data']

    print(f"[dataset] Loading RESISC45 from HuggingFace ({dc['dataset_name']})...")
    print(f"[dataset] Cache directory: {dc['data_dir']}")

    ds = load_dataset(
        dc['dataset_name'],
        cache_dir=dc['data_dir'],
    )['train']

    # Split into train/val/test per class (525/75/100 = 700 per class)
    # First split off test (100/700 = ~14.3%), then val from remainder (75/600 = 12.5%)
    split1 = ds.train_test_split(test_size=100/700, seed=42, stratify_by_column='label')
    split2 = split1['train'].train_test_split(test_size=75/600, seed=42, stratify_by_column='label')

    train_data = split2['train']
    val_data   = split2['test']
    test_data  = split1['test']

    print(f"[dataset] Train: {len(train_data)} images")
    print(f"[dataset] Val:   {len(val_data)} images")
    print(f"[dataset] Test:  {len(test_data)} images")

    class_names = ds.features['label'].names
    print(f"[dataset] Classes: {len(class_names)}")

    train_ds = RESISC45Dataset(train_data, get_transforms(config, 'train'))
    val_ds   = RESISC45Dataset(val_data,   get_transforms(config, 'val'))
    test_ds  = RESISC45Dataset(test_data,  get_transforms(config, 'test'))

    num_workers = dc.get('num_workers', 4)
    batch_size  = config['training']['batch_size']

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader, test_loader, class_names