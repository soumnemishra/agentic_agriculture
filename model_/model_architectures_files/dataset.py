
# prepare the image for feeding into the model 

#this tells about the data set handling 

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from sklearn.utils.class_weight import compute_class_weight
import config

class AddGaussianNoise(object):
    """Custom transform to simulate drone camera sensor noise with a probability."""
    def __init__(self, std=0.03, p=0.3): # 30% chance to apply noise
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:
            noise = torch.randn(tensor.size()) * self.std
            return torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor

drone_augmentation_v2 = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomApply([v2.RandomRotation(degrees=72)], p=0.5),
    v2.RandomApply([v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2))], p=0.5),
    v2.RandomApply([v2.ColorJitter(brightness=0.1, contrast=0.1)], p=0.3),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    AddGaussianNoise(std=0.03, p=0.3)
])

data_transforms = {
    'train': v2.Compose([
        v2.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        drone_augmentation_v2,
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'valid': v2.Compose([
        v2.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

def get_dataloaders():
    image_datasets = {
        'train': datasets.ImageFolder(config.TRAIN_DIR, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(config.VALID_DIR, transform=data_transforms['valid'])
    }

    dataloaders = {
        # Using num_workers=0 is safer on Windows to avoid multiprocessing crashes
        'train': DataLoader(image_datasets['train'], batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False),
        'valid': DataLoader(image_datasets['valid'], batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    }
    
    return dataloaders, image_datasets

def get_class_weights(train_dataset):
    labels = [label for _, label in train_dataset.samples]
    
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    weights = np.clip(weights, a_min=None, a_max=1.5)
    return torch.FloatTensor(weights).to(config.DEVICE)
