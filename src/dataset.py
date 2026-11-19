import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),     
        transforms.RandomVerticalFlip(p=0.5),       
        transforms.RandomRotation(degrees=15),     
        transforms.ColorJitter(
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.2, 
            hue=0.1
        ),                                         
        transforms.ToTensor(), 
        normalize                    
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    return train_transform, val_transform

def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Train or validation directory not found in {data_dir}")

    train_transform, val_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        root=train_dir, 
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=val_dir, 
        transform=val_transform
    )

    class_names = train_dataset.classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,    
        drop_last=True      
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, class_names

if __name__ == "__main__":
    sample_data_dir = "./data"

    try:
        train_loader, val_loader, classes = get_dataloaders(sample_data_dir, batch_size=4)
        images, labels = next(iter(train_loader))
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of labels shape: {labels.shape}")
        print(f"Classes: {classes}")
    except FileNotFoundError as e:
        print(e)