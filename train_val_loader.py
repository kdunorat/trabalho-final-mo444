import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import numpy as np


def generate_train_val(output_folder: str):
    torch.cuda.empty_cache()
    # Data augmentation and normalization for training
    data_transforms = {
        'Train': transforms.Compose([
            #transforms.Resize((256, 256)),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Validation': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load datasets
    train_dataset = datasets.ImageFolder(os.path.join(output_folder, 'Train'), data_transforms['Train'])
    validation_dataset = datasets.ImageFolder(os.path.join(output_folder, 'Validation'), data_transforms['Validation'])
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=4, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=24, shuffle=False, num_workers=4, pin_memory=True)


    return train_loader, validation_loader