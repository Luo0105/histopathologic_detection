# File: src/data_setup.py
# This file contains a helper function to create train and validation dataloaders.

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader

# Import our custom dataset class from the other file in this folder
from .dataset import CancerDataset

def create_dataloaders(data_path, batch_size, num_workers=0):
    """
    Creates training and validation dataloaders.

    Args:
        data_path (str): Path to the main data folder.
        batch_size (int): The number of samples per batch.
        num_workers (int): The number of subprocesses to use for data loading.

    Returns:
        A tuple of (train_loader, val_loader).
    """
    # 1. Load data labels
    df_labels = pd.read_csv(os.path.join(data_path, 'train_labels.csv'))
    
    # 2. Split data into training and validation sets
    train_df, val_df = train_test_split(
        df_labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_labels['label']
    )
    
    print(f"Data split complete. Training set size: {len(train_df)}, Validation set size: {len(val_df)}")
    
    # 3. Define standard image transforms
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 4. Create Dataset instances
    train_dataset = CancerDataset(df=train_df, data_path=data_path, transform=data_transforms)
    val_dataset = CancerDataset(df=val_df, data_path=data_path, transform=data_transforms)
    
    # 5. Create DataLoader instances
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True # Helps speed up data transfer to GPU
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print("✅ DataLoaders created successfully!")
    return train_loader, val_loader
