# File: src/dataset.py
# This file contains the definition of our custom dataset class.

import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class CancerDataset(Dataset):
    """Custom Dataset for histopathologic cancer detection."""
    
    def __init__(self, df, data_path, transform=None):
        """
        Initializes the dataset.
        Args:
            df (pd.DataFrame): DataFrame with image ids and labels.
            data_path (str): Path to the main data folder.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.data_path = data_path
        self.transform = transform
        
    def __len__(self):
        """Returns the total number of samples."""
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Fetches the sample (image and label) at the given index.
        """
        # 1. Get image id and label from the dataframe
        image_id = self.df.iloc[idx]['id']
        label = self.df.iloc[idx]['label']
        
        # 2. Construct the full image path
        image_path = os.path.join(self.data_path, 'train', f'{image_id}.tif')
        
        # 3. Read the image
        image = Image.open(image_path).convert("RGB") # Ensure image is in RGB format
        
        # 4. Apply transforms if they exist
        if self.transform:
            image = self.transform(image)
            
        return image, label
