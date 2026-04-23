import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms
import requests
from io import BytesIO


# Section 2: Image Transform

def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

class CaptionDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, max_length=512):
        self.df = pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = get_image_transform()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image (try local first, fall back to URL)
        try:
            image = Image.open(row['image_file']).convert('RGB')
        except:
            print("Loading image from local failed. Trying to download from web url")
            response = requests.get(row['url'], timeout=5)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Preprocess image
        pixel_values = self.transform(image)
        
        # Tokenize caption
        encoding = self.tokenizer(
            row['caption'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'pixel_values': pixel_values,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

