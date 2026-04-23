

# ========================================================================================================================
#                      dataset checking
# ========================================================================================================================

# import pandas as pd
# df = pd.read_parquet("dataset/conceptual-captions-200k.parquet")
# print(df.columns.tolist())
# print(df.head())

# from transformers import AutoTokenizer
# from vlm_train.datasets import CaptionDataset
# from torch.utils.data import DataLoader

# tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
# dataset = CaptionDataset("dataset/conceptual-captions-200k.parquet", tokenizer)

# print(f"Dataset size: {len(dataset)}")

# # Test loading one sample
# sample = dataset[0]
# print(f"SAMPLE DATA: {sample}\n\n")
# print(f"Sample Keys: {sample.keys()}")
# print(f"pixel_values shape: {sample['pixel_values'].shape}")
# print(f"input_ids shape: {sample['input_ids'].shape}")
# print(f"Caption: {tokenizer.decode(sample['input_ids'])}")



# ========================================================================================================================
#                      Testing METHOD-1: Inference
# ========================================================================================================================

import torch
from PIL import Image
from torchvision import transforms
from vlm_train.networks.lm_to_vlm import VLM

# Load and preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

image_path = "<PUT_YOUR_IMAGE_PATH>" ## TODO: Put yout image path

image = Image.open(image_path).convert("RGB")
pixel_values = transform(image).unsqueeze(0) # add batch dim -> (1, 3, 224, 224)

model = VLM()
print("Model Loaded")

output = model.generate(pixel_values, prompt="What is in this image?")
print("Output:", output)



