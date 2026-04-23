import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


class ViTEncoder(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224"):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]
        return patch_embeddings
