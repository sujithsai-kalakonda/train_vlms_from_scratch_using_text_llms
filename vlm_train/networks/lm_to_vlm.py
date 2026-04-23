import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from vlm_train.networks.vit_encoder import ViTEncoder
from vlm_train.networks.q_former import QFormer

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


## Adapter MLP
class Adapter(nn.Module):
    """
    # What it does:
    - Takes Q-Former output: (batch, 32, 768)
    - Projects to LLM's embedding size: (batch, 32, 576)
    - Why 576? That's SmolLM-135M's hidden size

    - Two linear layers with GELU — gives it some capacity to learn the mapping, not just a single matrix multiply
    - This is the "translator" between visual space and language space.
    """

    def __init__(self, input_dim=768, output_dim=576):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.GELU(), nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class VLM(nn.Module):
    """
    What's happening:

    - Instantiates all 4 components: ViT, Q-Former, LLM, Adapter
    - self.lm.config.hidden_size — reads the LLM's dimension automatically (576 for SmolLM-135M) so we don't hardcode it
    - for param in self.vit.parameters(): param.requires_grad = False — freezes the ViT. We never update its weights. It already knows how to see images, no need to retrain it. Saves memory + compute.
    """

    def __init__(self, lm_name="HuggingFaceTB/SmolLM-135M-Instruct"):
        super().__init__()
        self.vit = ViTEncoder()
        self.q_former = QFormer()

        self.lm = AutoModelForCausalLM.from_pretrained(lm_name)
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)

        lm_hidden_size = self.lm.config.hidden_size
        self.adapter = Adapter(input_dim=768, output_dim=lm_hidden_size)

        for param in self.vit.parameters():
            param.requires_grad = False

    def forward(self, pixel_values, input_ids, attention_mask=None):
        """
        The full pipeline in 6 lines:
            - self.vit(pixel_values) → (batch, 196, 768) patch embeddings
            - self.q_former.encode_image(...) → (batch, 32, 768) compressed queries
            - self.adapter(...) → (batch, 32, 576) projected to LLM space
            - self.lm.model.embed_tokens(input_ids) → converts text token IDs to embeddings (batch, seq, 576)
            - torch.cat([visual_tokens, text_embeddings], dim=1) → (batch, 32+seq, 576) — visual tokens prepended to text
            - Pass the combined embeddings to LLM → get logits/loss
            - The LLM sees: [32 visual tokens][text tokens] — it learns to use the visual prefix when generating.
        """

        patch_embeddings = self.vit(pixel_values)
        query_tokens = self.q_former.encode_image(patch_embeddings)
        visual_tokens = self.adapter(query_tokens)

        text_embeddings = self.lm.model.embed_tokens(input_ids)
        inputs_embeds = torch.cat([visual_tokens, text_embeddings], dim=1).to(
            self.lm.dtype
        )

        outputs = self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        return outputs

    @torch.no_grad()
    def generate(self, pixel_values, prompt="", max_new_tokens=100):
        """
        What's different from forward:
            - @torch.no_grad() — decorator that disables gradient tracking. We're just running inference, not training. Saves memory.
            - self.tokenizer(prompt, return_tensors="pt") — converts text string → token IDs tensor
            - .to(pixel_values.device) — makes sure text tensor is on same device as image (MPS/CPU)
            - self.lm.generate(...) — LLM autoregressively generates tokens one by one until max_new_tokens
            - self.tokenizer.decode(output_ids[0], ...) — converts token IDs back to readable text string

        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(pixel_values.device)

        patch_embeddings = self.vit(pixel_values)
        query_tokens = self.q_former.encode_image(patch_embeddings)
        visual_tokens = self.adapter(query_tokens)

        text_embeddings = self.lm.model.embed_tokens(inputs.input_ids)
        inputs_embeds = torch.cat([visual_tokens, text_embeddings], dim=1).to(
            self.lm.dtype
        )

        output_ids = self.lm.generate(
            inputs_embeds=inputs_embeds, max_new_tokens=max_new_tokens
        )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
