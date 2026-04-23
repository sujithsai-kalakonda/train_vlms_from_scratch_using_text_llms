# VLM from Scratch — Using a Text-Only LLM

Building a Vision Language Model (VLM) from scratch by plugging a vision encoder into a small text-only LLM, without any pretrained multimodal weights.

Based on: [Let's train Vision Language Models from scratch using just Text-Only LLMs](https://www.youtube.com/watch?v=Oj27kALfvr0)

---

## Architecture

```
Image → ViT Encoder → Q-Former → Adapter (MLP) → SmolLM-135M → Text
         (frozen)     (32 queries)  (projection)   (pretrained)
```

| Component | Model | Role |
|---|---|---|
| ViT Encoder | `google/vit-base-patch16-224` | Splits image into 196 patch embeddings `(196, 768)` |
| Q-Former | DistilBERT + cross-attention | Compresses 196 patches → 32 query tokens `(32, 768)` |
| Adapter | 2-layer MLP | Projects visual tokens into LLM embedding space `(32, 576)` |
| Language Model | `SmolLM-135M-Instruct` | Generates text from visual + text tokens |

The LLM sees: `[32 visual tokens | text tokens]` and generates a caption.

---

## Project Structure

```
vision/
├── __init__.py
├── pyproject.toml
├── test_pipeline.py                  # End-to-end inference test
├── train_stage1.py                   # Stage 1 training (coming soon)
├── dataset/
│   └── conceptual-captions-200k.parquet
└── vlm_train/
    ├── __init__.py
    ├── networks/
    │   ├── __init__.py
    │   ├── vit_encoder.py            # ViT patch embedding extractor
    │   ├── q_former.py               # Q-Former with cross-attention
    │   └── lm_to_vlm.py             # Adapter MLP + VLM class
    ├── datasets/
    │   └── __init__.py               # CaptionDataset loader
    └── utils/
        └── filter_dataset.py         # Dataset download utility
```

---

## Setup

**1. Create and activate a virtual environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

**2. Install dependencies**
```bash
pip install -e .
```

**3. Download the dataset**
```bash
python vlm_train/utils/filter_dataset.py
```
Saves ~200k image-caption pairs to `dataset/conceptual-captions-200k.parquet`.

---

## Run

**Check the dataset pipeline**
```bash
python test_pipeline.py
```

**Test full inference** — open `test_pipeline.py`, uncomment the inference block at the top, update the image path to point to your own image, then:
```bash
python test_pipeline.py
```

> Note: The model is **not trained yet**. Output will be empty or garbage until Stage 1 and Stage 2 training are done. This just validates the full pipeline runs end-to-end.

---

## What's Next

- **Stage 1** — Train Q-Former with CLIP-style contrastive loss to align image and text embeddings
- **Stage 2** — Fine-tune Adapter + SmolLM with LoRA for image captioning
