# vit-capsnet-resisc45

Vision Transformer with a true dynamic-routing Capsule Network head, trained from scratch on NWPU-RESISC45 (31,500 satellite images, 45 scene classes).

This project implements and compares three model architectures on the same dataset and training setup, changing only the classification head:

| Stage | Architecture | Status |
|---|---|---|
| 1 | ViT + MLP Classifier | 🔲 Pending |
| 2 | ViT + Capsule Network (dynamic routing) | 🔲 Pending |
| 3 | Dual-scale ViT + Capsule Network | 🔲 Pending |

---

## Why Capsule Networks for Satellite Imagery

Standard classifiers (MLP, softmax) know *what* features are present but lose spatial relationships between them. A parking lot is a large rectangle covered in small rectangles — the arrangement defines the class, not just the components.

Capsule Networks (Sabour et al. 2017) represent entities as vectors where direction encodes spatial properties and length encodes existence confidence. Dynamic routing by agreement routes lower-level capsules to higher-level ones based on whether their predictions agree — naturally suited to scenes where spatial arrangement is the discriminating signal.

---

## Dataset

**NWPU-RESISC45** — 31,500 images, 45 remote sensing scene classes, 700 images per class, 256×256 RGB. Sourced from Google Earth across 100+ countries.

Official splits: 525 train / 75 val / 100 test per class.

Loaded automatically via HuggingFace datasets (`jonathan-roberts1/NWPU-RESISC45`).

---

## Architecture

### ViT Encoder (all stages)
- Patch size: 16×16 → 256 patches per image
- Embedding dim: 512, depth: 10 layers, heads: 8
- Learnable CLS token + positional embeddings
- MLP feedforward dim: 1024

### Stage 1 — MLP Classifier
- CLS token → 512 → 512 → 256 → 128 → 45
- Loss: CrossEntropyLoss

### Stage 2 — Capsule Network Head
- CLS token → PrimaryCapsules (32 capsules × 8-dim)
- Dynamic routing by agreement (3 iterations)
- 45 digit capsules × 16-dim
- Loss: MarginLoss (Sabour et al. 2017, eq. 4)

### Stage 3 — Dual-scale ViT + Capsule Network
- Two ViT encoders in parallel: patch=16 (coarse) + patch=8 (fine)
- CLS tokens concatenated → 1024-dim fused feature
- Same Capsule head as Stage 2
- Loss: MarginLoss

---

## Setup

```bash
git clone https://github.com/OhmKulkarni/vit-capsnet-resisc45.git
cd vit-capsnet-resisc45
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Mac/Linux
pip install -r requirements.txt
```

No Kaggle API or manual download needed. The dataset downloads automatically on first training run (~500MB, cached locally).

---

## Training

```bash
# Stage 1 — set mode: vit_mlp in config.yaml
python train.py

# Stage 2 — set mode: vit_capsule in config.yaml
python train.py

# Stage 3 — set mode: multiscale_capsule in config.yaml
python train.py

# Resume a interrupted run
python train.py --resume
```

Checkpoints saved to `checkpoints/best_<mode>.pth` and `checkpoints/last_<mode>.pth`.

---

## Evaluation

```bash
# Evaluate current mode
python evaluate.py

# Evaluate all trained stages and print comparison
python evaluate.py --all

# Print comparison table only (no inference)
python evaluate.py --compare
```

---

## Inference

```bash
# Single image
python inference.py --image path/to/image.jpg

# Directory of images
python inference.py --dir path/to/folder/

# Specify stage explicitly
python inference.py --image image.jpg --mode vit_capsule
```

---

## Results

*Training in progress — results will be updated here.*

---

## References

- Cheng et al. (2017). *Remote Sensing Image Scene Classification: Benchmark and State of the Art.* IEEE. — NWPU-RESISC45 dataset paper.
- Sabour et al. (2017). *Dynamic Routing Between Capsules.* NeurIPS. — Capsule Network architecture.
- Dosovitskiy et al. (2020). *An Image is Worth 16x16 Words.* ICLR. — Vision Transformer.