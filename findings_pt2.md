# ViT + Capsule Network — Experimental Findings Log
**Dataset:** NWPU-RESISC45 (45 classes, 256×256 RGB, trained from scratch)
**Split:** ~23,625 train / ~3,375 val / 4,500 test (100 per class)

---

## Stage 1 — `vit_mlp` (Baseline)
**Architecture:** ViT (dim=512, depth=10, heads=8) + MLPClassifier  
**Loss:** CrossEntropyLoss  
**Key config:** lr=1e-4, weight_decay=0.05, batch=128 (effective), epochs=100

### Results
| Metric | Value |
|---|---|
| Test Accuracy | 83.96% |
| Macro F1 | 0.8388 |
| Train Acc (final) | 96.73% |
| Overfit Gap | ~12.5% |

### Observations
- Model was still slowly improving at epoch 100 — early stopping never triggered
- Strong on visually distinct classes: sea ice (0.98), cloud (0.97), golf course (0.95)
- Weakest classes were all structurally complex — defined by spatial relationships between components rather than texture or colour:
  - **palace** F1=0.614 (confused with church, commercial area)
  - **basketball court** F1=0.641 (confused with tennis court, ground track field)
  - **railway station** F1=0.673 (confused with airport, industrial area)
  - **airport** F1=0.696 (confused with runway, industrial area)
- These four classes became the *watch classes* for evaluating whether capsule routing genuinely helps

---

## Stage 2 — `vit_capsule_run1` (Routing Collapse)
**Architecture:** ViT + PrimaryCapsules + CapsuleNetwork (dynamic routing, 3 iterations)  
**Loss:** MarginLoss (m+=0.9, m-=0.1, λ=0.5)  
**Key config:** primary_caps=32, digit_caps_dim=16, W init=0.01, no warmup, weight_decay=0.05

### Results
| Metric | Value |
|---|---|
| Test Accuracy | 77.20% |
| Macro F1 | 0.7697 |
| Train Acc (final) | 87.85% |
| Overfit Gap | ~10.5% |

### What went wrong
**Routing collapse in early epochs.** At epoch 10, val accuracy was only 5.8% — Stage 1 was already at 55.8% by the same point. The model effectively had no working classification head for the first 30–40 epochs, wasting the high-LR phase of cosine annealing where the most learning should happen.

**Root causes identified:**
1. **W matrix init too small (`* 0.01`)** — primary capsule vectors were too close to zero for routing agreement signals to be meaningful early on
2. **Only 32 primary capsules for 45 classes** — an underdetermined routing system (fewer inputs than outputs), making the W matrix rank-deficient
3. **No warmup** — the ViT encoder and capsule head tried to co-adapt from random initialisation simultaneously, preventing either from bootstrapping cleanly

### Fixes applied
- `primary_caps_channels`: 32 → 128 (exceeds num_classes, fixes rank deficiency)
- W matrix init: `* 0.01` → `* 0.1`
- Added `capsule_warmup_epochs: 10` — ViT encoder frozen for first 10 epochs so routing bootstraps with fixed features

---

## Stage 2 — `vit_capsule_run2` (Routing Bootstraps, Overfitting)
**Architecture:** Same as run1  
**Key config:** primary_caps=128, digit_caps_dim=16, W init=0.1, warmup=10 epochs, weight_decay=0.05

### Results
| Metric | Value |
|---|---|
| Test Accuracy | 84.02% |
| Macro F1 | 0.8399 |
| Train Acc (final) | 98.5% |
| Overfit Gap | ~15% |

**Watch class comparison vs Stage 1:**
| Class | Stage 1 | Run2 | Delta |
|---|---|---|---|
| palace | 0.614 | 0.682 | +0.069 ✅ |
| basketball court | 0.641 | 0.712 | +0.071 ✅ |
| airport | 0.696 | 0.739 | +0.043 ✅ |
| railway station | 0.673 | 0.646 | -0.027 ❌ |

### Observations
- Training curve now healthy — sharp improvement at epoch 10 when encoder unfroze confirms warmup fix worked
- Overall accuracy near-tied with Stage 1 (+0.06%) — the capsule hypothesis is *partially* confirmed
- 22 classes improved, 22 regressed — gains on hard structural classes are being cancelled by regressions on easy ones
- **Overfitting is now the dominant problem:** train acc 98.5% vs val acc ~82%, gap of ~15%
- Root cause: W matrix alone has 128×8×45×16 = **737,280 parameters** being trained on only 23,000 samples

### Fixes applied for run3
- `digit_caps_dim`: 16 → 8 (halves W matrix to 368,640 params)
- Added `caps_dropout: 0.2` — dropout on primary capsule vectors before routing
- `weight_decay`: 0.05 → 0.1 — stronger L2 regularisation

---

## Stage 2 — `vit_capsule_run3` (Regularisation — Plateau)
**Architecture:** Same  
**Key config:** primary_caps=128, digit_caps_dim=8, caps_dropout=0.2, W init=0.1, warmup=10, weight_decay=0.1

### Results
| Metric | Value |
|---|---|
| Test Accuracy | 84.04% |
| Macro F1 | 0.8396 |
| Train Acc (final) | 98.29% |
| Overfit Gap | ~15% |

**Watch class comparison:**
| Class | Stage 1 | Run2 | Run3 | vs Stage 1 |
|---|---|---|---|---|
| palace | 0.614 | 0.682 | 0.642 | +0.028 ✅ |
| basketball court | 0.641 | 0.712 | 0.656 | +0.015 ✅ |
| airport | 0.696 | 0.739 | 0.715 | +0.019 ✅ |
| railway station | 0.673 | 0.646 | 0.657 | -0.017 ❌ |

### Observations
- Regularisation partially worked: several easy classes recovered vs run2
- But watch class peaks from run2 regressed (palace 0.682→0.642, basketball court 0.712→0.656)
- Overfitting gap did **not** narrow despite higher weight_decay and dropout — still ~15%
- Overall accuracy essentially identical across all three runs: 83.96, 84.02, 84.04 — hard ceiling hit

### Root cause analysis
All three runs converge to the same performance ceiling. The pattern reveals a fundamental tension: configurations that help structural classes hurt texture/appearance classes, and vice versa. This is not a regularisation problem — it is a **feature representation problem**.

The ViT CLS token compresses 256 patches into one global vector before the capsule head sees it. Routing has nothing spatially meaningful to route between — it is finding agreement between dimensions of a globally-pooled feature, not between spatial parts. This is the motivation for Stage 3.

---

## Stage 3 — `multiscale_capsule` (Dual-scale CLS Concatenation)
**Architecture:** Coarse ViT (patch=16, depth=10) + Fine ViT (patch=8, depth=2) in parallel → CLS tokens concatenated (1024-dim) → PrimaryCapsules(128) → CapsuleNetwork  
**Loss:** MarginLoss  
**Key config:** primary_caps=128, digit_caps_dim=8, caps_dropout=0.2, warmup=10, weight_decay=0.1, batch=16, accum=8 (effective batch=128), gradient_checkpoint_fine=True

### Results
| Metric | Value |
|---|---|
| Test Accuracy | 85.16% |
| Macro F1 | 0.8505 |
| Train Acc (final) | 97.38% |
| Overfit Gap | ~13.4% |
| Best Val Acc | 84.15% (epoch 99) |

**Watch class comparison:**
| Class | Stage 1 | Stage 2 (run3) | Stage 3 | vs Stage 1 |
|---|---|---|---|---|
| palace | 0.614 | 0.642 | 0.685 | +0.071 ✅ |
| basketball court | 0.641 | 0.656 | 0.763 | +0.122 ✅ |
| airport | 0.696 | 0.715 | 0.786 | +0.090 ✅ |
| railway station | 0.673 | 0.657 | 0.681 | +0.007 ✅ |

**Top improvements vs Stage 1 (29 classes improved, 15 regressed):**
| Class | Stage 1 | Stage 3 | Delta |
|---|---|---|---|
| basketball court | 0.641 | 0.763 | +0.122 |
| airport | 0.696 | 0.786 | +0.090 |
| palace | 0.614 | 0.685 | +0.071 |
| overpass | 0.825 | 0.887 | +0.061 |
| industrial area | 0.729 | 0.781 | +0.052 |
| roundabout | 0.735 | 0.787 | +0.052 |
| storage tank | 0.765 | 0.816 | +0.051 |
| mountain | 0.856 | 0.905 | +0.050 |

**Notable regressions vs Stage 1:**
| Class | Stage 1 | Stage 3 | Delta |
|---|---|---|---|
| mobile home park | 0.931 | 0.888 | -0.044 |
| harbor | 0.937 | 0.896 | -0.041 |
| freeway | 0.857 | 0.820 | -0.037 |
| church | 0.758 | 0.721 | -0.037 |
| ship | 0.857 | 0.821 | -0.037 |

### Observations
- First stage to cleanly beat the baseline — +1.2% accuracy, +0.012 Macro F1
- Dual-scale hypothesis confirmed: all four watch classes improved vs Stage 1 for the first time
- 29/45 classes improved vs Stage 1 — much healthier than Stage 2's even 22/22 split
- Railway station finally moved positive (+0.007) but remains one of the two lowest-F1 classes at 0.681
- Overfitting gap narrowed slightly from ~15% to ~13.4% despite double the encoder parameters
- Regressions concentrated in texture-dominant classes where the MLP's global features were sufficient and capsule routing adds noise

### Root cause analysis
Stage 3 validated the dual-scale hypothesis but remains constrained — both CLS tokens are globally pooled summaries. Even at two resolutions, routing works with two global vectors rather than spatially-explicit patch-level features. The texture/structure tradeoff persists across all stages.

**This motivates Stage 4.** Rather than concatenating two CLS tokens, Stage 4 feeds all 1024 patch tokens from the fine encoder directly into the capsule head — each patch token becomes one primary capsule, giving routing spatially-grounded inputs exactly as in the original Sabour et al. design.

---

## Stage 4 — `patch_capsule` (Spatial Patch-level Primary Capsules)
**Architecture:** Coarse ViT (patch=16, depth=10) CLS token for global context + Fine ViT (patch=8, depth=2) patch tokens (1024) as primary capsules → CapsuleNetwork → global context injection from coarse CLS → digit capsule lengths  
**Loss:** MarginLoss  
**Key config:** Inheriting Stage 3 best settings. num_primary_caps=1024 (=num_patches_fine), digit_caps_dim=8, caps_dropout=0.2, warmup=10, weight_decay=0.1

### Design
This is the most architecturally faithful implementation of Sabour et al. applied to this problem:

| Stage | Primary capsule inputs | Spatial? |
|---|---|---|
| Stage 2 | 128 projections of 1 CLS token | ❌ |
| Stage 3 | 128 projections of 2 CLS tokens | ❌ |
| Stage 4 | 1024 patch tokens (one per spatial location) | ✅ |

Each of the 1024 patch capsules encodes local visual features at a specific (x, y) location in the image. The routing algorithm asks: do the right components agree on the presence of a particular class? The coarse CLS token is injected after routing as global context — a runway patch looks like a freeway patch in isolation, but the surrounding scene disambiguates them.

**Note on W matrix size:** Now (1, 1024, 45, 8, 8) — 8× larger than Stage 3. VRAM pressure is higher; batch size may need reducing to 16 if OOM occurs at epoch 11 (encoder unfreeze).

### Hypothesis
If spatial routing genuinely helps: further improvement on structural classes beyond Stage 3, potentially breaking through the ~85% ceiling. Palace and railway station in particular should finally see meaningful gains.

If the ceiling persists near 85%: the limiting factor is the overfitting gap (~13%) driven by dataset size relative to model capacity, not the routing architecture. At that point, stronger augmentation (CutMix, MixUp) or a pretrained ViT backbone would be the logical next step.

### Results
*Pending — training not yet run*