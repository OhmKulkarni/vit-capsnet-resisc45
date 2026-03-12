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
- 22 classes improved, 22 regressed — gains on hard structural classes are being cancelled by regressions on easy ones (golf course -0.040, thermal power station -0.044, sparse residential -0.045)
- **Overfitting is now the dominant problem:** train acc 98.5% vs val acc ~82%, gap of ~15% (worse than Stage 1's 12.5%)
- Root cause: W matrix alone has 128×8×45×16 = **737,280 parameters** being trained on only 23,000 samples — more parameters than the entire MLP head, with less inductive bias to constrain them. The routing is memorising training-set agreement patterns that don't generalise.

### Fixes applied for run3
- `digit_caps_dim`: 16 → 8 (halves W matrix to 368,640 params — more appropriate for dataset size)
- Added `caps_dropout: 0.2` — dropout on primary capsule vectors before routing prevents co-adaptation between routing paths
- `weight_decay`: 0.05 → 0.1 — stronger L2 regularisation on all parameters including W

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

**Watch class comparison vs Stage 1:**
| Class | Stage 1 | Run2 | Run3 | vs Stage 1 |
|---|---|---|---|---|
| palace | 0.614 | 0.682 | 0.642 | +0.028 ✅ |
| basketball court | 0.641 | 0.712 | 0.656 | +0.015 ✅ |
| airport | 0.696 | 0.739 | 0.715 | +0.019 ✅ |
| railway station | 0.673 | 0.646 | 0.657 | -0.017 ❌ |

### Observations
- Regularisation partially worked: golf course recovered to 0.935 (+0.021 vs run2), ground track field +0.057, airplane +0.044, dense residential +0.036
- But the watch classes that run2 had cracked — palace and basketball court — regressed back from run2's peaks (palace 0.682→0.642, basketball court 0.712→0.656)
- Overfitting gap did **not** narrow despite higher weight_decay and dropout — still ~15%
- Overall accuracy is essentially identical across all three runs: 83.96, 84.02, 84.04 — the model has hit a hard ceiling

### Root cause analysis
All three runs are converging to the same performance ceiling. The pattern across runs reveals a fundamental tension: configurations that help structural classes hurt texture/appearance classes, and vice versa. This is not a regularisation problem — it is a **feature representation problem**.

The ViT CLS token is a single global vector summarising the entire image. Dynamic routing's strength is finding part-whole relationships *within* that representation — but if the representation itself doesn't encode spatial part information separately, routing has nothing meaningful to route between. A single CLS token compresses 256 patches into one vector before the capsule head ever sees it. The routing is essentially trying to find spatial structure in a globally-pooled feature, which explains why gains are marginal and unstable across runs.

**This is exactly the motivation for Stage 3.** The dual-scale architecture feeds two separate CLS tokens (coarse patch=16 + fine patch=8) into the capsule head, giving routing access to features computed at different spatial resolutions. The fine encoder's CLS token will carry more local structure; the coarse encoder's will carry global context. The 1024-dim concatenated input gives routing genuinely different signal sources to route between — which is what single-scale Stage 2 has been missing.

---

## Stage 3 — `multiscale_capsule` (Upcoming)
**Architecture:** Coarse ViT (patch=16) + Fine ViT (patch=8) in parallel → CLS tokens concatenated (1024-dim) → PrimaryCapsules → CapsuleNetwork  
**Key config:** primary_caps=128, digit_caps_dim=8, caps_dropout=0.2, warmup=10, weight_decay=0.1 (inheriting run3 best settings)

### Hypothesis
The dual-scale input gives the capsule routing genuinely distinct signal sources — global layout from the coarse encoder, local detail from the fine encoder. For the hard structural classes (palace, basketball court, airport, railway station), the fine encoder should capture component-level features (windows, court markings, gate structures, platform details) that the coarse encoder misses, while the coarse encoder preserves the overall spatial arrangement. Routing across 1024-dim concatenated features should produce more stable agreement signals than routing across a single 512-dim CLS token.

**Expected outcome:**
- Overall accuracy: 86–89% — a meaningful step above the ~84% ceiling hit by single-scale capsule runs
- Watch classes: all four should improve vs Stage 1, including railway station which has stubbornly regressed in every Stage 2 run
- The overfitting gap may widen slightly due to double the encoder parameters (~2× VRAM), but the richer input representation should produce more generalisable routing
- If Stage 3 also plateaus near 84%, the conclusion is that CLS-token-based routing cannot exploit spatial part-whole relationships regardless of resolution, and a spatially-explicit capsule input (e.g. from patch tokens rather than CLS) would be required