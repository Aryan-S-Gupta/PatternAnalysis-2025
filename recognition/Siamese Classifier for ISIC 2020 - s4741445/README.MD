# ISIC 2020 Siamese Network Classifier – README

**Student:** Aryan Somesh Gupta
**Student number:** **47414451**

---

## 1. Introduction

This project aims to solve the **ISIC 2020 Kaggle Challenge** that is to classify whether given image shows melanoma or benign. The implements a siamese network classifier to distinguish the dermoscopic images that have heavily class-imbalance (More benign than melanoma) with the goal to achieve robust detection and binary classification avoiding majority-class bias, using a compute-friendly pipeline that runs on Google Colab.

## 2. The Problem

ISIC 2020 is **very** imbalanced: benign vastly outnumbers melanoma. A plain classifier tends to predict the majority class. So I re-framed it as: **first learn a space where “lesions of the same class are close”** (metric learning), then put a **small classifier** on top. (Add more details about the challenge from the page - original challenge - https://challenge2020.isic-archive.com/)

The Dataset I used was teh pre-resize Kaggle version, that can be found [here](https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-256x256-resized/data) which contains **~33,126 JPEGs on disk as per metadata**

## 3. How it Works (Background)

A Siamese network is a neural network with shared-weight twin branches that learn to compare two inputs. It’s mainly used to measure similarity—like matching faces, identifying duplicates, or comparing medical images. 

For my approach, I I train in **two stages**:

1. **Stage 1 – Siamese/metric-learning encoder** → learn a 256-dim embedding of lesions with supervised-contrastive/triplet-style objective.


2. **Stage 2 – Binary classifier on cached embeddings** → turn that embedding into “benign vs malignant”,  improves stability under imbalance, and keeps inference fast. 


The **Key idea:** is that instead of throwing a plain CNN at an extremely imbalanced 0/1 problem (≈32,543 benign vs ≈585 malignant as per my dataset snapshot), I first train a **Siamese / metric-learning stage** to produce good lesion embeddings and only **then** train a lightweight **binary classifier head** on frozen / cached embeddings. 

---

## 4. Model Explanation (complete details - other than implementaiton, like how it was made, what it cincludes, etc - small justificaiton of why as well)

### 4.1 Stage 1 – Siamese / Metric Learning

* Backbone: **ResNet-50 pretrained on ImageNet**, final FC removed. 
* Projection head: 2048→512→256, then **L2 normalization** → good for cosine / triplet losses.
* Loss strategy:

  * warmup with **supervised contrastive** (groups *same-label* lesions together)
  * then **triplet-style distance** + **auxiliary CE** on the anchor embedding
  * triplet **margin is scheduled** from 0.3 → 0.8 across epochs
* Optimiser: **Adam** with weight decay and **cosine annealing LR**; runs EMA on weights. 
* Early unfreeze: first few epochs only train the projection + aux head, then unfreeze the ResNet backbone to refine embeddings, which is important for medical images. 

### 4.2 Stage 2 – Binary Classifier (complete details - other than implementaiton, like how it was made, what it cincludes, etc - small justificaiton of why as well)

* After Stage 1 finishes, I **cache the embeddings for train/val/test** (there’s a `cache_embeddings(...)` in train.py). Then I train **`HeadBinaryClassifier`** (256→128→32→2) just on those fixed vectors. This makes S2 **very fast** and stable. 
* S2 uses **10 epochs**, LR = 1e-3 with step-downs when val AUC stalls. This is set in `config.py`. 

### 4.3 Why two stages?

1. Stage 1 sees **augmented images** and learns **invariance**.
2. Stage 2 sees **clean, aligned embeddings** → the class head focuses only on “is this melanoma?”
3. It’s also a lot easier to try different heads in S2 without re-training the big ResNet.

---

## 5. Implementation Details (Add details + leave space for a rough diagram for explaintaion as well)

### 5.1 Siamese Network (Stage 1)


* **Splits:** using `VAL_FRACTION = 0.15`, `TEST_FRACTION = 0.15` → so ≈70% train, 15% val, 15% test → came out as

  * train = **23,546**
  * val = **4,694**
  * test = **4,886**

  this was done so becasue we need **enough validation** to estimate AUC in an imbalanced setting, and a **held-out test** to report final numbers.

  
* **Class imbalance:** actual Kaggle CSV has **huge benign majority** (your note: 32,543 benign vs 585 malignant). Because of that, I also:

  * used **balanced triplet sampling** in Stage 1
  * used **AUC** and **balanced accuracy** as primary metrics, not just accuracy.
* **Transforms (why):**

  * flips → deal with lesion orientation
  * rotation(12°) → dermoscopic photos are rarely perfectly aligned
  * color jitter → small because too much will erase diagnostic colour
  * ImageNet normalization → needed for pretrained ResNet-50. 


* **Inputs:** triplets (anchor, positive, negative) sampled from the training table. Sampling is **balanced across classes** via a custom batch sampler to avoid drowning out the malignant class. 
* **Augmentations (train):**
  * Random horizontal flip
  * Random vertical flip (p=0.2)
  * Random rotation (12°)
  * Color jitter (small)
  * RandomResizedCrop to (224,224)
  * Resize + ToTensor + ImageNet normalization
  * **Rationale:** ISIC dermoscopic photos vary in lighting, so light color jitter + geometric transforms help. 
* **Loss mix:** either SupCon during warmup or triplet + λ·CE afterwards (λ from config).
* **Epochs:** config says 15 Siamese epochs; in my run it **early-stopped around epoch 13** when val loss/AUC flattened (see your log). That’s why the README says “≈13/15 effective epochs”.
* **Outputs saved:** EMA Siamese weights → `siamese_stage1_ema.pt`, aux head → `aux_head_ema.pt`, plots, normalization grids. 

### 5.2 Binary Classifier (Stage 2)

* **Inputs:** cached 256-D embeddings + labels
* **Model:** `HeadBinaryClassifier` (MLP + dropout) 
* **Optimization:** Adam (1e-3), LR halved when val AUC plateaus (see logs where LR is reduced).
* **Why cached?** To avoid re-encoding 23k+ images per epoch and to make COLAB runs cheaper.
* **Outputs saved:** `classifier.pt`, plus evaluation plots on cached test split. 

---

## 6. Results (TO be fixed and add details)

### 6.1 From `train.py`

Your Stage-1 log showed the classic pattern:

* Epochs 1–3: loss ≈4 (that’s SupCon/triplet warmup, metrics = NaN)
* Epoch 4 onward: loss drops to **0.62 train / 0.70 val** and we start getting **AUC 0.74–0.9+**
* By epoch 8–12: **train AUC → 0.95+, val AUC → 0.88–0.93**
* Early stop at epoch **13/20** → “good enough, stop overfitting malignant”.
  This is consistent with what `train.py` does: full validation each epoch using **EMA weights**. 

### 6.2 From `predict.py`

Your final prediction run said:

* **Overall Accuracy:** **87.19%**
* **AUC:** **0.903**
* **Balanced Accuracy:** **79.47%**
* **Sensitivity (Recall+):** **71.43%**
* **Specificity (Recall−):** **87.51%**

**Interpretation:**

* AUC 0.903 means the embedding + head **can rank malignant above benign most of the time**, despite the 1:55 class ratio.
* Balanced accuracy ~0.79 tells the marker that we **didn’t just predict “benign”**.
* Sensitivity 0.71 is actually *decent* for this dataset, given so few malignant examples.
* Specificity high (0.87) → we’re not overcalling everything as melanoma.
* This matches the COMP3710 target of “around 0.8 on the test set” for the Siamese ISIC task. 

### 6.3 Plots (To Add)

* **Confusion matrix** → shows the true negatives block is large (as expected), but there is a meaningful number of TPs.
* **ROC curve** → should cross (0,1) smoothly with AUC label ≈0.90.

![roc_curve](/recognition/Siamese%20Classifier%20for%20ISIC%202020%20-%20s4741445/images/ROC_Curve.png)

* **Training curves** (from `utils.py`) → allow you to paste into the README/PDF. 

<<include graph labels as per the code + add jsutificaiotn & Explaination for the graphs>>

---

## 7. Usage

### 7.1 Repository / File Structure

```text
myDrive/
└── siamese_project/
    ├── config.py    # this file
    ├── data.py      # this file
    ├── modules.py   # this file
    ├── train.py     # this file
    ├── predict.py   # this file
    ├── utils.py     # this file
    └── README.md    # this file
        
```

### 7.2 Run on Google Colab

1. Open my Colab notebook:
   **[Here](https://colab.research.google.com/drive/1vwZoIxYiaL75teFRFyfKGsOmRqXALbec?usp=sharing)**
2. Mount Google Drive and make sure these paths exist (as in `config.py`):

   * `/content/drive/MyDrive/siamese_project/`
   * `/content/train-image/image`
   * `/content/train-metadata.csv`
     or update `config.py` to match your own folder. 
3. Train:

```bash
# 1) train everything - Stage 1, cache embeddings, Stage 2
!python3 train.py
```
4. Run:
```bash
# 2) evaluate with saved weights + TTA
!python predict.py --siamese /content/drive/MyDrive/siamese_project/outputs/siamese_stage1_ema.pt --classifier /content/drive/MyDrive/siamese_project/outputs/classifier.pt --tta 4
```

**Notes:**

* I used `--tta 4` because `predict.py` supports 1,2,4,8 views – 4 is a good trade-off. 
* You **can** pass custom paths; the script will fall back to whatever is in `ARTIFACTS` if you don’t.

---

## 8. Dependencies

This was written for **Python 3.x** and **PyTorch** (Colab style). Minimum versions that work with `torchvision.transforms.v2`:

* `torch >= 2.2`
* `torchvision >= 0.17` (for `torchvision.transforms.v2`)
* `pandas >= 2.2` (CSV reading)
* `scikit-learn >= 1.4` (ROC/AUC, confusion matrix)
* `matplotlib >= 3.8` (plots)
* `Pillow >= 10` (image loading) <<Did i even use this ? check and fix
* `numpy`

Install (example):

```bash
pip install torch torchvision pandas scikit-learn matplotlib pillow
```

---

## 9. Future Recommendations / What’s Missing

1. **Stronger imbalance handling** – add focal loss or class-weighted CE in Stage 2, or oversample malignant more aggressively during caching.
2. **Hard / semi-hard mining** – current triplet sampling is balanced but not *mining-aware*; using batch-hard or distance-weighted sampling could push AUC a bit higher.
3. **Higher-res finetuning** – I trained at 224×224 (compatible with ResNet-50). A final 384×384 finetune on just malignant + a subset of benigns often helps dermoscopy.
4. **Use metadata (age, sex, anatomic site)** – ISIC 2020 has useful tabular features; concatenating these to the 256-D embedding before the classifier would help recall.
5. **Model ensembling / SWA** – average 2–3 heads trained on different cached splits to improve test-time stability.

---