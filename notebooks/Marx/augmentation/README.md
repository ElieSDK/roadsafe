# Augmentation Pipeline

This folder provides a **virtual data augmentation pipeline** for the road damage dataset.
It allows you to train models on an **expanded dataset (~27,000 samples)** generated on the fly from the original ~9,000 images, without ever saving augmented files to disk.

---

Make sure the dependencies are installed :

```
pip install -r notebooks/Marx/augmentation/requirements.txt
```

---

## 📂 Folder Overview

```
notebooks/Marx/augmentation/
├── init.py # exports only the public API: make_virtual_loader
├── api.py # core augmentation pipeline (rotation, flip, contrast, blur, noise)
├── data_utils.py # helper to auto-download/unpack dataset from Zenodo
├── dataset.py # (legacy utilities, not needed for normal use)
├── preview.py # CLI script to preview originals + augmented variants
├── requirements.txt # pinned dependencies
├── transforms.py # augmentation transform definitions
└── virtual_loader.py # dataset wrapper for repeated/virtual epochs
```


---

## ⚙️ How It Works

- The **original dataset (~9,122 images)** is stored at Zenodo:
  [s_1024.zip](https://zenodo.org/records/11449977/files/s_1024.zip?download=1)

- When you run the pipeline, `data_utils.ensure_dataset()` will:
  1. Check if the dataset folder `s_1024/` already exists and contains images
  2. If not, automatically **download + unzip** the dataset from Zenodo

- The augmentation pipeline (`make_virtual_loader`) then:
  - Applies **random transformations**:
    rotation, horizontal flip, stronger contrast, Gaussian/Motion/Median blur, and sensor-like noise
  - With probability `p_augment=0.7` → ~70% of samples are augmented, ~30% left original
  - Virtually repeats the dataset `repeats=3` times per epoch → ~27,000 samples/epoch
  - **Nothing is saved to disk**; all augmentation is done in memory on the fly

---

## 🚀 Using the Pipeline in Training

Import the loader and use it directly in your training code:

```
from notebooks.Marx.augmentation import make_virtual_loader
from notebooks.Marx.augmentation.data_utils import ensure_dataset

# Ensure dataset is present (auto-download if missing)
DATA_DIR = ensure_dataset("notebooks/Marx/augmentation/s_1024")

# Build training loader (virtual ~27k samples per epoch)
train_loader = make_virtual_loader(
    data_dir=DATA_DIR,
    repeats=3,          # 9,122 × 3 ≈ 27k samples per epoch
    batch_size=32,
    p_augment=0.7,      # ~70% augmented, ~30% original
    max_rotate_deg=10,
    hflip_p=0.5,
    contrast_limit=0.5,
    noise_p=0.5,
    blur_p=0.5,
    to_tensor=True      # outputs CHW torch tensors in [0,1]
)

print("virtual epoch size:", len(train_loader.dataset))  # ~27k

for xb, yb in train_loader:
    # xb: augmented batch
    # yb: dummy labels (all 0, dataset is unlabeled)
    # training step...
    break
```

## 👀 Previewing Augmentation Results

If you want to see how the augmentations look, use the preview CLI.
It will:

Pick --num_images random originals

Save the original + --per_image_variants augmented variants

Write them to aug_preview/ (gitignored)

bash
```
  python notebooks/Marx/augmentation/preview.py \
    --data_dir notebooks/Marx/augmentation/s_1024 \
    --out_dir notebooks/Marx/augmentation/aug_preview \
    --num_images 4 \
    --per_image_variants 3 \
    --random
```


## Implementation exemple for your code when you want to apply augmentation:

```
# 1. Import
from notebooks.Marx.augmentation import make_virtual_loader
from notebooks.Marx.augmentation.data_utils import ensure_dataset

# 2. Ensure dataset is present (auto-downloads if missing)
DATA_DIR = ensure_dataset("notebooks/Marx/augmentation/s_1024")

# 3. Create the augmented DataLoader
train_loader = make_virtual_loader(
    data_dir=DATA_DIR,
    repeats=3,       # ~9,000 → ~27,000 samples/epoch (virtual)
    batch_size=32,
    p_augment=0.7,   # 70% augmented, 30% original
    to_tensor=True   # returns torch tensors (CHW, [0,1])
)

# 4. Use in training loop
for xb, yb in train_loader:
    # xb: augmented batch of images
    # yb: dummy labels (all 0, since dataset is unlabeled)
    # >>> your training step here <<<
    break

```

## 📌 Summary

Input: ~9,122 images (s_1024/, auto-downloaded if missing)

Output (virtual): ~27,000 samples/epoch via augmentation (make_virtual_loader)

Training: use make_virtual_loader in your training scripts (no files written)

Preview: run preview.py to generate a few augmented examples in aug_preview/
