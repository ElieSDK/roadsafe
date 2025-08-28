# Augmentation Pipeline (with Labels)

This folder provides a **virtual data augmentation pipeline** for the road surface dataset.
It allows you to train models on an **expanded dataset (~27,000 samples per epoch)** generated **on the fly** from the original ~9,000 images, without saving augmented files to disk.

---

# ⚡ Quickstart (3 steps)

## 1. install dependencies
```
pip install -r notebooks/Marx/augmentation/requirements.txt
```

## 2. ensure dataset is downloaded + extracted (~9k images)
```
python -c "from notebooks.Marx.augmentation.data_utils import ensure_dataset; print(ensure_dataset('notebooks/Marx/augmentation/s_1024/unknown'))"
```

## 3. preview a few augmentations with labels
```
python notebooks/Marx/augmentation/preview.py \
  --data_dir notebooks/Marx/augmentation/s_1024/unknown \
  --csv_path notebooks/Marx/augmentation/streetSurfaceVis_v1_0.csv \
  --out_dir notebooks/Marx/augmentation/aug_preview \
  --num_images 4 \
  --per_image_variants 3 \
  --random
  ```

  ## 📂 Folder Overview

```
notebooks/Marx/augmentation/
├── __init__.py        # exports main API
├── api.py             # unlabeled augmentation pipeline
├── data_utils.py      # helper to auto-download/unpack dataset from Zenodo
├── dataset.py         # (legacy utilities, not needed for normal use)
├── supervised.py      # labeled augmentation pipeline (with CSV)
├── preview.py         # CLI script to preview originals + augmented variants
├── transforms.py      # augmentation transform definitions
├── virtual_loader.py  # dataset wrapper for repeated/virtual epochs
├── requirements.txt   # dependencies
└── streetSurfaceVis_v1_0.csv # labels file (image_id, surface_type, surface_quality)
```

---

## 📊 Dataset

- The **original dataset (~9,122 images)** is stored at Zenodo:
  [s_1024.zip](https://zenodo.org/records/11449977/files/s_1024.zip?download=1)
- When first used, data_utils.ensure_dataset() automatically downloads + extracts it into:
```
notebooks/Marx/augmentation/s_1024/unknown/
```

### Labels are taken from streetSurfaceVis_v1_0.csv with two targets:

#### surface_type:
```
{asphalt=0, concrete=1, paving_stones=2, unpaved=3, sett=4}
```

#### surface_quality:
```
{excellent=0, good=1, intermediate=2, bad=3, very_bad=4}
```

---

# ⚙️ How It Works

Every image is resized by 50% before use.

Augmentations applied (with probability p_augment=0.7):
rotation, horizontal flip, stronger contrast, Gaussian/Motion/Median blur, and sensor-like noise.

Virtually repeats the dataset repeats=3 → ~27,000 samples per epoch.

Virtual = nothing is saved; augmentations happen in memory.

Test set = resize only (no augmentation).


# 🚀 Using in Training (Recommended)
```
from notebooks.Marx.augmentation.data_utils import ensure_dataset
from notebooks.Marx.augmentation import make_supervised_loaders

# ensure dataset is present
DATA_DIR = ensure_dataset("notebooks/Marx/augmentation/s_1024/unknown")
CSV_PATH = "notebooks/Marx/augmentation/streetSurfaceVis_v1_0.csv"

# build loaders
train_loader, test_loader, meta = make_supervised_loaders(
    data_dir=DATA_DIR,
    csv_path=CSV_PATH,
    batch_size=32,
    repeats=3,      # ~27k samples/epoch
    p_augment=0.7,
    to_tensor=True
)

xb, (y_type, y_quality) = next(iter(train_loader))
print(xb.shape, y_type.shape, y_quality.shape, meta)

```
- xb: batch of images (tensors, resized, possibly augmented)

- y_type: surface type labels (0..4)

- y_quality: surface quality labels (0..4)

---

## 📦 Exporting Arrays (Optional for Grading)

### If you need NumPy arrays instead of DataLoaders:
```
from notebooks.Marx.augmentation import materialize_arrays
from notebooks.Marx.augmentation.data_utils import ensure_dataset

DATA_DIR = ensure_dataset("notebooks/Marx/augmentation/s_1024/unknown")
CSV_PATH = "notebooks/Marx/augmentation/streetSurfaceVis_v1_0.csv"

X_train, Y_train, X_test, Y_test = materialize_arrays(
    data_dir=DATA_DIR,
    csv_path=CSV_PATH,
    repeats=3,
    p_augment=0.7,
    train_limit=3000,   # caps for quick tests
    test_limit=300
)

print(X_train.shape)                          # e.g. (3000, H, W, 3)
print(Y_train["surface_type"].shape)          # (3000,)
print(Y_train["surface_quality"].shape)       # (3000,)

```
⚠️ Arrays can be heavy → use train_limit / test_limit to keep it light.
For real training, prefer loaders.


## 👀 Previewing Augmentation Results

If you want to see how the augmentations look, use the preview CLI.
It will:

Pick --num_images random originals

Save the original + --per_image_variants augmented variants

Write them to aug_preview/ (gitignored)


```
python notebooks/Marx/augmentation/preview.py \
  --data_dir notebooks/Marx/augmentation/s_1024/unknown \
  --csv_path notebooks/Marx/augmentation/streetSurfaceVis_v1_0.csv \
  --out_dir notebooks/Marx/augmentation/aug_preview \
  --num_images 4 \
  --per_image_variants 3 \
  --random
```

- Picks --num_images random originals
- Saves original + --per_image_variants augmented variants each
- Writes them to aug_preview/
- Prints a summary of class distributions for surface type & quality

# 📌 Summary

Input: ~9,122 images + CSV labels

Output (virtual): ~27,000 samples per epoch via augmentation

Targets:

surface_type (0..4)

surface_quality (0..4)

Use make_supervised_loaders for training

Use materialize_arrays (with caps) if arrays are required

Use preview.py for quick visualization
