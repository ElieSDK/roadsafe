# notebooks/Marx/augmentation/supervised.py
import os, cv2, random
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd
from albumentations import (
    Compose, Rotate, HorizontalFlip, RandomBrightnessContrast,
    ISONoise, OneOf, GaussianBlur, MotionBlur, MedianBlur
)
from albumentations.pytorch import ToTensorV2

# ===== Label maps (two targets) =====
SURFACE_TYPE_MAP = {"asphalt":0,"concrete":1,"paving_stones":2,"unpaved":3,"sett":4}
SURFACE_QUALITY_MAP = {"excellent":0,"good":1,"intermediate":2,"bad":3,"very_bad":4}

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp",".JPG",".PNG"}

def _find_by_stem(root: Path, stem: str) -> Path | None:
    # Fast path: try <root>/<stem>.<ext>
    for ext in IMG_EXTS:
        p = root / f"{stem}{ext}"
        if p.exists():
            return p
    # Fallback recursive
    for p in root.rglob("*"):
        if p.suffix in IMG_EXTS and p.stem == stem:
            return p
    return None

def _resize_half(img_rgb):
    h, w = img_rgb.shape[:2]
    return cv2.resize(img_rgb, (w//2, h//2), interpolation=cv2.INTER_AREA)

def build_aug(
    max_rotate_deg=10, hflip_p=0.5, contrast_limit=0.5,
    noise_p=0.5, blur_p=0.5, to_tensor=True
):
    ops = [
        OneOf([
            GaussianBlur(blur_limit=(3,7), p=1.0),
            MotionBlur(blur_limit=(3,7),  p=1.0),
            MedianBlur(blur_limit=5,      p=1.0),
        ], p=blur_p),
        ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=noise_p),
        Rotate(limit=max_rotate_deg, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        HorizontalFlip(p=hflip_p),
        RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=contrast_limit, p=1.0),
    ]
    if to_tensor:
        ops.append(ToTensorV2())
    return Compose(ops)

class LabeledDataset:
    """
    Reads labels from CSV and pairs them to image paths by 'mapillary_image_id'.
    Resizes every image by 50%. Train may augment; Test stays clean.
    Returns (image, (y_type, y_quality)).
    """
    def __init__(
        self,
        data_dir: str,
        csv_path: str,
        id_col: str = "mapillary_image_id",
        augment: bool = True,
        p_augment: float = 0.7,
        max_rotate_deg: int = 10,
        hflip_p: float = 0.5,
        contrast_limit: float = 0.5,
        noise_p: float = 0.5,
        blur_p: float = 0.5,
        to_tensor: bool = True,
    ):
        self.root = Path(data_dir)
        df = pd.read_csv(csv_path)

        # Ensure required columns exist
        if id_col not in df.columns:
            raise ValueError(f"CSV missing id_col '{id_col}'. Columns: {list(df.columns)}")
        for col in ("surface_type","surface_quality"):
            if col not in df.columns:
                raise ValueError(f"CSV missing '{col}'. Columns: {list(df.columns)}")

        ids  = df[id_col].astype(str).tolist()
        tval = df["surface_type"].astype(str).str.lower().map(SURFACE_TYPE_MAP).tolist()
        qval = df["surface_quality"].astype(str).str.lower().map(SURFACE_QUALITY_MAP).tolist()

        # Match images on disk and keep valid labels
        samples: List[Tuple[Path, int, int]] = []
        for i, t, q in zip(ids, tval, qval):
            p = _find_by_stem(self.root, i)
            if p is not None and (t is not None) and (q is not None):
                samples.append((p, int(t), int(q)))

        if not samples:
            raise RuntimeError(f"No labeled images matched under {self.root}")

        self.paths  = [p for (p, _, _) in samples]
        self.y_type = [t for (_, t, _) in samples]
        self.y_qual = [q for (_, _, q) in samples]

        self.augment = augment
        self.p_augment = p_augment
        self.to_tensor = to_tensor
        self.tfm_aug = build_aug(max_rotate_deg, hflip_p, contrast_limit, noise_p, blur_p, to_tensor)
        self.tfm_id  = Compose([ToTensorV2()]) if to_tensor else None

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        y_t = self.y_type[i]
        y_q = self.y_qual[i]
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = _resize_half(rgb)  # 50% resize

        if self.augment and random.random() < self.p_augment:
            x = self.tfm_aug(image=rgb)["image"]
        else:
            x = self.tfm_id(image=rgb)["image"] if self.to_tensor else rgb
        return x, (y_t, y_q)

def make_supervised_loaders(
    data_dir: str,
    csv_path: str,
    batch_size: int = 32,
    repeats: int = 3,                 # virtual 3× for training
    p_augment: float = 0.7,
    to_tensor: bool = True,
    num_workers: int = 0,
    train_limit: int | None = None,   # optional caps for quick tests
    test_limit: int | None = None,
):
    """
    Returns (train_loader, test_loader, metadata_dict)
    - 80/20 stratified split on surface_type
    - Train: virtual repeats × on-the-fly aug
    - Test: 50% resize only
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, ConcatDataset

    # Base dataset once to get labels for stratification
    ds_full = LabeledDataset(
        data_dir=data_dir, csv_path=csv_path,
        augment=False, p_augment=p_augment, to_tensor=to_tensor
    )
    X_idx = np.arange(len(ds_full))
    y_all = np.array(ds_full.y_type)  # stratify by surface_type

    idx_tr, idx_te = train_test_split(
        X_idx, test_size=0.2, random_state=42, stratify=y_all
    )

    # Optional size caps (random subset)
    rng = np.random.default_rng(42)
    if train_limit is not None and train_limit < len(idx_tr):
        idx_tr = rng.choice(idx_tr, size=train_limit, replace=False)
    if test_limit is not None and test_limit < len(idx_te):
        idx_te = rng.choice(idx_te, size=test_limit, replace=False)

    class SubsetDS:
        def __init__(self, base_kwargs, indices, augment: bool):
            self.base = LabeledDataset(**base_kwargs, augment=augment)
            self.indices = list(indices)
        def __len__(self): return len(self.indices)
        def __getitem__(self, j):
            return self.base[self.indices[j]]

    base_kwargs = dict(
        data_dir=data_dir, csv_path=csv_path, p_augment=p_augment, to_tensor=to_tensor
    )

    ds_train = SubsetDS(base_kwargs, idx_tr, augment=True)
    ds_test  = SubsetDS(base_kwargs, idx_te, augment=False)

    # virtual 3× train set
    from torch.utils.data import ConcatDataset
    train_virtual = ConcatDataset([ds_train] * max(1, repeats))

    train_loader = DataLoader(train_virtual, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(ds_test,      batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    meta = {
        "surface_type_map": SURFACE_TYPE_MAP,
        "surface_quality_map": SURFACE_QUALITY_MAP,
        "train_len": len(train_virtual),
        "test_len":  len(ds_test),
    }
    return train_loader, test_loader, meta

def materialize_arrays(
    data_dir: str,
    csv_path: str,
    repeats: int = 3,
    p_augment: float = 0.7,
    num_workers: int = 0,
    train_limit: int | None = None,
    test_limit: int | None = None,
):
    """
    Returns:
      X_train, Y_train, X_test, Y_test
    Where Y_* is a dict:
      - 'surface_type'    -> np.ndarray[int64]
      - 'surface_quality' -> np.ndarray[int64]

    NOTE: This stores all samples in memory. Prefer DataLoaders for real training.
    """
    import numpy as np

    train_loader, test_loader, _ = make_supervised_loaders(
        data_dir=data_dir,
        csv_path=csv_path,
        batch_size=1,                 # collect one by one (robust)
        repeats=repeats,
        p_augment=p_augment,
        to_tensor=False,              # we want NumPy HWC
        num_workers=num_workers,
        train_limit=train_limit,
        test_limit=test_limit,
    )

    def _collect(loader):
        X_list, y_type_list, y_qual_list = [], [], []
        for xb, yb in loader:
            img = xb[0] if isinstance(xb, (list, tuple)) else xb[0]
            yt, yq = yb[0] if isinstance(yb, (list, tuple)) else yb
            X_list.append(img)
            y_type_list.append(int(yt))
            y_qual_list.append(int(yq))
        X = np.stack(X_list, axis=0)
        Y = {
            "surface_type":   np.asarray(y_type_list, dtype=np.int64),
            "surface_quality":np.asarray(y_qual_list, dtype=np.int64),
        }
        return X, Y

    X_train, Y_train = _collect(train_loader)
    X_test,  Y_test  = _collect(test_loader)
    return X_train, Y_train, X_test, Y_test
