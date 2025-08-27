import random, cv2
from pathlib import Path
from albumentations import (
    Compose, Rotate, HorizontalFlip, RandomBrightnessContrast,
    ISONoise, OneOf, GaussianBlur, MotionBlur, MedianBlur
)

from albumentations.pytorch import ToTensorV2

_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp",".JPG",".PNG"}

# ---- dataset defined at top-level so it's picklable ----
class FlatImageDataset:
    def __init__(self, root: str, tfm_aug, tfm_id, p_augment: float, to_tensor: bool):
        root = Path(root)
        self.paths = sorted(p for p in root.rglob("*") if p.suffix in _EXTS)
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
        self.tfm_aug = tfm_aug
        self.tfm_id = tfm_id
        self.p_augment = p_augment
        self.to_tensor = to_tensor

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if random.random() < self.p_augment:
            out = self.tfm_aug(image=img)["image"]
        else:
            out = self.tfm_id(image=img)["image"] if self.to_tensor else img
        return out, 0


def make_virtual_loader(
    data_dir: str,
    repeats: int = 3,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,             # safer default (no multiproc pickle issues)
    pin_memory: bool = True,

    # --- augmentation controls ---
    p_augment: float = 0.7,
    max_rotate_deg: int = 10,
    hflip_p: float = 0.5,
    contrast_limit: float = 0.5,
    noise_p: float = 0.5,
    blur_p: float = 0.5,
    to_tensor: bool = True,
):
    """
    Return DataLoader with ~repeats × dataset size samples/epoch.
    Random subset augmented each time; nothing saved to disk.
    """
    # aug pipeline
    aug_ops = [
        OneOf([
            GaussianBlur(blur_limit=(3,7), p=1.0),
            MotionBlur(blur_limit=(3,7),  p=1.0),
            MedianBlur(blur_limit=5,      p=1.0),
        ], p=blur_p),

        # ✅ robust noise across albumentations versions
        ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=noise_p),

        Rotate(limit=max_rotate_deg, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        HorizontalFlip(p=hflip_p),
        RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=contrast_limit, p=1.0),
    ]


    aug_to_tensor = [ToTensorV2()] if to_tensor else []
    tfm_aug = Compose(aug_ops + aug_to_tensor)
    tfm_id  = Compose(aug_to_tensor) if to_tensor else None

    # torch imports here (lazy)
    from torch.utils.data import DataLoader, ConcatDataset

    base = FlatImageDataset(data_dir, tfm_aug, tfm_id, p_augment, to_tensor)
    virt = ConcatDataset([base] * max(1, repeats))

    return DataLoader(
        virt,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,   # <= default now 0 for safety
        pin_memory=pin_memory
    )
