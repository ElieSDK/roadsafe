import cv2
from pathlib import Path

# Lazy import so torch is only required when you actually call this function
def _lazy_torch():
    from torch.utils.data import Dataset, DataLoader, ConcatDataset
    return Dataset, DataLoader, ConcatDataset

EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp",".JPG",".PNG"}

def make_virtual_loader(
    data_dir: str,
    repeats: int = 3,                 # ← “3× dataset” per epoch (no files written)
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    max_rotate_deg: int = 10,
    hflip_p: float = 0.5,
    contrast_limit: float = 0.3,
    to_tensor: bool = True,
):
    """
    Return a PyTorch DataLoader that yields augmented batches over the WHOLE dataset.
    - repeats: virtual multiplier (3 -> ~3x samples per epoch)
    - images are NOT saved to disk; aug is done on-the-fly each read
    """
    from albumentations import Compose, Rotate, HorizontalFlip, RandomBrightnessContrast
    if to_tensor:
        from albumentations.pytorch import ToTensorV2

    # augmentation pipeline
    tfm_list = [
        Rotate(limit=max_rotate_deg, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        HorizontalFlip(p=hflip_p),
        RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=contrast_limit, p=1.0),
    ]
    if to_tensor:
        tfm_list.append(ToTensorV2())
    tfm = Compose(tfm_list)

    Dataset, DataLoader, ConcatDataset = _lazy_torch()

    class FlatImageDataset(Dataset):
        def __init__(self, root: str):
            root = Path(root)
            self.paths = sorted(p for p in root.rglob("*") if p.suffix in EXTS)
            if not self.paths:
                raise RuntimeError(f"No images found in {root}")
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            p = self.paths[i]
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x = tfm(image=img)["image"]     # new random aug every access
            return x, 0                     # unlabeled → dummy label

    base = FlatImageDataset(data_dir)
    virt = ConcatDataset([base] * max(1, repeats))   # ← virtual 3× without copying files
    return DataLoader(virt, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)
