import cv2
from albumentations import Compose, Rotate, HorizontalFlip, RandomBrightnessContrast
from albumentations.pytorch import ToTensorV2

def get_train_transform(max_rotate_deg: int = 10, hflip_p: float = 0.5, contrast_limit: float = 0.3):
    """On-the-fly aug: rotation, horizontal flip, contrast. (No vertical flip for roads.)"""
    return Compose([
        Rotate(limit=max_rotate_deg, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        HorizontalFlip(p=hflip_p),
        RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=contrast_limit, p=1.0),
        ToTensorV2(),
    ])

def get_val_transform():
    """Validation/Test: no augmentation (just tensor conversion)."""
    return Compose([ToTensorV2()])
