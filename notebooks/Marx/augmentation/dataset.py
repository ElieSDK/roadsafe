import cv2
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision.datasets import ImageFolder

class AugImageFolder(Dataset):
    """
    Uses torchvision.ImageFolder and applies Albumentations in __getitem__.
    Directory layout expected:
        data/train/<class_name>/*.jpg
        data/val/<class_name>/*.jpg
    If you don’t have labels yet, put everything under a single subfolder (e.g., 'unknown/').
    """
    def __init__(self, root, transform=None):
        self.base = ImageFolder(root=root)
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        path, label = self.base.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label

def triple_dataset(ds, times: int = 3):
    """Virtually multiplies dataset length (no files created)."""
    return ConcatDataset([ds] * times)

def make_dataloader(root: str, transform, batch_size: int = 32, shuffle: bool = True,
                    num_workers: int = 4, pin_memory: bool = True):
    ds = AugImageFolder(root, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=pin_memory)
    return ds, loader
