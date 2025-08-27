#!/usr/bin/env python3
import argparse, os, cv2, random
from pathlib import Path
from albumentations import (
    Compose, Rotate, HorizontalFlip, RandomBrightnessContrast,
    ISONoise, OneOf, GaussianBlur, MotionBlur, MedianBlur
)

from notebooks.Marx.augmentation.data_utils import ensure_dataset



EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp",".JPG",".PNG"}

def build_aug(max_rotate_deg=10, hflip_p=0.5, contrast_limit=0.5,
              noise_p=0.5, blur_p=0.5):
    return Compose([
        OneOf([
            GaussianBlur(blur_limit=(3,7), p=1.0),
            MotionBlur(blur_limit=(3,7),  p=1.0),
            MedianBlur(blur_limit=5,      p=1.0),
        ], p=blur_p),
        ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=noise_p),
        Rotate(limit=max_rotate_deg, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        HorizontalFlip(p=hflip_p),
        RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=contrast_limit, p=1.0),
    ])

def find_images(root: Path):
    return sorted([p for p in root.rglob("*") if p.suffix in EXTS])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--per_image_variants", type=int, default=3,
                    help="How many augmented versions per original")
    ap.add_argument("--num_images", type=int, default=3,
                    help="How many originals to preview")
    ap.add_argument("--random", action="store_true",
                    help="Pick random originals instead of first N")
    args = ap.parse_args()

    src_dir = ensure_dataset(args.data_dir)  # auto-download if empty
    src = Path(src_dir)

    dst = Path(args.out_dir); dst.mkdir(parents=True, exist_ok=True)

    tfm = build_aug()

    imgs = find_images(src)
    if not imgs:
        print(f"[preview] No images found under {src}")
        return

    # choose originals
    if args.random:
        chosen = random.sample(imgs, min(args.num_images, len(imgs)))
    else:
        chosen = imgs[:args.num_images]

    for path in chosen:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        stem = path.stem

        # save ORIGINAL
        cv2.imwrite(str(dst / f"{stem}__orig.jpg"), bgr)

        # save VARIANTS
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for k in range(1, args.per_image_variants + 1):
            aug_rgb = tfm(image=rgb)["image"]
            aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(dst / f"{stem}__aug{k}.jpg"), aug_bgr)

    print(f"[preview] Saved {len(chosen)} originals + {args.per_image_variants} variants each in '{dst}'")

if __name__ == "__main__":
    main()
