# notebooks/Marx/augmentation/preview.py
#!/usr/bin/env python3
import argparse, cv2, random
from pathlib import Path
from collections import Counter

import pandas as pd
from albumentations import (
    Compose, Rotate, HorizontalFlip, RandomBrightnessContrast,
    ISONoise, OneOf, GaussianBlur, MotionBlur, MedianBlur
)

# Import label maps from the supervised module
from notebooks.Marx.augmentation.supervised import (
    SURFACE_TYPE_MAP, SURFACE_QUALITY_MAP
)

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".JPG", ".PNG"}

# Pretty names for printing
TYPE_NAMES = {v: k for k, v in SURFACE_TYPE_MAP.items()}
QUAL_NAMES = {v: k for k, v in SURFACE_QUALITY_MAP.items()}


def build_aug(max_rotate_deg=10, hflip_p=0.5, contrast_limit=0.5,
              noise_p=0.5, blur_p=0.5) -> Compose:
    """Augmentation recipe used for preview (mirrors training)."""
    return Compose([
        OneOf([
            GaussianBlur(blur_limit=(3, 7), p=1.0),
            MotionBlur(blur_limit=(3, 7),  p=1.0),
            MedianBlur(blur_limit=5,       p=1.0),
        ], p=blur_p),
        ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=noise_p),
        Rotate(limit=max_rotate_deg, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        HorizontalFlip(p=hflip_p),
        RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=contrast_limit, p=1.0),
    ])


def find_images(root: Path):
    """Recursively gather image files under root."""
    return sorted([p for p in root.rglob("*") if p.suffix in EXTS])


def load_label_map(csv_path: Path):
    """
    Build: stem -> (type_id, qual_id) from CSV.
    Expects columns: mapillary_image_id, surface_type, surface_quality
    """
    df = pd.read_csv(csv_path)
    m = {}
    if all(c in df.columns for c in ("mapillary_image_id", "surface_type", "surface_quality")):
        for _, r in df.iterrows():
            stem = str(r["mapillary_image_id"])
            t = SURFACE_TYPE_MAP.get(str(r["surface_type"]).lower(), None)
            q = SURFACE_QUALITY_MAP.get(str(r["surface_quality"]).lower(), None)
            if t is not None and q is not None:
                m[stem] = (t, q)
    return m


def filename_with_labels(stem: str, t: int | None, q: int | None,
                         kind: str, use_names: bool) -> str:
    """
    Build output filename:
      <stem>__<kind>[__typeX_qualY]  (ids)
      or
      <stem>__<kind>[__typeName_qualName] (names) if use_names=True
    """
    suffix = ""
    if t is not None and q is not None:
        if use_names:
            suffix = f"__{TYPE_NAMES.get(t, t)}_{QUAL_NAMES.get(q, q)}"
        else:
            suffix = f"__type{t}_qual{q}"
    return f"{stem}__{kind}{suffix}.jpg"


def main():
    ap = argparse.ArgumentParser(description="Preview virtual augmentations with optional labels.")
    ap.add_argument("--data_dir", required=True, help="Root folder with images.")
    ap.add_argument("--out_dir", required=True, help="Where to save preview images.")
    ap.add_argument("--per_image_variants", type=int, default=3,
                    help="How many augmented versions per original (default: 3).")
    ap.add_argument("--num_images", type=int, default=4,
                    help="How many originals to preview (default: 4).")
    ap.add_argument("--random", action="store_true",
                    help="Pick random originals instead of first N.")
    ap.add_argument("--csv_path", type=str, default=None,
                    help="Optional CSV to attach labels (surface_type, surface_quality).")
    ap.add_argument("--names_in_filenames", action="store_true",
                    help="If set, include class names in filenames instead of numeric ids.")
    args = ap.parse_args()

    src = Path(args.data_dir)
    dst = Path(args.out_dir); dst.mkdir(parents=True, exist_ok=True)
    tfm = build_aug()

    imgs = find_images(src)
    if not imgs:
        print(f"[preview] No images found under {src}")
        return

    # Optional label map from CSV
    label_map = {}
    if args.csv_path:
        csv_path = Path(args.csv_path)
        if csv_path.exists():
            label_map = load_label_map(csv_path)
        else:
            print(f"[preview] CSV not found at {csv_path}; continuing without labels.")

    # Pick originals
    chosen = (random.sample(imgs, min(args.num_images, len(imgs)))
              if args.random else imgs[:args.num_images])

    saved_info = []  # (filename, type_id, qual_id)
    for path in chosen:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        stem = path.stem

        # Optional labels
        t, q = label_map.get(stem, (None, None))

        # Save ORIGINAL
        orig_name = filename_with_labels(stem, t, q, "orig", args.names_in_filenames)
        cv2.imwrite(str(dst / orig_name), bgr)
        saved_info.append((orig_name, t, q))

        # Save VARIANTS
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for k in range(1, args.per_image_variants + 1):
            aug_rgb = tfm(image=rgb)["image"]
            aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)
            var_name = filename_with_labels(stem, t, q, f"aug{k}", args.names_in_filenames)
            cv2.imwrite(str(dst / var_name), aug_bgr)
            saved_info.append((var_name, t, q))

    # Summaries
    types = Counter([t for _, t, _ in saved_info if t is not None])
    quals = Counter([q for _, _, q in saved_info if q is not None])
    print(f"[preview] wrote {len(saved_info)} images to '{dst}'")

    if types:
        human_types = {TYPE_NAMES.get(k, str(k)): v for k, v in types.items()}
        print(f"[preview] type counts: {human_types}")
    if quals:
        human_quals = {QUAL_NAMES.get(k, str(k)): v for k, v in quals.items()}
        print(f"[preview] quality counts: {human_quals}")


if __name__ == "__main__":
    main()
