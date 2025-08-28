# notebooks/Marx/augmentation/data_utils.py
import os, zipfile, io, requests, shutil

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def _has_images(path: str, min_files: int) -> bool:
    count = 0
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                count += 1
                if count >= min_files:
                    return True
    return False

def ensure_dataset(
    base_dir: str = "notebooks/Marx/augmentation/unknown",
    url: str = "https://zenodo.org/records/11449977/files/s_1024.zip?download=1",
    min_files: int = 100,
) -> str:
    """
    Ensure dataset exists under <base_dir>/s_1024.
    - If images already there: return the path.
    - Else: download ZIP from `url` and extract, handling both
      cases where the ZIP contains a top-level 's_1024/' or raw files.
    Returns the final dataset directory path.
    """
    final_dir = os.path.join(base_dir, "s_1024")
    if _has_images(final_dir, min_files):
        return final_dir

    os.makedirs(base_dir, exist_ok=True)
    tmp_extract_dir = os.path.join(base_dir, "_extract_tmp")
    os.makedirs(tmp_extract_dir, exist_ok=True)

    print(f"[data] Downloading dataset to {tmp_extract_dir} ...")
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(tmp_extract_dir)

    # If ZIP contains a top-level 's_1024/' keep only its contents
    candidate = os.path.join(tmp_extract_dir, "s_1024")
    src_dir = candidate if os.path.isdir(candidate) else tmp_extract_dir

    # Move/merge into final_dir
    os.makedirs(final_dir, exist_ok=True)
    moved = 0
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                src = os.path.join(root, f)
                rel = os.path.relpath(root, src_dir)
                dst_sub = os.path.join(final_dir, rel)
                os.makedirs(dst_sub, exist_ok=True)
                shutil.move(src, os.path.join(dst_sub, f))
                moved += 1

    shutil.rmtree(tmp_extract_dir, ignore_errors=True)
    print(f"[data] Placed {moved} images into {final_dir}")
    return final_dir
