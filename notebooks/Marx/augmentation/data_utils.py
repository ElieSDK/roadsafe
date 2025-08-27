# notebooks/Marx/augmentation/data_utils.py
import os, zipfile, io, requests

def ensure_dataset(out_dir: str,
                   url: str = "https://zenodo.org/records/11449977/files/s_1024.zip?download=1",
                   min_files: int = 100):
    """
    Download+unzip the dataset into out_dir if it looks empty.
    Returns the out_dir path.
    """
    # if already populated, do nothing
    files = []
    for root, _, fnames in os.walk(out_dir):
        for f in fnames:
            if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp")):
                files.append(1)
                if len(files) >= min_files:
                    return out_dir

    os.makedirs(out_dir, exist_ok=True)
    print(f"[data] Downloading dataset to {out_dir} ...")
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(out_dir)
    print(f"[data] Extracted {len(z.namelist())} entries into {out_dir}")
    return out_dir
