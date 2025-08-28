
# 📍 Streamlit Photo + Location Uploader

A minimal, beautiful Streamlit front‑end where users can upload a photo and, at the same time, the app requests the browser for their **current GPS coordinates** (with permission). The image and a small JSON metadata file are saved to `./uploads/`.

## ✨ What it does
- Lets the user upload an image (PNG/JPG/JPEG)
- Requests **HTML5 Geolocation** via the browser (permission prompt)
- Displays the coordinates and a map preview
- Saves the image with a filename including timestamp + lat/lon
- Writes a JSON sidecar with the metadata

## 🚀 Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # or use your preferred env
pip install -r requirements.txt
streamlit run app.py
```

> If your browser blocks geolocation, enable it for `localhost`/the domain in site settings and reload.

## 🛡️ Privacy
- Geolocation is requested by your browser and shared only if you allow it.
- Everything is stored locally in the `uploads/` folder.
