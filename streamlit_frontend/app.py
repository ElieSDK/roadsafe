import os
from pathlib import Path
from datetime import datetime

import streamlit as st
from PIL import Image, ExifTags
from streamlit_js_eval import get_geolocation
from streamlit_folium import st_folium
import folium

# ---------- Page config ----------
st.set_page_config(page_title="Photo + Location Uploader", page_icon="📍", layout="centered")

# ---------- Helpers: EXIF parsing ----------
def _to_deg(value):
    d = value[0][0] / value[0][1]
    m = value[1][0] / value[1][1]
    s = value[2][0] / value[2][1]
    return d + (m / 60.0) + (s / 3600.0)

def extract_exif_datetime_gps(pil_img):
    dt_original, lat, lon = None, None, None
    try:
        exif = pil_img._getexif()
        if not exif:
            return dt_original, lat, lon
        tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}

        dto = tag_map.get("DateTimeOriginal") or tag_map.get("DateTime")
        if dto:
            try:
                dt_original = datetime.strptime(dto, "%Y:%m:%d %H:%M:%S").isoformat()
            except Exception:
                dt_original = str(dto)

        gps_info = tag_map.get("GPSInfo")
        if gps_info:
            gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_info.items()}
            gps_lat = gps.get("GPSLatitude"); gps_lat_ref = gps.get("GPSLatitudeRef")
            gps_lon = gps.get("GPSLongitude"); gps_lon_ref = gps.get("GPSLongitudeRef")
            if gps_lat and gps_lat_ref and gps_lon and gps_lon_ref:
                lat = _to_deg(gps_lat);  lon = _to_deg(gps_lon)
                if gps_lat_ref in ["S", "s"]: lat = -lat
                if gps_lon_ref in ["W", "w"]: lon = -lon
    except Exception:
        pass
    return dt_original, lat, lon

# ---------- UI ----------
st.markdown(
    """
    <style>
    .big-title {font-size: 2.2rem; font-weight: 800; margin-bottom: 0.25rem;}
    .subtle {opacity: 0.7; margin-top: 0; margin-bottom: 1rem;}
    .card {background: #111; border: 1px solid #2a2a2a; border-radius: 18px; padding: 18px;}
    </style>
    <div class="big-title">📷 Upload a photo — auto-capture your location</div>
    <p class="subtle">We follow these steps to determine your location:</p>
    <ol>
        <li><b>Check browser location:</b> If authorized, use your current GPS.</li>
        <li><b>Check photo metadata:</b> If no browser location, extract GPS from EXIF.</li>
        <li><b>Manual pin:</b> If no EXIF either, drop a pin on the map yourself.</li>
    </ol>
    """,
    unsafe_allow_html=True,
)

uploads_dir = Path("uploads"); uploads_dir.mkdir(exist_ok=True)

# Session state
if "picked_latlng" not in st.session_state: st.session_state.picked_latlng = None
if "coords_source" not in st.session_state: st.session_state.coords_source = None
if "latlon_ready" not in st.session_state: st.session_state.latlon_ready = None  # (lat, lon)
if "exif_datetime" not in st.session_state: st.session_state.exif_datetime = None

# ---------- File Uploader ----------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    st.caption("Drop an image here (PNG/JPG/JPEG)")
    st.markdown("</div>", unsafe_allow_html=True)

def save_image_and_meta(image: Image.Image, lat: float, lon: float, source: str, acc=None):
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    fname = f"{ts}_{lat:.6f}_{lon:.6f}.jpg"
    out_path = uploads_dir / fname
    image.save(out_path, format="JPEG", quality=92)
    meta = {
        "timestamp_saved_utc": ts,
        "exif_datetime": st.session_state.exif_datetime,
        "latitude": lat,
        "longitude": lon,
        "accuracy_m": acc,
        "coordinates_source": source,
        "filename": fname,
    }
    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        import json as _json; _json.dump(meta, f, ensure_ascii=False, indent=2)
    st.success(f"Saved ✅  ({lat:.6f}, {lon:.6f}) · source={source}")

if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Preview", use_container_width=True)

    # Reset state for a fresh upload
    st.session_state.latlon_ready = None
    st.session_state.coords_source = None
    st.session_state.exif_datetime, exif_lat, exif_lon = extract_exif_datetime_gps(image)

    # Step 1: Browser geolocation
    with st.spinner("Requesting your location from the browser..."):
        loc = get_geolocation()
    if loc and isinstance(loc, dict) and "coords" in loc and loc["coords"].get("latitude") is not None:
        coords = loc["coords"]
        lat, lon = coords.get("latitude"), coords.get("longitude")
        acc = coords.get("accuracy")
        st.session_state.latlon_ready = (lat, lon)
        st.session_state.coords_source = "browser"
        st.info(f"📍 Browser location: {lat:.6f}, {lon:.6f} (± ~{int(acc) if acc else '—'} m)")
        save_image_and_meta(image, lat, lon, "browser", acc=acc)
        st.stop()

    # Step 2: EXIF metadata
    if exif_lat is not None and exif_lon is not None:
        st.session_state.latlon_ready = (exif_lat, exif_lon)
        st.session_state.coords_source = "exif"
        st.info(f"🧭 EXIF location found: {exif_lat:.6f}, {exif_lon:.6f}")
        save_image_and_meta(image, exif_lat, exif_lon, "exif")
        st.stop()

    # Step 3: Manual pin
    st.warning("No browser location and no EXIF GPS. Please drop a pin on the map.")
    default_center = [35.681236, 139.767125]  # Tokyo Station

    # Build the map with a marker if we already have one saved in session_state
    m = folium.Map(location=st.session_state.picked_latlng or default_center, zoom_start=12)
    if st.session_state.picked_latlng:
        folium.Marker(st.session_state.picked_latlng, popup="Selected location").add_to(m)

    st.markdown("**Tip:** Click on the map to choose a location, then press 'Use this location'.")
    map_event = st_folium(m, width=700, height=420, key="manual_map")

    if map_event and map_event.get("last_clicked"):
        lat = map_event["last_clicked"]["lat"]; lon = map_event["last_clicked"]["lng"]
        st.session_state.picked_latlng = [lat, lon]
        st.rerun()  # refresh to show the marker on the same map

    col_a, col_b = st.columns([1, 2])
    with col_a:
        use_btn = st.button("✅ Use this location", disabled=st.session_state.picked_latlng is None)

    if use_btn and st.session_state.picked_latlng:
        lat, lon = st.session_state.picked_latlng
        st.session_state.latlon_ready = (lat, lon)
        st.session_state.coords_source = "manual"
        save_image_and_meta(image, lat, lon, "manual")
        st.stop()
