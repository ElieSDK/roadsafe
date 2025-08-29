# app.py
import io
import re
import json
import requests
from pathlib import Path
from datetime import datetime

import streamlit as st
from PIL import Image, ExifTags
from streamlit_js_eval import get_geolocation
from streamlit_folium import st_folium
import folium

# ---------- Page config ----------
st.set_page_config(page_title="Photo + Location Uploader", page_icon="📍", layout="centered")

# ---------- CSS (remove stray oval + styling) ----------
st.markdown(
    """
    <style>
      .big-title {font-size: 2.2rem; font-weight: 800; margin-bottom: 0.25rem;}
      .subtle {opacity: 0.7; margin-top: 0; margin-bottom: 1rem;}
      .card {background: #111; border: 1px solid #2a2a2a; border-radius: 18px; padding: 18px;}

      /* Hide any empty text inputs that sometimes render as a blank oval */
      div[data-testid="stTextInput"]:has(input:placeholder-shown) { display:none !important; }
      div[data-testid="stTextInput"] input:placeholder-shown { display:none !important; }

      div[data-testid="stFileUploader"] { margin-top: 0.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown(
    """
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

uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

# ---------- Session state (init once) ----------
st.session_state.setdefault("picked_latlng", None)     # list[lat, lon]
st.session_state.setdefault("coords_source", None)     # "browser" | "exif" | "manual"
st.session_state.setdefault("latlon_ready", None)      # tuple(lat, lon)
st.session_state.setdefault("exif_datetime", None)
st.session_state.setdefault("cur_file_sig", None)      # signature of current uploaded file

# ---------- EXIF helpers ----------
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

# ---------- Ward/city office lookup via OSM Overpass ----------
OSM_HEADERS = {"User-Agent": "roadsafe/1.0 (support@roadsafe.local)"}

@st.cache_data(show_spinner=False)
def _overpass(lat, lon, radius_m=2000):
    q = f"""
    [out:json][timeout:25];
    (
      node(around:{radius_m},{lat},{lon})[amenity=townhall];
      way(around:{radius_m},{lat},{lon})[amenity=townhall];
      relation(around:{radius_m},{lat},{lon})[amenity=townhall];

      node(around:{radius_m},{lat},{lon})[office=government][government=administrative];
      way(around:{radius_m},{lat},{lon})[office=government][government=administrative];
      relation(around:{radius_m},{lat},{lon})[office=government][government=administrative];
    );
    out center tags 20;
    """
    r = requests.post("https://overpass-api.de/api/interpreter", data=q.encode("utf-8"), headers=OSM_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json().get("elements", [])

def _haversine(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2
    R = 6371000.0
    dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*atan2(sqrt(a), sqrt(1-a))

def _extract_contact(tags):
    name = tags.get("name")
    addr = tags.get("addr:full")
    if not addr:
        parts = [tags.get(k, "") for k in [
            "addr:postcode","addr:state","addr:city","addr:district","addr:suburb","addr:street","addr:housenumber"
        ]]
        addr = " ".join([p for p in parts if p]).strip() or None
    email = tags.get("contact:email") or tags.get("email")
    website = tags.get("contact:website") or tags.get("website")
    phone = tags.get("contact:phone") or tags.get("phone")
    return name, addr, email, website, phone

def _scrape_email(website):
    try:
        html = requests.get(website, headers=OSM_HEADERS, timeout=10).text
        m = re.search(r"mailto:([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", html)
        return m.group(1) if m else None
    except Exception:
        return None

def find_nearest_office(lat, lon):
    for rad in (2000, 5000, 10000):
        elems = _overpass(lat, lon, rad)
        if not elems:
            continue

        def center_of(e):
            if "lat" in e and "lon" in e: return e["lat"], e["lon"]
            if "center" in e: return e["center"]["lat"], e["center"]["lon"]
            return None

        candidates = []
        for e in elems:
            c = center_of(e)
            if not c:
                continue
            d = _haversine(lat, lon, c[0], c[1])
            candidates.append((d, e))
        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0])
        best = candidates[0][1]
        tags = best.get("tags", {})
        name, addr, email, website, phone = _extract_contact(tags)
        if not email and website:
            email = _scrape_email(website)

        return {
            "name": name,
            "address": addr,
            "email": email,
            "website": website,
            "phone": phone,
            "distance_m": int(candidates[0][0]),
            "osm_id": best.get("id"),
            "osm_tags": tags,
        }
    return None

# ---------- Uploader ----------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    st.caption("Drop an image here (PNG/JPG/JPEG)")
    st.markdown("</div>", unsafe_allow_html=True)

def _file_signature(uploaded_file):
    # robust per-file signature → only reset state when this changes
    data = uploaded_file.getvalue()
    return f"{uploaded_file.name}:{len(data)}", data

def save_image_and_meta(image: Image.Image, lat: float, lon: float, source: str, acc=None, ward_info=None):
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
        "ward_office": ward_info or {},
    }
    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    st.success(f"Saved ✅  ({lat:.6f}, {lon:.6f}) · source={source}")

# ---------- Main flow ----------
if file is not None:
    # Compute file signature and only reset state if it's a different upload
    cur_sig, raw_bytes = _file_signature(file)
    if st.session_state.cur_file_sig != cur_sig:
        # new file → clear per-upload state ONCE
        st.session_state.cur_file_sig = cur_sig
        st.session_state.latlon_ready = None
        st.session_state.coords_source = None
        st.session_state.picked_latlng = None
        st.session_state.exif_datetime = None

    # Load PIL image from the raw bytes (so we can reuse bytes for signature)
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    st.image(image, caption="Preview", use_container_width=True)

    # Step 1: Browser geolocation
    with st.spinner("Requesting your location from the browser..."):
        loc = get_geolocation()
    if loc and isinstance(loc, dict) and "coords" in loc and loc["coords"].get("latitude") is not None:
        coords = loc["coords"]
        lat, lon = coords.get("latitude"), coords.get("longitude")
        acc = coords.get("accuracy")
        st.session_state.latlon_ready = (lat, lon)
        st.session_state.coords_source = "browser"

        ward = find_nearest_office(lat, lon)

        st.info(f"📍 Browser location: {lat:.6f}, {lon:.6f} (± ~{int(acc) if acc else '—'} m)")
        if ward:
            with st.expander("Closest ward/city office (auto)", expanded=True):
                st.write(ward)
        save_image_and_meta(image, lat, lon, "browser", acc=acc, ward_info=ward)
        st.stop()

    # Step 2: EXIF metadata
    st.session_state.exif_datetime, exif_lat, exif_lon = extract_exif_datetime_gps(image)
    if exif_lat is not None and exif_lon is not None:
        st.session_state.latlon_ready = (exif_lat, exif_lon)
        st.session_state.coords_source = "exif"

        ward = find_nearest_office(exif_lat, exif_lon)

        st.info(f"🧭 EXIF location found: {exif_lat:.6f}, {exif_lon:.6f}")
        if ward:
            with st.expander("Closest ward/city office (from EXIF)", expanded=True):
                st.write(ward)
        save_image_and_meta(image, exif_lat, exif_lon, "exif", ward_info=ward)
        st.stop()

    # Step 3: Manual pin (only reached if Step 1 and 2 failed)
    st.warning("No browser location and no EXIF GPS. Please drop a pin on the map.")
    default_center = [35.681236, 139.767125]  # Tokyo Station

    # Build map centered on last pick or default.
    m = folium.Map(location=st.session_state.picked_latlng or default_center, zoom_start=12)

    # Click helper to make older streamlit-folium versions record coordinates
    folium.LatLngPopup().add_to(m)

    # If already picked, show the marker.
    if st.session_state.picked_latlng:
        folium.Marker(
            st.session_state.picked_latlng,
            popup="Selected location",
            tooltip="Your chosen spot"
        ).add_to(m)

    # Render map and read last click (works on old versions too)
    map_event = st_folium(m, width=700, height=420, key="manual_map")

    # Capture click → store in session_state (no manual st.rerun())
    if map_event and map_event.get("last_clicked"):
        lat = map_event["last_clicked"]["lat"]
        lon = map_event["last_clicked"]["lng"]
        st.session_state.picked_latlng = [lat, lon]

    # Live feedback
    if st.session_state.picked_latlng:
        lat, lon = st.session_state.picked_latlng
        st.info(f"Selected: {lat:.6f}, {lon:.6f}")

    st.markdown("**Tip:** Click on the map to choose a location, then press ‘Use this location’.")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        use_btn = st.button("✅ Use this location", disabled=st.session_state.picked_latlng is None)

    if use_btn and st.session_state.picked_latlng:
        lat, lon = st.session_state.picked_latlng
        st.session_state.latlon_ready = (lat, lon)
        st.session_state.coords_source = "manual"

        ward = find_nearest_office(lat, lon)
        if ward:
            with st.expander("Closest ward/city office (manual)", expanded=True):
                st.write(ward)

        save_image_and_meta(image, lat, lon, "manual", ward_info=ward)
        st.stop()
