import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
from PIL import Image
import io
import gzip
import piexif
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import smtplib
from email.message import EmailMessage
from datetime import datetime
from huggingface_hub import hf_hub_download
import os

# ------------------- CONFIG -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SURFACE_TYPE_MAP = {"asphalt": 0, "concrete": 1, "paving_stones": 2, "unpaved": 3, "sett": 4}
SURFACE_TYPE_MAP_INV = {v: k for k, v in SURFACE_TYPE_MAP.items()}

SURFACE_QUALITY_MAP = {"excellent": 0, "good": 1, "intermediate": 2, "bad": 3, "very_bad": 4}
SURFACE_QUALITY_MAP_INV = {v: k for k, v in SURFACE_QUALITY_MAP.items()}

NUM_MATERIALS, NUM_QUALITIES = 5, 5

GMAIL_USER = st.secrets.get("gmail", {}).get("user")
GMAIL_PASSWORD = st.secrets.get("gmail", {}).get("app_password")

# ------------------- STYLES -------------------
st.markdown("""
<style>
/* General text */
body, p, div, .stText, .stMarkdown, .st-ag {
    font-size: 24px !important;
    line-height: 1.8 !important;
}

/* Alerts */
.stAlert, .stAlert * {
    font-size: 26px !important;
}

/* Buttons & uploaders */
.stFileUploader label, .stFileUploader button, .stButton button {
    font-size: 22px !important;
    padding: 0.75rem 1.5rem !important;
}

/* Map popup */
.leaflet-popup-content {
    font-size: 20px !important;
}

/* Colored info boxes */
.responsive-box {
    max-width: 100%;
    font-size: 32px !important;        /* bigger font */
    font-weight: bold !important;      /* bold text */
    padding: 20px 25px !important;    /* more padding */
    border-radius: 10px !important;
    color: white;
    margin-bottom: 20px !important;
    text-align: left;                  /* left-align the text */
}

/* Headers */
h1 {
    font-size: 44px !important;
}
h2 {
    font-size: 40px !important;
}

/* Footer links */
.footer a {
    font-size: 20px !important;
    text-decoration: none !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------- UTILITIES -------------------
def load_emails():
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "city.csv")
        return pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Could not read city.csv: {e}")
        return pd.DataFrame(columns=["city", "email"])

EMAIL_DF = load_emails()

# ------------------- MODEL -------------------
class MultiHeadEffNetB7(nn.Module):
    def __init__(self):
        super().__init__()
        base = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(base.children())[:-1])
        f = base.classifier[1].in_features
        self.mat = nn.Linear(f, NUM_MATERIALS)
        self.qual = nn.Linear(f, NUM_QUALITIES)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.mat(x), self.qual(x)

@st.cache_resource
def load_model():
    try:
        model = MultiHeadEffNetB7().to(device)
        local_path = hf_hub_download(repo_id="esdk/my-efficientnet-model", filename="efficientnet_fp16.pt.gz")
        with gzip.open(local_path, "rb") as f:
            buffer = io.BytesIO(f.read())
            state_dict = torch.load(buffer, map_location=device)
        fixed_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(fixed_state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def get_transform():
    return transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_gps_coords_from_bytes(file_bytes):
    try:
        exif_dict = piexif.load(file_bytes)
        gps = exif_dict.get("GPS")
        if not gps: return None
        def rat2deg(rat): return rat[0][0]/rat[0][1] + rat[1][0]/rat[1][1]/60 + rat[2][0]/rat[2][1]/3600
        lat = rat2deg(gps[piexif.GPSIFD.GPSLatitude])
        lon = rat2deg(gps[piexif.GPSIFD.GPSLongitude])
        if gps[piexif.GPSIFD.GPSLatitudeRef].decode() == "S": lat = -lat
        if gps[piexif.GPSIFD.GPSLongitudeRef].decode() == "W": lon = -lon
        return lat, lon
    except: return None

def get_image_timestamp(file_bytes):
    try:
        exif_dict = piexif.load(file_bytes)
        dt_bytes = exif_dict.get("0th").get(piexif.ImageIFD.DateTime)
        return dt_bytes.decode() if dt_bytes else None
    except: return None

def send_email(to_email, subject, body, attachment_bytes=None, attachment_name="image.jpg"):
    if not GMAIL_USER or not GMAIL_PASSWORD:
        return False, "Email credentials not found."
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = GMAIL_USER, to_email, subject
    msg.set_content(body, charset="utf-8")
    if attachment_bytes:
        msg.add_attachment(attachment_bytes, maintype="image", subtype="jpeg", filename=attachment_name)
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            server.send_message(msg)
        return True, "Email sent successfully"
    except Exception as e:
        return False, str(e)

# ------------------- UI -------------------
st.markdown("<h1 style='text-align:center;'>Street Surface Classification & GPS</h1>", unsafe_allow_html=True)

if "marker" not in st.session_state: st.session_state.marker = None
if "lat_lon" not in st.session_state: st.session_state.lat_lon = (None, None)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# --- Helper to display colored blocks ---
def display_surface_info(surface_type, surface_quality):
    st.markdown(f"""
        <div class="responsive-box" style="background-color:#198754;">
            <b>Surface Type:</b> {surface_type}
        </div>
    """, unsafe_allow_html=True)

    color_map = {
        "excellent": "#198754",
        "good": "#198754",
        "intermediate": "#ffc107",
        "bad": "#dc3545",
        "very_bad": "#dc3545"
    }
    color = color_map.get(surface_quality.lower(), "#6c757d")
    st.markdown(f"""
        <div class="responsive-box" style="background-color:{color};">
            <b>Surface Quality:</b> {surface_quality}
        </div>
    """, unsafe_allow_html=True)

# --- Main app logic ---
if uploaded_file:
    file_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.markdown("<h2 style='text-align:center;'>ANALYSIS</h2>", unsafe_allow_html=True)

    with st.spinner("Predicting..."):
        transform = get_transform()
        model = load_model()
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            out_type, out_qual = model(img_tensor)
            main_pred = SURFACE_TYPE_MAP_INV[out_type.argmax(1).item()]
            sub_pred = SURFACE_QUALITY_MAP_INV[out_qual.argmax(1).item()]

    main_pred_fmt = main_pred.replace("_", " ").capitalize()
    sub_pred_fmt = sub_pred.replace("_", " ").capitalize()

    display_surface_info(main_pred_fmt, sub_pred_fmt)

    if sub_pred in ["excellent", "good"]:
        st.warning("The road seems to be in good condition. Are you sure you want to report it?")

    # Timestamps
    img_timestamp = get_image_timestamp(file_bytes)
    upload_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # GPS
    st.markdown("<h2 style='text-align:center;'>GPS LOCATION</h2>", unsafe_allow_html=True)
    coords = get_gps_coords_from_bytes(file_bytes)

    if coords:
        st.session_state.marker = coords
        st.session_state.lat_lon = coords
        st.success(f"GPS metadata found: {coords[0]}, {coords[1]}")
    else:
        st.session_state.marker = None
        st.session_state.lat_lon = (None, None)
        st.info("No GPS metadata found. Please select a location on the map.")

    map_center = coords if coords else [3.1390, 101.6869]
    zoom_level = 16 if coords else 12
    m = folium.Map(location=map_center, zoom_start=zoom_level)

    if st.session_state.marker:
        folium.Marker(
            st.session_state.marker,
            tooltip="Detected Location",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)

    map_data = st_folium(m, width=900, height=600)
    if map_data and map_data.get("last_clicked"):
        st.session_state.marker = (map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"])

    # Reverse geocoding
    street_name, city_name = None, None
    if st.session_state.marker:
        lat, lon = st.session_state.marker
        st.info(f"Coordinates: {lat}, {lon}")
        try:
            geolocator = Nominatim(user_agent="street_app")
            location = geolocator.reverse((lat, lon), language='en')
            if location:
                addr = location.raw.get("address", {})
                street_name = addr.get("road")
                city_name = addr.get("city") or addr.get("town") or addr.get("suburb")
        except:
            pass

    if street_name: st.info(f"Street: {street_name}")
    if city_name:
        st.info(f"City: {city_name}")
        st.markdown("<h2 style='text-align:center;'>REPORT TO THE WARD CITY</h2>", unsafe_allow_html=True)

    # Email
    to_email = None
    if city_name:
        match = EMAIL_DF[EMAIL_DF["city"].str.lower() == city_name.lower()]
        if not match.empty:
            to_email = match.iloc[0]["email"]
            st.success(f"Sending report to: {to_email}")
        else:
            st.warning("No valid email found for this city. Cannot send report.")

    if to_email and st.button("Send Report via Email"):
        subject = "Street Surface Report"
        body = f"""Street Surface Report
Surface Type: {main_pred_fmt}
Surface Quality: {sub_pred_fmt}
Street Name: {street_name if street_name else 'Unknown'}
City: {city_name if city_name else 'Unknown'}
GPS: {lat if lat else 'Unknown'}, {lon if lon else 'Unknown'}
Picture taken: {img_timestamp if img_timestamp else 'Unknown'}
Upload time: {upload_timestamp}
"""
        success, info = send_email(to_email, subject, body, file_bytes, "street_image.jpg")
        if success: st.success("Report sent successfully!")
        else: st.error(f"Failed to send email: {info}")

# Footer
st.markdown("""
<div class="footer" style="margin-top: 50px; line-height:1.4; text-align:center;">
    <a href="https://www.linkedin.com/in/arina-w/" target="_blank">Wahab Arina</a>
    <span style="margin: 0 15px;">|</span>
    <a href="https://www.linkedin.com/in/eliesdk" target="_blank">Sadaka Elie</a>
    <span style="margin: 0 15px;">|</span>
    <a href="https://github.com/Marxi7" target="_blank">Scuderi Marcello</a>
</div>
""", unsafe_allow_html=True)
