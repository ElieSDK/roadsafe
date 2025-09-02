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

# ---------------- Load environment variables ----------------
GMAIL_USER = st.secrets.get("gmail", {}).get("user")
GMAIL_PASSWORD = st.secrets.get("gmail", {}).get("app_password")

# ---------------- Config ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SURFACE_TYPE_MAP = {"asphalt": 0, "concrete": 1, "paving_stones": 2, "unpaved": 3, "sett": 4}
SURFACE_TYPE_MAP_INV = {v: k for k, v in SURFACE_TYPE_MAP.items()}

SURFACE_QUALITY_MAP = {"excellent": 0, "good": 1, "intermediate": 2, "bad": 3, "very_bad": 4}
SURFACE_QUALITY_MAP_INV = {v: k for k, v in SURFACE_QUALITY_MAP.items()}

NUM_MATERIALS, NUM_QUALITIES = 5, 5

# ---------------- Load CSV ----------------
def load_emails():
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "city.csv")
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        st.error(f"Could not read city.csv: {e}")
        return pd.DataFrame(columns=["city", "email"])

EMAIL_DF = load_emails()

# ---------------- Multi-head EfficientNet-B7 ----------------
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

# ---------------- Model Loader ----------------
@st.cache_resource
def load_model():
    try:
        model = MultiHeadEffNetB7().to(device)
        local_path = hf_hub_download(
            repo_id="esdk/my-efficientnet-model",
            filename="efficientnet_fp16.pt.gz"
        )
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

# ---------------- Transform ----------------
def get_transform():
    return transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# ---------------- GPS & EXIF ----------------
def get_gps_coords_from_bytes(file_bytes):
    try:
        exif_dict = piexif.load(file_bytes)
        gps = exif_dict.get("GPS")
        if not gps:
            return None
        def rational_to_deg(rat):
            return rat[0][0]/rat[0][1] + rat[1][0]/rat[1][1]/60 + rat[2][0]/rat[2][1]/3600
        lat = rational_to_deg(gps[piexif.GPSIFD.GPSLatitude])
        lon = rational_to_deg(gps[piexif.GPSIFD.GPSLongitude])
        lat_ref = gps[piexif.GPSIFD.GPSLatitudeRef].decode()
        lon_ref = gps[piexif.GPSIFD.GPSLongitudeRef].decode()
        if lat_ref == "S": lat = -lat
        if lon_ref == "W": lon = -lon
        return lat, lon
    except:
        return None

def get_image_timestamp(file_bytes):
    try:
        exif_dict = piexif.load(file_bytes)
        dt_bytes = exif_dict.get("0th").get(piexif.ImageIFD.DateTime)
        if dt_bytes:
            return dt_bytes.decode()
        return None
    except:
        return None

# ---------------- Email ----------------
def send_email(to_email, subject, body, attachment_bytes=None, attachment_name="image.jpg"):
    if not GMAIL_USER or not GMAIL_PASSWORD:
        return False, "Email credentials not found. Please set them in Streamlit secrets."
    msg = EmailMessage()
    msg["From"] = GMAIL_USER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body, charset="utf-8")
    if attachment_bytes:
        msg.add_attachment(
            attachment_bytes,
            maintype="image",
            subtype="jpeg",
            filename=attachment_name
        )
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            server.send_message(msg)
        return True, "Email sent successfully"
    except Exception as e:
        return False, str(e)

# ---------------- Streamlit UI ----------------
st.title("Street Surface Classification & GPS")

# ---------------- Session State ----------------
if "marker" not in st.session_state:
    st.session_state.marker = None
if "lat_lon" not in st.session_state:
    st.session_state.lat_lon = (None, None)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ---------------- Prediction ----------------
    with st.spinner("Predicting..."):
        transform = get_transform()
        model = load_model()
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            out_type, out_qual = model(img_tensor)
            main_pred = SURFACE_TYPE_MAP_INV[out_type.argmax(1).item()]
            sub_pred = SURFACE_QUALITY_MAP_INV[out_qual.argmax(1).item()]

    # Format predictions
    main_pred_fmt = main_pred.replace("_", " ").capitalize()
    sub_pred_fmt = sub_pred.replace("_", " ").capitalize()

    st.success(f"Predicted Surface Type: **{main_pred_fmt}**")
    st.success(f"Predicted Surface Quality: **{sub_pred_fmt}**")

    if sub_pred in ["excellent", "good"]:
        st.warning("The road seems to be in good condition. Are you sure you want to report it?")

    # ---------------- Timestamps ----------------
    img_timestamp = get_image_timestamp(file_bytes)
    upload_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---------------- GPS ----------------
    coords = get_gps_coords_from_bytes(file_bytes)
    if coords:
        st.session_state.marker = coords
        st.session_state.lat_lon = coords
        st.success(f"GPS metadata found: {coords[0]}, {coords[1]}")
    else:
        st.session_state.marker = None         # <-- reset marker
        st.session_state.lat_lon = (None, None) # <-- reset lat/lon
        st.info("No GPS metadata found. Showing Tokyo map.")

    # ---------------- Folium Map ----------------
    map_center = coords if coords else [35.68, 139.76]  # Tokyo if no coords
    zoom_level = 16 if coords else 12
    m = folium.Map(location=map_center, zoom_start=zoom_level)

    marker_layer = folium.FeatureGroup(name="marker_layer")
    if st.session_state.marker:
        folium.Marker(
            st.session_state.marker,
            tooltip="Detected Location",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(marker_layer)
    marker_layer.add_to(m)

    map_data = st_folium(m, width=700, height=500)

    if map_data and map_data.get("last_clicked"):
        st.session_state.marker = (map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"])

    # ---------------- Reverse Geocoding ----------------
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
            street_name, city_name = None, None

    if street_name:
        st.info(f"Street: {street_name}")
    if city_name:
        st.info(f"City: {city_name}")

    # ---------------- Email ----------------
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
        if success:
            st.success("Report sent successfully!")
        else:
            st.error(f"Failed to send email: {info}")

# ---------------- Footer ----------------
st.markdown(
    """
    <div style="margin-top: 50px; font-size: 14px; text-align: center;">
        <a href="https://www.linkedin.com/in/arina-w/" target="_blank">Wahab Arina</a> |
        <a href="https://www.linkedin.com/in/eliesdk" target="_blank">Sadaka Elie</a> |
        <a href="https://github.com/Marxi7" target="_blank">Scuderi Marcello</a>
    </div>
    """,
    unsafe_allow_html=True
)
