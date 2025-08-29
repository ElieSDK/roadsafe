import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import folium
from streamlit_folium import st_folium
import piexif
import io
import smtplib
from email.message import EmailMessage
from geopy.geocoders import Nominatim
from email.header import Header

# ---------------- Config ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATHS = {
    "ResNet50": "best_model.pth",
    "EfficientNet": "efficientnet.pth"
}

SURFACE_TYPE_MAP = {"asphalt": 0, "concrete": 1, "paving_stones": 2, "unpaved": 3, "sett": 4}
SURFACE_TYPE_MAP_INV = {v: k for k, v in SURFACE_TYPE_MAP.items()}

SURFACE_QUALITY_MAP = {"excellent": 0, "good": 1, "intermediate": 2, "bad": 3, "very_bad": 4}
SURFACE_QUALITY_MAP_INV = {v: k for k, v in SURFACE_QUALITY_MAP.items()}


# ---------------- Model Loader ----------------
@st.cache_resource
def load_model(model_choice):
    checkpoint_path = MODEL_PATHS[model_choice]
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if model_choice == "ResNet50":
        backbone = models.resnet50(weights=None)
        in_feat = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif model_choice == "EfficientNet":
        backbone = models.efficientnet_b0(weights=None)
        in_feat = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

    backbone = backbone.to(device)
    fc_main = nn.Linear(in_feat, len(SURFACE_TYPE_MAP)).to(device)
    fc_sub = nn.Linear(in_feat, len(SURFACE_QUALITY_MAP)).to(device)

    if isinstance(checkpoint, dict) and "backbone" in checkpoint:
        backbone.load_state_dict(checkpoint["backbone"])
        fc_main.load_state_dict(checkpoint["fc_main"])
        fc_sub.load_state_dict(checkpoint["fc_sub"])
    else:
        backbone.load_state_dict(checkpoint, strict=False)

    backbone.eval()
    fc_main.eval()
    fc_sub.eval()
    return backbone, fc_main, fc_sub


# ---------------- Transform ----------------
transform = transforms.Compose([
    transforms.Resize((288, 512)),
    transforms.ToTensor()
])


# ---------------- GPS Extraction with piexif ----------------
def get_gps_coords_from_bytes(file_bytes):
    """Return (lat, lon) from JPEG bytes or None"""
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
        if lat_ref == "S":
            lat = -lat
        if lon_ref == "W":
            lon = -lon
        return lat, lon
    except:
        return None


# ---------------- Gmail Email Function ----------------
def send_email(to_email, subject, body, attachment_bytes=None, attachment_name="image.jpg"):
    import smtplib
    from email.message import EmailMessage

    gmail_user = "@gmail.com"
    gmail_password = "passrd"  # Gmail App Password

    # Clean dynamic text to remove non-breaking spaces
    subject = str(subject).replace('\xa0', ' ')
    body = str(body).replace('\xa0', ' ')

    msg = EmailMessage()
    msg["From"] = gmail_user
    msg["To"] = to_email
    msg["Subject"] = subject

    # Set the email body as UTF-8
    msg.set_content(body, charset="utf-8")

    # Attach image safely
    if attachment_bytes:
        msg.add_attachment(
            attachment_bytes,
            maintype="image",
            subtype="jpeg",
            filename=attachment_name
        )

    try:
        # Connect using SSL
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_user, gmail_password)
            server.send_message(msg)
        return True, "Email sent successfully"
    except Exception as e:
        return False, str(e)


# ---------------- Streamlit UI ----------------
st.title("Street Surface Classification & GPS")
st.write("Upload an image, choose a model, predict surface type & quality, and optionally send to city ward.")

# Model selector
model_choice = st.selectbox("Choose Model:", list(MODEL_PATHS.keys()))
backbone, fc_main, fc_sub = load_model(model_choice)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])
if uploaded_file is not None:
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ---------------- Prediction ----------------
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = backbone(img_tensor)
        out_main = fc_main(features)
        out_sub = fc_sub(features)
        main_pred = SURFACE_TYPE_MAP_INV[out_main.argmax(1).item()]
        sub_pred = SURFACE_QUALITY_MAP_INV[out_sub.argmax(1).item()]

    st.success(f"**Model:** {model_choice}")
    st.success(f"Predicted Surface Type: **{main_pred}**")
    st.success(f"Predicted Surface Quality: **{sub_pred}**")

    # ---------------- GPS & Map ----------------
    coords = get_gps_coords_from_bytes(file_bytes)
    if coords:
        lat, lon = coords
        m = folium.Map(location=[lat, lon], zoom_start=16)
        folium.Marker([lat, lon], tooltip="Uploaded Image Location").add_to(m)
        st_folium(m, width=700, height=500)

        geolocator = Nominatim(user_agent="street_app")
        try:
            location = geolocator.reverse((lat, lon), language='en')
            city_name = location.raw.get("address", {}).get("city") or location.raw.get("address", {}).get("town") or location.raw.get("address", {}).get("village")
        except:
            city_name = None

        if city_name:
            st.info(f"City detected: **{city_name}**")
            csv_df = pd.read_csv("csv.csv")  # must have 'city' and 'email' columns
            if city_name in csv_df['city'].values:
                if st.button(f"Send this picture to {city_name} ward?"):
                    to_email = csv_df.loc[csv_df['city'] == city_name, 'email'].values[0]
                    subject = f"Street Surface Report for {city_name}"
                    body = "Please find attached the street surface report image."
                    uploaded_file.seek(0)
                    success, info = send_email(to_email, subject, body, uploaded_file.read(), attachment_name="street_image.jpg")
                    if success:
                        st.success(f"Email sent to {to_email}!")
                    else:
                        st.error(f"Failed to send email: {info}")
    else:
        st.warning("No GPS metadata found in image.")
