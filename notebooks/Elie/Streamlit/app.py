import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io
import piexif
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import smtplib
from email.message import EmailMessage

# ---------------- Config ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATHS = {
    "ResNet50": "best_model.pth",
    "EfficientNet-B7": "efficientnet.pt"
}
DEFAULT_MODEL = "EfficientNet-B7"

SURFACE_TYPE_MAP = {"asphalt": 0, "concrete": 1, "paving_stones": 2, "unpaved": 3, "sett": 4}
SURFACE_TYPE_MAP_INV = {v: k for k, v in SURFACE_TYPE_MAP.items()}

SURFACE_QUALITY_MAP = {"excellent": 0, "good": 1, "intermediate": 2, "bad": 3, "very_bad": 4}
SURFACE_QUALITY_MAP_INV = {v: k for k, v in SURFACE_QUALITY_MAP.items()}

# ---------------- Multi-head Models ----------------
class MultiHeadResNet50(nn.Module):
    def __init__(self, num_types=len(SURFACE_TYPE_MAP), num_qual=len(SURFACE_QUALITY_MAP)):
        super().__init__()
        base = models.resnet50(weights=None)
        in_feat = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.fc_type = nn.Linear(in_feat, num_types)
        self.fc_qual = nn.Linear(in_feat, num_qual)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc_type(features), self.fc_qual(features)

# ======== Working EfficientNet Model ========
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights

NUM_MATERIALS, NUM_QUALITIES = 5, 5

class MultiHeadEffNetB7(nn.Module):
    def __init__(self):
        super().__init__()
        base = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(base.children())[:-1])
        f = base.classifier[1].in_features  # 2560
        self.mat = nn.Linear(f, NUM_MATERIALS)
        self.qual = nn.Linear(f, NUM_QUALITIES)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.mat(x), self.qual(x)

# ---------------- Model Loader ----------------
@st.cache_resource
def load_model(model_choice):
    if model_choice == "ResNet50":
        model = MultiHeadResNet50().to(device)
        state_dict = torch.load(MODEL_PATHS[model_choice], map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:  # EfficientNet-B7
        model = MultiHeadEffNetB7().to(device)
        state_dict = torch.load(MODEL_PATHS[model_choice], map_location=device)
        fixed_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(fixed_state_dict)
    model.eval()
    return model

# ---------------- Streamlit UI ----------------
st.title("Street Surface Classification & GPS")
st.write("Upload an image, choose a model, predict surface type & quality, and optionally send to city ward.")

# Model selector
model_choice = st.selectbox("Choose Model:", list(MODEL_PATHS.keys()), index=list(MODEL_PATHS.keys()).index(DEFAULT_MODEL))
model = load_model(model_choice)

# ---------------- Transform ----------------
if model_choice == "EfficientNet-B7":
    input_size = 600
else:
    input_size = 224

transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------- GPS Extraction ----------------
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
        if lat_ref == "S":
            lat = -lat
        if lon_ref == "W":
            lon = -lon
        return lat, lon
    except:
        return None

# ---------------- Gmail Email Function ----------------
def send_email(to_email, subject, body, attachment_bytes=None, attachment_name="image.jpg"):
    gmail_user = "your_email@gmail.com"
    gmail_password = "your_app_password"

    msg = EmailMessage()
    msg["From"] = gmail_user
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
            server.login(gmail_user, gmail_password)
            server.send_message(msg)
        return True, "Email sent successfully"
    except Exception as e:
        return False, str(e)

# ---------------- File Upload & Prediction ----------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])
if uploaded_file is not None:
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out_type, out_qual = model(img_tensor)
        main_pred = SURFACE_TYPE_MAP_INV[out_type.argmax(1).item()]
        sub_pred = SURFACE_QUALITY_MAP_INV[out_qual.argmax(1).item()]

    st.success(f"**Model:** {model_choice}")
    st.success(f"Predicted Surface Type: **{main_pred}**")
    st.success(f"Predicted Surface Quality: **{sub_pred}**")

    # GPS & Map
    coords = get_gps_coords_from_bytes(file_bytes)
    if coords:
        lat, lon = coords
        m = folium.Map(location=[lat, lon], zoom_start=16)
        folium.Marker([lat, lon], tooltip="Uploaded Image Location").add_to(m)
        st_folium(m, width=700, height=500)

        geolocator = Nominatim(user_agent="street_app")
        try:
            location = geolocator.reverse((lat, lon), language='en')
            city_name = location.raw.get("address", {}).get("city") or \
                        location.raw.get("address", {}).get("town") or \
                        location.raw.get("address", {}).get("village")
        except:
            city_name = None

        if city_name:
            st.info(f"City detected: **{city_name}**")
            csv_df = pd.read_csv("csv.csv")
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
