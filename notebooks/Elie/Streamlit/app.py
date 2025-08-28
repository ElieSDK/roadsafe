import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

#Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "best_model.pth"

SURFACE_TYPE_MAP = {"asphalt":0,"concrete":1,"paving_stones":2,"unpaved":3,"sett":4}
SURFACE_TYPE_MAP_INV = {v:k for k,v in SURFACE_TYPE_MAP.items()}

SURFACE_QUALITY_MAP = {"excellent":0,"good":1,"intermediate":2,"bad":3,"very_bad":4}
SURFACE_QUALITY_MAP_INV = {v:k for k,v in SURFACE_QUALITY_MAP.items()}

#Modele
@st.cache_resource
def load_model():
    checkpoint = torch.load(checkpoint_path, map_location=device)

    backbone = models.resnet50(weights=None)
    in_feat = backbone.fc.in_features
    backbone.fc = nn.Identity()
    backbone = backbone.to(device)

    fc_main = nn.Linear(in_feat, len(SURFACE_TYPE_MAP)).to(device)
    fc_sub = nn.Linear(in_feat, len(SURFACE_QUALITY_MAP)).to(device)

    backbone.load_state_dict(checkpoint['backbone'])
    fc_main.load_state_dict(checkpoint['fc_main'])
    fc_sub.load_state_dict(checkpoint['fc_sub'])

    backbone.eval()
    fc_main.eval()
    fc_sub.eval()

    return backbone, fc_main, fc_sub

backbone, fc_main, fc_sub = load_model()

#Image Transformations
transform = transforms.Compose([
    transforms.Resize((288,512)),
    transforms.ToTensor()
])

#Streamlit UI
st.title("Street Surface Classification")
st.write("Upload an image to predict surface type and quality.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    #Image Transformation
    img_tensor = transform(image).unsqueeze(0).to(device)

    #Prediction
    with torch.no_grad():
        features = backbone(img_tensor)
        out_main = fc_main(features)
        out_sub = fc_sub(features)

        main_pred = SURFACE_TYPE_MAP_INV[out_main.argmax(1).item()]
        sub_pred = SURFACE_QUALITY_MAP_INV[out_sub.argmax(1).item()]

    st.success(f"Predicted Surface Type: **{main_pred}**")
    st.success(f"Predicted Surface Quality: **{sub_pred}**")
