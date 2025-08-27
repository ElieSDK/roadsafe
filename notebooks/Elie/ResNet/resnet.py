import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from PIL import Image
from google.colab import drive
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# ---------------- Google Drive ----------------
drive.mount('/content/drive')

csv_path = '/content/drive/MyDrive/streetSurfaceVis_v1_0.csv'
img_folder = '/content/drive/MyDrive/s_1024'

SURFACE_TYPE_MAP = {"asphalt":0,"concrete":1,"paving_stones":2,"unpaved":3,"sett":4}
SURFACE_QUALITY_MAP = {"excellent":0,"good":1,"intermediate":2,"bad":3,"very_bad":4}

# ---------------- Transforms ----------------
transform = transforms.Compose([
    transforms.Resize((288, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# ---------------- Load Dataset ----------------
df = pd.read_csv(csv_path)
images_list, main_labels_list, sub_labels_list = [], [], []

for _, row in df.iterrows():
    path = os.path.join(img_folder, f"{row['mapillary_image_id']}.jpg")
    if os.path.exists(path):
        img = Image.open(path).convert('RGB')
        img_t = transform(img)
        images_list.append(img_t)
        main_labels_list.append(SURFACE_TYPE_MAP[row['surface_type']])
        sub_labels_list.append(SURFACE_QUALITY_MAP[row['surface_quality']])

# Convert to tensors
images_tensor = torch.stack(images_list)
main_labels_tensor = torch.tensor(main_labels_list)
sub_labels_tensor = torch.tensor(sub_labels_list)

dataset = TensorDataset(images_tensor, main_labels_tensor, sub_labels_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# ---------------- Model ----------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backbone = models.resnet18(pretrained=True)
in_feat = backbone.fc.in_features
backbone.fc = nn.Identity()
backbone = backbone.to(device)

fc_main = nn.Linear(in_feat, len(SURFACE_TYPE_MAP)).to(device)
fc_sub = nn.Linear(in_feat, len(SURFACE_QUALITY_MAP)).to(device)

optimizer = optim.Adam(list(backbone.parameters()) + list(fc_main.parameters()) + list(fc_sub.parameters()), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# ---------------- Training Loop ----------------
epochs = 2
for epoch in range(epochs):
    backbone.train()
    fc_main.train()
    fc_sub.train()
    running_loss = 0.0
    for images_batch, main_batch, sub_batch in dataloader:
        images_batch = images_batch.to(device)
        main_batch = main_batch.to(device)
        sub_batch = sub_batch.to(device)

        optimizer.zero_grad()
        features = backbone(images_batch)
        main_out = fc_main(features)
        sub_out = fc_sub(features)
        loss = criterion(main_out, main_batch) + criterion(sub_out, sub_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")

# ---------------- Prediction Function ----------------
def predict_image(img_path, backbone, fc_main, fc_sub, transform, device):
    img = Image.open(img_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)
    backbone.eval(); fc_main.eval(); fc_sub.eval()
    with torch.no_grad():
        feat = backbone(img_t)
        main_idx = fc_main(feat).argmax(dim=1).item()
        sub_idx = fc_sub(feat).argmax(dim=1).item()
    main_class = [k for k,v in SURFACE_TYPE_MAP.items() if v==main_idx][0]
    sub_class = [k for k,v in SURFACE_QUALITY_MAP.items() if v==sub_idx][0]
    return img, main_class, sub_class

# Example usage: find an existing image to predict on
sample_img_path = None
for index, row in df.iterrows():
    potential_path = os.path.join(img_folder, f"{row['mapillary_image_id']}.jpg")
    if os.path.exists(potential_path):
        sample_img_path = potential_path
        print(f"Found existing image: {sample_img_path}")
        break

if sample_img_path:
    img, main_class, sub_class = predict_image(sample_img_path, backbone, fc_main, fc_sub, transform, device)

    print("Predicted surface type:", main_class)
    print("Predicted surface quality:", sub_class)
    plt.imshow(img)
    plt.title(f"Type: {main_class}, Quality: {sub_class}")
    plt.axis('off')
    plt.show()
else:
    print("No existing image found in the specified folder based on the dataframe.")

# ---------------- Evaluation ----------------
def final_evaluation(backbone, fc_main, fc_sub, dataloader, device, task="main"):
    y_true, y_pred = [], []
    backbone.eval(); fc_main.eval(); fc_sub.eval()
    with torch.no_grad():
        for images_batch, main_batch, sub_batch in dataloader:
            images_batch = images_batch.to(device)
            main_batch = main_batch.to(device)
            sub_batch = sub_batch.to(device)
            features = backbone(images_batch)
            main_out = fc_main(features)
            sub_out = fc_sub(features)
            if task == "main":
                preds = main_out.argmax(dim=1)
                y_true.extend(main_batch.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
            else:
                preds = sub_out.argmax(dim=1)
                y_true.extend(sub_batch.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
    acc = (sum([yt==yp for yt, yp in zip(y_true, y_pred)])/len(y_true))
    print(f"{task.capitalize()} Accuracy: {acc:.2%}")
    target_names = list(SURFACE_TYPE_MAP.keys()) if task=="main" else list(SURFACE_QUALITY_MAP.keys())
    print(classification_report(y_true, y_pred, target_names=target_names))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix ({task})")
    plt.show()

# Example evaluation
final_evaluation(backbone, fc_main, fc_sub, dataloader, device, task="main")
final_evaluation(backbone, fc_main, fc_sub, dataloader, device, task="sub")s
