import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
from torch.utils.data import Dataset
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

# ---------------- Dataset Loader Function ----------------
def load_dataset(csv_file, img_folder, transform):
    df = pd.read_csv(csv_file)
    images = []
    main_labels = []
    sub_labels = []
    for _, row in df.iterrows():
        path = os.path.join(img_folder, f"{row['mapillary_image_id']}.jpg")
        if os.path.exists(path):
            images.append(path)
            main_labels.append(SURFACE_TYPE_MAP[row['surface_type']])
            sub_labels.append(SURFACE_QUALITY_MAP[row['surface_quality']])
    return images, main_labels, sub_labels

images, main_labels, sub_labels = load_dataset(csv_path, img_folder, transform)

# ---------------- Simple Dataset for DataLoader ----------------
class SimpleDataset(Dataset):
    def __init__(self, images, main_labels, sub_labels, transform):
        self.images = images
        self.main_labels = main_labels
        self.sub_labels = sub_labels
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, (self.main_labels[idx], self.sub_labels[idx])

dataset = SimpleDataset(images, main_labels, sub_labels, transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# ---------------- Model ----------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backbone = models.resnet18(pretrained=True)
in_feat = backbone.fc.in_features
backbone.fc = nn.Identity()

fc_main = nn.Linear(in_feat, len(SURFACE_TYPE_MAP)).to(device)
fc_sub = nn.Linear(in_feat, len(SURFACE_QUALITY_MAP)).to(device)

# Combined parameters for optimizer
params = list(backbone.parameters()) + list(fc_main.parameters()) + list(fc_sub.parameters())
optimizer = optim.Adam(params, lr=1e-4)
criterion = nn.CrossEntropyLoss()

# ---------------- Training Loop ----------------
epochs = 2
for epoch in range(epochs):
    backbone.train()
    fc_main.train()
    fc_sub.train()
    running_loss = 0.0
    for images_batch, (main_batch, sub_batch) in dataloader:
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

# Example usage
sample_img_id = images[0].split('/')[-1].replace('.jpg','')
img_path = os.path.join(img_folder, f"{sample_img_id}.jpg")
img, main_class, sub_class = predict_image(img_path, backbone, fc_main, fc_sub, transform, device)

print("Predicted surface type:", main_class)
print("Predicted surface quality:", sub_class)
plt.imshow(img)
plt.title(f"Type: {main_class}, Quality: {sub_class}")
plt.axis('off')
plt.show()

# ---------------- Final Evaluation ----------------
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def final_evaluation(backbone, fc_main, fc_sub, dataloader, device, task="main"):
    y_true, y_pred = [], []
    backbone.eval(); fc_main.eval(); fc_sub.eval()
    with torch.no_grad():
        for images, (main_labels, sub_labels) in dataloader:
            images = images.to(device)
            main_labels = main_labels.to(device)
            sub_labels = sub_labels.to(device)

            features = backbone(images)
            main_out = fc_main(features)
            sub_out = fc_sub(features)

            if task == "main":
                preds = main_out.argmax(dim=1)
                y_true.extend(main_labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
            else:
                preds = sub_out.argmax(dim=1)
                y_true.extend(sub_labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

    # Accuracy
    acc = (sum([yt == yp for yt, yp in zip(y_true, y_pred)]) / len(y_true))
    print(f"{task.capitalize()} Accuracy: {acc:.2%}")

    # Classification report
    if task == "main":
        target_names = list(SURFACE_TYPE_MAP.keys())
    else:
        target_names = list(SURFACE_QUALITY_MAP.keys())
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix ({task})")
    plt.show()


# Example usage after training
final_evaluation(backbone, fc_main, fc_sub, dataloader, device, task="main")  # for surface type
final_evaluation(backbone, fc_main, fc_sub, dataloader, device, task="sub")   # for surface quality
