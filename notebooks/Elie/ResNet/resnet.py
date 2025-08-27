# ---------------- Complete Augmentation + Dataset with Safety Check ----------------
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from PIL import Image
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

# ---------------- Google Drive ----------------
drive.mount('/content/drive')

# ---------------- Config ----------------
csv_path = '/content/drive/MyDrive/streetSurfaceVis_v1_0.csv'
img_folder = '/content/drive/MyDrive/s_1024'

SURFACE_TYPE_MAP = {"asphalt":0,"concrete":1,"paving_stones":2,"unpaved":3,"sett":4}
SURFACE_QUALITY_MAP = {"excellent":0,"good":1,"intermediate":2,"bad":3,"very_bad":4}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------- Augmentation ----------------
augmentations = A.Compose([
    A.Rotate(limit=(2,20), p=1.0),
    A.GaussNoise(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.Blur(blur_limit=3, p=0.3),
    A.Resize(288,512),
    ToTensorV2()
])

# ---------------- Load CSV and generate augmented images ----------------
df = pd.read_csv(csv_path)
images_list, main_labels_list, sub_labels_list = [], [], []
missing_files = []

n_aug = 3  # number of augmented images per original

for _, row in df.iterrows():
    path = os.path.join(img_folder, f"{row['mapillary_image_id']}.jpg")
    if not os.path.exists(path):
        missing_files.append(row['mapillary_image_id'])
        continue  # skip missing files

    img = Image.open(path).convert('RGB')
    img_np = np.array(img)

    # original image
    aug_img = augmentations(image=img_np)['image']
    images_list.append(aug_img)
    main_labels_list.append(SURFACE_TYPE_MAP[row['surface_type']])
    sub_labels_list.append(SURFACE_QUALITY_MAP[row['surface_quality']])

    # 3 augmented versions
    for _ in range(n_aug):
        aug_img = augmentations(image=img_np)['image']
        images_list.append(aug_img)
        main_labels_list.append(SURFACE_TYPE_MAP[row['surface_type']])
        sub_labels_list.append(SURFACE_QUALITY_MAP[row['surface_quality']])

print(f"⚠️ Skipped {len(missing_files)} missing images")
if missing_files:
    print("Missing image IDs:", missing_files)

# ---------------- Convert to tensors and create TensorDataset ----------------
images_tensor = torch.stack(images_list)
main_labels_tensor = torch.tensor(main_labels_list)
sub_labels_tensor = torch.tensor(sub_labels_list)

dataset = TensorDataset(images_tensor, main_labels_tensor, sub_labels_tensor)

# ---------------- Train/Val/Test Split ----------------
n_total = len(dataset)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val
train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16)
test_loader = DataLoader(test_set, batch_size=16)

# ---------------- Model ----------------
backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
in_feat = backbone.fc.in_features
backbone.fc = nn.Identity()
backbone = backbone.to(device)

fc_main = nn.Linear(in_feat, len(SURFACE_TYPE_MAP)).to(device)
fc_sub = nn.Linear(in_feat, len(SURFACE_QUALITY_MAP)).to(device)

optimizer = optim.Adam(
    list(backbone.parameters()) + list(fc_main.parameters()) + list(fc_sub.parameters()),
    lr=1e-4
)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ---------------- Training Loop (1 epoch example) ----------------
best_val_loss = float('inf')
epochs = 1

for epoch in range(epochs):
    backbone.train(); fc_main.train(); fc_sub.train()
    train_loss = 0.0
    for images, main_labels, sub_labels in train_loader:
        images = images.to(device).float()  # Add .float() here
        main_labels = main_labels.to(device)
        sub_labels = sub_labels.to(device)

        optimizer.zero_grad()
        features = backbone(images)
        out_main = fc_main(features)
        out_sub = fc_sub(features)
        loss = criterion(out_main, main_labels) + criterion(out_sub, sub_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    backbone.eval(); fc_main.eval(); fc_sub.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, main_labels, sub_labels in val_loader:
            images = images.to(device).float() # Add .float() here
            main_labels = main_labels.to(device)
            sub_labels = sub_labels.to(device)
            features = backbone(images)
            out_main = fc_main(features)
            out_sub = fc_sub(features)
            loss = criterion(out_main, main_labels) + criterion(out_sub, sub_labels)
            val_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'backbone': backbone.state_dict(),
            'fc_main': fc_main.state_dict(),
            'fc_sub': fc_sub.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, "best_model.pth")
        print("✅ Model checkpoint saved.")

# ---------------- Display 3 example augmented images ----------------
for i in range(3):
    plt.imshow(np.transpose(images_tensor[i].numpy(), (1,2,0)))
    plt.title(f"Type: {main_labels_tensor[i].item()}, Quality: {sub_labels_tensor[i].item()}")
    plt.axis('off')
    plt.show()
