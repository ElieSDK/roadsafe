import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
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

# ---------------- Transforms (with cropping) ----------------
def crop_center_lower(img):
    """Crop middle and lower half of image"""
    width, height = img.size
    return img.crop((0.25 * width, 0.5 * height, 0.75 * width, height))

transform = transforms.Compose([
    #transforms.Lambda(crop_center_lower),
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torchvision.models import resnet18, ResNet18_Weights
backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

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

# Learning rate scheduler (reduce LR if no improvement for 5 epochs)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ---------------- Training Loop with Checkpoints ----------------
best_val_loss = float("inf")
epochs = 1

for epoch in range(epochs):
    # ---- Training ----
    backbone.train(); fc_main.train(); fc_sub.train()
    train_loss = 0.0
    for images_batch, main_batch, sub_batch in train_loader:
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
        train_loss += loss.item()

# ---- Validation ----
    backbone.eval(); fc_main.eval(); fc_sub.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images_batch, main_batch, sub_batch in val_loader:
            images_batch = images_batch.to(device)
            main_batch = main_batch.to(device)
            sub_batch = sub_batch.to(device)
            features = backbone(images_batch)
            main_out = fc_main(features)
            sub_out = fc_sub(features)
            loss = criterion(main_out, main_batch) + criterion(sub_out, sub_batch)
            val_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# ---- Scheduler step ----
    scheduler.step(avg_val_loss)

# ---- Save checkpoint if val improved ----
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

print("Training finished.")

# ---------------- Load Best Model ----------------
checkpoint = torch.load("best_model.pth", map_location=device)
backbone.load_state_dict(checkpoint['backbone'])
fc_main.load_state_dict(checkpoint['fc_main'])
fc_sub.load_state_dict(checkpoint['fc_sub'])

backbone.eval()
fc_main.eval()
fc_sub.eval()

# ---------------- Prediction for Single Image ----------------
def predict_single_image(img_path, backbone, fc_main, fc_sub, transform, device):
    """
    Predict surface type and quality for a single image.
    """
    img = Image.open(img_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = backbone(img_t)
        main_idx = fc_main(features).argmax(dim=1).item()
        sub_idx = fc_sub(features).argmax(dim=1).item()

    main_class = [k for k,v in SURFACE_TYPE_MAP.items() if v==main_idx][0]
    sub_class = [k for k,v in SURFACE_QUALITY_MAP.items() if v==sub_idx][0]

    # Display image
    plt.imshow(img)
    plt.title(f"Type: {main_class}, Quality: {sub_class}")
    plt.axis('off')
    plt.show()

    print("Predicted surface type:", main_class)
    print("Predicted surface quality:", sub_class)
    return main_class, sub_class

# ---------------- Evaluation for Dataset ----------------
def evaluate_dataset(backbone, fc_main, fc_sub, dataloader, device, task="main"):
    """
    Evaluate model on a dataset (train/val/test) for either 'main' (type) or 'sub' (quality).
    """
    y_true, y_pred = [], []

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

    # Accuracy
    acc = sum([yt==yp for yt, yp in zip(y_true, y_pred)]) / len(y_true)
    print(f"{task.capitalize()} Accuracy: {acc:.2%}")

    # Classification report
    target_names = list(SURFACE_TYPE_MAP.keys()) if task=="main" else list(SURFACE_QUALITY_MAP.keys())
    # Get unique labels present in the true labels
    unique_labels = sorted(list(set(y_true)))
    print(classification_report(y_true, y_pred, target_names=target_names, labels=unique_labels))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=[target_names[i] for i in unique_labels])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix ({task})")
    plt.show()

# ---------------- Example Usage ----------------
# Predict single image
example_img_path = "/content/drive/MyDrive/s_1024/003.jpeg"  # replace with your image
predict_single_image(example_img_path, backbone, fc_main, fc_sub, transform, device)

# Evaluate datasets
print("=== Evaluating Test Set ===")
evaluate_dataset(backbone, fc_main, fc_sub, test_loader, device, task="main")  # surface type
evaluate_dataset(backbone, fc_main, fc_sub, test_loader, device, task="sub")   # surface quality
