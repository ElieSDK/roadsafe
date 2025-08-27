import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from google.colab import drive
import pandas as pd
from matplotlib import pyplot as plt

#https://zenodo.org/records/11449977

# Mount Google Drive to access dataset
drive.mount('/content/drive')

# Paths to CSV metadata and image folder
csv_path = '/content/drive/MyDrive/streetSurfaceVis_v1_0.csv'
img_folder = '/content/drive/MyDrive/s_1024'

# Dictionaries mapping surface type/quality to numeric labels
SURFACE_TYPE_MAP = {"asphalt":0,"concrete":1,"paving_stones":2,"unpaved":3,"sett":4}
SURFACE_QUALITY_MAP = {"excellent":0,"good":1,"intermediate":2,"bad":3,"very_bad":4}

#Custom Dataset

'''
Loads dataframe from CSV.
Ensures images exist.

Returns:
Transformed image tensor.
Labels: (main_label = type, sub_label = quality).
'''

class SurfaceDataset(Dataset):
    def __init__(self, csv_file, img_folder, transform=None):
        self.df = pd.read_csv(csv_file)              # read csv into dataframe
        self.img_folder = img_folder
        self.transform = transform
        # Keep only rows where image file exists in folder
        self.df = self.df[self.df['mapillary_image_id'].apply(
            lambda x: os.path.exists(os.path.join(self.img_folder, str(x) + ".jpg"))
        )].reset_index(drop=True)

    def __len__(self):
        return len(self.df)                          # number of samples

    def __getitem__(self, idx):
        row = self.df.iloc[idx]                      # pick row
        img_path = os.path.join(self.img_folder, str(row['mapillary_image_id']) + ".jpg")
        img = Image.open(img_path).convert('RGB')    # load image
        if self.transform:
            img = self.transform(img)                # apply transforms
        # map labels to integers
        main_label = SURFACE_TYPE_MAP[row['surface_type']]
        sub_label = SURFACE_QUALITY_MAP[row['surface_quality']]
        return img, (main_label, sub_label)

# Transformations applied to each image (resize, convert to tensor, normalize for ResNet)
transform = transforms.Compose([
    transforms.Resize((288, 512)),   # (height, width) = (576/2, 1024/2)
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# Dataset and dataloader
dataset = SurfaceDataset(csv_path, img_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model with shared ResNet backbone and two heads (multi-task learning)
class DoubleHeadResNet(nn.Module):
    def __init__(self, num_main, num_sub):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)    # load pretrained ResNet18
        in_feat = self.backbone.fc.in_features              # get number of features before final FC
        self.backbone.fc = nn.Identity()                    # remove original classifier
        self.fc_main = nn.Linear(in_feat, num_main)         # head for surface type
        self.fc_sub = nn.Linear(in_feat, num_sub)           # head for surface quality
    def forward(self, x):
        feat = self.backbone(x)                             # extract features
        return self.fc_main(feat), self.fc_sub(feat)        # output predictions for both heads

# Initialize model
model = DoubleHeadResNet(
    num_main=len(SURFACE_TYPE_MAP),
    num_sub=len(SURFACE_QUALITY_MAP)
).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ---------------- Training Loop ----------------
epochs = 2  # number of epochs you want

best_val_acc = 0.0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, (main_labels, sub_labels) in dataloader:
        images = images.to(device)
        main_labels = main_labels.to(device)
        sub_labels = sub_labels.to(device)

        optimizer.zero_grad()
        main_out, sub_out = model(images)
        loss = criterion(main_out, main_labels) + criterion(sub_out, sub_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Evaluate on validation set
    val_acc_type, val_acc_quality = evaluate(model, val_loader, device)
    val_acc = (val_acc_type + val_acc_quality)/2  # simple average

    # Save model if validation improves
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  -> New best model saved with val acc: {best_val_acc:.2%}")

# Function to predict surface type and quality from an image
def predict_image(img_path, model, transform, device):
    img = Image.open(img_path).convert('RGB')       # load image
    img_t = transform(img).unsqueeze(0).to(device)  # preprocess and add batch dimension

    model.eval()
    with torch.no_grad():
        main_out, sub_out = model(img_t)
        main_idx = main_out.argmax(dim=1).item()
        sub_idx = sub_out.argmax(dim=1).item()

    # decode indices back to labels
    main_class = [k for k,v in SURFACE_TYPE_MAP.items() if v==main_idx][0]
    sub_class = [k for k,v in SURFACE_QUALITY_MAP.items() if v==sub_idx][0]

    return img, main_class, sub_class


# Example usage
sample_img_id = dataset.df.iloc[0]['mapillary_image_id']
img_path = os.path.join(img_folder, str(sample_img_id) + ".jpg")

img, main_class, sub_class = predict_image(img_path, model, transform, device)

#Custom picture
#img, main_class, sub_class = predict_image("/content/drive/MyDrive/s_1024/002.jpg", model, transform, device)


print("Predicted surface type:", main_class)
print("Predicted surface quality:", sub_class)

plt.imshow(img)
plt.title(f"Type: {main_class}, Quality: {sub_class}")
plt.axis('off')
plt.show()

# ---------------- Final Evaluation ----------------
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def final_evaluation(model, dataloader, device, task="main"):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for images, (main_labels, sub_labels) in dataloader:
            images = images.to(device)
            main_labels = main_labels.to(device)
            sub_labels = sub_labels.to(device) # Corrected: Removed the -1 shift

            main_out, sub_out = model(images)

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
    print(classification_report(y_true, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix ({task})")
    plt.show()


# Example usage after training
final_evaluation(model, dataloader, device, task="main")  # for surface type
final_evaluation(model, dataloader, device, task="sub")   # for surface quality
