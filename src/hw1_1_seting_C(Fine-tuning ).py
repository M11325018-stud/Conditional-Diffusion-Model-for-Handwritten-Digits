import os
import sys
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm   # <-- 加入進度條


# ======================
# Dataset
# ======================
class OfficeHomeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        label = int(row["label"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ======================
# Load backbone
# ======================
def load_dino_backbone(ckpt_path, num_classes=65):
    resnet = models.resnet50(weights=None)
    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features, num_classes)

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 取出 student 權重
    state_dict = ckpt["student"] if "student" in ckpt else ckpt
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module.backbone."):
            new_state[k[len("module.backbone."):]] = v
    resnet.load_state_dict(new_state, strict=False)
    return resnet


# ======================
# Train & Validate
# ======================
def train_and_validate(train_csv, train_dir, val_csv, val_dir, ckpt_path, out_dir,
                       epochs=50, batch_size=32, lr=1e-3, num_workers=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    train_set = OfficeHomeDataset(train_csv, train_dir, transform)
    val_set = OfficeHomeDataset(val_csv, val_dir, transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    model = load_dino_backbone(ckpt_path).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(out_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", ncols=100)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_loss = total_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        correct, total = 0, 0
        vbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", ncols=100)
        with torch.no_grad():
            for imgs, labels in vbar:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total

        print(f"Epoch {epoch+1}/{epochs} | Loss={avg_loss:.4f} | Val Acc={acc:.4f}")

        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))

    print(f"Finished training. Best Val Acc = {best_acc:.4f}")


# ======================
# Main
# ======================
if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python3 finetune.py <train_csv> <train_dir> <val_csv> <val_dir> <dino_ckpt> <out_dir>")
        sys.exit(1)

    train_csv, train_dir, val_csv, val_dir, dino_ckpt, out_dir = sys.argv[1:]
    train_and_validate(train_csv, train_dir, val_csv, val_dir, dino_ckpt, out_dir)
