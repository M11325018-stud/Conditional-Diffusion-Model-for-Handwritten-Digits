import os
import sys
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


# ======================
# Dataset
# ======================
class OfficeHomeTestDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # 如果有 label 欄位，回傳 label，否則回傳 None
        label = row["label"] if "label" in self.df.columns else None
        return image, int(row["id"]), row["filename"], label


# ======================
# Load model
# ======================
def load_model(ckpt_path, num_classes=65, device="cpu"):
    resnet = models.resnet50(weights=None)
    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features, num_classes)

    state_dict = torch.load(ckpt_path, map_location=device)
    resnet.load_state_dict(state_dict, strict=True)
    resnet = resnet.to(device)
    resnet.eval()
    return resnet


# ======================
# Inference
# ======================
def inference(test_csv, test_dir, ckpt_path, output_csv, batch_size=64, num_workers=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    dataset = OfficeHomeTestDataset(test_csv, test_dir, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers, pin_memory=True)

    model = load_model(ckpt_path, device=device)

    results = []
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, ids, filenames, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().numpy()

            for i in range(len(ids)):
                results.append([ids[i], filenames[i], preds[i]])

                # 如果有 ground truth，就計算正確率
                if labels[i] is not None and not pd.isna(labels[i]):
                    if preds[i] == int(labels[i]):
                        correct += 1
                    total += 1

    # 輸出結果
    df_out = pd.DataFrame(results, columns=["id", "filename", "label"])
    df_out.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")

    # 顯示 Accuracy
    if total > 0:
        acc = correct / total
        print(f"Accuracy = {correct}/{total} = {acc:.4f}")


# ======================
# Main
# ======================
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 inference_hw1_1.py <test_csv> <test_dir> <output_csv> <best_model.pth>")
        sys.exit(1)

    test_csv, test_dir, output_csv, ckpt_path = sys.argv[1:]
    inference(test_csv, test_dir, ckpt_path, output_csv)
