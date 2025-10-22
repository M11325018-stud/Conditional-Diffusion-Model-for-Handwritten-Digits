#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SegNet inference (TA interface).
Usage (TA via hw1_2.sh):
    python3 inference_hw1_2.py <input_dir> <output_dir> [--ckpt CKPT] [--tta] [--vis]

- <input_dir>: folder containing files named 'xxxx_sat.jpg'
- <output_dir>: where to save 'xxxx_mask.png'
- --ckpt: checkpoint path (default: ./best_model.pth)
- --tta: enable horizontal-flip test-time augmentation
- --vis: additionally save colorized results as 'xxxx_color.png'
"""

import os
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn

# ---------- DeepGlobe palette ----------
ID2RGB = {
    0: (0, 255, 255),   # Urban
    1: (255, 255, 0),   # Agriculture
    2: (255, 0, 255),   # Rangeland
    3: (0, 255, 0),     # Forest
    4: (0, 0, 255),     # Water
    5: (255, 255, 255), # Barren
    6: (0, 0, 0),       # Unknown
}
IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD  = [0.229, 0.224, 0.225]

# ---------- SegNet ----------
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class SegNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # encoder
        self.enc11 = conv_block(3, 64);  self.enc12 = conv_block(64, 64); self.pool1 = nn.MaxPool2d(2,2, return_indices=True)
        self.enc21 = conv_block(64, 128); self.enc22 = conv_block(128, 128); self.pool2 = nn.MaxPool2d(2,2, return_indices=True)
        self.enc31 = conv_block(128, 256); self.enc32 = conv_block(256, 256); self.enc33 = conv_block(256, 256); self.pool3 = nn.MaxPool2d(2,2, return_indices=True)
        self.enc41 = conv_block(256, 512); self.enc42 = conv_block(512, 512); self.enc43 = conv_block(512, 512); self.pool4 = nn.MaxPool2d(2,2, return_indices=True)
        self.enc51 = conv_block(512, 512); self.enc52 = conv_block(512, 512); self.enc53 = conv_block(512, 512); self.pool5 = nn.MaxPool2d(2,2, return_indices=True)
        # decoder
        self.unpool5 = nn.MaxUnpool2d(2,2); self.dec53 = conv_block(512,512); self.dec52 = conv_block(512,512); self.dec51 = conv_block(512,512)
        self.unpool4 = nn.MaxUnpool2d(2,2); self.dec43 = conv_block(512,512); self.dec42 = conv_block(512,512); self.dec41 = conv_block(512,256)
        self.unpool3 = nn.MaxUnpool2d(2,2); self.dec33 = conv_block(256,256); self.dec32 = conv_block(256,256); self.dec31 = conv_block(256,128)
        self.unpool2 = nn.MaxUnpool2d(2,2); self.dec22 = conv_block(128,128); self.dec21 = conv_block(128,64)
        self.unpool1 = nn.MaxUnpool2d(2,2); self.dec12 = conv_block(64,64); self.dec11 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.enc11(x); x1 = self.enc12(x1); s1 = x1.size(); x1p, i1 = self.pool1(x1)
        x2 = self.enc21(x1p); x2 = self.enc22(x2); s2 = x2.size(); x2p, i2 = self.pool2(x2)
        x3 = self.enc31(x2p); x3 = self.enc32(x3); x3 = self.enc33(x3); s3 = x3.size(); x3p, i3 = self.pool3(x3)
        x4 = self.enc41(x3p); x4 = self.enc42(x4); x4 = self.enc43(x4); s4 = x4.size(); x4p, i4 = self.pool4(x4)
        x5 = self.enc51(x4p); x5 = self.enc52(x5); x5 = self.enc53(x5); s5 = x5.size(); x5p, i5 = self.pool5(x5)

        d5 = self.unpool5(x5p, i5, output_size=s5); d5 = self.dec53(d5); d5 = self.dec52(d5); d5 = self.dec51(d5)
        d4 = self.unpool4(d5, i4, output_size=s4); d4 = self.dec43(d4); d4 = self.dec42(d4); d4 = self.dec41(d4)
        d3 = self.unpool3(d4, i3, output_size=s3); d3 = self.dec33(d3); d3 = self.dec32(d3); d3 = self.dec31(d3)
        d2 = self.unpool2(d3, i2, output_size=s2); d2 = self.dec22(d2); d2 = self.dec21(d2)
        d1 = self.unpool1(d2, i1, output_size=s1); d1 = self.dec12(d1)
        return self.dec11(d1)

# ---------- helpers ----------
def normalize_img(img):
    img = img.astype(np.float32) / 255.0
    img = (img - np.array(IMNET_MEAN, dtype=np.float32)) / np.array(IMNET_STD, dtype=np.float32)
    return img.transpose(2, 0, 1).astype(np.float32)

def ids_to_color(ids_np):
    h, w = ids_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for k, c in ID2RGB.items():
        rgb[ids_np == k] = c
    return rgb

def safe_load_state(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device)

# ---------- main ----------
def parse_args():
    ap = argparse.ArgumentParser("HW1-2 SegNet inference")
    ap.add_argument("input_dir", type=str, help="testing images dir (xxxx_sat.jpg)")
    ap.add_argument("output_dir", type=str, help="output dir (save xxxx_mask.png)")
    ap.add_argument("--ckpt", type=str, default="./best_model.pth", help="checkpoint path")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--tta", action="store_true", help="enable horizontal flip TTA")
    ap.add_argument("--vis", action="store_true", help="also save colorized results as xxxx_color.png")
    return ap.parse_args()

def main():
    args = parse_args()
    inp = Path(args.input_dir)
    outp = Path(args.output_dir)
    outp.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegNet(num_classes=7).to(device)

    ckpt = os.path.expanduser(args.ckpt)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = safe_load_state(ckpt, device)

    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    img_paths = sorted(inp.glob("*_sat.jpg"))
    with torch.no_grad():
        for p in img_paths:
            bgr = cv2.imread(str(p))
            orig_h, orig_w = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_resized = cv2.resize(rgb, (args.img_size, args.img_size))

            x = normalize_img(rgb_resized)
            x = torch.from_numpy(x).unsqueeze(0).to(device)

            logits = model(x)
            if args.tta:
                xf = torch.flip(x, dims=[3])
                lf = model(xf)
                lf = torch.flip(lf, dims=[3])
                logits = 0.5 * (logits + lf)

            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
            pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            # save灰階 id mask (0–6)
            stem = p.stem.replace("_sat", "")
            cv2.imwrite(str(outp / f"{stem}_mask.png"), pred)

            # optional 彩色視覺化
            if args.vis:
                color = ids_to_color(pred)
                cv2.imwrite(str(outp / f"{stem}_color.png"), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main()
