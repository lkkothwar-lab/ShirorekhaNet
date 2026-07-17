import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class LineSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=512, is_train=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.is_train = is_train
        self.files = sorted([f for f in os.listdir(img_dir)
                             if f.lower().endswith(('.png','.jpg','.jpeg'))])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = cv2.imread(os.path.join(self.img_dir, fname))
        mask = cv2.imread(os.path.join(self.mask_dir, fname), cv2.IMREAD_GRAYSCALE)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)

        img_t = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        mask_t = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        mask_t = (mask_t > 0.5).float()
        return img_t, mask_t


class ShirorekhaDataset(LineSegDataset):
    """Returns 4‑channel input: RGB + adaptive‑thresholded binary channel."""
    def __getitem__(self, idx):
        img_t, mask_t = super().__getitem__(idx)
        img_np = (img_t.permute(1,2,0).numpy() * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        bin_t = torch.from_numpy(binary).unsqueeze(0).float() / 255.0
        return torch.cat([img_t, bin_t], dim=0), mask_t
