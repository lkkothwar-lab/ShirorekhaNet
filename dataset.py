import os
import cv2
import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset

class SanskritLineDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the RGB images.
            mask_dir (str): Path to the corresponding masks.
            transform (albumentations.Compose, optional): Optional transforms to be applied.
        """
        def __init__(self, image_dir, mask_dir, img_size=512, is_train=True, augmentations=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.is_train = is_train
        self.augmentations = augmentations

        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.mask_filenames = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        # Ensure image and mask files match
        if len(self.image_filenames) != len(self.mask_filenames):
            raise ValueError("Number of images and masks do not match.")

        # Optional: Verify corresponding filenames (e.g., img_001.png and mask_001.png)
        # This assumes a consistent naming convention between image and mask files.
        # If your filenames are not identical (e.g., 'image_1.png' vs 'mask_1.png'), you'll need
        # a mapping logic here, e.g., by stripping prefixes/suffixes.
        # For this example, we assume filenames are identical or can be mapped directly.
        for img_fn, mask_fn in zip(self.image_filenames, self.mask_filenames):
            if os.path.splitext(img_fn)[0] != os.path.splitext(mask_fn)[0]:
                print(f"Warning: Image {img_fn} does not seem to match mask {mask_fn}")

        self.binarize = lambda x: cv2.adaptiveThreshold(
            x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        mask_name = self.mask_filenames[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        img_orig = cv2.imread(img_path) # Read in BGR
        mask_orig = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # Read mask as grayscale

        if img_orig is None or mask_orig is None:
            raise FileNotFoundError(f"Could not load image {img_path} or mask {mask_path}")

        # Convert BGR to RGB for consistency
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

        # Resize images and masks (assuming they are not pre-resized or need to be to a common size)
        img_resized = cv2.resize(img_orig, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask_orig, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST) # Use NEAREST for masks

        # Apply augmentations if training and defined
        if self.is_train and self.augmentations:
            augmented = self.augmentations(image=img_resized, mask=mask_resized)
            img_processed = augmented['image']
            mask_processed = augmented['mask']
        else:
            img_processed = img_resized
            mask_processed = mask_resized

        # Denoising
        denoised = cv2.fastNlMeansDenoisingColored(img_processed, None, 10, 10, 7, 21)

        # Binarization
        gray = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)
        binary = self.binarize(gray)

        # Convert to tensors
        image_tensor = torch.from_numpy(denoised).permute(2,0,1).float() / 255.0
        binary_tensor = torch.from_numpy(binary).unsqueeze(0).float() / 255.0

        # Process mask and convert to tensor
        mask_numpy_processed = (mask_resized > 128).astype(np.float32) # Ensure mask is binary (0 or 1)
        mask_tensor = torch.from_numpy(mask_numpy_processed).unsqueeze(0) # Add channel dimension for mask

        return torch.cat([image_tensor, binary_tensor], dim=0), mask_tensor
    pass

def get_transforms(img_size=512):
    """Returns the Albumentations transforms for training and validation."""
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        # ... [Add any other augmentations you used] ...
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return train_transform, val_transform
print(f"✅ SanskritLineDataset defined")