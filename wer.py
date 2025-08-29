"""
Semantic Segmentation Training Script (U-Net Example)
----------------------------------------------------

How to use this script:

1. Prepare your dataset:
   - Place images in 'images/' and masks in 'masks/' folders.
   - Each mask should have the same filename as its corresponding image.

2. Install dependencies:
   pip install torch torchvision pillow numpy

3. Run the script:
   python segmentation_training.py

4. You can change the loss function at the marked section.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from glob import glob
import os

# --------- Data Augmentation ---------
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# --------- Dataset Class ---------
class SegmentationDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        if self.transform:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)
        mask = (mask > 0.5).float()  # Binarize mask if needed
        return image, mask

# --------- U-Net Model ---------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.center = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        center = self.center(self.pool4(enc4))
        dec4 = self.dec4(torch.cat([self.up4(center), enc4], 1))
        dec3 = self.dec3(torch.cat([self.up3(dec4), enc3], 1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], 1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], 1))
        return torch.sigmoid(self.final(dec1))

# --------- Loss Functions ---------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth
    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')
    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# --------- Training & Validation Loops ---------
def train(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, masks in dataloader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = loss_fn(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, masks)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# --------- Main Usage ---------
if __name__ == "__main__":
    # Set these paths to your image and mask folders
    IMG_DIR = 'images/'
    MASK_DIR = 'masks/'

    # Get file paths
    img_paths = sorted(glob(os.path.join(IMG_DIR, '*.png')))
    mask_paths = sorted(glob(os.path.join(MASK_DIR, '*.png')))

    if len(img_paths) == 0 or len(mask_paths) == 0:
        print("No images or masks found. Please check your paths and dataset.")
        exit()

    # Split dataset into train/val (simple split, can be improved)
    split_idx = int(0.8 * len(img_paths))
    train_imgs, val_imgs = img_paths[:split_idx], img_paths[split_idx:]
    train_masks, val_masks = mask_paths[:split_idx], mask_paths[split_idx:]

    # Create DataLoaders
    train_dataset = SegmentationDataset(train_imgs, train_masks, transform=train_transforms)
    val_dataset = SegmentationDataset(val_imgs, val_masks, transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Device, model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    # --- CHOOSE LOSS FUNCTION HERE ---
    # WARNING: You have all three loss functions defined, but only the last one (BCELoss) will be used!
    # Comment out the ones you don't want to use:
    
    # loss_fn = DiceLoss()      # For Dice Loss
    # loss_fn = FocalLoss()     # For Focal Loss
    loss_fn = nn.BCELoss()      # For Binary Cross Entropy ← This is what's actually being used

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, loss_fn, optimizer, device)
        val_loss = validate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), "unet_model.pth")
    print("Training complete. Model saved as 'unet_model.pth'.")