import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# --- U-Net Model ---
class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = torch.nn.MaxPool2d(2)
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = torch.nn.MaxPool2d(2)
        self.center = DoubleConv(512, 1024)
        self.up4 = torch.nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = torch.nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = torch.nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = torch.nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.final = torch.nn.Conv2d(64, out_channels, 1)
    
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

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🛰️  SATELLITE FEATURE SEGMENTATION (Optimized for 0.7 threshold)")
print(f"Using device: {device}")

# Load model
model = UNet().to(device)
model.load_state_dict(torch.load("unet_model.pth", map_location=device))
model.eval()
print("✅ Model loaded successfully")

# Preprocessing (no normalization - matches training)
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load image
img_path = "test_images/test_image.png"
image = Image.open(img_path).convert("RGB")
input_tensor = preprocess(image).unsqueeze(0).to(device)

print(f"📷 Satellite image loaded: {image.size}")

# Predict
with torch.no_grad():
    pred = model(input_tensor)
    pred_np = pred.squeeze().cpu().numpy()

print(f"📊 Prediction range: [{pred_np.min():.4f}, {pred_np.max():.4f}]")
print(f"📊 Prediction mean: {pred_np.mean():.4f}")

# Test key thresholds around 0.7
thresholds = [0.65, 0.7, 0.75, 0.8]
print(f"\n🎯 THRESHOLD COMPARISON:")

results = {}
for thresh in thresholds:
    mask = (pred_np > thresh).astype(np.uint8)
    positive_pixels = np.sum(mask)
    coverage = (positive_pixels / mask.size) * 100
    results[thresh] = (mask, positive_pixels, coverage)
    
    if coverage > 80:
        status = "Too broad - captures too much"
    elif coverage > 50:
        status = "Moderate coverage - good for large features"
    elif coverage > 20:
        status = "Selective - good for specific features"
    elif coverage > 5:
        status = "Highly selective - small features only"
    else:
        status = "Too selective - may miss features"
    
    print(f"   Threshold {thresh}: {positive_pixels:5d} pixels ({coverage:5.1f}%) - {status}")

# Use 0.7 as primary threshold (user found it works best)
optimal_threshold = 0.7
optimal_mask, optimal_pixels, optimal_coverage = results[optimal_threshold]

print(f"\n✨ USING OPTIMAL THRESHOLD: {optimal_threshold}")
print(f"   Coverage: {optimal_coverage:.1f}%")
print(f"   Pixels: {optimal_pixels:,}")

# Create output directory
os.makedirs("test_images", exist_ok=True)

# Save the optimal mask
final_mask = (optimal_mask * 255).astype(np.uint8)
mask_img = Image.fromarray(final_mask)
mask_img.save("test_images/geographical_features_mask.png")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Top row: Original, Raw prediction, Optimal mask
axes[0,0].imshow(image)
axes[0,0].set_title("Original Satellite Image", fontsize=14, fontweight='bold')
axes[0,0].axis('off')

im = axes[0,1].imshow(pred_np, cmap='hot')
axes[0,1].set_title("Raw Prediction Heatmap", fontsize=14, fontweight='bold')
axes[0,1].axis('off')
plt.colorbar(im, ax=axes[0,1], fraction=0.046, pad=0.04)

axes[0,2].imshow(final_mask, cmap='gray')
axes[0,2].set_title(f"Geographical Features (Threshold {optimal_threshold})", fontsize=14, fontweight='bold')
axes[0,2].axis('off')

# Bottom row: Threshold comparisons
comparison_thresholds = [0.65, 0.75, 0.8]
for i, thresh in enumerate(comparison_thresholds):
    mask, _, coverage = results[thresh]
    display_mask = (mask * 255).astype(np.uint8)
    
    axes[1,i].imshow(display_mask, cmap='gray')
    axes[1,i].set_title(f"Threshold {thresh}\n({coverage:.1f}% coverage)", fontsize=12)
    axes[1,i].axis('off')

plt.tight_layout()
plt.savefig("test_images/threshold_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# Create overlay visualization
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Satellite Image", fontsize=14, fontweight='bold')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(final_mask, cmap='gray')
plt.title(f"Detected Features (Threshold {optimal_threshold})", fontsize=14, fontweight='bold')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(image)
plt.imshow(optimal_mask, cmap='Reds', alpha=0.5)
plt.title(f"Features Overlay\n({optimal_coverage:.1f}% coverage)", fontsize=14, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.savefig("test_images/final_result.png", dpi=300, bbox_inches='tight')
plt.show()

# Save different versions
for thresh in [0.65, 0.7, 0.75]:
    mask, _, coverage = results[thresh]
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img.save(f"test_images/features_threshold_{thresh:.2f}.png")
    print(f"💾 Saved: features_threshold_{thresh:.2f}.png ({coverage:.1f}% coverage)")

print(f"\n✅ SEGMENTATION COMPLETE!")
print(f"📁 Output files:")
print(f"   • geographical_features_mask.png - Main result")
print(f"   • threshold_comparison.png - Comparison of different thresholds")
print(f"   • final_result.png - Side-by-side visualization")
print(f"   • features_threshold_X.XX.png - Individual threshold results")

print(f"\n💡 Your optimal threshold of 0.7 successfully differentiates geographical features!")
print(f"   This threshold captures {optimal_coverage:.1f}% of the image as features.")
