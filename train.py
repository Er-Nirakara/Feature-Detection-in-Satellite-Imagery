import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm

import config
from model import LightweightUNet

# 1. SSD-Based Lazy Loading Dataset
class LazyTileDataset(Dataset):
    def __init__(self, cache_dir):
        # Find all image tensor files
        self.img_files = glob.glob(os.path.join(cache_dir, '*_img.npy'))
        
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        mask_path = img_path.replace('_img.npy', '_mask.npy')
        
        # Load from SSD
        img = np.load(img_path)
        mask = np.load(mask_path)
        
        # Numpy shape is (H, W, C). PyTorch expects (C, H, W).
        img = img.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)
        
        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# 2. Custom Loss Function (BCE + Dice Loss)
class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(BCEDiceLoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # Standard BCE
        bce_loss = self.bce(inputs, targets)
        
        # Dice Loss
        inputs_sigmoid = torch.sigmoid(inputs)
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice
        
        return bce_loss + dice_loss

# 3. Jaccard Index (IoU) Metric
def calculate_iou(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds) > threshold
    targets = targets > 0.5
    
    intersection = (preds & targets).float().sum((1, 2))
    union = (preds | targets).float().sum((1, 2))
    
    iou = (intersection + 1e-5) / (union + 1e-5)
    return iou.mean().item()

# 4. Training Loop
def main():
    print("--- Step 1: Initialize paths and model parameters ---")
    
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Used device: {device}")
    
    # Load Model
    model = LightweightUNet(in_channels=config.IN_CHANNELS, out_classes=config.OUT_CLASSES).to(device)
    print(f"Total Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Prepare DataLoader
    full_dataset = LazyTileDataset(config.CACHE_DIR)
    
    # Split into 90% Train / 10% Validation
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # num_workers=0 ensures Windows doesn't duplicate RAM usage
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Hyperparameters from PRD
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    print("\n--- Step 2: Start Model Training ---")
    os.makedirs('checkpoints', exist_ok=True)
    
    best_iou = 0.0
    
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        # Training Phase
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{config.EPOCHS} [Train]", leave=False)
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_iou += calculate_iou(outputs, masks)
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        print(f"Epoch {epoch}/{config.EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}")
        
        # Checkpoint Saving
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), f"checkpoints/unet_best_poc.pth")
            print("-> Model Checkpoint Saved!")

if __name__ == "__main__":
    main()