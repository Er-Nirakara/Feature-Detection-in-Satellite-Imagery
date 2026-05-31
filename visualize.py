import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

import config
from model import LightweightUNet
from train import LazyTileDataset

def mask_to_rgb(mask):
    """
    Converts our 3-channel binary mask into a colorful RGB image for visualization.
    Buildings (Ch 0) -> Red
    Roads (Ch 1)     -> Blue
    Vegetation(Ch 2) -> Green
    """
    h, w = mask.shape[1], mask.shape[2]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    
    rgb[:, :, 0] = mask[0] # Red = Buildings
    rgb[:, :, 2] = mask[1] # Blue = Roads
    rgb[:, :, 1] = mask[2] # Green = Vegetation
    
    return rgb

def main():
    print("--- Phase 6: Visualizing Model Predictions ---")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load the architecture and the trained weights
    model = LightweightUNet(in_channels=config.IN_CHANNELS, out_classes=config.OUT_CLASSES)
    model_path = "checkpoints/unet_best_poc.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}. Did the model save during training?")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Set to evaluation mode
    
    # 2. Load the dataset
    dataset = LazyTileDataset(config.CACHE_DIR)
    
    # 3. Pick 3 random micro-tensors to visualize
    # Seed for reproducible "random" tiles that hopefully have some features
    random.seed(42) 
    indices = random.sample(range(len(dataset)), 3)
    
    # Setup matplotlib figure
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    plt.suptitle("Satellite Image Feature Detection: Ground Truth vs. Prediction", fontsize=16, y=0.95)
    
    for i, idx in enumerate(indices):
        img_tensor, gt_mask_tensor = dataset[idx]
        
        # 4. Predict
        with torch.no_grad():
            # Add batch dimension: (C, H, W) -> (1, C, H, W)
            input_tensor = img_tensor.unsqueeze(0).to(device)
            output = model(input_tensor)
            
            # Apply Sigmoid and threshold at 50% confidence
            pred_mask = (torch.sigmoid(output[0]) > 0.5).cpu().numpy()
            
        # 5. Format arrays for Matplotlib (H, W, Channels)
        # Take only the first 3 channels (RGB) for the original image display
        img_rgb = img_tensor[:3].numpy().transpose(1, 2, 0)
        
        # 6. Plotting
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title("Original Satellite (RGB)")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask_to_rgb(gt_mask_tensor.numpy()))
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(mask_to_rgb(pred_mask))
        axes[i, 2].set_title("U-Net Prediction")
        axes[i, 2].axis('off')
        
    # Save the plot to your folder
    plt.tight_layout()
    save_path = "visualization_output.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Success! Saved visualization to: {save_path}")
    
    # Open the window to show you the result
    plt.show()

if __name__ == "__main__":
    main()

    