import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class LightweightUNet(nn.Module):
    def __init__(self, in_channels=4, out_classes=3):
        super(LightweightUNet, self).__init__()
        
        # Encoder (Downsampling) - Starting with 16 filters instead of 64
        self.inc = DoubleConv(in_channels, 16)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        
        # Bottleneck
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # Decoder (Upsampling)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128) # 128 + 128 (from skip connection)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(64, 32)
        
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(32, 16)
        
        # Output Layer (1x1 Convolution mapping to the 3 Target Classes)
        self.outc = nn.Conv2d(16, out_classes, kernel_size=1)

    def forward(self, x):
        # Encoder passes
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder passes with skip connections
        x = self.up1(x5)
        x = self.conv1(torch.cat([x, x4], dim=1))
        
        x = self.up2(x)
        x = self.conv2(torch.cat([x, x3], dim=1))
        
        x = self.up3(x)
        x = self.conv3(torch.cat([x, x2], dim=1))
        
        x = self.up4(x)
        x = self.conv4(torch.cat([x, x1], dim=1))
        
        logits = self.outc(x)
        return logits

# Quick test to ensure the model compiles and fits VRAM constraints
if __name__ == "__main__":
    model = LightweightUNet(in_channels=4, out_classes=3)
    print(f"Total Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Simulate a batch of 4 (Batch, Channels, Height, Width)
    dummy_input = torch.randn(4, 4, 128, 128)
    output = model(dummy_input)
    print(f"Output shape: {output.shape} (Expected: [4, 3, 128, 128])")