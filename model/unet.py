import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder
        self.encoder1 = self.conv_block(3, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoding
        enc1 = self.encoder1(x)                # 64x64x64
        enc2 = self.encoder2(self.pool(enc1))  # 32x32x128
        enc3 = self.encoder3(self.pool(enc2))  # 16x16x256
        enc4 = self.encoder4(self.pool(enc3))  # 8x8x512
        
        # Decoding
        dec3 = self.upconv3(enc4)     # 16x16x256
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)    # 16x16x256
        
        dec2 = self.upconv2(dec3)     # 32x32x128
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)    # 32x32x128
        
        dec1 = self.upconv1(dec2)     # 64x64x64
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)    # 64x64x64
        
        out = self.final_conv(dec1)   # 64x64x3
        
        return out
    
    def to(self, device):
        super(UNet, self).to(device)
        return self