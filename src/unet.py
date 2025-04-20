import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    input_format = "flat_channels"  # 输入为 [B, T×C, H, W]
    def __init__(self, in_channels=8, num_classes=2, base_c=64):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, base_c)
        self.enc2 = conv_block(base_c, base_c * 2)
        self.enc3 = conv_block(base_c * 2, base_c * 4)
        self.enc4 = conv_block(base_c * 4, base_c * 8)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(base_c * 8, base_c * 16)

        self.up4 = nn.ConvTranspose2d(base_c * 16, base_c * 8, kernel_size=2, stride=2)
        self.dec4 = conv_block(base_c * 16, base_c * 8)
        self.up3 = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        self.dec3 = conv_block(base_c * 8, base_c * 4)
        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(base_c * 4, base_c * 2)
        self.up1 = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.dec1 = conv_block(base_c * 2, base_c)

        self.out_conv = nn.Conv2d(base_c, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out_conv(d1)
        return {"out": out}  # 返回标准 dict
