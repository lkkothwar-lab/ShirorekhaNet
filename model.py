import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x + self.gamma * (attention * x)


class MultiScaleInception(nn.Module):
    def __init__(self, in_channels, dropout_rate=0.1):
        super().__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 32, 1, bias=False)
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 1, bias=False), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.Dropout2d(dropout_rate))
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 1, bias=False), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, padding=2, bias=False), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.Dropout2d(dropout_rate))
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, 32, 1, bias=False), nn.BatchNorm2d(32),
            ChannelAttention(32, ratio=8))
        self.output_proj = nn.Sequential(
            nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.shortcut = nn.Sequential()
        if in_channels != 128:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, 128, 1, bias=False),
                nn.BatchNorm2d(128))
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        branches = [self.branch1x1(x), self.branch3x3(x),
                    self.branch5x5(x), self.branch_pool(x)]
        concat = torch.cat(branches, dim=1)
        proj = self.output_proj(concat)
        out = proj + identity
        return self.final_relu(out)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_inception=True):
        super().__init__()
        self.use_inception = use_inception
        if use_inception:
            self.conv1 = MultiScaleInception(in_channels)
            mid_channels = 128
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
            mid_channels = out_channels

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels))

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or (use_inception and out_channels != mid_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels))
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return self.final_relu(out)


class ShirorekhaNet(nn.Module):
    def __init__(self, in_channels=4, use_inception=True):
        super().__init__()
        self.enc1 = ResidualBlock(in_channels, 64, use_inception)
        self.enc2 = ResidualBlock(64, 128, use_inception)
        self.enc3 = ResidualBlock(128, 256, use_inception)
        self.enc4 = ResidualBlock(256, 512, use_inception)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlock(512, 512, use_inception)

        self.upconv4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec4 = ResidualBlock(1024, 512, use_inception)   # 512 + 512
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ResidualBlock(512, 256, use_inception)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ResidualBlock(256, 128, use_inception)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ResidualBlock(128, 64, use_inception)

        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1))

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool(e1)
        e2 = self.enc2(p1); p2 = self.pool(e2)
        e3 = self.enc3(p2); p3 = self.pool(e3)
        e4 = self.enc4(p3); p4 = self.pool(e4)
        b = self.bottleneck(p4)

        d4 = self.upconv4(b); d4 = torch.cat([d4, e4], 1); d4 = self.dec4(d4)
        d3 = self.upconv3(d4); d3 = torch.cat([d3, e3], 1); d3 = self.dec3(d3)
        d2 = self.upconv2(d3); d2 = torch.cat([d2, e2], 1); d2 = self.dec2(d2)
        d1 = self.upconv1(d2); d1 = torch.cat([d1, e1], 1); d1 = self.dec1(d1)
        return self.out_conv(d1)
