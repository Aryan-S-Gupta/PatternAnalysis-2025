import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import config


class ResNet50Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()  # 2048-d
        self.backbone = m
        self.proj = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256),
        )

    def forward(self, x):
        f = self.backbone(x)        # [B,2048]
        z = self.proj(f)            # [B,256]
        return nn.functional.normalize(z, p=2, dim=1)


class SiameseTriplet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = ResNet50Embedder()

    def forward_once(self, x):
        return self.embed(x)


class HeadBinaryClassifier(nn.Module):
    def __init__(self, in_dim=256, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(inplace=True), nn.Dropout(0.25),
            nn.Linear(128,  32),    nn.ReLU(inplace=True), nn.Dropout(0.10),
            nn.Linear(32, num_classes),
        )

    def forward(self, z):
        return self.net(z)
