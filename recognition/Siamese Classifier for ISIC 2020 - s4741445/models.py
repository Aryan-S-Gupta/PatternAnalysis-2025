import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import config


class ResNet50Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        if config.PRETRAINED_BACKBONE:
            m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            m = resnet50(weights=None)
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
