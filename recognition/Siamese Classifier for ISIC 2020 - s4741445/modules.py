import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import config


class ResNet50Embedder(nn.Module):
    # Extracts 2048-d features from ResNet50 and projects to 256-d embedding
    def __init__(self):
        super().__init__()
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()  # remove final classifier
        self.backbone = m
        self.proj = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256),
        )

    def forward(self, x):
        f = self.backbone(x)        # feature vector
        z = self.proj(f)            # projected embedding
        return nn.functional.normalize(z, p=2, dim=1)  # L2 normalize


class SiameseTriplet(nn.Module):
    # Siamese model for triplet loss (anchor, positive, negative)
    def __init__(self):
        super().__init__()
        self.embed = ResNet50Embedder()

    def forward_once(self, x):
        return self.embed(x)  # single forward pass for one image


class HeadBinaryClassifier(nn.Module):
    # Small MLP classifier on top of embeddings
    def __init__(self, in_dim=256, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(inplace=True), nn.Dropout(0.25),
            nn.Linear(128, 32), nn.ReLU(inplace=True), nn.Dropout(0.10),
            nn.Linear(32, num_classes),
        )

    def forward(self, z):
        return self.net(z)  # class logits
