from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(pretrained: bool = True):
    """Build an EfficientNet-B0 binary classifier."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model


def freeze_backbone(model, freeze: bool = True):
    """Freeze feature extraction layers for a transfer-learning baseline."""
    for parameter in model.features.parameters():
        parameter.requires_grad = not freeze
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    return model
