"""Shared frozen ResNet-50 backbone for all ResNet classifiers."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

from app.ai.shared_backbone import get_device


logger = logging.getLogger(__name__)

_backbone: Optional[nn.Module] = None
_preprocess = None
_device: Optional[torch.device] = None


def get_resnet50_backbone() -> Tuple[nn.Module, object, torch.device]:
    """Return one frozen ImageNet ResNet-50 backbone shared by all heads."""
    global _backbone, _preprocess, _device

    if _backbone is None:
        _device = get_device()
        weights = models.ResNet50_Weights.DEFAULT
        _preprocess = weights.transforms()
        full_model = models.resnet50(weights=weights)
        _backbone = nn.Sequential(*list(full_model.children())[:-1]).to(_device)
        _backbone.eval()
        for parameter in _backbone.parameters():
            parameter.requires_grad_(False)
        logger.info("[SharedResNet] ResNet-50 backbone ready on %s.", _device)

    return _backbone, _preprocess, _device
