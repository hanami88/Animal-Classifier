"""
resnet50_linear.py — ResNet-50 Linear Probe
"""
import io
import logging
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage

from app.ai.base import Classifier, ClassifierResult, PredictionResultItem
from app.ai.shared_backbone import BREED_CATALOGUE
from app.ai.shared_resnet_backbone import get_resnet50_backbone

logger = logging.getLogger(__name__)

WEIGHTS_FILE = Path(__file__).parent.parent.parent / "weights" / "resnet50_linear.pt"
NUM_CLASSES = 37
MODEL_VERSION = "resnet-50-linear"

class ResNet50LinearClassifier(Classifier):
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self._backbone, self._preprocess, self._device = get_resnet50_backbone()

        if not WEIGHTS_FILE.exists():
            raise FileNotFoundError(f"ResNet-50 Linear weights not found: {WEIGHTS_FILE}")

        self._model = nn.Linear(2048, NUM_CLASSES).to(self._device)
        self._model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=self._device))
        self._model.eval()

    def _encode_tta(self, img: PILImage.Image) -> torch.Tensor:
        from app.services.image import (
            get_raw_view, get_denoise_view, get_clahe_view, get_optimal_view, get_geom_view
        )
        views = [
            get_raw_view(img),
            get_denoise_view(img),
            get_clahe_view(img),
            get_optimal_view(img),
            get_geom_view(img)
        ]
        tensors = [self._preprocess(v).unsqueeze(0).to(self._device) for v in views]
        batch_tensors = torch.cat(tensors, dim=0) 
        with torch.no_grad():
            feats = self._backbone(batch_tensors).squeeze(-1).squeeze(-1) 
        return feats.mean(dim=0, keepdim=True)

    def predict(self, image: Union[str, bytes, PILImage.Image]) -> ClassifierResult:
        if isinstance(image, str): pil = PILImage.open(image).convert("RGB")
        elif isinstance(image, bytes): pil = PILImage.open(io.BytesIO(image)).convert("RGB")
        else: pil = image.convert("RGB")

        img_feat = self._encode_tta(pil)
        
        with torch.no_grad():
            logits = self._model(img_feat)
            probs = F.softmax(logits.squeeze(0), dim=0)

        k = min(self.top_k, len(BREED_CATALOGUE))
        top_probs, top_idx = torch.topk(probs, k=k)

        top_k_results: List[PredictionResultItem] = []
        for prob, idx in zip(top_probs.detach().cpu().tolist(), top_idx.detach().cpu().tolist()):
            breed_id, breed_name, species, _ = BREED_CATALOGUE[idx]
            top_k_results.append(PredictionResultItem(
                breed_id=breed_id, breed_name=breed_name,
                species=species, score=round(prob, 6)
            ))

        best = top_k_results[0]
        return ClassifierResult(
            predicted_breed_id=best.breed_id, score=round(best.score, 6),
            top_k_results=top_k_results, model_version=MODEL_VERSION
        )
