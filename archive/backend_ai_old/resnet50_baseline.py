"""
resnet50_baseline.py — ResNet-50 Baseline (Raw Image + Linear Probe)
"""
import io
import logging
from pathlib import Path
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage

from app.ai.base import Classifier, ClassifierResult, PredictionResultItem
from app.ai.shared_backbone import BREED_CATALOGUE
from app.ai.shared_resnet_backbone import get_resnet50_backbone

logger = logging.getLogger(__name__)

WEIGHTS_FILE = Path(__file__).parent.parent.parent / "weights" / "resnet50_baseline.pt"
NUM_CLASSES = 37
MODEL_VERSION = "resnet-50-baseline"

class ResNet50BaselineClassifier(Classifier):
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self._backbone, self._preprocess, self._device = get_resnet50_backbone()

        if not WEIGHTS_FILE.exists():
            raise FileNotFoundError(f"ResNet-50 Baseline weights not found: {WEIGHTS_FILE}")

        self._model = nn.Linear(2048, NUM_CLASSES).to(self._device)
        self._model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=self._device))
        self._model.eval()

    def _encode_image(self, img: PILImage.Image) -> torch.Tensor:
        image_input = self._preprocess(img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            img_feat = self._backbone(image_input).squeeze(-1).squeeze(-1)
            return img_feat
            
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
        embeddings = [self._encode_image(v) for v in views]
        stacked = torch.cat(embeddings, dim=0)   # [5, 2048]
        return stacked.mean(dim=0, keepdim=True)

    def predict(self, image: Union[str, bytes, PILImage.Image]) -> ClassifierResult:
        if isinstance(image, str): pil = PILImage.open(image).convert("RGB")
        elif isinstance(image, bytes): pil = PILImage.open(io.BytesIO(image)).convert("RGB")
        else: pil = image.convert("RGB")
        
        with torch.no_grad():
            img_feat = self._encode_image(pil)
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
