"""
clip_zero_shot.py — CLIP Zero-Shot Baseline classifier using shared backbone.

Pipeline: image → [TTA 5-view] → CLIP image encoder → cosine similarity
          with enriched text embeddings → softmax → top-K results

Accuracy (benchmark Full Trainval + TTA 5-view): 87.76%
"""
from __future__ import annotations

import io
import logging
from typing import List, Optional, Union
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from app.ai.base import Classifier, ClassifierResult, PredictionResultItem
from app.ai.shared_backbone import (
    BREED_CATALOGUE, get_clip_backbone, get_text_features, get_logit_scale,
    projected_features,
)

from app.services.image import (
    get_raw_view, get_denoise_view, get_clahe_view, get_optimal_view, get_geom_view
)

logger = logging.getLogger(__name__)

MODEL_VERSION = "clip-vit-b32-zero-shot-tta"


class ZeroShotClassifier(Classifier):
    """
    CLIP Zero-Shot Baseline classifier.
    No training required — pure prompt-based classification.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self._model, self._processor, self._device = get_clip_backbone()
        self._text_features = get_text_features()  # [37, 512]
        self._logit_scale = get_logit_scale()
        logger.info("[ZeroShot] Ready.")

    def _load_image(self, image: Union[str, bytes, PILImage.Image]) -> PILImage.Image:
        if isinstance(image, str):
            return PILImage.open(image).convert("RGB")
        elif isinstance(image, bytes):
            return PILImage.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, PILImage.Image):
            return image.convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _encode_image(self, img: PILImage.Image) -> torch.Tensor:
        """Encode a single PIL image → L2-normalized [1, 512]."""
        inputs = self._processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.get_image_features(**inputs)
            img_feats = projected_features(out)
            return F.normalize(img_feats, dim=-1)

    def _encode_tta(self, img: PILImage.Image) -> torch.Tensor:
        """5-view TTA → average L2-normalized embedding [1, 512]."""
        views = [
            get_raw_view(img),
            get_denoise_view(img),
            get_clahe_view(img),
            get_optimal_view(img),
            get_geom_view(img)
        ]
        embeddings = [self._encode_image(v) for v in views]
        stacked = torch.cat(embeddings, dim=0)   # [5, 512]
        return F.normalize(stacked.mean(dim=0, keepdim=True), dim=-1)

    def predict(self, image: Union[str, bytes, PILImage.Image]) -> ClassifierResult:
        pil = self._load_image(image)
        img_feat = self._encode_tta(pil)  # [1, 512]

        logits = (img_feat @ self._text_features.T).squeeze(0)  # [37]
        probs = F.softmax(logits * self._logit_scale, dim=0)

        k = min(self.top_k, len(BREED_CATALOGUE))
        top_probs, top_idx = torch.topk(probs, k=k)

        top_k: List[PredictionResultItem] = []
        for prob, idx in zip(top_probs.detach().cpu().tolist(), top_idx.detach().cpu().tolist()):
            breed_id, breed_name, species, _ = BREED_CATALOGUE[idx]
            top_k.append(PredictionResultItem(
                breed_id=breed_id, breed_name=breed_name,
                species=species, score=round(prob, 6)
            ))

        best = top_k[0]
        return ClassifierResult(
            predicted_breed_id=best.breed_id,
            score=round(best.score, 6),
            top_k_results=top_k,
            model_version=MODEL_VERSION,
        )
