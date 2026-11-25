"""
tip_adapter.py — CLIP Tip-Adapter (Training-Free Cache Model)
"""
import io
import logging
from pathlib import Path
from typing import List, Union

import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from app.ai.base import Classifier, ClassifierResult, PredictionResultItem
from app.ai.shared_backbone import (
    BREED_CATALOGUE, get_clip_backbone, get_text_features, get_logit_scale,
    projected_features,
)

logger = logging.getLogger(__name__)

CACHE_WEIGHTS = Path(__file__).parent.parent.parent / "weights" / "tip_adapter_cache.pt"
MODEL_VERSION = "tip-adapter-free-v1"

class TipAdapterClassifier(Classifier):
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self._model, self._processor, self._device = get_clip_backbone()
        self.logit_scale = get_logit_scale()
        self.text_features = get_text_features()

        if not CACHE_WEIGHTS.exists():
            raise FileNotFoundError(f"Missing cache weights: {CACHE_WEIGHTS}")

        
        # Load Support Cache (W_1 and Labels)
        cache_data = torch.load(CACHE_WEIGHTS, map_location=self._device)
        self.W_1 = cache_data["W_1"] # [N, 512]
        self.labels = cache_data["labels"] # [N]
        
        # Hyperparameters based on grid search in benchmark (Val Acc: 88.11%)
        self.alpha = 1.0
        self.beta = 10.0

    def _encode_image(self, img: PILImage.Image) -> torch.Tensor:
        """Encode a single PIL image +' L2-normalized [1, 512]."""
        image_input = self._processor(images=img, return_tensors="pt")["pixel_values"].to(self._device)
        with torch.no_grad():
            out = self._model.get_image_features(pixel_values=image_input)
            image_features = projected_features(out)
            return image_features / image_features.norm(dim=-1, keepdim=True)

    def _encode_tta(self, img: PILImage.Image) -> torch.Tensor:
        """5-view TTA +' average L2-normalized embedding [1, 512]."""
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
        stacked = torch.cat(embeddings, dim=0)   # [5, 512]
        return F.normalize(stacked.mean(dim=0, keepdim=True), dim=-1)

    def predict(self, image: Union[str, bytes, PILImage.Image]) -> ClassifierResult:
        if isinstance(image, str): pil = PILImage.open(image).convert("RGB")
        elif isinstance(image, bytes): pil = PILImage.open(io.BytesIO(image)).convert("RGB")
        else: pil = image.convert("RGB")

        with torch.no_grad():
            image_features = self._encode_tta(pil)
            
            # Text branch logits
            clip_logits = (image_features @ self.text_features.t()) * self.logit_scale
            
            # Cache branch logits
            w1_norm = self.W_1 / self.W_1.norm(dim=-1, keepdim=True)
            aff = torch.exp(-self.beta * (1 - (image_features @ w1_norm.t())))
            K_avg = len(self.W_1) / float(len(BREED_CATALOGUE))
            w2 = F.one_hot(self.labels, num_classes=len(BREED_CATALOGUE)).float().to(self._device)
            cache_logits = ((aff @ w2) / K_avg) * self.logit_scale
            
            # Combine
            logits = clip_logits + cache_logits * self.alpha
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
            predicted_breed_id=best.breed_id,
            score=round(best.score, 6),
            top_k_results=top_k_results,
            model_version=MODEL_VERSION,
        )
