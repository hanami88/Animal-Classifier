"""
clip_adapter.py — CLIP + Feature Adapter classifier.

Architecture:
    image → [TTA 5-view] → CLIP image features [512]
          → Bottleneck MLP (512→64→512, ReLU, residual α=0.2)
          → L2 normalize
          → cosine similarity with text features × logit_scale → top-K

Weights: backend/weights/clip_adapter_full.pt  (AdapterClassifier state_dict)
Accuracy (benchmark Full Trainval + TTA 5-view): 90.27%
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage

from app.ai.base import Classifier, ClassifierResult, PredictionResultItem
from app.ai.shared_backbone import (
    BREED_CATALOGUE, get_clip_backbone, get_text_features, get_logit_scale,
    projected_features,
)

logger = logging.getLogger(__name__)

WEIGHTS_FILE = Path(__file__).parent.parent.parent / "weights" / "clip_adapter_full.pt"
MODEL_VERSION = "clip-vit-b32-clip-adapter-tta"



class _CLIPAdapter(nn.Module):
    """Bottleneck MLP with residual blend — matches export_weights.py architecture."""
    def __init__(self, input_dim: int = 512, bottleneck_dim: int = 64, alpha: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, input_dim),
        )
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.mlp(x)
        x_adapted = self.alpha * res + (1.0 - self.alpha) * x
        return x_adapted / x_adapted.norm(dim=-1, keepdim=True)


class _AdapterClassifier(nn.Module):
    """Full classifier: adapter + text-feature cosine head."""
    def __init__(self, adapter: _CLIPAdapter, text_features: torch.Tensor, logit_scale: float):
        super().__init__()
        self.adapter = adapter
        self.register_buffer("text_features", text_features.clone())
        self.logit_scale = logit_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.adapter(x) @ self.text_features.t()) * self.logit_scale


class CLIPAdapterClassifier(Classifier):
    """CLIP + Feature Adapter (Residual MLP) classifier."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self._clip, self._processor, self._device = get_clip_backbone()
        text_feats = get_text_features()
        logit_scale = get_logit_scale()

        # Reconstruct model architecture and load weights
        adapter = _CLIPAdapter(input_dim=512, bottleneck_dim=64, alpha=0.2).to(self._device)

        if not WEIGHTS_FILE.exists():
            raise FileNotFoundError(
                f"CLIP-Adapter weights not found: {WEIGHTS_FILE}\n"
                "Run: python scripts/export_weights.py"
            )
        state = torch.load(WEIGHTS_FILE, map_location=self._device)
        adapter.load_state_dict(state)
        
        self._model = _AdapterClassifier(adapter, text_feats, logit_scale).to(self._device)
        self._model.eval()
        logger.info(f"[CLIPAdapter] Loaded weights from {WEIGHTS_FILE.name}")

    def _load_image(self, image: Union[str, bytes, PILImage.Image]) -> PILImage.Image:
        if isinstance(image, str):
            return PILImage.open(image).convert("RGB")
        elif isinstance(image, bytes):
            return PILImage.open(io.BytesIO(image)).convert("RGB")
        return image.convert("RGB")

    def _encode_image(self, img: PILImage.Image) -> torch.Tensor:
        inputs = self._processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._clip.get_image_features(**inputs)
            feats = projected_features(out)
            return F.normalize(feats, dim=-1)

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
        pil = self._load_image(image)
        img_feat = self._encode_tta(pil)  # [1, 512]

        with torch.no_grad():
            logits = self._model(img_feat).squeeze(0)  # [37]
        probs = F.softmax(logits, dim=0)

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
