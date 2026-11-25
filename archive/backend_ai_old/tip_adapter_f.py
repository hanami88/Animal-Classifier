"""
tip_adapter_f.py — CLIP + Tip-Adapter-F (Fine-Tuned Cache Adapter) classifier.

Architecture:
    image → [TTA 5-view] → CLIP image features [512]
          → TipAdapterF.forward(x):
              logits_zs   = x @ W_text.T * logit_scale          (zero-shot branch)
              A           = exp(-β * (1 - x @ W_1_norm.T))      (affinity matrix)
              logits_cache = (A @ W_2) / K_avg * logit_scale    (cache branch)
              return logits_zs + α * logits_cache
          → argmax → top-K

Weights: backend/weights/tip_adapter_f_full.pt  (TipAdapterF state_dict)
Accuracy (benchmark Full Trainval + TTA 5-view): 89.56%
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

WEIGHTS_FILE = Path(__file__).parent.parent.parent / "weights" / "tip_adapter_f_full.pt"
NUM_CLASSES = 37
MODEL_VERSION = "clip-vit-b32-tip-adapter-f-tta"



class _TipAdapterF(nn.Module):
    """Fine-tuned cache model — matches export_weights.py architecture."""
    def __init__(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: float,
        alpha: float = 1.0,
        beta: float = 10.0,
    ):
        super().__init__()
        self.W_1 = nn.Parameter(support_features.clone())
        self.register_buffer("W_2", F.one_hot(support_labels, num_classes=NUM_CLASSES).float())
        self.register_buffer("W_text", text_features.clone())
        self.alpha = alpha
        self.beta = beta
        self.logit_scale = logit_scale
        self.register_buffer("K_avg", torch.tensor(len(support_features) / float(NUM_CLASSES)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits_zs = (x @ self.W_text.t()) * self.logit_scale
        w1_norm = self.W_1 / self.W_1.norm(dim=-1, keepdim=True)
        A = torch.exp(-self.beta * (1.0 - x @ w1_norm.t()))
        logits_cache = ((A @ self.W_2) / self.K_avg) * self.logit_scale
        return logits_zs + self.alpha * logits_cache


class TipAdapterFClassifier(Classifier):
    """CLIP + Tip-Adapter-F (Fine-Tuned Cache Model) classifier."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self._clip, self._processor, self._device = get_clip_backbone()
        text_feats = get_text_features()
        logit_scale = get_logit_scale()

        if not WEIGHTS_FILE.exists():
            raise FileNotFoundError(
                f"Tip-Adapter-F weights not found: {WEIGHTS_FILE}\n"
                "Run: python scripts/export_weights.py"
            )

        # Load state dict to reconstruct model dimensions
        state = torch.load(WEIGHTS_FILE, map_location=self._device)
        if not torch.allclose(
            state["W_text"].to(text_feats.device), text_feats, atol=1e-5, rtol=1e-5
        ):
            raise ValueError(
                "Tip-Adapter-F checkpoint was trained with different text features. "
                "Re-export CLIP weights with scripts/export_clip_weights.py."
            )
        w1_shape = state["W_1"].shape  # [n_support, 512]
        n_support = w1_shape[0]
        w2_shape = state["W_2"].shape  # [n_support, 37]

        # Reconstruct dummy support tensors (values will be overwritten by load_state_dict)
        dummy_feats = torch.zeros(n_support, 512)
        dummy_labels = torch.zeros(n_support, dtype=torch.long)
        logit_scale_val = state.get("logit_scale", logit_scale)
        if isinstance(logit_scale_val, torch.Tensor):
            logit_scale_val = logit_scale_val.item()

        self._model = _TipAdapterF(
            dummy_feats, dummy_labels, text_feats, logit_scale_val
        ).to(self._device)
        self._model.load_state_dict(state)
        self._model.eval()
        logger.info(f"[TipAdapterF] Loaded weights from {WEIGHTS_FILE.name} (support: {n_support})")

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
