"""
coop.py — CLIP + Prompt Tuning (CoOp) classifier.

Architecture:
    At init: inject learnable ctx vectors into CLIP's token embedding layer
    image → [TTA 5-view] → CLIP image features [512]
          → CoOp text features via custom embedding forward
            [sos, ctx_0..ctx_3, class_tokens, eos] → CLIP text encoder → [37, 512]
          → cosine similarity × logit_scale.exp() → top-K

Weights: backend/weights/coop_context_full.pt  (ctx tensor [4, 512])
Accuracy (benchmark Full Trainval + TTA 5-view): 91.61%
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import List, Union

import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from app.ai.base import Classifier, ClassifierResult, PredictionResultItem
from app.ai.shared_backbone import (
    BREED_CATALOGUE, OXFORD_CLASSES, get_clip_backbone, projected_features
)

logger = logging.getLogger(__name__)

WEIGHTS_FILE = Path(__file__).parent.parent.parent / "weights" / "coop_context_full.pt"
NUM_CTX = 4
MODEL_VERSION = "clip-vit-b32-coop-tta"


CLASS_NAMES = OXFORD_CLASSES


def _construct_coop_input_ids(
    class_names: List[str],
    processor,
    num_ctx: int = 4,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Build input_ids with placeholder tokens for learnable ctx vectors."""
    result = []
    for name in class_names:
        tokens = processor.tokenizer(name, return_tensors="pt")["input_ids"][0]
        sos        = tokens[:1]
        name_tokens = tokens[1:-1]
        eos        = tokens[-1:]
        ctx_tokens = torch.tensor([49400 + i for i in range(num_ctx)])
        seq = torch.cat([sos, ctx_tokens, name_tokens, eos])
        pad_len = 77 - len(seq)
        if pad_len > 0:
            seq = torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)])
        else:
            seq = seq[:77]
            seq[-1] = eos[0]
        result.append(seq)
    return torch.stack(result).to(device)


class CoOpClassifier(Classifier):
    """
    CLIP + Context Optimization (CoOp) Prompt Tuning classifier.
    Injects 4 learnable context vectors into CLIP's text token embeddings.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self._clip, self._processor, self._device = get_clip_backbone()

        if not WEIGHTS_FILE.exists():
            raise FileNotFoundError(
                f"CoOp context weights not found: {WEIGHTS_FILE}\n"
                "Run: python scripts/export_weights.py"
            )

        # Load ctx vectors [4, 512]
        ctx_data = torch.load(WEIGHTS_FILE, map_location=self._device)
        if ctx_data.shape != (NUM_CTX, 512):
            raise ValueError(f"Unexpected ctx shape: {ctx_data.shape}, expected [{NUM_CTX}, 512]")
        self._ctx = ctx_data.to(self._device)

        # Build input_ids once (constant structure, only embedding values change)
        self._coop_input_ids = _construct_coop_input_ids(
            CLASS_NAMES, self._processor, NUM_CTX, self._device
        )

        # Patch CLIP token embedding to inject learned context
        self._patch_embeddings()

        # Pre-compute & cache text features (ctx is fixed after loading)
        self._text_features = self._build_text_features()
        logger.info("[CoOp] Loaded ctx weights and built text features.")

    def _patch_embeddings(self):
        """Monkey-patch token embedding forward to inject ctx vectors."""
        ctx = self._ctx
        orig_fwd = self._clip.text_model.embeddings.token_embedding.forward

        def custom_fwd(input_ids, *args, **kwargs):
            emb = orig_fwd(input_ids, *args, **kwargs).clone()
            for i in range(NUM_CTX):
                mask = (input_ids == (49400 + i))
                if mask.any():
                    emb[mask] = ctx[i].to(emb.dtype)
            return emb

        self._clip.text_model.embeddings.token_embedding.forward = custom_fwd

    def _build_text_features(self) -> torch.Tensor:
        """Compute CoOp text features with learned context vectors."""
        with torch.no_grad():
            tf = self._clip.get_text_features(input_ids=self._coop_input_ids)
            tf = projected_features(tf)
            return F.normalize(tf, dim=-1)  # [37, 512]

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

        logit_scale = self._clip.logit_scale.exp()
        logits = (img_feat @ self._text_features.t()) * logit_scale
        logits = logits.squeeze(0)  # [37]
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
