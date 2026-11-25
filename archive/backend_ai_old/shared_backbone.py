"""
shared_backbone.py — Singleton CLIP ViT-B/32 backbone shared across all classifiers.

All five methods (Zero-Shot, Tip-Adapter Free, Tip-Adapter-F, CLIP-Adapter,
and CoOp) load the same CLIP model instance to avoid duplicate model loads.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from app.ai.common import BREED_CATALOGUE, WEIGHTS_DIR, get_device

logger = logging.getLogger(__name__)

MODEL_ID = "openai/clip-vit-base-patch32"
MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"

# Prompt template used by the benchmark notebook.
PROMPT_TEMPLATES = [
    "Picture of a pet with breed: {c}, which is a type of animal."
]

# Exact class names from torchvision.datasets.OxfordIIITPet.classes
OXFORD_CLASSES = [
    'Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 
    'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair', 
    'Chihuahua', 'Egyptian Mau', 'English Cocker Spaniel', 'English Setter', 
    'German Shorthaired', 'Great Pyrenees', 'Havanese', 'Japanese Chin', 
    'Keeshond', 'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'Newfoundland', 
    'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard', 
    'Samoyed', 'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx', 
    'Staffordshire Bull Terrier', 'Wheaten Terrier', 'Yorkshire Terrier'
]

# ─── Singleton state ──────────────────────────────────────────────────────────
_clip_model: Optional[CLIPModel] = None
_clip_processor: Optional[CLIPProcessor] = None
_text_features_enr: Optional[torch.Tensor] = None  # [37, 512] enriched
_device: Optional[torch.device] = None
_artifacts_verified = False


def projected_features(output) -> torch.Tensor:
    """Unwrap the already-projected output returned by CLIP get_*_features."""
    if isinstance(output, torch.Tensor):
        return output
    for attribute in ("text_embeds", "image_embeds", "pooler_output"):
        value = getattr(output, attribute, None)
        if isinstance(value, torch.Tensor):
            return value
    raise TypeError(f"Unsupported CLIP feature output: {type(output)!r}")


def verify_clip_artifacts() -> None:
    """Fail fast when deployed CLIP files do not match the export manifest."""
    global _artifacts_verified
    if _artifacts_verified:
        return

    manifest_path = WEIGHTS_DIR / "clip_benchmark_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing CLIP artifact manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_id") != MODEL_ID or manifest.get("model_revision") != MODEL_REVISION:
        raise ValueError("CLIP artifact manifest targets a different backbone")

    for name, expected_hash in manifest.get("artifacts", {}).items():
        if Path(name).name != name:
            raise ValueError(f"Invalid artifact name in manifest: {name}")
        artifact_path = WEIGHTS_DIR / name
        if not artifact_path.exists():
            raise FileNotFoundError(f"Missing CLIP artifact: {artifact_path}")
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if digest != expected_hash:
            raise ValueError(f"CLIP artifact hash mismatch: {name}")

    _artifacts_verified = True


def get_clip_backbone() -> Tuple[CLIPModel, CLIPProcessor, torch.device]:
    """Return singleton CLIP model + processor, loading once on first call."""
    global _clip_model, _clip_processor, _device

    if _clip_model is None:
        verify_clip_artifacts()
        _device = get_device()
        logger.info(f"[SharedBackbone] Loading {MODEL_ID} on {_device}...")
        try:
            _clip_model = CLIPModel.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                local_files_only=True,
                use_safetensors=False,
            ).to(_device)
            _clip_processor = CLIPProcessor.from_pretrained(
                MODEL_ID, revision=MODEL_REVISION, local_files_only=True
            )
        except OSError:
            logger.info("[SharedBackbone] Local cache missing; downloading pinned CLIP files...")
            _clip_model = CLIPModel.from_pretrained(
                MODEL_ID, revision=MODEL_REVISION, use_safetensors=False
            ).to(_device)
            _clip_processor = CLIPProcessor.from_pretrained(
                MODEL_ID, revision=MODEL_REVISION
            )
        _clip_model.eval()
        logger.info("[SharedBackbone] CLIP backbone ready.")

    return _clip_model, _clip_processor, _device


def get_text_features() -> torch.Tensor:
    """
    Return cached enriched text embeddings [37, 512] — L2 normalized.
    Exported from the benchmark's single enriched prompt per breed.
    """
    global _text_features_enr

    if _text_features_enr is None:
        verify_clip_artifacts()
        weights_pt = WEIGHTS_DIR / "text_features_enr.pt"

        if weights_pt.exists():
            logger.info("[SharedBackbone] Loading benchmark text_features_enr.pt...")
            _text_features_enr = torch.load(
                weights_pt, map_location=get_device(), weights_only=False
            )
        else:
            raise FileNotFoundError(
                f"Missing benchmark artifact: {weights_pt}. "
                "Run scripts/export_clip_weights.py before starting the API."
            )

        if _text_features_enr.shape != (len(OXFORD_CLASSES), 512):
            raise ValueError(
                f"Invalid text feature shape: {_text_features_enr.shape}; expected (37, 512)"
            )
        _text_features_enr = _text_features_enr.float()
        max_norm_error = (_text_features_enr.norm(dim=-1) - 1.0).abs().max().item()
        if not torch.isfinite(_text_features_enr).all() or max_norm_error > 1e-4:
            raise ValueError("Invalid benchmark text feature values")

    return _text_features_enr


def _build_text_features_from_scratch() -> torch.Tensor:
    """Compute text embeddings using the benchmark prompt."""
    model, processor, device = get_clip_backbone()

    prompts = []
    for c in OXFORD_CLASSES:
        for tmpl in PROMPT_TEMPLATES:
            prompts.append(tmpl.format(c=c))

    inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.get_text_features(**inputs)
        text_feats = projected_features(out)
        text_feats = F.normalize(text_feats, dim=-1)

    n_breeds = len(BREED_CATALOGUE)
    n_templates = len(PROMPT_TEMPLATES)
    text_feats = text_feats.view(n_breeds, n_templates, -1).mean(dim=1)
    return F.normalize(text_feats, dim=-1)


def get_logit_scale() -> float:
    """Return CLIP logit scale (exp of learned parameter)."""
    model, _, _ = get_clip_backbone()
    return model.logit_scale.exp().item()
