"""
Grounding-DINO wrapper.

Open-vocabulary object localization for the Critic. Mock detections are
available only when explicitly allowed by the caller.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

_gdino_model = None
_gdino_device = "cpu"
_GDINO_AVAILABLE = False


def _try_import_gdino() -> bool:
    global _GDINO_AVAILABLE
    if _GDINO_AVAILABLE:
        return True
    try:
        from groundingdino.util.inference import load_model, predict  # noqa: F401

        _GDINO_AVAILABLE = True
        return True
    except ImportError:
        return False


def _load_gdino() -> None:
    global _gdino_model, _gdino_device
    if _gdino_model is not None:
        return
    from groundingdino.util.inference import load_model
    import torch

    checkpoint_dir = Path(__file__).parent / "checkpoints"
    config_path = checkpoint_dir / "GroundingDINO_SwinT_OGC.py"
    weights_path = checkpoint_dir / "groundingdino_swint_ogc.pth"

    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            "Grounding-DINO checkpoints not found. Place "
            "GroundingDINO_SwinT_OGC.py and groundingdino_swint_ogc.pth "
            f"in {checkpoint_dir}."
        )

    _gdino_device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[GroundingDINO] Loading model on {_gdino_device}")
    _gdino_model = load_model(str(config_path), str(weights_path), device=_gdino_device)
    logger.info("[GroundingDINO] Model loaded")


def detect(
    image: Image.Image,
    text_prompt: str,
    box_threshold: float = 0.30,
    text_threshold: float = 0.25,
    allow_mock: bool = False,
) -> list[dict]:
    """
    Return detections as:
      {"label": str, "confidence": float, "bbox": [x1, y1, x2, y2]}
    with normalized bbox coordinates in [0, 1].
    """
    if not _try_import_gdino():
        if allow_mock:
            logger.warning("[GroundingDINO] Not installed. Using mock detector.")
            return _mock_detect(image, text_prompt)
        raise RuntimeError("Grounding-DINO is not installed")

    try:
        _load_gdino()
        from groundingdino.util.inference import predict
        import groundingdino.datasets.transforms as T

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_tensor, _ = transform(image, None)
        boxes, logits, phrases = predict(
            model=_gdino_model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=_gdino_device,
        )

        results = []
        for box, logit, phrase in zip(boxes.tolist(), logits.tolist(), phrases):
            cx, cy, width, height = box
            x1 = cx - width / 2
            y1 = cy - height / 2
            x2 = cx + width / 2
            y2 = cy + height / 2
            results.append(
                {
                    "label": phrase,
                    "confidence": float(logit),
                    "bbox": [
                        max(0.0, float(x1)),
                        max(0.0, float(y1)),
                        min(1.0, float(x2)),
                        min(1.0, float(y2)),
                    ],
                }
            )
        return results

    except Exception:
        if not allow_mock:
            raise
        logger.exception("[GroundingDINO] Inference failed. Using mock detector.")
        return _mock_detect(image, text_prompt)


def _mock_detect(image: Image.Image, text_prompt: str) -> list[dict]:
    """
    Deterministic mock detector for demos and tests.

    It preserves multiword object phrases separated with periods, matching the
    Critic prompt format: "object one . object two".
    """
    phrases = [p.strip() for p in text_prompt.split(".") if p.strip()]
    if len(phrases) < 2:
        tokens = text_prompt.split()
        phrases = [
            tokens[0] if tokens else "object1",
            tokens[-1] if len(tokens) > 1 else "object2",
        ]

    return [
        {
            "label": phrases[0],
            "confidence": 0.85,
            "bbox": [0.05, 0.2, 0.45, 0.8],
        },
        {
            "label": phrases[1],
            "confidence": 0.82,
            "bbox": [0.55, 0.2, 0.95, 0.8],
        },
    ]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def best_match(detections: list[dict], query: str) -> Optional[dict]:
    """Pick the detection whose label best matches the query string."""
    if not detections:
        return None

    query_tokens = _tokens(query)
    if not query_tokens:
        return None

    scored = []
    for det in detections:
        label_tokens = _tokens(str(det.get("label", "")))
        overlap = len(query_tokens & label_tokens)
        confidence = float(det.get("confidence", 0.0))
        scored.append((overlap, confidence, det))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if scored[0][0] <= 0:
        return None
    return scored[0][2]
