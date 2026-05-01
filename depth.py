"""
Depth Anything V2 wrapper.

Returns an H x W numpy array of relative depth where larger values are treated
as farther from the camera. Mock depth is available only when explicitly
allowed by the caller.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_DEPTH_AVAILABLE = False
_depth_pipe = None


def _try_import_depth() -> bool:
    global _DEPTH_AVAILABLE
    if _DEPTH_AVAILABLE:
        return True
    try:
        from transformers import pipeline as hf_pipeline  # noqa: F401

        _DEPTH_AVAILABLE = True
        return True
    except ImportError:
        return False


def _load_depth_pipe() -> None:
    global _depth_pipe
    if _depth_pipe is not None:
        return
    from transformers import pipeline as hf_pipeline

    logger.info("[DepthAnything] Loading Depth Anything V2")
    _depth_pipe = hf_pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
    )
    logger.info("[DepthAnything] Model loaded")


def estimate_depth(image: Image.Image, allow_mock: bool = False) -> np.ndarray:
    """
    Return a float32 numpy array (H, W) of relative depth.
    Values are normalized to [0, 1].
    """
    if not _try_import_depth():
        if allow_mock:
            logger.warning("[DepthAnything] transformers not installed. Using mock depth.")
            return _mock_depth(image)
        raise RuntimeError("Depth Anything dependencies are not installed")

    try:
        _load_depth_pipe()
        result = _depth_pipe(image)
        depth = np.array(result["depth"], dtype=np.float32)
        d_min, d_max = float(depth.min()), float(depth.max())
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)
        return depth.astype(np.float32)
    except Exception:
        if not allow_mock:
            raise
        logger.exception("[DepthAnything] Inference failed. Using mock depth.")
        return _mock_depth(image)


def _mock_depth(image: Image.Image) -> np.ndarray:
    """Mock depth: horizontal gradient (left=near, right=far)."""
    w, h = image.size
    grad = np.linspace(0, 1, w, dtype=np.float32)
    return np.tile(grad, (h, 1))


def median_depth_in_box(depth_map: np.ndarray, bbox: list) -> float:
    """Compute median depth inside a normalized [x1, y1, x2, y2] box."""
    h, w = depth_map.shape
    x1 = max(0, min(w, int(float(bbox[0]) * w)))
    y1 = max(0, min(h, int(float(bbox[1]) * h)))
    x2 = max(0, min(w, int(float(bbox[2]) * w)))
    y2 = max(0, min(h, int(float(bbox[3]) * h)))
    if x2 <= x1 or y2 <= y1:
        return 0.5
    region = depth_map[y1:y2, x1:x2]
    return float(np.median(region))
