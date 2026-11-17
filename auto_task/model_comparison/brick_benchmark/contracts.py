"""Original-image pixel contracts shared by all five methods."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CLASSES = ('cake', 'banana', 'doll', 'clock')


@dataclass
class Instance:
    category: str
    score: float
    mask: Any
    corners: Optional[List[List[float]]] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Record:
    image_id: Any
    image_path: Path
    width: int
    height: int
    scene_id: str
    instances: List[Instance]


@dataclass
class Dataset:
    classes: Tuple[str, ...]
    records: List[Record]
    manifest_path: Path
    fingerprint: str
    split: str


def validate_instance(instance, height, width):
    import math
    import numpy as np
    if not isinstance(instance, Instance):
        raise ValueError('backend must return Instance records')
    if instance.category not in CLASSES + ('unknown',):
        raise ValueError('unknown class mapping: ' + str(instance.category))
    if not math.isfinite(float(instance.score)) or not 0 <= instance.score <= 1:
        raise ValueError('score must be finite in [0,1]')
    mask = instance.mask
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != (height, width):
        raise ValueError('mask must be bool in original H x W image coordinates')
    if not mask.any():
        raise ValueError('empty instance mask')
    if instance.corners is not None:
        points = np.asarray(instance.corners, dtype=float)
        if (points.shape != (4, 2) or not np.isfinite(points).all() or
                (points < 0).any() or (points[:, 0] >= width).any() or
                (points[:, 1] >= height).any()):
            raise ValueError('corners must be four finite original-image points')
        import cv2
        if not cv2.isContourConvex(points.astype(np.float32).reshape(-1, 1, 2)):
            raise ValueError('corners must be cyclic convex points')
