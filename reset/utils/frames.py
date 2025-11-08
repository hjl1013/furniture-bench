"""Frame and image processing utility functions."""
from typing import Dict, List

import numpy as np


def prepare_rgb_frames(obs: Dict) -> List[np.ndarray]:
    """Extract and prepare RGB frames from observation."""
    frames: List[np.ndarray] = []
    for key in [
        "color_image1",
        "color_image2",
        "color_image3",
        "image1",
        "image2",
        "image3",
    ]:
        if key not in obs:
            continue
        img = np.asarray(obs[key])
        if img.ndim != 3:
            continue
        if img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
            img = np.moveaxis(img, 0, -1)
        if img.shape[-1] not in (3, 4):
            continue
        if img.dtype != np.uint8:
            img = img.astype(np.float32)
            if img.max() <= 1.0:
                img *= 255.0
            img = np.clip(img, 0.0, 255.0).astype(np.uint8)
        frames.append(img)
    return frames

