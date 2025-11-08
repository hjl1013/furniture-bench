"""File I/O utility functions."""
from pathlib import Path
from typing import Iterable

import numpy as np


def iter_annotation_files(dataset_path: Path) -> Iterable[Path]:
    """Iterate over annotation files."""
    if dataset_path.is_file():
        yield dataset_path
        return

    if dataset_path.suffix == ".pkl":
        yield dataset_path
        return

    for pkl_path in sorted(dataset_path.rglob("*.pkl")):
        if pkl_path.is_file():
            yield pkl_path


def to_numpy(array_like) -> np.ndarray:
    """Convert to numpy array."""
    if isinstance(array_like, np.ndarray):
        return array_like.astype(np.float32, copy=True)
    return np.array(array_like, dtype=np.float32)

