"""File I/O utility functions."""
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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
    """Convert to numpy array, handling PyTorch tensors (including CUDA tensors)."""
    if isinstance(array_like, np.ndarray):
        return array_like.astype(np.float32, copy=True)
    
    # Handle PyTorch tensors
    if TORCH_AVAILABLE and isinstance(array_like, torch.Tensor):
        # Move to CPU if on CUDA, then convert to numpy
        if array_like.is_cuda:
            array_like = array_like.cpu()
        return array_like.numpy().astype(np.float32)
    
    return np.array(array_like, dtype=np.float32)

