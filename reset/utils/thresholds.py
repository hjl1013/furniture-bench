"""Threshold preparation utility functions."""
from typing import Dict, Optional

from furniture_bench.config import config


def prepare_thresholds(furniture_name: str, gripper_close_threshold: Optional[float] = None, 
                      gripper_close_ratio: float = 0.9, contact_tolerance: float = 0.01,
                      min_consecutive_steps: int = 5) -> Dict[str, float]:
    """
    Prepare thresholds for extraction.
    
    Args:
        furniture_name: Name of furniture type
        gripper_close_threshold: Absolute threshold (m) for gripper width when considered closed
        gripper_close_ratio: Ratio of max gripper width used when threshold is not provided
        contact_tolerance: Distance threshold for contact detection
        min_consecutive_steps: Minimum consecutive steps for action detection
    
    Returns:
        Dictionary with thresholds
    """
    max_width = config["robot"]["max_gripper_width"].get(furniture_name)
    if max_width is None:
        raise ValueError(f"Unknown furniture '{furniture_name}' for gripper width lookup")

    if gripper_close_threshold is not None:
        gripper_thresh = gripper_close_threshold
    else:
        gripper_thresh = max_width * gripper_close_ratio

    return {
        "gripper": gripper_thresh,
        "contact_tolerance": contact_tolerance,
        "min_steps": min_consecutive_steps,
    }
