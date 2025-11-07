import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Optional, Union
import sys
import random

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from furniture_bench.utils import transform as T

# define the paths to the extracted information
GRASP_PATH = PROJECT_ROOT / "reset/extracted_info/grasp_summary.pkl"
INITIAL_STATE_PATH = PROJECT_ROOT / "reset/extracted_info/initial_state.pkl"
OBJECT_AFFORDANCE_PATH = PROJECT_ROOT / "reset/extracted_info/object_affordance_trajectories.json"

# Cache loaded data
_grasp_summary_cache: Optional[Dict] = None
_initial_state_cache: Optional[List[Dict]] = None
_object_affordance_cache: Optional[Dict] = None


def _load_grasp_summary(grasp_path: Optional[Path] = None) -> Dict:
    """Load grasp summary from pickle file."""
    global _grasp_summary_cache
    if _grasp_summary_cache is not None:
        return _grasp_summary_cache
    
    path = grasp_path or GRASP_PATH
    path = Path(path).expanduser().resolve()
    
    if not path.is_file():
        raise FileNotFoundError(f"Grasp summary file not found: {path}")
    
    with open(path, "rb") as f:
        _grasp_summary_cache = pickle.load(f)
    
    return _grasp_summary_cache


def _load_initial_state(initial_state_path: Optional[Path] = None) -> List[Dict]:
    """Load initial state from pickle file."""
    global _initial_state_cache
    if _initial_state_cache is not None:
        return _initial_state_cache
    
    path = initial_state_path or INITIAL_STATE_PATH
    path = Path(path).expanduser().resolve()
    
    if not path.is_file():
        raise FileNotFoundError(f"Initial state file not found: {path}")
    
    with open(path, "rb") as f:
        _initial_state_cache = pickle.load(f)
    
    return _initial_state_cache


def _load_object_affordance(affordance_path: Optional[Path] = None) -> Dict:
    """Load object affordance trajectories from JSON file."""
    global _object_affordance_cache
    if _object_affordance_cache is not None:
        return _object_affordance_cache
    
    path = affordance_path or OBJECT_AFFORDANCE_PATH
    path = Path(path).expanduser().resolve()
    
    if not path.is_file():
        raise FileNotFoundError(f"Object affordance file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        _object_affordance_cache = json.load(f)
    
    return _object_affordance_cache


def get_initial_state(initial_state_path: Optional[Path] = None) -> Union[Dict, List[Dict]]:
    """
    Return the initial state(s) of the environment.
    
    Args:
        initial_state_path: Optional path to initial state file (defaults to INITIAL_STATE_PATH)
    
    Returns:
        List of initial state dictionaries, each containing:
        - robot_state: Dictionary with robot state information
        - parts_poses: Array of part poses (in AprilTag frame)
    """
    initial_states = _load_initial_state(initial_state_path)
    initial_state = random.choice(initial_states)
    
    return initial_state


def get_grasp_eef_pose(
    part_name: str,
    pose: np.ndarray,
    furniture_name: Optional[str] = None,
    mode: int = 0,
    grasp_path: Optional[Path] = None
) -> np.ndarray:
    """
    Compute grasp end-effector pose from part pose using extracted grasp information.
    
    The relative pose stored is: rel_mat = inv(ee_mat) @ part_mat
    So to get EE pose from part pose: ee_mat = part_mat @ inv(rel_mat)
    
    Args:
        part_name: Name of the part to grasp
        pose: 7D pose vector [x, y, z, qx, qy, qz, qw] of the part (in AprilTag frame)
        furniture_name: Optional furniture name. If None, searches all furniture types.
        mode: Cluster mode to use (0 = largest/dominant cluster, 1+ = other clusters sorted by size)
        grasp_path: Optional path to grasp summary file (defaults to GRASP_PATH)
    
    Returns:
        grasp_eef_pose: 7D pose vector [x, y, z, qx, qy, qz, qw] of end-effector (in AprilTag frame)
    """
    summary = _load_grasp_summary(grasp_path)
    
    # Find the part in summary
    part_stats = None
    found_furniture = None
    
    if furniture_name:
        if furniture_name not in summary:
            raise ValueError(f"Furniture '{furniture_name}' not found in grasp summary")
        if part_name not in summary[furniture_name]:
            raise ValueError(f"Part '{part_name}' not found in furniture '{furniture_name}'")
        part_stats = summary[furniture_name][part_name]
        found_furniture = furniture_name
    else:
        # Search all furniture types
        for furn_name, parts_data in summary.items():
            if part_name in parts_data:
                part_stats = parts_data[part_name]
                found_furniture = furn_name
                break
        
        if part_stats is None:
            available_parts = []
            for furn_name, parts_data in summary.items():
                available_parts.extend([f"{furn_name}:{p}" for p in parts_data.keys()])
            raise ValueError(
                f"Part '{part_name}' not found in any furniture type. "
                f"Available parts: {', '.join(available_parts)}"
            )
    
    # Get cluster grasps
    cluster_grasps = part_stats.get("cluster_grasps", [])
    if not cluster_grasps:
        raise ValueError(f"No cluster grasps found for part '{part_name}' in furniture '{found_furniture}'")
    
    # Select cluster based on mode
    # Filter out noise clusters (cluster_id == -1) for mode selection
    valid_clusters = [cg for cg in cluster_grasps if cg["cluster_id"] >= 0]
    if not valid_clusters:
        # Fall back to noise cluster if no valid clusters
        valid_clusters = cluster_grasps
    
    if mode < 0 or mode >= len(valid_clusters):
        raise ValueError(
            f"Mode {mode} out of range [0, {len(valid_clusters)}). "
            f"Available modes: 0-{len(valid_clusters)-1}"
        )
    
    selected_cluster = valid_clusters[mode]
    rel_pos = np.array(selected_cluster["relative_position"], dtype=np.float64)
    rel_quat = np.array(selected_cluster["relative_quaternion"], dtype=np.float64)
    
    # Normalize quaternion
    rel_quat = rel_quat / (np.linalg.norm(rel_quat) + 1e-10)
    
    # Convert poses to transformation matrices
    part_mat = T.pose2mat(pose)
    rel_mat = T.pose2mat(np.concatenate([rel_pos, rel_quat]))
    
    # Compute EE pose: ee_mat = part_mat @ inv(rel_mat)
    # rel_mat transforms from EE frame to part frame, so inv(rel_mat) transforms from part to EE
    rel_mat_inv = np.linalg.inv(rel_mat)
    ee_mat = part_mat @ rel_mat_inv
    
    # Convert back to pose vector
    ee_pos, ee_quat = T.mat2pose(ee_mat)
    ee_pose = np.concatenate([ee_pos, ee_quat])
    
    return ee_pose


def get_object_affordance(
    base_part: str,
    target_part: str,
    furniture_name: Optional[str] = None,
    mode: int = 0,
    affordance_path: Optional[Path] = None
) -> np.ndarray:
    """
    Return the object affordance trajectory (relative pose trajectory) between base and target parts.
    
    Args:
        base_part: Name of the base part
        target_part: Name of the target part
        furniture_name: Optional furniture name. If None, searches all furniture types.
        mode: Trajectory mode to use (0 = first trajectory, 1+ = other trajectories)
        affordance_path: Optional path to affordance file (defaults to OBJECT_AFFORDANCE_PATH)
    
    Returns:
        object_affordance: Nx7 array of relative poses [x, y, z, qx, qy, qz, qw],
                          where each row is the pose of target_part in base_part's frame
    """
    affordance_data = _load_object_affordance(affordance_path)
    
    # Construct pair key (format: "base_part__target_part")
    pair_key = f"{base_part}__{target_part}"
    
    # Find the pair in affordance data
    trajectories = None
    found_furniture = None
    
    if furniture_name:
        if furniture_name not in affordance_data:
            raise ValueError(f"Furniture '{furniture_name}' not found in affordance data")
        if pair_key not in affordance_data[furniture_name]:
            raise ValueError(
                f"Pair '{pair_key}' not found in furniture '{furniture_name}'. "
                f"Available pairs: {list(affordance_data[furniture_name].keys())}"
            )
        trajectories = affordance_data[furniture_name][pair_key]
        found_furniture = furniture_name
    else:
        # Search all furniture types
        for furn_name, pairs_data in affordance_data.items():
            if pair_key in pairs_data:
                trajectories = pairs_data[pair_key]
                found_furniture = furn_name
                break
        
        if trajectories is None:
            available_pairs = []
            for furn_name, pairs_data in affordance_data.items():
                available_pairs.extend([f"{furn_name}:{p}" for p in pairs_data.keys()])
            raise ValueError(
                f"Pair '{pair_key}' not found in any furniture type. "
                f"Available pairs: {', '.join(available_pairs)}"
            )
    
    if not trajectories:
        raise ValueError(f"No trajectories found for pair '{pair_key}' in furniture '{found_furniture}'")
    
    if mode < 0 or mode >= len(trajectories):
        raise ValueError(
            f"Mode {mode} out of range [0, {len(trajectories)}). "
            f"Available modes: 0-{len(trajectories)-1}"
        )
    
    # Convert trajectory to numpy array
    trajectory = np.array(trajectories[mode], dtype=np.float64)
    
    return trajectory


def clear_cache():
    """Clear cached data to force reload from files."""
    global _grasp_summary_cache, _initial_state_cache, _object_affordance_cache
    _grasp_summary_cache = None
    _initial_state_cache = None
    _object_affordance_cache = None