"""
Extract distances between objects (parts) and gripper per frame.

For each frame in a demonstration, computes and saves:
- Distance between gripper and each part
- Distance between each pair of parts (optional)

Saves results as a pickle file with a structured format.
"""
from functools import lru_cache
from pathlib import Path
import argparse
import pickle
from typing import Dict, List, Tuple, Optional
import os
import numpy as np
from tqdm import tqdm

from furniture_bench.config import config
from furniture_bench.furniture import furniture_factory
from furniture_bench.utils import transform as T

# Try to reuse mesh-loading helpers from extract_grasp
try:
    from .extract_grasp import (
        _load_part_mesh_trimesh,
        _load_gripper_mesh_trimesh,
        _detect_contact_mesh,
        TRIMESH_AVAILABLE,
        _prepare_thresholds,
    )
except Exception:
    try:
        from reset.extract_from_demo.extract_grasp import (
            _load_part_mesh_trimesh,
            _load_gripper_mesh_trimesh,
            _detect_contact_mesh,
            TRIMESH_AVAILABLE,
            _prepare_thresholds,
        )
    except Exception:
        _load_part_mesh_trimesh = None
        _load_gripper_mesh_trimesh = None
        _detect_contact_mesh = None
        _prepare_thresholds = None
        TRIMESH_AVAILABLE = False

try:
    import trimesh
except Exception:
    trimesh = None
    TRIMESH_AVAILABLE = False


def _iter_annotation_files(dataset_path: Path):
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


def _to_numpy(array_like) -> np.ndarray:
    """Convert to numpy array."""
    if isinstance(array_like, np.ndarray):
        return array_like.astype(np.float32, copy=True)
    return np.array(array_like, dtype=np.float32)


def _compute_mesh_distance(
    mesh_a: "trimesh.Trimesh",
    mesh_b: "trimesh.Trimesh",
) -> float:
    """
    Compute minimum distance between two meshes.
    
    Args:
        mesh_a: First mesh
        mesh_b: Second mesh
    
    Returns:
        Minimum distance between meshes, or inf if computation fails
    """
    try:
        manager = trimesh.collision.CollisionManager()
        manager.add_object("a", mesh_a)
        min_dist = manager.min_distance_single(mesh_b)
        return float(min_dist)
    except Exception:
        # Fallback to proximity-based distance
        try:
            # Use closest_point for direct mesh-to-mesh distance
            closest_points, distances, _ = trimesh.proximity.closest_point(mesh_a, mesh_b.vertices)
            if len(distances) > 0:
                return float(np.min(distances))
            else:
                return float(np.inf)
        except Exception:
            # Last resort: use AABB center distance
            try:
                aabb_a = mesh_a.bounds
                aabb_b = mesh_b.bounds
                center_a = aabb_a.mean(axis=0)
                center_b = aabb_b.mean(axis=0)
                return float(np.linalg.norm(center_a - center_b))
            except Exception:
                return float(np.inf)


def _extract_distances(
    observations: List[Dict],
    furniture_name: str,
    part_names: List[str],
    thresholds: Dict[str, float],
    robot_from_april: np.ndarray,
    include_part_part: bool = True,
) -> Dict:
    """
    Extract distances between gripper and parts, and between parts, for each frame.
    
    Args:
        observations: List of observation dictionaries
        furniture_name: Name of furniture type
        part_names: List of part names
        thresholds: Dictionary with gripper threshold, etc.
        robot_from_april: Transformation matrix from AprilTag to robot frame
        include_part_part: Whether to compute part-part distances
    
    Returns:
        Dictionary containing:
        - furniture: Furniture name
        - num_frames: Number of frames processed
        - part_names: List of part names
        - frames: List of frame data, each containing:
            - frame_idx: Frame index
            - gripper_part_distances: Dict mapping part_name -> distance
            - part_part_distances: Dict mapping (part_i, part_j) -> distance (if include_part_part)
            - gripper_width: Gripper width
            - gripper_closed: Whether gripper is closed
    """
    if not TRIMESH_AVAILABLE or trimesh is None:
        raise ImportError(
            "Mesh-based detection requires trimesh. Install with: pip install trimesh"
        )
    
    if _load_part_mesh_trimesh is None or _load_gripper_mesh_trimesh is None:
        raise ImportError("Mesh loading functions not available")
    
    num_parts = len(part_names)
    
    # Load meshes
    gripper_mesh = _load_gripper_mesh_trimesh()
    
    furniture_conf = config["furniture"].get(furniture_name, {})
    part_meshes = {}
    for part_name in part_names:
        part_conf = furniture_conf.get(part_name)
        if not isinstance(part_conf, dict) or "asset_file" not in part_conf:
            print(f"Warning: missing asset file for part '{part_name}', skipping")
            continue
        try:
            part_meshes[part_name] = _load_part_mesh_trimesh(part_conf["asset_file"])
        except Exception as e:
            print(f"Warning: failed to load mesh for part '{part_name}': {e}")
    
    if not part_meshes:
        raise ValueError(f"No part meshes could be loaded for {furniture_name}")
    
    # Process each frame
    frames_data = []
    
    for step_idx, obs in enumerate(tqdm(observations, desc="Processing frames")):
        robot_state = obs.get("robot_state")
        parts_poses = obs.get("parts_poses")
        
        if robot_state is None or parts_poses is None:
            continue
        
        ee_pos = _to_numpy(robot_state.get("ee_pos"))
        ee_quat = _to_numpy(robot_state.get("ee_quat"))
        gripper_width = float(np.asarray(robot_state.get("gripper_width")))
        ee_pose_vec = np.concatenate([ee_pos, ee_quat])
        ee_pose_mat = T.pose2mat(ee_pose_vec)
        
        parts_array = _to_numpy(parts_poses).reshape(num_parts, 7)
        
        # Check if gripper is closed
        gripper_closed = gripper_width < thresholds["gripper"]
        
        # Transform gripper mesh to current pose
        gripper_transformed = gripper_mesh.copy()
        gripper_transformed.apply_transform(ee_pose_mat)
        
        # Compute distances between gripper and each part
        gripper_part_distances = {}
        for part_idx, part_name in enumerate(part_names):
            if part_name not in part_meshes:
                gripper_part_distances[part_name] = float(np.inf)
                continue
            
            part_pose_vec = parts_array[part_idx]
            part_pose_mat_april = T.pose2mat(part_pose_vec)
            part_pose_mat_robot = robot_from_april @ part_pose_mat_april
            
            # Transform part mesh to current pose
            part_transformed = part_meshes[part_name].copy()
            part_transformed.apply_transform(part_pose_mat_robot)
            
            # Compute distance
            distance = _compute_mesh_distance(gripper_transformed, part_transformed)
            gripper_part_distances[part_name] = distance
        
        # Compute distances between each pair of parts (if requested)
        part_part_distances = {}
        if include_part_part:
            for i in range(num_parts):
                for j in range(i + 1, num_parts):
                    name_i = part_names[i]
                    name_j = part_names[j]
                    
                    if name_i not in part_meshes or name_j not in part_meshes:
                        part_part_distances[(name_i, name_j)] = float(np.inf)
                        continue
                    
                    part_i_pose_april = parts_array[i]
                    part_j_pose_april = parts_array[j]
                    
                    part_i_mat_robot = robot_from_april @ T.pose2mat(part_i_pose_april)
                    part_j_mat_robot = robot_from_april @ T.pose2mat(part_j_pose_april)
                    
                    mesh_i = part_meshes[name_i].copy()
                    mesh_j = part_meshes[name_j].copy()
                    mesh_i.apply_transform(part_i_mat_robot)
                    mesh_j.apply_transform(part_j_mat_robot)
                    
                    distance = _compute_mesh_distance(mesh_i, mesh_j)
                    part_part_distances[(name_i, name_j)] = distance
        
        # Store frame data
        frame_data = {
            "frame_idx": step_idx,
            "gripper_part_distances": gripper_part_distances,
            "gripper_width": gripper_width,
            "gripper_closed": gripper_closed,
        }
        
        if include_part_part:
            frame_data["part_part_distances"] = part_part_distances
        
        frames_data.append(frame_data)
    
    return {
        "furniture": furniture_name,
        "num_frames": len(frames_data),
        "part_names": part_names,
        "frames": frames_data,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract distances between objects and gripper per frame"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset directory or single .pkl file (processes only first file)",
    )
    parser.add_argument(
        "--furniture",
        default=None,
        help="Override furniture name (otherwise from file)",
    )
    parser.add_argument(
        "--gripper-close-threshold",
        type=float,
        default=None,
        help="Absolute threshold (m) for gripper width when considered closed",
    )
    parser.add_argument(
        "--gripper-close-ratio",
        type=float,
        default=0.9,
        help="Ratio of max gripper width used when threshold is not provided",
    )
    parser.add_argument(
        "--no-part-part",
        action="store_true",
        help="Skip computing part-part distances (only compute gripper-part distances)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save output pkl file (default: distances.pkl in dataset directory)",
    )
    parser.add_argument(
        "--contact-tolerance",
        type=float,
        default=0.01,
        help="Distance threshold (m) for collision detection. Default: 0.01",
    )
    parser.add_argument(
        "--min-consecutive-steps",
        type=int,
        default=5,
        help="Minimum consecutive steps to consider an action valid",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not TRIMESH_AVAILABLE or trimesh is None:
        raise ImportError(
            "trimesh is required. Install with: pip install trimesh"
        )
    
    if _load_part_mesh_trimesh is None or _load_gripper_mesh_trimesh is None:
        raise ImportError(
            "Mesh loading functions not available. Ensure extract_grasp.py is importable."
        )
    
    dataset_root = Path(args.dataset).expanduser().resolve()
    annotation_files = list(_iter_annotation_files(dataset_root))
    
    if not annotation_files:
        raise FileNotFoundError(f"No .pkl files found under {dataset_root}")
    
    # Process only the first file
    annotation_path = annotation_files[0]
    print(f"Processing: {annotation_path}")
    
    with open(annotation_path, "rb") as f:
        data = pickle.load(f)
    
    furniture_name = args.furniture or data.get("furniture")
    if furniture_name is None:
        raise ValueError(
            f"Furniture name not provided via --furniture or dataset metadata"
        )
    
    furniture = furniture_factory(furniture_name)
    part_names = [part.name for part in furniture.parts]
    
    thresholds = _prepare_thresholds(furniture_name, args)
    robot_from_april = config["robot"]["tag_base_from_robot_base"]
    
    observations = data.get("observations", [])
    if not observations:
        raise ValueError("No observations found in dataset")
    
    print(f"Processing {len(observations)} observations...")
    print(f"Furniture: {furniture_name}")
    print(f"Parts: {part_names}")
    print(f"Include part-part distances: {not args.no_part_part}")
    
    distances_data = _extract_distances(
        observations,
        furniture_name,
        part_names,
        thresholds,
        robot_from_april,
        include_part_part=not args.no_part_part,
    )
    
    # Determine output path
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / "distances.pkl"
    
    # Save as pickle file
    with open(output_path, "wb") as f:
        pickle.dump(distances_data, f)
    
    print(f"\nSaved distances to {output_path}")
    print(f"Total frames: {distances_data['num_frames']}")
    
    # Print summary statistics
    if distances_data["frames"]:
        first_frame = distances_data["frames"][0]
        print("\nSample frame data (frame 0):")
        print(f"  Gripper width: {first_frame['gripper_width']:.4f} m")
        print(f"  Gripper closed: {first_frame['gripper_closed']}")
        print("  Gripper-part distances:")
        for part_name, dist in first_frame["gripper_part_distances"].items():
            print(f"    {part_name}: {dist:.4f} m")
        
        if "part_part_distances" in first_frame:
            print("  Part-part distances:")
            for (part_i, part_j), dist in first_frame["part_part_distances"].items():
                print(f"    {part_i} <-> {part_j}: {dist:.4f} m")


if __name__ == "__main__":
    main()

