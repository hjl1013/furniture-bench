"""
Unified extraction script combining distance computation and all-in-one extraction.

Extracts:
1. Distances between gripper and parts, and between parts (saved as distances.pkl)
2. MOVE and INTERACT actions with grasp poses and object affordances (saved as all_in_one_extraction.json and extracted_info.pkl)

MOVE actions include: grasped part, initial/final poses, grasp pose (relative gripper pose)
INTERACT actions include: target/base parts, initial/final poses, grasp pose, object affordance trajectory
"""
from pathlib import Path
import argparse
import json
import pickle
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

from furniture_bench.config import config
from furniture_bench.furniture import furniture_factory
from furniture_bench.utils import transform as T

from reset.utils.file_io import iter_annotation_files, to_numpy
from reset.utils.pose_utils import (
    compute_relative_pose,
    compute_relative_pose_between_parts,
    pose_vec_to_mat,
)
from reset.utils.mesh_utils import (
    load_part_mesh_trimesh,
    load_gripper_mesh_trimesh,
    compute_mesh_distance,
    TRIMESH_AVAILABLE,
)
from reset.utils.thresholds import prepare_thresholds

try:
    import trimesh  # type: ignore[import]
except Exception:
    trimesh = None
    TRIMESH_AVAILABLE = False


def detect_grasped_part_from_distances(
    gripper_part_distances: Dict[str, float],
    gripper_closed: bool,
    contact_tolerance: float,
) -> Optional[str]:
    """
    Detect which part (if any) the gripper is grasping based on pre-computed distances.
    Selects the part with minimum distance that satisfies contact_tolerance.
    """
    if not gripper_closed:
        return None
    
    min_dist = float('inf')
    grasped_part = None
    
    for part_name, distance in gripper_part_distances.items():
        if distance < contact_tolerance and distance < min_dist:
            min_dist = distance
            grasped_part = part_name
    
    return grasped_part


def extract_distances(
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
        Dictionary containing distance data for all frames
    """
    if not TRIMESH_AVAILABLE or trimesh is None:
        raise ImportError(
            "Mesh-based detection requires trimesh. Install with: pip install trimesh"
        )
    
    num_parts = len(part_names)
    
    # Load meshes
    gripper_mesh = load_gripper_mesh_trimesh()
    
    furniture_conf = config["furniture"].get(furniture_name, {})
    part_meshes = {}
    for part_name in part_names:
        part_conf = furniture_conf.get(part_name)
        if not isinstance(part_conf, dict) or "asset_file" not in part_conf:
            print(f"Warning: missing asset file for part '{part_name}', skipping")
            continue
        try:
            part_meshes[part_name] = load_part_mesh_trimesh(part_conf["asset_file"])
        except Exception as e:
            print(f"Warning: failed to load mesh for part '{part_name}': {e}")
    
    if not part_meshes:
        raise ValueError(f"No part meshes could be loaded for {furniture_name}")
    
    # Process each frame
    frames_data = []
    
    for step_idx, obs in enumerate(tqdm(observations, desc="Computing distances")):
        robot_state = obs.get("robot_state")
        parts_poses = obs.get("parts_poses")
        
        if robot_state is None or parts_poses is None:
            continue
        
        ee_pos = to_numpy(robot_state.get("ee_pos"))
        ee_quat = to_numpy(robot_state.get("ee_quat"))
        gripper_width = float(np.asarray(robot_state.get("gripper_width")))
        ee_pose_vec = np.concatenate([ee_pos, ee_quat])
        ee_pose_mat = T.pose2mat(ee_pose_vec)
        
        parts_array = to_numpy(parts_poses).reshape(num_parts, 7)
        
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
            distance = compute_mesh_distance(gripper_transformed, part_transformed)
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
                    
                    distance = compute_mesh_distance(mesh_i, mesh_j)
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


def extract_all_in_one(
    distances_data: Dict,
    observations: List[Dict],
    contact_tolerance: float,
    min_consecutive_steps: int = 5,
) -> List[Dict]:
    """
    Extract MOVE and INTERACT actions with grasp poses and object affordances.
    
    Args:
        distances_data: Dictionary containing frames with distances
        observations: List of observation dictionaries (for getting part poses and robot state)
        contact_tolerance: Distance threshold for collision detection
        min_consecutive_steps: Minimum number of consecutive steps to consider an action
    
    Returns:
        List of action dictionaries with full information
    """
    frames_data = distances_data["frames"]
    part_names = distances_data["part_names"]
    num_parts = len(part_names)
    
    # Get robot_from_april transformation
    robot_from_april = config["robot"]["tag_base_from_robot_base"]
    
    # Create mapping from frame_idx to frame data
    frame_idx_to_data = {frame["frame_idx"]: frame for frame in frames_data}
    
    # State tracking
    actions = []
    current_action = None
    prev_parts_array = None
    
    for step_idx, obs in enumerate(tqdm(observations, desc="Extracting actions")):
        robot_state = obs.get("robot_state")
        parts_poses = obs.get("parts_poses")
        
        if robot_state is None or parts_poses is None:
            continue
        
        # Get distance data for this frame
        frame_data = frame_idx_to_data.get(step_idx)
        if frame_data is None:
            continue
        
        parts_array = to_numpy(parts_poses).reshape(num_parts, 7)
        
        # Get robot state
        ee_pos = to_numpy(robot_state.get("ee_pos"))
        ee_quat = to_numpy(robot_state.get("ee_quat"))
        ee_pose_vec = np.concatenate([ee_pos, ee_quat])
        ee_pose_mat = T.pose2mat(ee_pose_vec)
        
        gripper_closed = frame_data["gripper_closed"]
        gripper_part_distances = frame_data["gripper_part_distances"]
        part_part_distances = frame_data.get("part_part_distances", {})
        
        # Detect grasped part using pre-computed distances
        grasped_part_name = detect_grasped_part_from_distances(
            gripper_part_distances,
            gripper_closed,
            contact_tolerance,
        )
        
        grasped_part_idx = None
        if grasped_part_name is not None:
            grasped_part_idx = part_names.index(grasped_part_name)
        
        # Determine if grasped part is interacting with another part
        # For INTERACT: select the closest part to the grasped part that satisfies contact_tolerance
        interacting_part_idx = None
        if grasped_part_name is not None:
            min_interaction_dist = float('inf')
            closest_part_name = None
            
            # Find the closest part to the grasped part that is within contact_tolerance
            for (part_i, part_j), distance in part_part_distances.items():
                if part_i == grasped_part_name:
                    if distance < contact_tolerance and distance < min_interaction_dist:
                        min_interaction_dist = distance
                        closest_part_name = part_j
                elif part_j == grasped_part_name:
                    if distance < contact_tolerance and distance < min_interaction_dist:
                        min_interaction_dist = distance
                        closest_part_name = part_i
            
            if closest_part_name is not None:
                interacting_part_idx = part_names.index(closest_part_name)
        
        # State machine to track actions
        if current_action is None:
            # No current action - check if we should start one
            if grasped_part_idx is not None:
                if interacting_part_idx is not None:
                    # Start INTERACT action
                    target_pose_vec = parts_array[grasped_part_idx]
                    base_pose_vec = parts_array[interacting_part_idx]
                    
                    # Convert to robot frame for grasp pose computation
                    target_pose_mat_april = T.pose2mat(target_pose_vec)
                    target_pose_mat_robot = robot_from_april @ target_pose_mat_april
                    
                    # Compute grasp pose (relative pose of gripper w.r.t. target part)
                    rel_pos, rel_quat = compute_relative_pose(ee_pose_mat, target_pose_mat_robot)
                    
                    current_action = {
                        "type": "INTERACT",
                        "target_part": part_names[grasped_part_idx],
                        "base_part": part_names[interacting_part_idx],
                        "start_step": step_idx,
                        "target_initial_pose": target_pose_vec.tolist(),
                        "base_initial_pose": base_pose_vec.tolist(),
                        "grasp_pose": {
                            "relative_position": rel_pos.tolist(),
                            "relative_quaternion": rel_quat.tolist(),
                        },
                        "affordance_trajectory": [
                            compute_relative_pose_between_parts(base_pose_vec, target_pose_vec).tolist()
                        ],
                        "consecutive_steps": 1,
                    }
                else:
                    # Start MOVE action
                    grasped_pose_vec = parts_array[grasped_part_idx]
                    
                    # Convert to robot frame for grasp pose computation
                    grasped_pose_mat_april = T.pose2mat(grasped_pose_vec)
                    grasped_pose_mat_robot = robot_from_april @ grasped_pose_mat_april
                    
                    # Compute grasp pose (relative pose of gripper w.r.t. grasped part)
                    rel_pos, rel_quat = compute_relative_pose(ee_pose_mat, grasped_pose_mat_robot)
                    
                    current_action = {
                        "type": "MOVE",
                        "grasped_part": part_names[grasped_part_idx],
                        "start_step": step_idx,
                        "initial_pose": grasped_pose_vec.tolist(),
                        "grasp_pose": {
                            "relative_position": rel_pos.tolist(),
                            "relative_quaternion": rel_quat.tolist(),
                        },
                        "consecutive_steps": 1,
                    }
        else:
            # We have a current action - check if it should continue or end
            action_type = current_action["type"]
            
            if action_type == "MOVE":
                # Check if still grasping the same part without interaction
                if (
                    grasped_part_idx is not None
                    and part_names[grasped_part_idx] == current_action["grasped_part"]
                    and interacting_part_idx is None
                ):
                    # Continue MOVE
                    current_action["consecutive_steps"] += 1
                else:
                    # End MOVE action
                    if current_action["consecutive_steps"] >= min_consecutive_steps:
                        # Record end pose from previous step (last valid state)
                        if prev_parts_array is not None:
                            prev_grasped_idx = part_names.index(current_action["grasped_part"])
                            current_action["end_pose"] = prev_parts_array[prev_grasped_idx].tolist()
                        else:
                            # Fallback to current step if no previous data
                            prev_grasped_idx = part_names.index(current_action["grasped_part"])
                            current_action["end_pose"] = parts_array[prev_grasped_idx].tolist()
                        current_action["end_step"] = step_idx - 1
                        # Remove internal tracking field
                        del current_action["consecutive_steps"]
                        actions.append(current_action)
                    
                    current_action = None
                    
                    # Check if we should start a new action
                    if grasped_part_idx is not None:
                        if interacting_part_idx is not None:
                            # Start INTERACT
                            target_pose_vec = parts_array[grasped_part_idx]
                            base_pose_vec = parts_array[interacting_part_idx]
                            
                            target_pose_mat_april = T.pose2mat(target_pose_vec)
                            target_pose_mat_robot = robot_from_april @ target_pose_mat_april
                            
                            rel_pos, rel_quat = compute_relative_pose(ee_pose_mat, target_pose_mat_robot)
                            
                            current_action = {
                                "type": "INTERACT",
                                "target_part": part_names[grasped_part_idx],
                                "base_part": part_names[interacting_part_idx],
                                "start_step": step_idx,
                                "target_initial_pose": target_pose_vec.tolist(),
                                "base_initial_pose": base_pose_vec.tolist(),
                                "grasp_pose": {
                                    "relative_position": rel_pos.tolist(),
                                    "relative_quaternion": rel_quat.tolist(),
                                },
                                "affordance_trajectory": [
                                    compute_relative_pose_between_parts(base_pose_vec, target_pose_vec).tolist()
                                ],
                                "consecutive_steps": 1,
                            }
                        else:
                            # Start MOVE
                            grasped_pose_vec = parts_array[grasped_part_idx]
                            
                            grasped_pose_mat_april = T.pose2mat(grasped_pose_vec)
                            grasped_pose_mat_robot = robot_from_april @ grasped_pose_mat_april
                            
                            rel_pos, rel_quat = compute_relative_pose(ee_pose_mat, grasped_pose_mat_robot)
                            
                            current_action = {
                                "type": "MOVE",
                                "grasped_part": part_names[grasped_part_idx],
                                "start_step": step_idx,
                                "initial_pose": grasped_pose_vec.tolist(),
                                "grasp_pose": {
                                    "relative_position": rel_pos.tolist(),
                                    "relative_quaternion": rel_quat.tolist(),
                                },
                                "consecutive_steps": 1,
                            }
            
            elif action_type == "INTERACT":
                # Check if still grasping target part and interacting with base part
                target_part_name = current_action["target_part"]
                base_part_name = current_action["base_part"]
                
                still_grasping_target = (
                    grasped_part_idx is not None
                    and part_names[grasped_part_idx] == target_part_name
                )
                
                still_interacting = False
                if still_grasping_target:
                    # Check if target-base distance is still within contact_tolerance
                    collision_key1 = (target_part_name, base_part_name)
                    collision_key2 = (base_part_name, target_part_name)
                    
                    target_base_distance = None
                    if collision_key1 in part_part_distances:
                        target_base_distance = part_part_distances[collision_key1]
                    elif collision_key2 in part_part_distances:
                        target_base_distance = part_part_distances[collision_key2]
                    
                    if target_base_distance is not None:
                        still_interacting = target_base_distance < contact_tolerance
                
                if still_grasping_target and still_interacting:
                    # Continue INTERACT - add to affordance trajectory
                    target_pose_vec = parts_array[grasped_part_idx]
                    base_pose_vec = parts_array[part_names.index(base_part_name)]
                    rel_pose = compute_relative_pose_between_parts(base_pose_vec, target_pose_vec)
                    current_action["affordance_trajectory"].append(rel_pose.tolist())
                    current_action["consecutive_steps"] += 1
                else:
                    # End INTERACT action
                    if current_action["consecutive_steps"] >= min_consecutive_steps:
                        # Record end poses from previous step (last valid state)
                        target_idx = part_names.index(target_part_name)
                        base_idx = part_names.index(base_part_name)
                        if prev_parts_array is not None:
                            current_action["target_end_pose"] = prev_parts_array[target_idx].tolist()
                            current_action["base_end_pose"] = prev_parts_array[base_idx].tolist()
                        else:
                            # Fallback to current step if no previous data
                            current_action["target_end_pose"] = parts_array[target_idx].tolist()
                            current_action["base_end_pose"] = parts_array[base_idx].tolist()
                        current_action["end_step"] = step_idx - 1
                        # Remove internal tracking field
                        del current_action["consecutive_steps"]
                        actions.append(current_action)
                    
                    current_action = None
                    
                    # Check if we should start a new action
                    if grasped_part_idx is not None:
                        if interacting_part_idx is not None:
                            # Start INTERACT
                            target_pose_vec = parts_array[grasped_part_idx]
                            base_pose_vec = parts_array[interacting_part_idx]
                            
                            target_pose_mat_april = T.pose2mat(target_pose_vec)
                            target_pose_mat_robot = robot_from_april @ target_pose_mat_april
                            
                            rel_pos, rel_quat = compute_relative_pose(ee_pose_mat, target_pose_mat_robot)
                            
                            current_action = {
                                "type": "INTERACT",
                                "target_part": part_names[grasped_part_idx],
                                "base_part": part_names[interacting_part_idx],
                                "start_step": step_idx,
                                "target_initial_pose": target_pose_vec.tolist(),
                                "base_initial_pose": base_pose_vec.tolist(),
                                "grasp_pose": {
                                    "relative_position": rel_pos.tolist(),
                                    "relative_quaternion": rel_quat.tolist(),
                                },
                                "affordance_trajectory": [
                                    compute_relative_pose_between_parts(base_pose_vec, target_pose_vec).tolist()
                                ],
                                "consecutive_steps": 1,
                            }
                        else:
                            # Start MOVE
                            grasped_pose_vec = parts_array[grasped_part_idx]
                            
                            grasped_pose_mat_april = T.pose2mat(grasped_pose_vec)
                            grasped_pose_mat_robot = robot_from_april @ grasped_pose_mat_april
                            
                            rel_pos, rel_quat = compute_relative_pose(ee_pose_mat, grasped_pose_mat_robot)
                            
                            current_action = {
                                "type": "MOVE",
                                "grasped_part": part_names[grasped_part_idx],
                                "start_step": step_idx,
                                "initial_pose": grasped_pose_vec.tolist(),
                                "grasp_pose": {
                                    "relative_position": rel_pos.tolist(),
                                    "relative_quaternion": rel_quat.tolist(),
                                },
                                "consecutive_steps": 1,
                            }
        
        # Update previous parts array for next iteration
        prev_parts_array = parts_array.copy()
    
    # Flush any remaining action
    if current_action is not None:
        if current_action["consecutive_steps"] >= min_consecutive_steps:
            action_type = current_action["type"]
            last_step = len(observations) - 1
            last_obs = observations[last_step]
            parts_poses = last_obs.get("parts_poses")
            if parts_poses is not None:
                parts_array = to_numpy(parts_poses).reshape(num_parts, 7)
                
                if action_type == "MOVE":
                    grasped_idx = part_names.index(current_action["grasped_part"])
                    current_action["end_pose"] = parts_array[grasped_idx].tolist()
                elif action_type == "INTERACT":
                    target_idx = part_names.index(current_action["target_part"])
                    base_idx = part_names.index(current_action["base_part"])
                    current_action["target_end_pose"] = parts_array[target_idx].tolist()
                    current_action["base_end_pose"] = parts_array[base_idx].tolist()
                
                current_action["end_step"] = last_step
                del current_action["consecutive_steps"]
                actions.append(current_action)
    
    return actions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified extraction: distances and all-in-one extraction"
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
        "--contact-tolerance",
        type=float,
        default=0.01,
        help="Distance threshold (m) for collision detection. Default: 0.01",
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
        "--min-consecutive-steps",
        type=int,
        default=5,
        help="Minimum consecutive steps to consider an action valid",
    )
    parser.add_argument(
        "--no-part-part",
        action="store_true",
        help="Skip computing part-part distances (only compute gripper-part distances)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to save output files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not TRIMESH_AVAILABLE or trimesh is None:
        raise ImportError(
            "trimesh is required. Install with: pip install trimesh"
        )
    
    dataset_root = Path(args.dataset).expanduser().resolve()
    annotation_files = list(iter_annotation_files(dataset_root))
    
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
    
    thresholds = prepare_thresholds(
        furniture_name,
        gripper_close_threshold=args.gripper_close_threshold,
        gripper_close_ratio=args.gripper_close_ratio,
        contact_tolerance=args.contact_tolerance,
        min_consecutive_steps=args.min_consecutive_steps,
    )
    robot_from_april = config["robot"]["tag_base_from_robot_base"]
    
    observations = data.get("observations", [])
    if not observations:
        raise ValueError("No observations found in dataset")
    
    print(f"Processing {len(observations)} observations...")
    print(f"Furniture: {furniture_name}")
    print(f"Parts: {part_names}")
    print(f"Include part-part distances: {not args.no_part_part}")
    
    # Step 1: Extract distances
    print("\n" + "="*60)
    print("Step 1: Extracting distances...")
    print("="*60)
    distances_data = extract_distances(
        observations,
        furniture_name,
        part_names,
        thresholds,
        robot_from_april,
        include_part_part=not args.no_part_part,
    )
    
    # Step 2: Extract actions using distances
    print("\n" + "="*60)
    print("Step 2: Extracting actions...")
    print("="*60)
    actions = extract_all_in_one(
        distances_data,
        observations,
        args.contact_tolerance,
        args.min_consecutive_steps,
    )
    
    # Prepare output directory
    output_dir = Path(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save distances.pkl
    distances_path = output_dir / "distances.pkl"
    with open(distances_path, "wb") as f:
        pickle.dump(distances_data, f)
    print(f"\nSaved distances to {distances_path}")
    
    # Prepare extraction data
    extraction_data = {
        "furniture": furniture_name,
        "num_actions": len(actions),
        "actions": actions,
    }
    
    # Save extracted_info.pkl (same format as JSON but as pickle)
    pkl_path = output_dir / "extracted_info.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(extraction_data, f)
    print(f"Saved extraction results to {pkl_path}")
    
    # Print summary
    print(f"\nFound {len(actions)} action(s):")
    for i, action in enumerate(actions):
        action_type = action["type"]
        if action_type == "MOVE":
            print(
                f"  {i+1}. MOVE: {action['grasped_part']} "
                f"(steps {action['start_step']}-{action['end_step']})"
            )
        elif action_type == "INTERACT":
            print(
                f"  {i+1}. INTERACT: {action['target_part']} with {action['base_part']} "
                f"(steps {action['start_step']}-{action['end_step']})"
            )


if __name__ == "__main__":
    main()

