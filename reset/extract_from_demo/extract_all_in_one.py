"""
All-in-one extraction script combining MOVE/INTERACT detection, grasp poses, and object affordances.

Uses pre-computed distance data for fast processing and extracts:
- MOVE actions: grasped part, initial/final poses, grasp pose (relative gripper pose)
- INTERACT actions: target/base parts, initial/final poses, grasp pose, object affordance trajectory
"""
from pathlib import Path
import argparse
import json
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

from furniture_bench.config import config
from furniture_bench.furniture import furniture_factory
from furniture_bench.utils import transform as T


def _to_numpy(array_like) -> np.ndarray:
    """Convert to numpy array."""
    if isinstance(array_like, np.ndarray):
        return array_like.astype(np.float32, copy=True)
    return np.array(array_like, dtype=np.float32)


def _compute_relative_pose(ee_pose: np.ndarray, part_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute relative pose of part in gripper frame.
    
    Args:
        ee_pose: End-effector pose matrix (4x4)
        part_pose: Part pose matrix (4x4)
    
    Returns:
        Tuple of (relative_position, relative_quaternion)
    """
    rel_mat = np.linalg.inv(ee_pose) @ part_pose
    rel_pos = rel_mat[:3, 3]
    rel_quat = T.mat2quat(rel_mat[:3, :3])
    return rel_pos, rel_quat


def _compute_relative_pose_between_parts(a_pose_vec: np.ndarray, b_pose_vec: np.ndarray) -> np.ndarray:
    """
    Compute relative pose of B in A's frame.
    
    Args:
        a_pose_vec: Part A pose vector [x,y,z,qx,qy,qz,qw]
        b_pose_vec: Part B pose vector [x,y,z,qx,qy,qz,qw]
    
    Returns:
        Relative pose vector [x,y,z,qx,qy,qz,qw] of B in A's frame
    """
    a_mat = T.pose2mat(a_pose_vec)
    b_mat = T.pose2mat(b_pose_vec)
    rel = np.linalg.inv(a_mat) @ b_mat
    pos, quat = T.mat2pose(rel)
    return np.concatenate([pos, quat])


def _detect_grasped_part_from_distances(
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
    
    # Find part with minimum distance that's below threshold
    min_dist = float('inf')
    grasped_part = None
    
    for part_name, distance in gripper_part_distances.items():
        if distance < contact_tolerance and distance < min_dist:
            min_dist = distance
            grasped_part = part_name
    
    return grasped_part


def _extract_all_in_one(
    distances_data: Dict,
    observations: List[Dict],
    contact_tolerance: float,
    min_consecutive_steps: int = 5,
) -> List[Dict]:
    """
    Extract MOVE and INTERACT actions with grasp poses and object affordances.
    
    Args:
        distances_data: Dictionary from extract_distance.py containing frames with distances
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
    
    for step_idx, obs in enumerate(tqdm(observations, desc="Processing frames")):
        robot_state = obs.get("robot_state")
        parts_poses = obs.get("parts_poses")
        
        if robot_state is None or parts_poses is None:
            continue
        
        # Get distance data for this frame
        frame_data = frame_idx_to_data.get(step_idx)
        if frame_data is None:
            continue
        
        parts_array = _to_numpy(parts_poses).reshape(num_parts, 7)
        
        # Get robot state
        ee_pos = _to_numpy(robot_state.get("ee_pos"))
        ee_quat = _to_numpy(robot_state.get("ee_quat"))
        ee_pose_vec = np.concatenate([ee_pos, ee_quat])
        ee_pose_mat = T.pose2mat(ee_pose_vec)
        
        gripper_closed = frame_data["gripper_closed"]
        gripper_part_distances = frame_data["gripper_part_distances"]
        part_part_distances = frame_data.get("part_part_distances", {})
        
        # Detect grasped part using pre-computed distances
        grasped_part_name = _detect_grasped_part_from_distances(
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
                    rel_pos, rel_quat = _compute_relative_pose(ee_pose_mat, target_pose_mat_robot)
                    
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
                            _compute_relative_pose_between_parts(base_pose_vec, target_pose_vec).tolist()
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
                    rel_pos, rel_quat = _compute_relative_pose(ee_pose_mat, grasped_pose_mat_robot)
                    
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
                            
                            rel_pos, rel_quat = _compute_relative_pose(ee_pose_mat, target_pose_mat_robot)
                            
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
                                    _compute_relative_pose_between_parts(base_pose_vec, target_pose_vec).tolist()
                                ],
                                "consecutive_steps": 1,
                            }
                        else:
                            # Start MOVE
                            grasped_pose_vec = parts_array[grasped_part_idx]
                            
                            grasped_pose_mat_april = T.pose2mat(grasped_pose_vec)
                            grasped_pose_mat_robot = robot_from_april @ grasped_pose_mat_april
                            
                            rel_pos, rel_quat = _compute_relative_pose(ee_pose_mat, grasped_pose_mat_robot)
                            
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
                    rel_pose = _compute_relative_pose_between_parts(base_pose_vec, target_pose_vec)
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
                            
                            rel_pos, rel_quat = _compute_relative_pose(ee_pose_mat, target_pose_mat_robot)
                            
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
                                    _compute_relative_pose_between_parts(base_pose_vec, target_pose_vec).tolist()
                                ],
                                "consecutive_steps": 1,
                            }
                        else:
                            # Start MOVE
                            grasped_pose_vec = parts_array[grasped_part_idx]
                            
                            grasped_pose_mat_april = T.pose2mat(grasped_pose_vec)
                            grasped_pose_mat_robot = robot_from_april @ grasped_pose_mat_april
                            
                            rel_pos, rel_quat = _compute_relative_pose(ee_pose_mat, grasped_pose_mat_robot)
                            
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
                parts_array = _to_numpy(parts_poses).reshape(num_parts, 7)
                
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="All-in-one extraction: MOVE/INTERACT actions with grasp poses and object affordances"
    )
    parser.add_argument(
        "--distances",
        type=Path,
        required=True,
        help="Path to distances.pkl file from extract_distance.py",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset directory or single .pkl file (for getting part poses and robot state)",
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
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to save output JSON file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load distance data
    distances_path = Path(args.distances).expanduser().resolve()
    if not distances_path.is_file():
        raise FileNotFoundError(f"Distances file not found: {distances_path}")
    
    print(f"Loading distances from: {distances_path}")
    with open(distances_path, "rb") as f:
        distances_data = pickle.load(f)
    
    furniture_name = distances_data.get("furniture")
    if furniture_name is None:
        raise ValueError("Furniture name not found in distances data")
    
    print(f"Furniture: {furniture_name}")
    print(f"Number of frames in distance data: {distances_data['num_frames']}")
    
    # Load dataset for part poses and robot state
    dataset_root = Path(args.dataset).expanduser().resolve()
    annotation_files = list(_iter_annotation_files(dataset_root))
    
    if not annotation_files:
        raise FileNotFoundError(f"No .pkl files found under {dataset_root}")
    
    annotation_path = annotation_files[0]
    print(f"Loading dataset from: {annotation_path}")
    
    with open(annotation_path, "rb") as f:
        data = pickle.load(f)
    
    observations = data.get("observations", [])
    if not observations:
        raise ValueError("No observations found in dataset")
    
    print(f"Processing {len(observations)} observations...")
    
    # Extract all information
    actions = _extract_all_in_one(
        distances_data,
        observations,
        args.contact_tolerance,
        args.min_consecutive_steps,
    )
    
    # Prepare output
    output_data = {
        "furniture": furniture_name,
        "num_actions": len(actions),
        "actions": actions,
    }
    
    # Determine output path
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "all_in_one_extraction.json"
    
    # Save JSON
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved extraction results to {output_path}")
    print(f"Found {len(actions)} action(s):")
    for i, action in enumerate(actions):
        action_type = action["type"]
        if action_type == "MOVE":
            print(
                f"  {i+1}. MOVE: {action['grasped_part']} "
                f"(steps {action['start_step']}-{action['end_step']})"
            )
            print(f"      Grasp pose: rel_pos={action['grasp_pose']['relative_position']}")
        elif action_type == "INTERACT":
            print(
                f"  {i+1}. INTERACT: {action['target_part']} with {action['base_part']} "
                f"(steps {action['start_step']}-{action['end_step']})"
            )
            print(f"      Grasp pose: rel_pos={action['grasp_pose']['relative_position']}")
            print(f"      Affordance trajectory length: {len(action['affordance_trajectory'])}")


if __name__ == "__main__":
    main()

