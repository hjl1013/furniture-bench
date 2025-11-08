"""
Extract manipulation order from demonstrations.

Detects two types of actions:
- MOVE: Robot moves a part from one place to another without interacting with other parts
- INTERACT: Robot manipulates a part that collides with another part

For each action, records:
- MOVE: grasped part, initial pose, end pose
- INTERACT: target part (grasped), base part (interacted with), initial poses, end poses
"""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import argparse
import json
import pickle
from typing import Dict, List, Tuple, Optional, Set
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
        _prepare_rgb_frames,
    )
except Exception:
    try:
        from reset.extract_from_demo.extract_grasp import (
            _load_part_mesh_trimesh,
            _load_gripper_mesh_trimesh,
            _detect_contact_mesh,
            TRIMESH_AVAILABLE,
            _prepare_thresholds,
            _prepare_rgb_frames,
        )
    except Exception:
        _load_part_mesh_trimesh = None
        _load_gripper_mesh_trimesh = None
        _detect_contact_mesh = None
        _prepare_thresholds = None
        _prepare_rgb_frames = None
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


def _detect_part_collisions(
    part_meshes: Dict[str, "trimesh.Trimesh"],
    part_names: List[str],
    parts_array: np.ndarray,
    robot_from_april: np.ndarray,
    contact_tolerance: float,
) -> Set[Tuple[int, int]]:
    """
    Detect collisions between parts.
    
    Args:
        part_meshes: Dictionary mapping part names to trimesh meshes
        part_names: List of part names
        parts_array: Array of shape (num_parts, 7) with poses in AprilTag frame
        robot_from_april: Transformation matrix from AprilTag to robot frame
        contact_tolerance: Distance threshold for collision detection
    
    Returns:
        Set of tuples (i, j) where i < j and parts i and j are colliding
    """
    num_parts = len(part_names)
    collisions = set()
    
    for i in range(num_parts):
        for j in range(i + 1, num_parts):
            name_i = part_names[i]
            name_j = part_names[j]
            
            if name_i not in part_meshes or name_j not in part_meshes:
                continue
            
            part_i_pose_april = parts_array[i]
            part_j_pose_april = parts_array[j]
            
            part_i_mat_robot = robot_from_april @ T.pose2mat(part_i_pose_april)
            part_j_mat_robot = robot_from_april @ T.pose2mat(part_j_pose_april)
            
            mesh_i = part_meshes[name_i].copy()
            mesh_j = part_meshes[name_j].copy()
            mesh_i.apply_transform(part_i_mat_robot)
            mesh_j.apply_transform(part_j_mat_robot)
            
            manager = trimesh.collision.CollisionManager()
            manager.add_object("a", mesh_i)
            try:
                min_dist = manager.min_distance_single(mesh_j)
            except Exception:
                # Fallback to AABB distance
                try:
                    aabb_i = mesh_i.bounds
                    aabb_j = mesh_j.bounds
                    center_i = aabb_i.mean(axis=0)
                    center_j = aabb_j.mean(axis=0)
                    min_dist = float(np.linalg.norm(center_i - center_j))
                except Exception:
                    min_dist = float(np.inf)
            
            if min_dist < contact_tolerance:
                collisions.add((i, j))
    
    return collisions


def _detect_grasped_part(
    gripper_mesh: "trimesh.Trimesh",
    part_meshes: Dict[str, "trimesh.Trimesh"],
    part_names: List[str],
    parts_array: np.ndarray,
    ee_pose_vec: np.ndarray,
    robot_from_april: np.ndarray,
    contact_tolerance: float,
    gripper_closed: bool,
) -> Optional[int]:
    """
    Detect which part (if any) the gripper is grasping.
    
    Args:
        gripper_mesh: Gripper mesh in local frame
        part_meshes: Dictionary mapping part names to trimesh meshes
        part_names: List of part names
        parts_array: Array of shape (num_parts, 7) with poses in AprilTag frame
        ee_pose_vec: 7D pose vector [x,y,z,qx,qy,qz,qw] of end-effector in robot frame
        robot_from_april: Transformation matrix from AprilTag to robot frame
        contact_tolerance: Distance threshold for contact detection
        gripper_closed: Whether gripper is closed
    
    Returns:
        Index of grasped part, or None if no part is grasped
    """
    if not gripper_closed:
        return None
    
    num_parts = len(part_names)
    
    for part_idx in range(num_parts):
        part_name = part_names[part_idx]
        
        if part_name not in part_meshes:
            continue
        
        part_pose_vec = parts_array[part_idx]
        part_pose_mat_april = T.pose2mat(part_pose_vec)
        part_pose_mat_robot = robot_from_april @ part_pose_mat_april
        
        # Convert part pose matrix to pose vector in robot frame
        part_pos_robot, part_quat_robot = T.mat2pose(part_pose_mat_robot)
        part_pose_vec_robot = np.concatenate([part_pos_robot, part_quat_robot])
        
        # Check contact
        has_contact, _ = _detect_contact_mesh(
            gripper_mesh,
            part_meshes[part_name],
            ee_pose_vec,
            part_pose_vec_robot,
            contact_tolerance,
            visualize=False,
            observation_images=None,
        )
        
        if has_contact:
            return part_idx
    
    return None


def _extract_manipulation_order(
    observations: List[Dict],
    furniture_name: str,
    part_names: List[str],
    thresholds: Dict[str, float],
    robot_from_april: np.ndarray,
    contact_tolerance: float,
    min_consecutive_steps: int = 5,
) -> List[Dict]:
    """
    Extract manipulation order from observations.
    
    Args:
        observations: List of observation dictionaries
        furniture_name: Name of furniture type
        part_names: List of part names
        thresholds: Dictionary with gripper threshold, etc.
        robot_from_april: Transformation matrix from AprilTag to robot frame
        contact_tolerance: Distance threshold for collision detection
        min_consecutive_steps: Minimum number of consecutive steps to consider an action
    
    Returns:
        List of action dictionaries, each with type, parts, and poses
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
    
    # State tracking
    actions = []
    current_action = None
    prev_parts_array = None
    
    for step_idx, obs in enumerate(tqdm(observations, desc="Processing frames")):
        robot_state = obs.get("robot_state")
        parts_poses = obs.get("parts_poses")
        
        if robot_state is None or parts_poses is None:
            continue
        
        ee_pos = _to_numpy(robot_state.get("ee_pos"))
        ee_quat = _to_numpy(robot_state.get("ee_quat"))
        gripper_width = float(np.asarray(robot_state.get("gripper_width")))
        ee_pose_vec = np.concatenate([ee_pos, ee_quat])
        
        parts_array = _to_numpy(parts_poses).reshape(num_parts, 7)
        
        # Check if gripper is closed
        gripper_closed = gripper_width < thresholds["gripper"]
        
        # Detect grasped part
        grasped_part_idx = _detect_grasped_part(
            gripper_mesh,
            part_meshes,
            part_names,
            parts_array,
            ee_pose_vec,
            robot_from_april,
            contact_tolerance,
            gripper_closed,
        )
        
        # Detect part-part collisions
        part_collisions = _detect_part_collisions(
            part_meshes,
            part_names,
            parts_array,
            robot_from_april,
            contact_tolerance,
        )
        
        # Determine if grasped part is colliding with another part
        interacting_part_idx = None
        if grasped_part_idx is not None:
            for (i, j) in part_collisions:
                if i == grasped_part_idx:
                    interacting_part_idx = j
                    break
                elif j == grasped_part_idx:
                    interacting_part_idx = i
                    break
        
        # State machine to track actions
        if current_action is None:
            # No current action - check if we should start one
            if grasped_part_idx is not None:
                if interacting_part_idx is not None:
                    # Start INTERACT action
                    current_action = {
                        "type": "INTERACT",
                        "target_part": part_names[grasped_part_idx],
                        "base_part": part_names[interacting_part_idx],
                        "start_step": step_idx,
                        "target_initial_pose": parts_array[grasped_part_idx].tolist(),
                        "base_initial_pose": parts_array[interacting_part_idx].tolist(),
                        "consecutive_steps": 1,
                    }
                else:
                    # Start MOVE action
                    current_action = {
                        "type": "MOVE",
                        "grasped_part": part_names[grasped_part_idx],
                        "start_step": step_idx,
                        "initial_pose": parts_array[grasped_part_idx].tolist(),
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
                            current_action = {
                                "type": "INTERACT",
                                "target_part": part_names[grasped_part_idx],
                                "base_part": part_names[interacting_part_idx],
                                "start_step": step_idx,
                                "target_initial_pose": parts_array[grasped_part_idx].tolist(),
                                "base_initial_pose": parts_array[interacting_part_idx].tolist(),
                                "consecutive_steps": 1,
                            }
                        else:
                            current_action = {
                                "type": "MOVE",
                                "grasped_part": part_names[grasped_part_idx],
                                "start_step": step_idx,
                                "initial_pose": parts_array[grasped_part_idx].tolist(),
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
                    # Check if target is still colliding with base
                    target_idx = grasped_part_idx
                    base_idx = part_names.index(base_part_name)
                    # Ensure correct order (i < j)
                    if target_idx < base_idx:
                        still_interacting = (target_idx, base_idx) in part_collisions
                    else:
                        still_interacting = (base_idx, target_idx) in part_collisions
                
                if still_grasping_target and still_interacting:
                    # Continue INTERACT
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
                            current_action = {
                                "type": "INTERACT",
                                "target_part": part_names[grasped_part_idx],
                                "base_part": part_names[interacting_part_idx],
                                "start_step": step_idx,
                                "target_initial_pose": parts_array[grasped_part_idx].tolist(),
                                "base_initial_pose": parts_array[interacting_part_idx].tolist(),
                                "consecutive_steps": 1,
                            }
                        else:
                            current_action = {
                                "type": "MOVE",
                                "grasped_part": part_names[grasped_part_idx],
                                "start_step": step_idx,
                                "initial_pose": parts_array[grasped_part_idx].tolist(),
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


def _render_action_video(
    observations: List[Dict],
    actions: List[Dict],
    window_title: str,
    play_speed_hz: float = 30.0,
):
    """
    Render demonstration video with action labels overlayed.
    
    Args:
        observations: List of observation dictionaries
        actions: List of action dictionaries with start_step and end_step
        window_title: Title for the video window
        play_speed_hz: Playback speed in Hz
    """
    if _prepare_rgb_frames is None:
        raise ImportError("_prepare_rgb_frames not available from extract_grasp")
    
    try:
        import cv2  # type: ignore[import]
    except ImportError:
        raise ImportError("cv2 is required for video rendering. Install with: pip install opencv-python")
    
    wait_ms = max(1, int(1000.0 / max(play_speed_hz, 1e-3)))
    
    # Build a mapping from frame index to active action
    frame_to_action = {}
    for action in actions:
        start_step = action.get("start_step", 0)
        end_step = action.get("end_step", len(observations) - 1)
        for step_idx in range(start_step, end_step + 1):
            if step_idx not in frame_to_action:
                frame_to_action[step_idx] = []
            frame_to_action[step_idx].append(action)
    
    for step_idx, obs in enumerate(observations):
        frames = _prepare_rgb_frames(obs)
        if not frames:
            continue
        
        bgr_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
        frame_bgr = np.hstack(bgr_frames)
        
        # Get active actions for this frame
        active_actions = frame_to_action.get(step_idx, [])
        
        display = np.ascontiguousarray(frame_bgr)
        
        # Display frame number
        cv2.putText(
            display,
            f"frame: {step_idx}",
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=(0, 255, 255),
            thickness=2,
        )
        
        # Display active actions
        y_offset = 60
        if active_actions:
            for action_idx, action in enumerate(active_actions):
                action_type = action["type"]
                
                if action_type == "MOVE":
                    action_text = f"MOVE: {action['grasped_part']}"
                    color = (0, 255, 0)  # Green
                elif action_type == "INTERACT":
                    action_text = f"INTERACT: {action['target_part']} <-> {action['base_part']}"
                    color = (0, 165, 255)  # Orange
                else:
                    action_text = f"{action_type}"
                    color = (255, 255, 255)  # White
                
                cv2.putText(
                    display,
                    action_text,
                    org=(10, y_offset),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=color,
                    thickness=2,
                )
                y_offset += 30
        else:
            cv2.putText(
                display,
                "action: none",
                org=(10, y_offset),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(128, 128, 128),
                thickness=2,
            )
        
        cv2.imshow(window_title, display)
        key = cv2.waitKey(wait_ms)
        if key in (27, ord("q")):
            break
    
    cv2.destroyWindow(window_title)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract manipulation order from demonstrations"
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
        help="Distance threshold (m) for collision detection. Default: 0.015",
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
        "--output",
        type=Path,
        default=None,
        help="Directory to save output JSON file",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render video with action labels overlayed after processing",
    )
    parser.add_argument(
        "--play-speed-hz",
        type=float,
        default=30.0,
        help="Playback speed for rendered video in Hz (default: 30.0)",
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
    
    # Process only the first file (as per user's note: "only one demonstration")
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
    
    actions = _extract_manipulation_order(
        observations,
        furniture_name,
        part_names,
        thresholds,
        robot_from_april,
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
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / "manipulation_order.json"
    
    # Save JSON
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved manipulation order to {output_path.name}")
    print(f"Found {len(actions)} action(s):")
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
    
    # Render video if requested
    if args.render:
        window_title = f"{furniture_name} - Manipulation Order"
        if hasattr(annotation_path, "stem"):
            window_title += f" - {annotation_path.stem}"
        print(f"\nRendering video with action labels...")
        print("Press 'q' or ESC to quit the video")
        _render_action_video(
            observations,
            actions,
            window_title=window_title,
            play_speed_hz=args.play_speed_hz,
        )


if __name__ == "__main__":
    main()

