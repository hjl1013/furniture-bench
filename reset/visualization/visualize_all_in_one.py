"""
Visualize all-in-one extraction results.

Shows:
1. Full demonstration video with action labels
2. For each action:
   - Action-specific video clip
   - Grasp pose visualization (3D)
   - Affordance trajectory visualization (if INTERACT)
"""
import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Try to import helper functions
try:
    from reset.extract_from_demo.extract_grasp import (  # type: ignore[import]
        _prepare_rgb_frames,
        _load_gripper_mesh,
        _load_part_mesh,
        _ensure_open3d,
        _quat_normalize,
        _pose_vec_to_mat,
        _mesh_with_transform,
        _load_part_mesh_trimesh,
        _trimesh_to_open3d,
    )
except Exception:
    try:
        from extract_grasp import (  # type: ignore[import]
            _prepare_rgb_frames,
            _load_gripper_mesh,
            _load_part_mesh,
            _ensure_open3d,
            _quat_normalize,
            _pose_vec_to_mat,
            _mesh_with_transform,
            _load_part_mesh_trimesh,
            _trimesh_to_open3d,
        )
    except Exception:
        _prepare_rgb_frames = None
        _load_gripper_mesh = None
        _load_part_mesh = None
        _ensure_open3d = None
        _quat_normalize = None
        _pose_vec_to_mat = None
        _mesh_with_transform = None
        _load_part_mesh_trimesh = None
        _trimesh_to_open3d = None

from furniture_bench.config import config
from furniture_bench.utils import transform as T


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


def _render_action_video(
    observations: List[Dict],
    actions: List[Dict],
    window_title: str,
    play_speed_hz: float = 30.0,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
):
    """
    Render demonstration video with action labels overlayed.
    
    Args:
        observations: List of observation dictionaries
        actions: List of action dictionaries with start_step and end_step
        window_title: Title for the video window
        play_speed_hz: Playback speed in Hz
        start_frame: Optional start frame (for action-specific clips)
        end_frame: Optional end frame (for action-specific clips)
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
    
    # Determine frame range
    frame_start = start_frame if start_frame is not None else 0
    frame_end = end_frame if end_frame is not None else len(observations)
    
    for step_idx in range(frame_start, frame_end):
        if step_idx >= len(observations):
            break
        
        obs = observations[step_idx]
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


def _visualize_grasp_pose(
    furniture_name: str,
    part_name: str,
    grasp_pose: Dict[str, List[float]],
    use_viser: bool = False,
):
    """
    Visualize a single grasp pose in 3D.
    
    Args:
        furniture_name: Name of furniture
        part_name: Name of part being grasped
        grasp_pose: Dictionary with "relative_position" and "relative_quaternion"
        use_viser: Whether to use viser for visualization
    """
    if _load_gripper_mesh is None or _load_part_mesh is None or _ensure_open3d is None:
        raise ImportError("Mesh loading functions not available")
    
    o3d = _ensure_open3d()
    
    # Load meshes
    try:
        gripper_mesh = _load_gripper_mesh()
    except Exception as e:
        print(f"Warning: Could not load gripper mesh: {e}")
        return
    
    furniture_conf = config["furniture"].get(furniture_name, {})
    part_conf = furniture_conf.get(part_name)
    if not isinstance(part_conf, dict) or "asset_file" not in part_conf:
        print(f"Warning: Could not find asset file for part '{part_name}'")
        return
    
    try:
        part_mesh = _load_part_mesh(part_conf["asset_file"])
    except Exception as e:
        print(f"Warning: Could not load part mesh: {e}")
        return
    
    # Get relative pose
    rel_pos = np.array(grasp_pose["relative_position"], dtype=np.float64)
    rel_quat = np.array(grasp_pose["relative_quaternion"], dtype=np.float64)
    rel_quat = _quat_normalize(rel_quat)
    
    # Compute transforms
    rel_T = _pose_vec_to_mat(rel_pos, rel_quat)
    rel_T_inv = np.linalg.inv(rel_T)
    
    # Part at origin, gripper positioned relative to part
    part_T = np.eye(4)
    ee_T = part_T @ rel_T_inv
    
    # Color meshes
    part_mesh_colored = _mesh_with_transform(part_mesh, part_T, color=(0.9, 0.7, 0.3))
    gripper_mesh_colored = _mesh_with_transform(gripper_mesh, ee_T, color=(0.3, 0.5, 0.9))
    
    # Create coordinate frames
    part_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05).transform(ee_T)
    
    geometries = [part_mesh_colored, gripper_mesh_colored, part_frame, gripper_frame]
    
    window_title = f"Grasp Pose: {part_name}"
    print(f"\nVisualizing grasp pose for {part_name}")
    print("Close the window to continue...")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_title,
        width=1280,
        height=720,
    )


def _visualize_affordance_trajectory(
    furniture_name: str,
    base_part: str,
    target_part: str,
    affordance_trajectory: List[List[float]],
    fps: float = 30.0,
    show_trace: bool = True,
    use_viser: bool = False,
):
    """
    Visualize object affordance trajectory.
    
    Args:
        furniture_name: Name of furniture
        base_part: Name of base part
        target_part: Name of target part
        affordance_trajectory: List of relative pose vectors [x,y,z,qx,qy,qz,qw] (target in base frame)
        fps: Playback frames per second
        show_trace: Whether to show trajectory trace
        use_viser: Whether to use viser for visualization
    """
    if _load_part_mesh is None or _ensure_open3d is None:
        raise ImportError("Mesh loading functions not available")
    
    # Convert to numpy array
    traj_arr = np.asarray(affordance_trajectory, dtype=np.float64)
    
    # Load meshes
    furniture_conf = config["furniture"].get(furniture_name, {})
    base_conf = furniture_conf.get(base_part)
    target_conf = furniture_conf.get(target_part)
    
    if base_conf is None or "asset_file" not in base_conf:
        raise KeyError(f"Base part '{base_part}' has no asset_file in config")
    if target_conf is None or "asset_file" not in target_conf:
        raise KeyError(f"Target part '{target_part}' has no asset_file in config")
    
    if _load_part_mesh_trimesh is None or _trimesh_to_open3d is None:
        raise ImportError("Could not import mesh loading functions")
    
    mesh_base_tm = _load_part_mesh_trimesh(base_conf["asset_file"])
    mesh_target_tm = _load_part_mesh_trimesh(target_conf["asset_file"])
    
    base_vertices_orig = mesh_base_tm.vertices.copy()
    target_vertices_orig = mesh_target_tm.vertices.copy()
    
    o3d = _ensure_open3d()
    
    # Convert to open3d
    mesh_base_o3d = _trimesh_to_open3d(mesh_base_tm)
    mesh_target_o3d = _trimesh_to_open3d(mesh_target_tm)
    
    mesh_base_o3d.compute_vertex_normals()
    mesh_target_o3d.compute_vertex_normals()
    
    # Color meshes
    mesh_base_o3d.paint_uniform_color([0.8, 0.7, 0.3])  # orange
    mesh_target_o3d.paint_uniform_color([0.2, 0.5, 0.9])  # blue
    
    # Base part stays fixed at origin
    mesh_base_o3d.vertices = o3d.utility.Vector3dVector(base_vertices_orig)
    mesh_base_o3d.compute_vertex_normals()
    
    # Prepare trajectory line set
    if show_trace:
        points = [list(np.zeros(3))]
        lines = []
        colors = []
        traj_lineset = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        traj_lineset.colors = o3d.utility.Vector3dVector(colors)
    else:
        traj_lineset = None
    
    # Create coordinate frame for base
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Affordance: {base_part} <- {target_part}")
    vis.add_geometry(mesh_base_o3d)
    vis.add_geometry(mesh_target_o3d)
    vis.add_geometry(base_frame)
    if traj_lineset is not None:
        vis.add_geometry(traj_lineset)
    
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    
    # Animation loop
    frame_dt = 1.0 / max(1, fps)
    prev_time = time.time()
    points_accum = []
    
    def _apply_transform_to_vertices(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
        homo = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)], axis=1)
        transformed = (transform @ homo.T).T[:, :3]
        return transformed
    
    for idx in range(traj_arr.shape[0]):
        rel_pose = traj_arr[idx]
        rel_mat = T.pose2mat(rel_pose)
        
        # Apply transform to original target vertices
        transformed_vertices = _apply_transform_to_vertices(target_vertices_orig, rel_mat)
        mesh_target_o3d.vertices = o3d.utility.Vector3dVector(transformed_vertices)
        mesh_target_o3d.compute_vertex_normals()
        
        # Update trace
        if traj_lineset is not None:
            origin_in_target_frame = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            origin_in_base_frame_homo = rel_mat @ origin_in_target_frame
            w = origin_in_base_frame_homo[3]
            if np.abs(w) > 1e-10:
                point = origin_in_base_frame_homo[:3] / w
            else:
                point = origin_in_base_frame_homo[:3]
            points_accum.append(point.tolist())
            
            if len(points_accum) >= 2:
                pts = np.asarray(points_accum)
                lines = [[i, i + 1] for i in range(pts.shape[0] - 1)]
                colors = [[0.9, 0.1, 0.1] for _ in range(len(lines))]
                traj_lineset.points = o3d.utility.Vector3dVector(pts)
                traj_lineset.lines = o3d.utility.Vector2iVector(lines)
                traj_lineset.colors = o3d.utility.Vector3dVector(colors)
                vis.update_geometry(traj_lineset)
        
        vis.update_geometry(mesh_target_o3d)
        vis.poll_events()
        vis.update_renderer()
        
        elapsed = time.time() - prev_time
        sleep = max(0.0, frame_dt - elapsed)
        time.sleep(sleep)
        prev_time = time.time()
    
    print("Animation finished — window remains open for inspection. Close it to continue...")
    while True:
        try:
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.02)
        except KeyboardInterrupt:
            break
        if not vis.poll_events():
            break
    
    vis.destroy_window()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize all-in-one extraction results"
    )
    parser.add_argument(
        "--extraction-file",
        type=Path,
        required=True,
        help="Path to all_in_one_extraction.json file",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to dataset directory or .pkl file (auto-detected if not provided)",
    )
    parser.add_argument(
        "--play-speed-hz",
        type=float,
        default=30.0,
        help="Playback speed for rendered video in Hz (default: 30.0)",
    )
    parser.add_argument(
        "--affordance-fps",
        type=float,
        default=30.0,
        help="Playback speed for affordance animation in fps (default: 30.0)",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Do not show trajectory trace in affordance visualization",
    )
    parser.add_argument(
        "--skip-full-video",
        action="store_true",
        help="Skip showing the full demonstration video",
    )
    return parser.parse_args()


def find_dataset_file(extraction_file_path: Path) -> Optional[Path]:
    """Try to find the dataset file associated with the extraction file."""
    parent_dir = extraction_file_path.parent
    pkl_files = list(parent_dir.glob("*.pkl"))
    if pkl_files:
        return pkl_files[0]
    
    for parent in parent_dir.parents:
        pkl_files = list(parent.glob("*.pkl"))
        if pkl_files:
            return pkl_files[0]
    
    return None


def main():
    args = parse_args()
    
    # Load extraction file
    extraction_path = Path(args.extraction_file).expanduser().resolve()
    if not extraction_path.is_file():
        raise FileNotFoundError(f"Extraction file not found: {extraction_path}")
    
    print(f"Loading extraction file: {extraction_path}")
    with open(extraction_path, "r", encoding="utf-8") as f:
        extraction_data = json.load(f)
    
    furniture_name = extraction_data.get("furniture")
    actions = extraction_data.get("actions", [])
    num_actions = extraction_data.get("num_actions", len(actions))
    
    print(f"Furniture: {furniture_name}")
    print(f"Number of actions: {num_actions}")
    
    # Find dataset file
    if args.dataset is not None:
        dataset_path = Path(args.dataset).expanduser().resolve()
    else:
        dataset_path = find_dataset_file(extraction_path)
        if dataset_path is None:
            raise FileNotFoundError(
                "Could not find dataset file. Please specify --dataset or ensure "
                "a .pkl file exists in the same directory as the extraction file."
            )
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Load dataset
    if dataset_path.is_file() and dataset_path.suffix == ".pkl":
        annotation_files = [dataset_path]
    else:
        annotation_files = list(_iter_annotation_files(dataset_path))
    
    if not annotation_files:
        raise FileNotFoundError(f"No .pkl files found at {dataset_path}")
    
    annotation_path = annotation_files[0]
    print(f"Using annotation file: {annotation_path}")
    
    with open(annotation_path, "rb") as f:
        data = pickle.load(f)
    
    observations = data.get("observations", [])
    if not observations:
        raise ValueError("No observations found in dataset")
    
    print(f"Loaded {len(observations)} observations")
    
    # Print action summary
    print("\nActions:")
    for i, action in enumerate(actions):
        action_type = action["type"]
        if action_type == "MOVE":
            print(
                f"  {i+1}. MOVE: {action['grasped_part']} "
                f"(steps {action.get('start_step', '?')}-{action.get('end_step', '?')})"
            )
        elif action_type == "INTERACT":
            print(
                f"  {i+1}. INTERACT: {action['target_part']} with {action['base_part']} "
                f"(steps {action.get('start_step', '?')}-{action.get('end_step', '?')})"
            )
    
    # 1. Show full video
    if not args.skip_full_video:
        window_title = f"{furniture_name} - Full Demonstration"
        if hasattr(annotation_path, "stem"):
            window_title += f" - {annotation_path.stem}"
        
        print(f"\n{'='*60}")
        print("1. Showing full demonstration video...")
        print(f"{'='*60}")
        print("Press 'q' or ESC to quit the video")
        
        _render_action_video(
            observations,
            actions,
            window_title=window_title,
            play_speed_hz=args.play_speed_hz,
        )
    
    # 2. Show each action individually
    for action_idx, action in enumerate(actions):
        action_type = action["type"]
        start_step = action.get("start_step", 0)
        end_step = action.get("end_step", len(observations) - 1)
        
        print(f"\n{'='*60}")
        print(f"Action {action_idx + 1}/{num_actions}: {action_type}")
        print(f"{'='*60}")
        
        if action_type == "MOVE":
            print(f"  Grasped part: {action['grasped_part']}")
            print(f"  Steps: {start_step}-{end_step}")
        elif action_type == "INTERACT":
            print(f"  Target part: {action['target_part']}")
            print(f"  Base part: {action['base_part']}")
            print(f"  Steps: {start_step}-{end_step}")
        
        # 2a. Show action-specific video
        print(f"\n2a. Showing action video clip...")
        print("Press 'q' or ESC to quit the video")
        
        action_window_title = f"{furniture_name} - Action {action_idx + 1}: {action_type}"
        _render_action_video(
            observations,
            [action],  # Only show this action
            window_title=action_window_title,
            play_speed_hz=args.play_speed_hz,
            start_frame=start_step,
            end_frame=end_step + 1,
        )
        
        # 2b. Visualize grasp pose
        print(f"\n2b. Visualizing grasp pose...")
        if action_type == "MOVE":
            part_name = action["grasped_part"]
        else:  # INTERACT
            part_name = action["target_part"]
        
        grasp_pose = action.get("grasp_pose")
        if grasp_pose:
            try:
                _visualize_grasp_pose(
                    furniture_name,
                    part_name,
                    grasp_pose,
                    use_viser=False,
                )
            except Exception as e:
                print(f"Warning: Could not visualize grasp pose: {e}")
        else:
            print("Warning: No grasp pose found in action")
        
        # 2c. Visualize affordance trajectory (if INTERACT)
        if action_type == "INTERACT":
            print(f"\n2c. Visualizing affordance trajectory...")
            affordance_trajectory = action.get("affordance_trajectory")
            if affordance_trajectory:
                try:
                    _visualize_affordance_trajectory(
                        furniture_name,
                        action["base_part"],
                        action["target_part"],
                        affordance_trajectory,
                        fps=args.affordance_fps,
                        show_trace=not args.no_trace,
                        use_viser=False,
                    )
                except Exception as e:
                    print(f"Warning: Could not visualize affordance trajectory: {e}")
            else:
                print("Warning: No affordance trajectory found in action")
        
        # Ask user if they want to continue
        if action_idx < len(actions) - 1:
            print(f"\nPress Enter to continue to next action, or Ctrl+C to exit...")
            try:
                input()
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == "__main__":
    main()

