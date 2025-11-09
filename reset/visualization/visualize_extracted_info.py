"""
Visualize extracted information from extracted_info.pkl.

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

# Import helper functions from utils
from reset.utils.frames import prepare_rgb_frames as _prepare_rgb_frames
from reset.utils.mesh_utils import (
    load_gripper_mesh as _load_gripper_mesh,
    load_part_mesh as _load_part_mesh,
    load_part_mesh_trimesh as _load_part_mesh_trimesh,
    trimesh_to_open3d as _trimesh_to_open3d,
    _ensure_open3d,
)
from reset.utils.pose_utils import (
    quat_normalize as _quat_normalize,
    pose_vec_to_mat as _pose_vec_to_mat,
)
from reset.utils.file_io import iter_annotation_files
import copy

def _mesh_with_transform(base_mesh, transform, color=None):
    """Apply transform and color to mesh."""
    mesh = copy.deepcopy(base_mesh)
    mesh.transform(transform)
    if color is not None:
        mesh.paint_uniform_color(color)
    return mesh

from furniture_bench.config import config
from furniture_bench.utils import transform as T


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
        raise ImportError("_prepare_rgb_frames not available")
    
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


def _add_grasp_pose_to_viser(
    server,
    furniture_name: str,
    part_name: str,
    grasp_pose: Dict[str, List[float]],
    action_idx: int,
    offset: np.ndarray,
):
    """
    Add a grasp pose visualization to an existing viser server.
    
    Args:
        server: Viser server instance
        furniture_name: Name of furniture
        part_name: Name of part being grasped
        grasp_pose: Dictionary with "relative_position" and "relative_quaternion"
                    representing gripper pose relative to part (gripper in part frame)
        action_idx: Index of the action (for unique path naming)
        offset: 3D offset to position this visualization
    """
    if _load_gripper_mesh is None or _load_part_mesh is None:
        raise ImportError("Mesh loading functions not available")
    
    # Load meshes
    try:
        gripper_mesh_o3d = _load_gripper_mesh()
    except Exception as e:
        print(f"Warning: Could not load gripper mesh: {e}")
        return
    
    furniture_conf = config["furniture"].get(furniture_name, {})
    part_conf = furniture_conf.get(part_name)
    if not isinstance(part_conf, dict) or "asset_file" not in part_conf:
        print(f"Warning: Could not find asset file for part '{part_name}'")
        return
    
    try:
        part_mesh_o3d = _load_part_mesh(part_conf["asset_file"])
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
    
    # Part at origin with offset, gripper positioned relative to part
    part_T = np.eye(4)
    part_T[:3, 3] = offset
    ee_T = part_T @ rel_T_inv
    
    from reset.visualization.viser_utils import (
        add_mesh_to_viser_scene,
        _open3d_to_viser_mesh_data,
    )
    
    # Add part mesh
    part_mesh_data = _open3d_to_viser_mesh_data(part_mesh_o3d, color=(0.9, 0.7, 0.3))
    add_mesh_to_viser_scene(server, f"/grasp_{action_idx}/part", part_mesh_data, transform=part_T)
    
    # Add gripper mesh
    gripper_mesh_data = _open3d_to_viser_mesh_data(gripper_mesh_o3d, color=(0.3, 0.5, 0.9))
    add_mesh_to_viser_scene(server, f"/grasp_{action_idx}/gripper", gripper_mesh_data, transform=ee_T)


def _visualize_grasp_pose(
    furniture_name: str,
    part_name: str,
    grasp_pose: Dict[str, List[float]],
    use_viser: bool = False,
):
    """
    Visualize a single grasp pose in 3D (Open3D only, for non-viser mode).
    
    Args:
        furniture_name: Name of furniture
        part_name: Name of part being grasped
        grasp_pose: Dictionary with "relative_position" and "relative_quaternion"
                    representing gripper pose relative to part (gripper in part frame)
        use_viser: Whether to use viser for visualization (not used here, kept for compatibility)
    """
    if _load_gripper_mesh is None or _load_part_mesh is None:
        raise ImportError("Mesh loading functions not available")
    
    # Load meshes
    try:
        gripper_mesh_o3d = _load_gripper_mesh()
    except Exception as e:
        print(f"Warning: Could not load gripper mesh: {e}")
        return
    
    furniture_conf = config["furniture"].get(furniture_name, {})
    part_conf = furniture_conf.get(part_name)
    if not isinstance(part_conf, dict) or "asset_file" not in part_conf:
        print(f"Warning: Could not find asset file for part '{part_name}'")
        return
    
    try:
        part_mesh_o3d = _load_part_mesh(part_conf["asset_file"])
    except Exception as e:
        print(f"Warning: Could not load part mesh: {e}")
        return
    
    # Get relative pose (gripper in part frame)
    rel_pos = np.array(grasp_pose["relative_position"], dtype=np.float64)
    rel_quat = np.array(grasp_pose["relative_quaternion"], dtype=np.float64)
    rel_quat = _quat_normalize(rel_quat)
    
    # Compute transforms
    rel_T = _pose_vec_to_mat(rel_pos, rel_quat)
    # rel_T now represents gripper in part frame (no inverse needed)
    
    # Part at origin, gripper positioned relative to part
    part_T = np.eye(4)
    ee_T = part_T @ rel_T  # gripper_pose = part_pose @ (gripper_in_part)
    
    o3d = _ensure_open3d()
    
    # Color meshes
    part_mesh_colored = _mesh_with_transform(part_mesh_o3d, part_T, color=(0.9, 0.7, 0.3))
    gripper_mesh_colored = _mesh_with_transform(gripper_mesh_o3d, ee_T, color=(0.3, 0.5, 0.9))
    
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


def _add_affordance_trajectory_to_viser(
    server,
    furniture_name: str,
    base_part: str,
    target_part: str,
    affordance_trajectory: List[List[float]],
    action_idx: int,
    offset: np.ndarray,
    show_trace: bool = True,
):
    """
    Add an affordance trajectory visualization to an existing viser server.
    
    Args:
        server: Viser server instance
        furniture_name: Name of furniture
        base_part: Name of base part
        target_part: Name of target part
        affordance_trajectory: List of relative pose vectors [x,y,z,qx,qy,qz,qw] (target in base frame)
        action_idx: Index of the action (for unique path naming)
        offset: 3D offset to position this visualization
        show_trace: Whether to show trajectory trace
    """
    if _load_part_mesh_trimesh is None:
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
    
    mesh_base_tm = _load_part_mesh_trimesh(base_conf["asset_file"])
    mesh_target_tm = _load_part_mesh_trimesh(target_conf["asset_file"])
    
    target_vertices_orig = mesh_target_tm.vertices.copy()
    
    def _apply_transform_to_vertices(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
        homo = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)], axis=1)
        transformed = (transform @ homo.T).T[:, :3]
        return transformed
    
    from reset.visualization.viser_utils import (
        add_mesh_to_viser_scene,
        _trimesh_to_viser_mesh_data,
    )
    
    # Add base mesh (initially hidden, will be centered when visible)
    # offset parameter is now always HIDE_OFFSET when passed in
    base_offset_T = np.eye(4)
    base_offset_T[:3, 3] = offset  # offset is HIDE_OFFSET when initially added
    base_mesh_data = _trimesh_to_viser_mesh_data(mesh_base_tm, color=(0.8, 0.7, 0.3))
    add_mesh_to_viser_scene(server, f"/affordance_{action_idx}/base", base_mesh_data, transform=base_offset_T)
    
    # Add target mesh at initial pose (initially hidden)
    if len(traj_arr) > 0:
        rel_pose = traj_arr[0]
        rel_mat = T.pose2mat(rel_pose)
        transformed_vertices = _apply_transform_to_vertices(target_vertices_orig, rel_mat)
        target_mesh_data = {
            "vertices": transformed_vertices + offset,  # offset is HIDE_OFFSET when initially added
            "faces": mesh_target_tm.faces,
            "vertex_colors": np.tile(np.array([0.2, 0.5, 0.9], dtype=np.float32), (len(transformed_vertices), 1)),
        }
        add_mesh_to_viser_scene(server, f"/affordance_{action_idx}/target", target_mesh_data)
        
        # Store trajectory data for animation
        server._affordance_trajectories = getattr(server, '_affordance_trajectories', {})
        server._affordance_trajectories[action_idx] = {
            'trajectory': traj_arr,
            'target_vertices_orig': target_vertices_orig,
            'target_faces': mesh_target_tm.faces,
            'show_trace': show_trace,
            'base_conf': base_conf,
            'target_conf': target_conf,
        }


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
    if _load_part_mesh_trimesh is None:
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
    
    mesh_base_tm = _load_part_mesh_trimesh(base_conf["asset_file"])
    mesh_target_tm = _load_part_mesh_trimesh(target_conf["asset_file"])
    
    base_vertices_orig = mesh_base_tm.vertices.copy()
    target_vertices_orig = mesh_target_tm.vertices.copy()
    
    def _apply_transform_to_vertices(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
        homo = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)], axis=1)
        transformed = (transform @ homo.T).T[:, :3]
        return transformed
    
    if use_viser:
        from reset.visualization.viser_utils import (
            create_viser_server,
            add_mesh_to_viser_scene,
            update_mesh_in_viser_scene,
            add_frame_to_viser_scene,
            add_line_set_to_viser_scene,
            update_line_set_in_viser_scene,
            _trimesh_to_viser_mesh_data,
            VISER_AVAILABLE,
        )
        if not VISER_AVAILABLE:
            raise ImportError("viser is required for viser visualization. Install with: pip install viser")
        
        server = create_viser_server(title=f"Affordance: {base_part} <- {target_part}")
        
        # Add base mesh (fixed at origin)
        base_mesh_data = _trimesh_to_viser_mesh_data(mesh_base_tm, color=(0.8, 0.7, 0.3))
        add_mesh_to_viser_scene(server, "/base", base_mesh_data)
        
        # Add target mesh (will be updated in animation)
        target_mesh_data = _trimesh_to_viser_mesh_data(mesh_target_tm, color=(0.2, 0.5, 0.9))
        add_mesh_to_viser_scene(server, "/target", target_mesh_data)
        
        # Animation loop
        frame_dt = 1.0 / max(1, fps)
        prev_time = time.time()
        points_accum = []
        
        print(f"\nAnimating trajectory ({len(traj_arr)} frames) with viser...")
        print("Open your browser to the URL shown above to view the visualization.")
        print("Press Ctrl+C to exit.")
        
        for idx in range(traj_arr.shape[0]):
            rel_pose = traj_arr[idx]
            rel_mat = T.pose2mat(rel_pose)
            
            # Update target mesh
            transformed_vertices = _apply_transform_to_vertices(target_vertices_orig, rel_mat)
            target_mesh_data_updated = {
                "vertices": transformed_vertices,
                "faces": target_mesh_data["faces"],
                "vertex_colors": target_mesh_data.get("vertex_colors"),
            }
            update_mesh_in_viser_scene(server, "/target", target_mesh_data_updated)
            
            # Update trace
            if show_trace:
                origin_in_target_frame = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
                origin_in_base_frame_homo = rel_mat @ origin_in_target_frame
                w = origin_in_base_frame_homo[3]
                if np.abs(w) > 1e-10:
                    point = origin_in_base_frame_homo[:3] / w
                else:
                    point = origin_in_base_frame_homo[:3]
                points_accum.append(point)
                
                if len(points_accum) >= 2:
                    pts = np.asarray(points_accum)
                    update_line_set_in_viser_scene(server, "/trace", pts, color=(0.9, 0.1, 0.1))
            
            # Maintain playback speed
            elapsed = time.time() - prev_time
            sleep = max(0.0, frame_dt - elapsed)
            time.sleep(sleep)
            prev_time = time.time()
        
        print("Animation finished — window remains open for inspection. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nExiting...")
    else:
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
        description="Visualize extracted information from extracted_info.pkl"
    )
    parser.add_argument(
        "--extraction-file",
        type=Path,
        required=True,
        help="Path to extracted_info.pkl file",
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
    parser.add_argument(
        "--use-viser",
        action="store_true",
        help="Use viser for visualization instead of Open3D",
    )
    return parser.parse_args()


def find_dataset_file(extraction_file_path: Path) -> Optional[Path]:
    """Try to find the dataset file associated with the extraction file."""
    parent_dir = extraction_file_path.parent
    pkl_files = list(parent_dir.glob("*.pkl"))
    if pkl_files:
        # Prefer dataset files, but if not found, return any pkl
        for pkl_file in pkl_files:
            if pkl_file.name != "extracted_info.pkl" and pkl_file.name != "distances.pkl":
                return pkl_file
        # Fallback to any pkl file
        if pkl_files:
            return pkl_files[0]
    
    for parent in parent_dir.parents:
        pkl_files = list(parent.glob("*.pkl"))
        if pkl_files:
            for pkl_file in pkl_files:
                if pkl_file.name != "extracted_info.pkl" and pkl_file.name != "distances.pkl":
                    return pkl_file
            if pkl_files:
                return pkl_files[0]
    
    return None


def load_extraction_file(extraction_path: Path) -> Dict:
    """Load extraction data from pickle file."""
    if not extraction_path.is_file():
        raise FileNotFoundError(f"Extraction file not found: {extraction_path}")
    
    print(f"Loading extraction file: {extraction_path}")
    with open(extraction_path, "rb") as f:
        extraction_data = pickle.load(f)
    
    return extraction_data


def main():
    args = parse_args()
    
    # Load extraction file (pkl format)
    extraction_path = Path(args.extraction_file).expanduser().resolve()
    extraction_data = load_extraction_file(extraction_path)
    
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
        annotation_files = list(iter_annotation_files(dataset_path))
    
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
    
    if args.use_viser:
        # Viser mode: create single server with all visualizations and buttons
        from reset.visualization.viser_utils import (
            create_viser_server,
            update_mesh_in_viser_scene,
            update_line_set_in_viser_scene,
            VISER_AVAILABLE,
        )
        if not VISER_AVAILABLE:
            raise ImportError("viser is required for viser visualization. Install with: pip install viser")
        
        server = create_viser_server(title=f"{furniture_name} - Extracted Information")
        
        # Single constant for hiding objects
        HIDE_OFFSET = np.array([0, 0, -1000])  # Move hidden objects far below
        
        # Track which action indices have grasps and affordances
        # Use sets for fast lookup
        grasp_action_indices_set = set()  # Set of action indices that have grasps
        affordance_action_indices_set = set()  # Set of action indices that have affordances
        
        # Store mesh data for re-adding when visibility changes
        server._grasp_mesh_data = {}  # action_idx -> (part_mesh_data, gripper_mesh_data, offset)
        server._affordance_mesh_data = {}  # action_idx -> mesh data
        
        # Add all grasps and affordances to the scene
        for action_idx, action in enumerate(actions):
            action_type = action["type"]
            
            if action_type == "MOVE":
                part_name = action["grasped_part"]
            else:  # INTERACT
                part_name = action["target_part"]
            
            grasp_pose = action.get("grasp_pose")
            if grasp_pose:
                try:
                    # Load and store mesh data
                    if _load_gripper_mesh is None or _load_part_mesh is None:
                        raise ImportError("Mesh loading functions not available")
                    
                    gripper_mesh_o3d = _load_gripper_mesh()
                    furniture_conf = config["furniture"].get(furniture_name, {})
                    part_conf = furniture_conf.get(part_name)
                    if isinstance(part_conf, dict) and "asset_file" in part_conf:
                        part_mesh_o3d = _load_part_mesh(part_conf["asset_file"])
                        
                        # Get relative pose (gripper in part frame)
                        rel_pos = np.array(grasp_pose["relative_position"], dtype=np.float64)
                        rel_quat = np.array(grasp_pose["relative_quaternion"], dtype=np.float64)
                        rel_quat = _quat_normalize(rel_quat)
                        
                        # Compute transforms - center at origin
                        rel_T = _pose_vec_to_mat(rel_pos, rel_quat)
                        # rel_T now represents gripper in part frame (no inverse needed)
                        
                        # Part at origin, gripper positioned relative to part
                        part_T = np.eye(4)  # Centered at origin
                        ee_T = part_T @ rel_T  # gripper_pose = part_pose @ (gripper_in_part)
                        
                        from reset.visualization.viser_utils import _open3d_to_viser_mesh_data
                        part_mesh_data = _open3d_to_viser_mesh_data(part_mesh_o3d, color=(0.9, 0.7, 0.3))
                        gripper_mesh_data = _open3d_to_viser_mesh_data(gripper_mesh_o3d, color=(0.3, 0.5, 0.9))
                        
                        # Store mesh data (with centered transforms)
                        server._grasp_mesh_data[action_idx] = (part_mesh_data, gripper_mesh_data, part_T, ee_T)
                        
                        # Add to scene (initially hidden)
                        from reset.visualization.viser_utils import add_mesh_to_viser_scene
                        part_T_hidden = part_T.copy()
                        part_T_hidden[:3, 3] += HIDE_OFFSET
                        ee_T_hidden = ee_T.copy()
                        ee_T_hidden[:3, 3] += HIDE_OFFSET
                        add_mesh_to_viser_scene(server, f"/grasp_{action_idx}/part", part_mesh_data, transform=part_T_hidden)
                        add_mesh_to_viser_scene(server, f"/grasp_{action_idx}/gripper", gripper_mesh_data, transform=ee_T_hidden)
                        
                        grasp_action_indices_set.add(action_idx)
                except Exception as e:
                    print(f"Warning: Could not add grasp pose for action {action_idx}: {e}")
            
            # Add affordance trajectory if INTERACT
            if action_type == "INTERACT":
                affordance_trajectory = action.get("affordance_trajectory")
                if affordance_trajectory:
                    try:
                        # Always center at origin (offset is always [0,0,0])
                        _add_affordance_trajectory_to_viser(
                            server,
                            furniture_name,
                            action["base_part"],
                            action["target_part"],
                            affordance_trajectory,
                            action_idx,
                            HIDE_OFFSET,  # Pass HIDE_OFFSET to start hidden
                            show_trace=not args.no_trace,
                        )
                        affordance_action_indices_set.add(action_idx)
                    except Exception as e:
                        print(f"Warning: Could not add affordance trajectory for action {action_idx}: {e}")
        
        # Create GUI controls for switching between visualizations
        # Use action indices directly (0 to num_actions-1)
        current_grasp_idx = server.gui.add_slider(
            label="Grasp Index (Action #)",
            min=0,
            max=max(0, num_actions - 1),
            step=1,
            initial_value=0,
        )
        
        current_affordance_idx = server.gui.add_slider(
            label="Affordance Index (Action #)",
            min=0,
            max=max(0, num_actions - 1),
            step=1,
            initial_value=0,
        )
        
        show_grasps = server.gui.add_checkbox(
            label="Show Grasps",
            initial_value=True,
        )
        
        show_affordances = server.gui.add_checkbox(
            label="Show Affordances",
            initial_value=True,
        )
        
        play_affordance = server.gui.add_button(
            label="Play Affordance Animation",
        )
        
        affordance_frame = server.gui.add_slider(
            label="Affordance Frame",
            min=0,
            max=100,
            step=1,
            initial_value=0,
        )
        
        # Store animation state
        server._animation_playing = False
        server._animation_frame = 0
        server._animation_action_idx = None
        
        def update_visibility():
            """Update visibility of grasps and affordances based on GUI state."""
            show_grasps_val = show_grasps.value
            show_affordances_val = show_affordances.value
            
            from reset.visualization.viser_utils import add_mesh_to_viser_scene, update_mesh_in_viser_scene
            
            # Hide/show all grasps by updating transforms - center visible ones at origin
            # Use action indices directly (0 to num_actions-1)
            selected_grasp_action_idx = current_grasp_idx.value
            
            for action_idx in range(num_actions):
                # Check if this action has a grasp
                has_grasp = action_idx in grasp_action_indices_set
                grasp_visible = show_grasps_val and has_grasp and (action_idx == selected_grasp_action_idx)
                
                if has_grasp and action_idx in server._grasp_mesh_data:
                    part_mesh_data, gripper_mesh_data, part_T_orig, ee_T_orig = server._grasp_mesh_data[action_idx]
                    
                    # Update transforms - center at origin if visible, hide otherwise
                    part_T = part_T_orig.copy()  # Already centered at origin
                    ee_T = ee_T_orig.copy()  # Already centered at origin
                    if not grasp_visible:
                        part_T[:3, 3] += HIDE_OFFSET
                        ee_T[:3, 3] += HIDE_OFFSET
                    
                    # Update meshes
                    try:
                        update_mesh_in_viser_scene(server, f"/grasp_{action_idx}/part", part_mesh_data, transform=part_T)
                        update_mesh_in_viser_scene(server, f"/grasp_{action_idx}/gripper", gripper_mesh_data, transform=ee_T)
                    except:
                        # Fallback: remove and re-add
                        try:
                            server.scene.remove(f"/grasp_{action_idx}/part")
                            server.scene.remove(f"/grasp_{action_idx}/gripper")
                        except:
                            pass
                        if grasp_visible:
                            add_mesh_to_viser_scene(server, f"/grasp_{action_idx}/part", part_mesh_data, transform=part_T)
                            add_mesh_to_viser_scene(server, f"/grasp_{action_idx}/gripper", gripper_mesh_data, transform=ee_T)
            
            # Hide/show ALL affordances by updating transforms - center visible one at origin
            # Use action indices directly (0 to num_actions-1)
            selected_affordance_action_idx = current_affordance_idx.value
            
            for action_idx in range(num_actions):
                # Check if this action has an affordance
                has_affordance = action_idx in affordance_action_indices_set
                affordance_visible = show_affordances_val and has_affordance and (action_idx == selected_affordance_action_idx)
                
                if has_affordance and hasattr(server, '_affordance_trajectories') and action_idx in server._affordance_trajectories:
                    traj_data = server._affordance_trajectories[action_idx]
                    
                    # Center at origin if visible, hide otherwise
                    base_offset = np.array([0.0, 0.0, 0.0]) if affordance_visible else HIDE_OFFSET
                    
                    # Update base mesh transform
                    base_offset_T = np.eye(4)
                    base_offset_T[:3, 3] = base_offset
                    
                    try:
                        from reset.visualization.viser_utils import _trimesh_to_viser_mesh_data
                        mesh_base_tm = _load_part_mesh_trimesh(traj_data['base_conf']['asset_file'])
                        base_mesh_data = _trimesh_to_viser_mesh_data(mesh_base_tm, color=(0.8, 0.7, 0.3))
                        update_mesh_in_viser_scene(server, f"/affordance_{action_idx}/base", base_mesh_data, transform=base_offset_T)
                    except:
                        pass
                    
                    # Update target mesh for ALL affordances (hide non-visible ones)
                    if affordance_visible:
                        # Only update frame for the visible/selected affordance
                        update_affordance_frame()
                    else:
                        # Hide target mesh for non-visible affordances
                        try:
                            target_mesh_data_hidden = {
                                "vertices": np.array([[0, 0, -1000], [0, 0, -1000], [0, 0, -1000]]),
                                "faces": np.array([[0, 1, 2]]),
                                "vertex_colors": np.array([[0.2, 0.5, 0.9], [0.2, 0.5, 0.9], [0.2, 0.5, 0.9]], dtype=np.float32),
                            }
                            update_mesh_in_viser_scene(server, f"/affordance_{action_idx}/target", target_mesh_data_hidden)
                            # Hide trace
                            update_line_set_in_viser_scene(server, f"/affordance_{action_idx}/trace", np.array([[0, 0, -1000], [0, 0, -1000]]), color=(0.9, 0.1, 0.1))
                        except:
                            pass
        
        def update_affordance_frame():
            """Update affordance animation frame for the currently selected affordance."""
            if not hasattr(server, '_affordance_trajectories'):
                return
            
            # Use action index directly
            action_idx = current_affordance_idx.value
            
            # Check if this action has an affordance
            if action_idx not in affordance_action_indices_set:
                return
            
            if action_idx not in server._affordance_trajectories:
                return
            
            # Check visibility - only update if this affordance is visible
            affordance_visible = show_affordances.value and (action_idx == current_affordance_idx.value)
            if not affordance_visible:
                return
            
            traj_data = server._affordance_trajectories[action_idx]
            traj_arr = traj_data['trajectory']
            frame_idx = min(int(affordance_frame.value), len(traj_arr) - 1)
            
            if frame_idx < len(traj_arr):
                rel_pose = traj_arr[frame_idx]
                rel_mat = T.pose2mat(rel_pose)
                
                # Always center at origin when visible
                base_offset = np.array([0.0, 0.0, 0.0])
                
                target_vertices_orig = traj_data['target_vertices_orig']
                
                def _apply_transform_to_vertices(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
                    homo = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)], axis=1)
                    transformed = (transform @ homo.T).T[:, :3]
                    return transformed
                
                transformed_vertices = _apply_transform_to_vertices(target_vertices_orig, rel_mat)
                target_mesh_data_updated = {
                    "vertices": transformed_vertices + base_offset,
                    "faces": traj_data['target_faces'],
                    "vertex_colors": np.tile(np.array([0.2, 0.5, 0.9], dtype=np.float32), (len(transformed_vertices), 1)),
                }
                update_mesh_in_viser_scene(server, f"/affordance_{action_idx}/target", target_mesh_data_updated)
                
                # Update trace
                if traj_data['show_trace']:
                    origin_in_target_frame = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
                    origin_in_base_frame_homo = rel_mat @ origin_in_target_frame
                    w = origin_in_base_frame_homo[3]
                    if np.abs(w) > 1e-10:
                        point = origin_in_base_frame_homo[:3] / w
                    else:
                        point = origin_in_base_frame_homo[:3]
                    point += base_offset
                    
                    # Get all points up to current frame
                    points_accum = []
                    for f_idx in range(frame_idx + 1):
                        rel_pose_f = traj_arr[f_idx]
                        rel_mat_f = T.pose2mat(rel_pose_f)
                        origin_in_base_frame_homo_f = rel_mat_f @ origin_in_target_frame
                        w_f = origin_in_base_frame_homo_f[3]
                        if np.abs(w_f) > 1e-10:
                            point_f = origin_in_base_frame_homo_f[:3] / w_f
                        else:
                            point_f = origin_in_base_frame_homo_f[:3]
                        points_accum.append(point_f + base_offset)
                    
                    if len(points_accum) >= 2:
                        pts = np.asarray(points_accum)
                        update_line_set_in_viser_scene(server, f"/affordance_{action_idx}/trace", pts, color=(0.9, 0.1, 0.1))
                else:
                    # Hide trace by moving it far away
                    try:
                        update_line_set_in_viser_scene(server, f"/affordance_{action_idx}/trace", np.array([[0, 0, -1000], [0, 0, -1000]]), color=(0.9, 0.1, 0.1))
                    except:
                        pass
        
        def on_play_click():
            """Handle play button click - toggle animation."""
            if not hasattr(server, '_affordance_trajectories'):
                return
            
            # Use action index directly
            action_idx = current_affordance_idx.value
            
            # Check if this action has an affordance
            if action_idx not in affordance_action_indices_set:
                return
            
            if action_idx not in server._affordance_trajectories:
                return
            
            traj_data = server._affordance_trajectories[action_idx]
            traj_arr = traj_data['trajectory']
            
            # Toggle animation
            server._animation_playing = not server._animation_playing
            server._animation_action_idx = action_idx
            
            if server._animation_playing:
                # Update max frame slider and reset to start
                affordance_frame.max = len(traj_arr) - 1
                affordance_frame.value = 0
                server._animation_frame = 0
                update_affordance_frame()  # Update immediately
            else:
                # Stop animation - frame stays at current position
                pass
        
        # Update affordance frame slider max when affordance index changes
        def on_affordance_idx_change(_):
            """Update affordance frame slider max when affordance index changes."""
            # Stop animation when switching affordances
            server._animation_playing = False
            server._animation_action_idx = None
            
            # Use action index directly
            action_idx = current_affordance_idx.value
            
            # Check if this action has an affordance
            if action_idx in affordance_action_indices_set and hasattr(server, '_affordance_trajectories') and action_idx in server._affordance_trajectories:
                traj_data = server._affordance_trajectories[action_idx]
                traj_arr = traj_data['trajectory']
                affordance_frame.max = len(traj_arr) - 1
                affordance_frame.value = 0
                server._animation_frame = 0
            else:
                # No affordance for this action, set max to 0
                affordance_frame.max = 0
                affordance_frame.value = 0
                server._animation_frame = 0
            
            update_visibility()
            update_affordance_frame()
        
        # Register callbacks
        current_grasp_idx.on_update(lambda _: update_visibility())
        current_affordance_idx.on_update(on_affordance_idx_change)
        show_grasps.on_update(lambda _: update_visibility())
        show_affordances.on_update(lambda _: update_visibility())
        affordance_frame.on_update(lambda _: update_affordance_frame())
        play_affordance.on_click(lambda _: on_play_click())
        
        # Initial visibility update and affordance frame setup
        # Find first action with affordance
        first_affordance_action_idx = None
        for action_idx in range(num_actions):
            if action_idx in affordance_action_indices_set:
                first_affordance_action_idx = action_idx
                break
        
        if first_affordance_action_idx is not None and hasattr(server, '_affordance_trajectories') and first_affordance_action_idx in server._affordance_trajectories:
            traj_data = server._affordance_trajectories[first_affordance_action_idx]
            traj_arr = traj_data['trajectory']
            affordance_frame.max = len(traj_arr) - 1
        else:
            affordance_frame.max = 0
        
        update_visibility()
        update_affordance_frame()
        
        print(f"\n{'='*60}")
        print("Viser Visualization Ready")
        print(f"{'='*60}")
        print("Open your browser to the URL shown above to view the visualization.")
        print("\nControls:")
        print("  - Use 'Grasp Index' slider to switch between grasp poses")
        print("  - Use 'Affordance Index' slider to switch between affordance trajectories")
        print("  - Use checkboxes to show/hide grasps and affordances")
        print("  - Use 'Affordance Frame' slider to scrub through affordance animation")
        print("  - Click 'Play Affordance Animation' to animate the selected affordance")
        print("  - Press Ctrl+C to exit")
        print()
        
        # Animation loop
        try:
            frame_dt = 1.0 / max(1, args.affordance_fps)
            prev_time = time.time()
            
            while True:
                # Handle affordance animation
                if server._animation_playing and server._animation_action_idx is not None:
                    # Use action index directly
                    current_action_idx = current_affordance_idx.value
                    
                    # Check if the selected affordance is still the one being animated and has an affordance
                    if (current_action_idx == server._animation_action_idx and 
                        current_action_idx in affordance_action_indices_set and
                        current_action_idx in server._affordance_trajectories):
                        traj_data = server._affordance_trajectories[current_action_idx]
                        traj_arr = traj_data['trajectory']
                        
                        elapsed = time.time() - prev_time
                        if elapsed >= frame_dt:
                            server._animation_frame = (server._animation_frame + 1) % len(traj_arr)
                            # Update the slider value and call update function directly
                            affordance_frame.value = server._animation_frame
                            update_affordance_frame()  # Call directly to ensure update
                            prev_time = time.time()
                    else:
                        # Affordance changed or doesn't exist, stop animation
                        server._animation_playing = False
                        server._animation_action_idx = None
                
                time.sleep(0.01)  # Small sleep to avoid busy waiting
        except KeyboardInterrupt:
            print("\nExiting...")
    else:
        # Open3D mode: show videos and individual visualizations
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

