#!/usr/bin/env python3
"""
Test script to visualize object affordance trajectory using get_object_affordance from get_info.py.

This script loads a trajectory using get_object_affordance and visualizes it as an animated
3D scene showing how the target part moves relative to the base part.
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from reset.scripts.get_info import get_object_affordance
from reset.extract_from_demo.extract_grasp import (
    _ensure_open3d,
    _load_part_mesh_trimesh,
    _trimesh_to_open3d,
    TRIMESH_AVAILABLE,
)
from furniture_bench.config import config
from furniture_bench.furniture import furniture_factory
from furniture_bench.utils import transform as T


def _apply_transform_to_vertices(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 homogeneous transformation matrix to 3D vertices.
    
    Args:
        vertices: (N, 3) array of 3D vertices
        transform: (4, 4) homogeneous transformation matrix
    
    Returns:
        (N, 3) array of transformed vertices
    """
    # Convert to homogeneous coordinates: (N, 3) -> (N, 4)
    homo = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)], axis=1)
    
    # Apply transformation: transform @ homo.T gives (4, N)
    # Each column is a transformed homogeneous point
    transformed_homo = (transform @ homo.T).T  # Results in (N, 4)
    
    # Normalize by w component (homogeneous coordinate normalization)
    w = transformed_homo[:, 3:4]  # (N, 1)
    # Avoid division by zero (shouldn't happen for valid transforms, but be safe)
    w_safe = np.where(np.abs(w) < 1e-10, np.ones_like(w), w)
    transformed = transformed_homo[:, :3] / w_safe
    
    return transformed


def visualize_affordance_trajectory(
    base_part: str,
    target_part: str,
    trajectory: np.ndarray,
    furniture_name: str,
    fps: float = 30.0,
    show_trace: bool = True,
    window_title: str = "Object Affordance Visualization",
    use_viser: bool = False,
):
    """
    Visualize an object affordance trajectory showing target part movement relative to base part.
    
    Args:
        base_part: Name of the base part (fixed at origin)
        target_part: Name of the target part (animated)
        trajectory: Nx7 array of relative poses [x, y, z, qx, qy, qz, qw],
                    where each row is the pose of target_part in base_part's frame
        furniture_name: Name of the furniture
        fps: Frames per second for animation playback
        show_trace: Whether to show trajectory trace line
        window_title: Title for the visualization window
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required for visualization. Install it with: pip install trimesh")
    
    o3d = _ensure_open3d()
    
    # Load meshes
    furniture_conf = config["furniture"].get(furniture_name, {})
    if not isinstance(furniture_conf, dict):
        raise KeyError(f"No furniture config for {furniture_name}")
    
    base_conf = furniture_conf.get(base_part)
    target_conf = furniture_conf.get(target_part)
    
    if base_conf is None or "asset_file" not in base_conf:
        raise KeyError(f"Base part '{base_part}' has no asset_file in config")
    if target_conf is None or "asset_file" not in target_conf:
        raise KeyError(f"Target part '{target_part}' has no asset_file in config")
    
    # Load trimesh meshes - these are in each part's local coordinate frame
    mesh_base_tm = _load_part_mesh_trimesh(base_conf["asset_file"])
    mesh_target_tm = _load_part_mesh_trimesh(target_conf["asset_file"])
    
    # Get original vertices from trimesh BEFORE converting to open3d
    # This ensures we have the true original vertices in each part's local coordinate frame
    base_vertices_orig = mesh_base_tm.vertices.copy()  # Base part vertices in base's local frame (at origin)
    target_vertices_orig = mesh_target_tm.vertices.copy()  # Target part vertices in target's local frame
    
    # Convert to open3d for visualization
    mesh_base_o3d = _trimesh_to_open3d(mesh_base_tm)
    mesh_target_o3d = _trimesh_to_open3d(mesh_target_tm)
    
    mesh_base_o3d.compute_vertex_normals()
    mesh_target_o3d.compute_vertex_normals()
    
    # Color base and target
    mesh_base_o3d.paint_uniform_color([0.8, 0.7, 0.3])  # Orange
    mesh_target_o3d.paint_uniform_color([0.2, 0.5, 0.9])  # Blue
    
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
        
        server = create_viser_server(title=window_title)
        
        # Add base mesh (fixed at origin)
        base_mesh_data = _trimesh_to_viser_mesh_data(mesh_base_tm, color=(0.8, 0.7, 0.3))
        add_mesh_to_viser_scene(server, "/base", base_mesh_data)
        
        # Add base frame
        identity_transform = np.eye(4)
        add_frame_to_viser_scene(server, "/base_frame", identity_transform, size=0.05)
        
        # Add target mesh (will be updated in animation)
        target_mesh_data = _trimesh_to_viser_mesh_data(mesh_target_tm, color=(0.2, 0.5, 0.9))
        add_mesh_to_viser_scene(server, "/target", target_mesh_data)
        
        # Animation loop
        frame_dt = 1.0 / max(1.0, fps)
        prev_time = time.time()
        points_accum = []
        
        print(f"\nAnimating trajectory ({len(trajectory)} frames) with viser...")
        print("Open your browser to the URL shown above to view the visualization.")
        print("Press Ctrl+C to exit.")
        
        for idx in range(len(trajectory)):
            rel_pose = trajectory[idx]
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
        
        print("\nAnimation finished — window remains open for inspection. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nExiting...")
    else:
        # Ensure base part is positioned at origin in its own coordinate frame
        # Base part defines the coordinate frame, so it stays fixed
        mesh_base_o3d.vertices = o3d.utility.Vector3dVector(base_vertices_orig)
        mesh_base_o3d.compute_vertex_normals()
        
        # Prepare trajectory line set
        if show_trace:
            points = []
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
        
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_title)
        vis.add_geometry(mesh_base_o3d)
        vis.add_geometry(mesh_target_o3d)
        vis.add_geometry(base_frame)
        if traj_lineset is not None:
            vis.add_geometry(traj_lineset)
        
        # Set render options
        opt = vis.get_render_option()
        opt.mesh_show_back_face = True
        
        # Animation loop
        frame_dt = 1.0 / max(1.0, fps)
        prev_time = time.time()
        
        # For each frame: apply the relative pose transform directly to original target vertices
        # The relative pose represents the pose of target part in base part's coordinate frame
        # We transform target vertices from target's local frame to base frame using this relative pose
        # IMPORTANT: Always use target_vertices_orig (original vertices in target's local frame) 
        #            and apply rel_mat transform - never accumulate transforms
        points_accum = []
        
        print(f"\nAnimating trajectory ({len(trajectory)} frames)...")
        print("Controls:")
        print("  - Mouse: Rotate (left drag), Pan (right drag), Zoom (scroll)")
        print("  - Press Ctrl+C to exit")
        print()
        
        for idx in range(len(trajectory)):
            # Get relative pose: pose of target part in base part's coordinate frame
            rel_pose = trajectory[idx]
            
            # Convert to 4x4 transformation matrix
            # rel_mat transforms points from target's local coordinate frame to base coordinate frame
            rel_mat = T.pose2mat(rel_pose)
            
            # Apply transform to ORIGINAL target vertices (in target's local frame)
            # This transforms them to base frame coordinates
            # Each frame applies this transform independently - no accumulation
            transformed_vertices = _apply_transform_to_vertices(target_vertices_orig, rel_mat)
            mesh_target_o3d.vertices = o3d.utility.Vector3dVector(transformed_vertices)
            mesh_target_o3d.compute_vertex_normals()
            
            # Update trace: transform origin of target part (in target's local frame) to base frame
            if traj_lineset is not None:
                origin_in_target_frame = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
                origin_in_base_frame_homo = rel_mat @ origin_in_target_frame
                # Normalize by w component
                w = origin_in_base_frame_homo[3]
                if np.abs(w) > 1e-10:
                    point = origin_in_base_frame_homo[:3] / w
                else:
                    point = origin_in_base_frame_homo[:3]  # fallback (shouldn't happen for affine)
                points_accum.append(point.tolist())
                
                # Rebuild line set
                pts = np.asarray(points_accum)
                if pts.shape[0] >= 2:
                    lines = [[i, i + 1] for i in range(pts.shape[0] - 1)]
                    colors = [[0.9, 0.1, 0.1] for _ in range(len(lines))]
                    traj_lineset.points = o3d.utility.Vector3dVector(pts)
                    traj_lineset.lines = o3d.utility.Vector2iVector(lines)
                    traj_lineset.colors = o3d.utility.Vector3dVector(colors)
                else:
                    traj_lineset.points = o3d.utility.Vector3dVector(pts)
                    traj_lineset.lines = o3d.utility.Vector2iVector([])
                    traj_lineset.colors = o3d.utility.Vector3dVector([])
                
                vis.update_geometry(traj_lineset)
            
            vis.update_geometry(mesh_target_o3d)
            vis.poll_events()
            vis.update_renderer()
            
            # Maintain playback speed but keep UI responsive
            elapsed = time.time() - prev_time
            sleep = max(0.0, frame_dt - elapsed)
            time.sleep(sleep)
            prev_time = time.time()
        
        print("\nAnimation finished — window remains open for inspection. Close it to exit.")
        # Keep window open until closed by user
        try:
            while True:
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.02)
        except KeyboardInterrupt:
            print("\nExiting...")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize object affordance trajectory using get_object_affordance"
    )
    parser.add_argument(
        "--base-part",
        type=str,
        required=True,
        help="Name of the base part (fixed at origin)"
    )
    parser.add_argument(
        "--target-part",
        type=str,
        required=True,
        help="Name of the target part (animated)"
    )
    parser.add_argument(
        "--furniture",
        type=str,
        default=None,
        help="Furniture name (optional, will search all if not provided)"
    )
    parser.add_argument(
        "--mode",
        type=int,
        default=0,
        help="Trajectory mode to use (0 = first trajectory, 1+ = other trajectories)"
    )
    parser.add_argument(
        "--affordance-path",
        type=str,
        default=None,
        help="Optional path to affordance file (defaults to extracted_info/object_affordance_trajectories.json)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Animation playback frames per second (default: 30.0)"
    )
    parser.add_argument(
        "--no-trace",
        dest="show_trace",
        action="store_false",
        help="Do not show trajectory trace line"
    )
    parser.add_argument(
        "--use-viser",
        action="store_true",
        help="Use viser for visualization instead of Open3D"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Testing get_object_affordance function")
    print("=" * 60)
    
    # Get trajectory using get_object_affordance
    print(f"\nLoading trajectory for pair '{args.base_part}' -> '{args.target_part}' (mode={args.mode})...")
    try:
        affordance_path = Path(args.affordance_path) if args.affordance_path else None
        trajectory = get_object_affordance(
            base_part=args.base_part,
            target_part=args.target_part,
            furniture_name=args.furniture,
            mode=args.mode,
            affordance_path=affordance_path,
        )
        print(f"Loaded trajectory with {len(trajectory)} frames")
    except Exception as e:
        print(f"ERROR: Failed to load trajectory: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Determine furniture name if not provided
    if args.furniture is None:
        # Try to infer from affordance data
        try:
            from reset.scripts.get_info import _load_object_affordance
            affordance_data = _load_object_affordance(affordance_path)
            # Find furniture that contains this pair
            pair_key = f"{args.base_part}__{args.target_part}"
            for furn_name, pairs_data in affordance_data.items():
                if pair_key in pairs_data:
                    args.furniture = furn_name
                    print(f"Inferred furniture name: {args.furniture}")
                    break
            
            if args.furniture is None:
                print(f"ERROR: Could not find pair '{pair_key}' in any furniture type")
                return 1
        except Exception as e:
            print(f"ERROR: Could not infer furniture name: {e}")
            return 1
    
    # Visualize
    window_title = f"Affordance: {args.base_part} <- {args.target_part} (mode={args.mode})"
    try:
        visualize_affordance_trajectory(
            base_part=args.base_part,
            target_part=args.target_part,
            trajectory=trajectory,
            furniture_name=args.furniture,
            fps=args.fps,
            show_trace=args.show_trace,
            window_title=window_title,
            use_viser=args.use_viser,
        )
    except Exception as e:
        print(f"ERROR during visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())

