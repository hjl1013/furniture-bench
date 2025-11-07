#!/usr/bin/env python3
"""
Test script to visualize a specific grasp pose using get_grasp_eef_pose from get_info.py.

This script loads a part pose, computes the end-effector pose using get_grasp_eef_pose,
and visualizes the part and gripper meshes in their relative positions.
"""
import argparse
import sys
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from reset.scripts.get_info import get_grasp_eef_pose
from reset.extract_from_demo.extract_grasp import (
    _ensure_open3d,
    _load_part_mesh,
    _load_gripper_mesh,
    _mesh_with_transform,
)
from furniture_bench.config import config
from furniture_bench.furniture import furniture_factory
import furniture_bench.utils.transform as T


def visualize_grasp(
    part_name: str,
    part_pose: np.ndarray,
    ee_pose: np.ndarray,
    furniture_name: str,
    window_title: str = "Grasp Visualization",
    use_viser: bool = False,
):
    """
    Visualize a part and end-effector in their poses.
    
    Args:
        part_name: Name of the part
        part_pose: 7D pose vector [x, y, z, qx, qy, qz, qw] of part (in AprilTag frame)
        ee_pose: 7D pose vector [x, y, z, qx, qy, qz, qw] of end-effector (in AprilTag frame)
        furniture_name: Name of the furniture
        window_title: Title for the visualization window
    """
    o3d = _ensure_open3d()
    
    # Load meshes
    part_mesh = None
    gripper_mesh = None
    
    # Load part mesh
    furniture_conf = config["furniture"].get(furniture_name, {})
    part_conf = furniture_conf.get(part_name)
    if isinstance(part_conf, dict) and "asset_file" in part_conf:
        try:
            part_mesh = _load_part_mesh(part_conf["asset_file"])
            print(f"Loaded part mesh for '{part_name}'")
        except Exception as exc:
            print(f"Warning: unable to load part mesh: {exc}")
    else:
        print(f"Warning: Part config missing asset file for '{part_name}'")
    
    # Load gripper mesh
    try:
        gripper_mesh = _load_gripper_mesh()
        print(f"Loaded gripper mesh")
    except Exception as exc:
        print(f"Warning: unable to load gripper mesh: {exc}")
    
    if part_mesh is None and gripper_mesh is None:
        raise ValueError("Could not load any meshes to visualize")
    
    # Convert poses to transformation matrices
    part_mat = T.pose2mat(part_pose)
    ee_mat = T.pose2mat(ee_pose)
    
    # Create geometries list
    geometries = []
    
    # Add part mesh
    if part_mesh is not None:
        part_mesh_transformed = _mesh_with_transform(
            part_mesh,
            part_mat,
            color=(0.9, 0.7, 0.3),  # Orange/yellow
        )
        geometries.append(part_mesh_transformed)
        
        # Add coordinate frame for part
        part_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.05
        ).transform(part_mat)
        geometries.append(part_frame)
    
    # Add gripper mesh
    if gripper_mesh is not None:
        gripper_mesh_transformed = _mesh_with_transform(
            gripper_mesh,
            ee_mat,
            color=(0.5, 0.7, 0.9),  # Blue
        )
        geometries.append(gripper_mesh_transformed)
        
        # Add coordinate frame for gripper
        gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.03
        ).transform(ee_mat)
        geometries.append(gripper_frame)
    
    if use_viser:
        from reset.visualization.viser_utils import (
            create_viser_server,
            add_mesh_to_viser_scene,
            add_frame_to_viser_scene,
            _open3d_to_viser_mesh_data,
            VISER_AVAILABLE,
        )
        if not VISER_AVAILABLE:
            raise ImportError("viser is required for viser visualization. Install with: pip install viser")
        
        server = create_viser_server(title=window_title)
        
        # Add part mesh
        if part_mesh is not None:
            part_mesh_data = _open3d_to_viser_mesh_data(part_mesh, color=(0.9, 0.7, 0.3))
            add_mesh_to_viser_scene(server, "/part", part_mesh_data, transform=part_mat)
            add_frame_to_viser_scene(server, "/part_frame", part_mat, size=0.05)
        
        # Add gripper mesh
        if gripper_mesh is not None:
            gripper_mesh_data = _open3d_to_viser_mesh_data(gripper_mesh, color=(0.5, 0.7, 0.9))
            add_mesh_to_viser_scene(server, "/gripper", gripper_mesh_data, transform=ee_mat)
            add_frame_to_viser_scene(server, "/gripper_frame", ee_mat, size=0.03)
        
        print(f"\nVisualizing grasp with viser...")
        print("Open your browser to the URL shown above to view the visualization.")
        print("Press Ctrl+C to exit.")
        
        try:
            import time
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nExiting...")
    else:
        # Visualize
        print(f"\nVisualizing grasp...")
        print("Controls:")
        print("  - Mouse: Rotate (left drag), Pan (right drag), Zoom (scroll)")
        print("  - Press 'Q' or close window to exit")
        print()
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_title,
            width=1280,
            height=720,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a specific grasp pose using get_grasp_eef_pose"
    )
    parser.add_argument(
        "--part-name",
        type=str,
        required=True,
        help="Name of the part to visualize grasp for"
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
        help="Grasp cluster mode (0 = largest/dominant cluster, 1+ = other clusters)"
    )
    parser.add_argument(
        "--part-pose",
        type=str,
        default=None,
        help="Part pose as comma-separated values [x,y,z,qx,qy,qz,qw]. If not provided, uses pose from initial state."
    )
    parser.add_argument(
        "--use-initial-state",
        action="store_true",
        help="Use part pose from initial state (requires --initial-state-index)"
    )
    parser.add_argument(
        "--initial-state-index",
        type=int,
        default=0,
        help="Index of initial state to use (default: 0)"
    )
    parser.add_argument(
        "--initial-state-path",
        type=str,
        default=None,
        help="Optional path to initial state file"
    )
    parser.add_argument(
        "--grasp-path",
        type=str,
        default=None,
        help="Optional path to grasp summary file"
    )
    parser.add_argument(
        "--use-viser",
        action="store_true",
        help="Use viser for visualization instead of Open3D"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Testing get_grasp_eef_pose function")
    print("=" * 60)
    
    # Get part pose
    if args.part_pose:
        # Parse part pose from string
        try:
            pose_values = [float(x.strip()) for x in args.part_pose.split(",")]
            if len(pose_values) != 7:
                raise ValueError("Part pose must have 7 values [x,y,z,qx,qy,qz,qw]")
            part_pose = np.array(pose_values, dtype=np.float64)
            print(f"\nUsing provided part pose: {part_pose}")
        except Exception as e:
            print(f"ERROR: Failed to parse part pose: {e}")
            return 1
    elif args.use_initial_state:
        # Load from initial state
        try:
            # Note: get_initial_state returns a single dict (randomly chosen)
            # To get a specific index, we need to load all states
            from reset.scripts.get_info import _load_initial_state
            initial_state_path = Path(args.initial_state_path) if args.initial_state_path else None
            initial_states = _load_initial_state(initial_state_path=initial_state_path)
            
            if args.initial_state_index < 0 or args.initial_state_index >= len(initial_states):
                print(f"ERROR: Initial state index {args.initial_state_index} out of range [0, {len(initial_states)})")
                return 1
            
            initial_state = initial_states[args.initial_state_index]
            parts_poses = np.asarray(initial_state["parts_poses"])
            
            # Find part index
            if args.furniture:
                furniture = furniture_factory(args.furniture)
                part_names = [part.name for part in furniture.parts]
            else:
                # Try to infer from initial state or search all furniture
                print("Warning: Furniture name not provided, trying to find part in all furniture types...")
                # For now, we'll need furniture name to get part index
                print("ERROR: Furniture name required when using initial state")
                return 1
            
            furniture = furniture_factory(args.furniture)
            part_names = [part.name for part in furniture.parts]
            
            if args.part_name not in part_names:
                print(f"ERROR: Part '{args.part_name}' not found in furniture '{args.furniture}'")
                print(f"Available parts: {part_names}")
                return 1
            
            part_idx = part_names.index(args.part_name)
            part_pose = parts_poses[part_idx]
            print(f"\nUsing part pose from initial state (index {args.initial_state_index}): {part_pose}")
        except Exception as e:
            print(f"ERROR: Failed to load part pose from initial state: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        # Use default pose at origin
        print("\nNo part pose provided, using default pose at origin")
        part_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    
    # Get end-effector pose using get_grasp_eef_pose
    print(f"\nComputing end-effector pose for part '{args.part_name}' (mode={args.mode})...")
    try:
        grasp_path = Path(args.grasp_path) if args.grasp_path else None
        ee_pose = get_grasp_eef_pose(
            part_name=args.part_name,
            pose=part_pose,
            furniture_name=args.furniture,
            mode=args.mode,
            grasp_path=grasp_path,
        )
        print(f"Computed end-effector pose: {ee_pose}")
    except Exception as e:
        print(f"ERROR: Failed to compute end-effector pose: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Determine furniture name if not provided
    if args.furniture is None:
        # Try to infer from grasp summary
        try:
            from reset.scripts.get_info import _load_grasp_summary
            summary = _load_grasp_summary(grasp_path)
            # Find furniture that contains this part
            for furn_name, parts_data in summary.items():
                if args.part_name in parts_data:
                    args.furniture = furn_name
                    print(f"Inferred furniture name: {args.furniture}")
                    break
            
            if args.furniture is None:
                print(f"ERROR: Could not find part '{args.part_name}' in any furniture type")
                return 1
        except Exception as e:
            print(f"ERROR: Could not infer furniture name: {e}")
            return 1
    
    # Visualize
    window_title = f"Grasp for {args.part_name} (mode={args.mode})"
    try:
        visualize_grasp(
            part_name=args.part_name,
            part_pose=part_pose,
            ee_pose=ee_pose,
            furniture_name=args.furniture,
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

