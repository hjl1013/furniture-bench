#!/usr/bin/env python3
"""Visualize object positions, robot origin frame, and EEF pose for each observation frame."""

import argparse
import copy
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from furniture_bench.config import config
from furniture_bench.furniture import furniture_factory
import furniture_bench.utils.transform as T
from reset.extract_from_demo.extract_grasp import (
    _iter_annotation_files,
    _to_numpy,
    _load_part_mesh,
    _load_gripper_mesh,
    _prepare_rgb_frames,
    _ensure_open3d,
    ASSETS_ROOT,
)


def visualize_frame(
    obs: Dict,
    furniture_name: str,
    part_names: List[str],
    robot_from_april: np.ndarray,
    frame_idx: int,
    show_images: bool = True,
    show_gripper: bool = True,
):
    """Visualize a single observation frame with part meshes, robot origin, and EEF pose."""
    o3d = _ensure_open3d()
    
    robot_state = obs.get("robot_state")
    parts_poses = obs.get("parts_poses")
    
    if robot_state is None or parts_poses is None:
        print(f"Warning: Frame {frame_idx} missing robot_state or parts_poses")
        return
    
    # Get EEF pose (already in robot frame)
    ee_pos = _to_numpy(robot_state.get("ee_pos"))
    ee_quat = _to_numpy(robot_state.get("ee_quat"))
    ee_pose_vec = np.concatenate([ee_pos, ee_quat])
    ee_T = T.pose2mat(ee_pose_vec)
    
    # Get part poses and convert from AprilTag frame to robot frame
    num_parts = len(part_names)
    parts_array = _to_numpy(parts_poses).reshape(num_parts, 7)
    
    # Load furniture config
    furniture_conf = config["furniture"].get(furniture_name, {})
    
    # Collect geometries for visualization
    geometries = []
    
    # 1. Robot origin frame (at identity - origin)
    robot_origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0]
    )
    geometries.append(robot_origin_frame)
    
    # 2. AprilTag base frame (in robot frame)
    # tag_base_from_robot_base transforms from robot frame to AprilTag frame
    # So to visualize AprilTag frame in robot frame, we need robot_from_april (the inverse)
    # But actually, tag_base_from_robot_base gives us the pose of AprilTag frame in robot frame
    # The origin of AprilTag in robot frame is tag_base_from_robot_base[:3, 3]
    # However, the orientation needs to be inverted since the matrix transforms FROM robot TO AprilTag
    # Actually, let's use robot_from_april which we already have
    april_tag_origin_robot = robot_from_april[:3, 3]
    april_tag_rotation_robot = robot_from_april[:3, :3]
    
    # Create coordinate frame at origin first, then transform it
    april_tag_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0]
    )
    # Transform the frame to AprilTag position and orientation in robot frame
    april_tag_frame.transform(robot_from_april)
    # Color AprilTag frame differently (yellow)
    april_tag_frame.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow
    geometries.append(april_tag_frame)
    
    # 3. EEF pose frame
    eef_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.08, origin=ee_T[:3, 3]
    )
    # Color EEF frame differently (green)
    eef_frame.paint_uniform_color([0.0, 1.0, 0.0])
    geometries.append(eef_frame)
    
    # 4. Load and visualize gripper mesh at EEF pose
    if show_gripper:
        try:
            gripper_mesh = _load_gripper_mesh()
            gripper_transformed = copy.deepcopy(gripper_mesh)
            gripper_transformed.transform(ee_T)
            gripper_transformed.paint_uniform_color([0.3, 0.5, 0.9])  # Blue
            geometries.append(gripper_transformed)
        except Exception as e:
            print(f"Warning: Could not load gripper mesh: {e}")
    
    # 5. Load and visualize part meshes at their poses
    part_colors = [
        [0.9, 0.7, 0.3],  # Orange
        [0.9, 0.3, 0.7],  # Pink
        [0.3, 0.9, 0.7],  # Cyan
        [0.7, 0.3, 0.9],  # Purple
        [0.7, 0.9, 0.3],  # Yellow-green
        [0.3, 0.7, 0.9],  # Light blue
    ]
    
    for part_idx, part_name in enumerate(part_names):
        part_pose_vec = parts_array[part_idx]
        part_pose_mat_april = T.pose2mat(part_pose_vec)
        part_pose_mat_robot = robot_from_april @ part_pose_mat_april
        
        # Load part mesh
        part_conf = furniture_conf.get(part_name)
        if not isinstance(part_conf, dict) or "asset_file" not in part_conf:
            print(f"Warning: Part '{part_name}' missing asset file, skipping")
            continue
        
        try:
            part_mesh = _load_part_mesh(part_conf["asset_file"])
        except Exception as e:
            print(f"Warning: Could not load mesh for part '{part_name}': {e}")
            continue
        
        # Transform part mesh to robot frame
        part_transformed = copy.deepcopy(part_mesh)
        part_transformed.transform(part_pose_mat_robot)
        
        # Color part mesh
        color = part_colors[part_idx % len(part_colors)]
        part_transformed.paint_uniform_color(color)
        geometries.append(part_transformed)
        
        # Add coordinate frame for part
        part_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.06, origin=part_pose_mat_robot[:3, 3]
        )
        # Color frame same as mesh
        part_frame.paint_uniform_color(color)
        geometries.append(part_frame)
    
    # Show observation images first if available
    if show_images:
        try:
            import cv2  # type: ignore[import]
            
            frames = _prepare_rgb_frames(obs)
            if frames:
                print(f"\nFrame {frame_idx} - Observation Images")
                print("  Press any key in the image window to continue to 3D visualization")
                
                # Stack images horizontally
                if len(frames) == 1:
                    stacked_img = frames[0]
                else:
                    stacked_img = np.hstack(frames)
                
                # Convert RGB to BGR for OpenCV
                display_img = cv2.cvtColor(stacked_img, cv2.COLOR_RGB2BGR)
                
                # Add text overlay
                cv2.putText(
                    display_img,
                    f"Frame: {frame_idx} | EEF: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                
                cv2.imshow("Observation Images", display_img)
                cv2.waitKey(0)
                cv2.destroyWindow("Observation Images")
        except ImportError:
            print("Warning: cv2 not available, skipping image display")
        except Exception as e:
            print(f"Warning: Failed to display images: {e}")
    
    # Print frame information
    # Get AprilTag frame info for printing (in robot frame)
    april_tag_origin = robot_from_april[:3, 3]
    april_tag_quat = T.mat2quat(robot_from_april[:3, :3])
    
    print(f"\n{'='*60}")
    print(f"Frame {frame_idx} - 3D Visualization")
    print(f"{'='*60}")
    print(f"Robot Origin Frame: [0, 0, 0] (RED - X, GREEN - Y, BLUE - Z)")
    print(f"AprilTag Base Frame:")
    print(f"  Position: [{april_tag_origin[0]:.4f}, {april_tag_origin[1]:.4f}, {april_tag_origin[2]:.4f}]")
    print(f"  Quaternion: [{april_tag_quat[0]:.4f}, {april_tag_quat[1]:.4f}, {april_tag_quat[2]:.4f}, {april_tag_quat[3]:.4f}]")
    print(f"  (YELLOW coordinate frame)")
    print(f"EEF Position: [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")
    print(f"EEF Quaternion: [{ee_quat[0]:.4f}, {ee_quat[1]:.4f}, {ee_quat[2]:.4f}, {ee_quat[3]:.4f}]")
    print(f"  (GREEN coordinate frame)")
    print(f"\nPart Poses (in Robot Frame):")
    for part_idx, part_name in enumerate(part_names):
        part_pose_vec = parts_array[part_idx]
        part_pose_mat_april = T.pose2mat(part_pose_vec)
        part_pose_mat_robot = robot_from_april @ part_pose_mat_april
        part_pos = part_pose_mat_robot[:3, 3]
        part_quat = T.mat2quat(part_pose_mat_robot[:3, :3])
        print(f"  {part_name}:")
        print(f"    Position: [{part_pos[0]:.4f}, {part_pos[1]:.4f}, {part_pos[2]:.4f}]")
        print(f"    Quaternion: [{part_quat[0]:.4f}, {part_quat[1]:.4f}, {part_quat[2]:.4f}, {part_quat[3]:.4f}]")
    
    print(f"\nVisualization Controls:")
    print(f"  - Mouse: Rotate (left drag), Pan (right drag), Zoom (scroll)")
    print(f"  - Press 'Q' or close window to continue to next frame")
    
    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Frame {frame_idx} - {furniture_name}",
        width=1280,
        height=720,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize object positions, robot origin frame, and EEF pose for each observation frame"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a dataset directory or a single pickle file"
    )
    parser.add_argument(
        "--furniture",
        default=None,
        help="Override furniture name (otherwise taken from dataset)"
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Start frame index (default: 0)"
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="End frame index (default: all frames)"
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Step size between frames (default: 1)"
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Don't show observation images, only 3D visualization"
    )
    parser.add_argument(
        "--no-gripper",
        action="store_true",
        help="Don't show gripper mesh, only coordinate frame"
    )
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset).expanduser().resolve()
    annotation_files = list(_iter_annotation_files(dataset_root))
    
    if not annotation_files:
        print(f"ERROR: No .pkl files found under {dataset_root}")
        sys.exit(1)
    
    # Load first annotation file
    annotation_path = annotation_files[0]
    print(f"Loading dataset from: {annotation_path}")
    
    with open(annotation_path, "rb") as f:
        data = pickle.load(f)
    
    furniture_name = args.furniture or data.get("furniture")
    if furniture_name is None:
        print("ERROR: Furniture name not provided via --furniture or dataset metadata")
        sys.exit(1)
    
    print(f"Furniture: {furniture_name}")
    
    furniture = furniture_factory(furniture_name)
    part_names = [part.name for part in furniture.parts]
    print(f"Parts: {part_names}")
    
    # Get transformation from AprilTag base to robot base
    robot_from_april = np.linalg.inv(config["robot"]["tag_base_from_robot_base"])
    print(f"\nRobot base transformation:")
    print(f"  tag_base_from_robot_base:\n{config['robot']['tag_base_from_robot_base']}")
    print(f"  robot_from_april (inverse):\n{robot_from_april}")
    
    observations = data.get("observations", [])
    if not observations:
        print("ERROR: No observations found in dataset")
        sys.exit(1)
    
    print(f"\nTotal observations: {len(observations)}")
    
    # Determine frame range
    start_frame = args.start_frame
    end_frame = args.end_frame if args.end_frame is not None else len(observations)
    end_frame = min(end_frame, len(observations))
    frame_step = args.frame_step
    
    print(f"Visualizing frames {start_frame} to {end_frame-1} (step: {frame_step})")
    print(f"Press Ctrl+C to stop\n")
    
    try:
        for frame_idx in range(start_frame, end_frame, frame_step):
            if frame_idx >= len(observations):
                break
            
            obs = observations[frame_idx]
            visualize_frame(
                obs,
                furniture_name,
                part_names,
                robot_from_april,
                frame_idx,
                show_images=not args.no_images,
                show_gripper=not args.no_gripper,
            )
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()