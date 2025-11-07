import argparse
import copy
import json
import pickle
import xml.etree.ElementTree as ET
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from tqdm import tqdm

import numpy as np

from furniture_bench.config import config
from furniture_bench.furniture import furniture_factory
import furniture_bench.utils.transform as T
from furniture_bench.utils.averageQuaternions import averageQuaternions

try:
    import trimesh  # type: ignore[import]
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    import yourdfpy  # type: ignore[import]
    YOURDFPY_AVAILABLE = True
except ImportError:
    YOURDFPY_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_ROOT = PROJECT_ROOT / "furniture_bench" / "assets"
GRIPPER_MESH_PATH = ASSETS_ROOT / "franka_description_ros" / "tools" / "gripper.stl"

_OPEN3D_MODULE = None


def _iter_annotation_files(dataset_path: Path) -> Iterable[Path]:
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
    if isinstance(array_like, np.ndarray):
        return array_like.astype(np.float32, copy=True)
    return np.array(array_like, dtype=np.float32)


def _compute_relative_pose(ee_pose: np.ndarray, part_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ee_mat = T.pose2mat(ee_pose)
    rel_mat = np.linalg.inv(ee_mat) @ part_pose
    rel_pos = rel_mat[:3, 3]
    rel_quat = T.mat2quat(rel_mat[:3, :3])
    return rel_pos, rel_quat


def _avg_quaternion(quaternions: np.ndarray) -> np.ndarray:
    if len(quaternions) == 0:
        raise ValueError("Cannot average an empty quaternion set")
    quats_wxyz = np.stack([T.convert_quat(q, "wxyz") for q in quaternions], axis=0)
    mean_wxyz = averageQuaternions(quats_wxyz)
    return T.convert_quat(mean_wxyz, "xyzw")


def _corr_block(data_a: np.ndarray, data_b: np.ndarray) -> np.ndarray:
    if len(data_a) <= 1 or len(data_b) <= 1:
        return np.full((data_a.shape[1], data_b.shape[1]), np.nan)
    stacked = np.hstack([data_a, data_b])
    corr = np.corrcoef(stacked, rowvar=False)
    return corr[: data_a.shape[1], data_a.shape[1] :]


def _prepare_thresholds(furniture_name: str, args) -> Dict[str, float]:
    max_width = config["robot"]["max_gripper_width"].get(furniture_name)
    if max_width is None:
        raise ValueError(f"Unknown furniture '{furniture_name}' for gripper width lookup")

    if args.gripper_close_threshold is not None:
        gripper_thresh = args.gripper_close_threshold
    else:
        gripper_thresh = max_width * args.gripper_close_ratio

    return {
        "gripper": gripper_thresh,
        "contact_tolerance": args.contact_tolerance,
        "min_steps": args.min_consecutive_steps,
    }


def _analyze_observations(
    furniture_name: str,
    observations: List[Dict],
    part_names: List[str],
    thresholds: Dict[str, float],
    robot_from_april: np.ndarray,
    aggregate_store: Dict[str, Dict[str, Dict[str, List[np.ndarray]]]],
    visualize_contact: bool = False,
    visualize_contact_limit: int = 10,
):
    """Analyze observations using mesh-based contact detection."""
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required for mesh-based contact detection. Install it with: pip install trimesh")
    
    num_parts = len(part_names)
    consecutive_counters = {idx: 0 for idx in range(num_parts)}
    visualization_count = 0
    
    # Load gripper mesh once
    gripper_mesh = _load_gripper_mesh_trimesh()
    
    # Load part meshes once
    furniture_conf = config["furniture"].get(furniture_name, {})
    part_meshes = {}
    for part_name in part_names:
        part_conf = furniture_conf.get(part_name)
        if not isinstance(part_conf, dict) or "asset_file" not in part_conf:
            print(f"Warning: Part config missing asset file for '{part_name}', skipping mesh-based detection")
            continue
        try:
            part_meshes[part_name] = _load_part_mesh_trimesh(part_conf["asset_file"])
        except Exception as exc:
            print(f"Warning: Unable to load mesh for part '{part_name}': {exc}, skipping mesh-based detection")
            continue
    
    if not part_meshes:
        raise ValueError(f"No part meshes could be loaded for {furniture_name}")

    for obs in tqdm(observations, desc="Processing observations"):
        robot_state = obs.get("robot_state")
        parts_poses = obs.get("parts_poses")
        if robot_state is None or parts_poses is None:
            continue

        ee_pos = _to_numpy(robot_state.get("ee_pos"))
        ee_quat = _to_numpy(robot_state.get("ee_quat"))
        gripper_width = float(np.asarray(robot_state.get("gripper_width")))
        ee_pose_vec = np.concatenate([ee_pos, ee_quat])

        parts_array = _to_numpy(parts_poses).reshape(num_parts, 7)

        # Check if gripper is closed (quick filter)
        gripper_closed = gripper_width < thresholds["gripper"]
        
        for part_idx in range(num_parts):
            part_name = part_names[part_idx]
            
            # Skip if mesh not available
            if part_name not in part_meshes:
                consecutive_counters[part_idx] = 0
                continue
            
            part_pose_vec = parts_array[part_idx]
            part_pose_mat_april = T.pose2mat(part_pose_vec)
            part_pose_mat_robot = robot_from_april @ part_pose_mat_april
            
            # Convert part pose matrix to pose vector in robot frame
            part_pos_robot, part_quat_robot = T.mat2pose(part_pose_mat_robot)
            part_pose_vec_robot = np.concatenate([part_pos_robot, part_quat_robot])
            
            # Use mesh-based contact detection
            should_visualize = visualize_contact and visualization_count < visualize_contact_limit
            
            # Extract observation images if visualizing
            observation_images = None
            if should_visualize:
                observation_images = _prepare_rgb_frames(obs)
            
            if gripper_closed:
                has_contact, min_dist = _detect_contact_mesh(
                    gripper_mesh,
                    part_meshes[part_name],
                    ee_pose_vec,
                    part_pose_vec_robot,
                    thresholds["contact_tolerance"],
                    visualize=should_visualize,
                    observation_images=observation_images,
                )
                if should_visualize and has_contact:
                    visualization_count += 1
            else:
                has_contact = False

            if has_contact:
                consecutive_counters[part_idx] += 1
            else:
                consecutive_counters[part_idx] = 0

            if consecutive_counters[part_idx] >= thresholds["min_steps"]:
                part_pos_robot = part_pose_mat_robot[:3, 3]
                rel_pos, rel_quat = _compute_relative_pose(ee_pose_vec, part_pose_mat_robot)

                part_store = aggregate_store[furniture_name][part_name]
                part_store["relative_pos"].append(rel_pos)
                part_store["relative_quat"].append(rel_quat)
                part_store["ee_pos"].append(ee_pos)
                part_store["ee_quat"].append(ee_quat)
                part_store["part_pos"].append(part_pos_robot)
                part_store["part_quat"].append(part_pose_vec[3:])


def _summarize(aggregate_store: Dict[str, Dict[str, Dict[str, List[np.ndarray]]]]):
    summary = {}
    for furniture_name, parts_data in aggregate_store.items():
        print(f"\n=== {furniture_name} ===")
        summary[furniture_name] = {}

        for part_name, samples in parts_data.items():
            if len(samples["relative_pos"]) == 0:
                continue

            rel_pos = np.vstack(samples["relative_pos"])
            rel_quat = np.vstack(samples["relative_quat"])
            ee_pos = np.vstack(samples["ee_pos"])
            ee_quat = np.vstack(samples["ee_quat"])
            part_pos = np.vstack(samples["part_pos"])
            part_quat = np.vstack(samples["part_quat"])

            rel_pos_mean = rel_pos.mean(axis=0)
            rel_pos_std = rel_pos.std(axis=0)
            rel_distance_mean = float(np.linalg.norm(rel_pos, axis=1).mean())
            rel_distance_std = float(np.linalg.norm(rel_pos, axis=1).std())

            mean_rel_quat = _avg_quaternion(rel_quat)

            pos_corr = _corr_block(ee_pos, part_pos)
            quat_corr = _corr_block(ee_quat, part_quat)

            part_pos_mean = part_pos.mean(axis=0)
            part_pos_std = part_pos.std(axis=0)
            mean_part_quat = _avg_quaternion(part_quat)

            print(
                f"- {part_name}: samples={len(rel_pos)}\n"
                f"  mean rel pos (m): {rel_pos_mean.round(4)}\n"
                f"  std rel pos (m): {rel_pos_std.round(4)}\n"
                f"  mean rel dist (m): {rel_distance_mean:.4f} (std {rel_distance_std:.4f})\n"
                f"  mean rel quat (xyzw): {np.round(mean_rel_quat, 4)}\n"
                f"  mean part pos (m): {part_pos_mean.round(4)}\n"
                f"  std part pos (m): {part_pos_std.round(4)}\n"
                f"  pos corr (ee vs part):\n{np.round(pos_corr, 3)}\n"
                f"  quat corr (ee vs part):\n{np.round(quat_corr, 3)}"
            )

            summary[furniture_name][part_name] = {
                "samples": int(len(rel_pos)),
                "mean_relative_position": rel_pos_mean.tolist(),
                "std_relative_position": rel_pos_std.tolist(),
                "mean_relative_distance": rel_distance_mean,
                "std_relative_distance": rel_distance_std,
                "mean_relative_quaternion": mean_rel_quat.tolist(),
                "position_correlation": np.nan_to_num(pos_corr).tolist(),
                "quaternion_correlation": np.nan_to_num(quat_corr).tolist(),
                "mean_part_position": part_pos_mean.tolist(),
                "std_part_position": part_pos_std.tolist(),
                "mean_part_quaternion": mean_part_quat.tolist(),
            }

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Extract grasp pose statistics from demonstrations using mesh-based contact detection")
    parser.add_argument("--dataset", required=True, help="Path to a dataset directory or a single pickle file")
    parser.add_argument("--furniture", default=None, help="Override furniture name (otherwise taken from dataset)")
    parser.add_argument("--contact-tolerance", type=float, default=0.015, help="Distance threshold (m) for mesh-based contact detection. Default: 0.015 (15mm)")
    parser.add_argument("--gripper-close-threshold", type=float, default=None, help="Absolute threshold (m) for gripper width when considered closed")
    parser.add_argument("--gripper-close-ratio", type=float, default=0.9, help="Ratio of max gripper width used when threshold is not provided")
    parser.add_argument("--min-consecutive-steps", type=int, default=1, help="Number of consecutive frames with contact before logging a grasp")
    parser.add_argument("--output", type=Path, default=None, help="Directory path to save summary.json and aggregate_store.json files")
    parser.add_argument("--render", action="store_true", help="Render sampled grasp poses based on computed statistics")
    parser.add_argument("--render-video", action="store_true", help="Render video of the demonstrations")
    parser.add_argument("--render-part", default=None, help="Name of a specific part to render (defaults to all)")
    parser.add_argument("--render-furniture", default=None, help="Furniture name to render (defaults to stats furniture)")
    parser.add_argument("--render-samples", type=int, default=5, help="Number of random samples to visualize per part")
    parser.add_argument("--render-spread-mult", type=float, default=1.0, help="Multiplier on positional std when sampling part poses")
    parser.add_argument("--visualize-contact", action="store_true", help="Visualize mesh-based contact detection (opens 3D viewer for each contact)")
    parser.add_argument("--visualize-contact-limit", type=int, default=10, help="Maximum number of contact visualizations to show (to avoid opening too many windows)")
    return parser.parse_args()


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return quat
    return quat / norm


def _pose_vec_to_mat(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    pose_vec = np.concatenate([pos, quat])
    return T.pose2mat(pose_vec)


def _ensure_open3d():
    global _OPEN3D_MODULE
    if _OPEN3D_MODULE is None:
        try:
            import open3d as o3d  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Rendering meshes requires the 'open3d' package. Install it or omit --render."
            ) from exc
        _OPEN3D_MODULE = o3d
    return _OPEN3D_MODULE


def _parse_urdf_mesh(asset_file: str) -> Tuple[Path, Optional[List[float]]]:
    urdf_path = (ASSETS_ROOT / asset_file).resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Asset URDF not found at {urdf_path}")

    tree = ET.parse(urdf_path)
    mesh_element = tree.find(".//mesh")
    if mesh_element is None:
        raise ValueError(f"No <mesh> element found in {urdf_path}")

    mesh_filename = mesh_element.get("filename")
    if mesh_filename is None:
        raise ValueError(f"Mesh filename missing in {urdf_path}")

    mesh_path = (urdf_path.parent / mesh_filename).resolve()
    scale_attr = mesh_element.get("scale")
    scale_values: Optional[List[float]] = None
    if scale_attr:
        scale_values = [float(val) for val in scale_attr.replace(",", " ").split() if val]

    return mesh_path, scale_values


def _apply_scale_trimesh(mesh: "trimesh.Trimesh", scale_values: Optional[List[float]]) -> "trimesh.Trimesh":
    """Apply scale to a trimesh mesh."""
    if not scale_values:
        return mesh
    
    if len(scale_values) == 1:
        scale_matrix = np.eye(4) * scale_values[0]
        scale_matrix[3, 3] = 1.0
    elif len(scale_values) == 3:
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = scale_values[0]
        scale_matrix[1, 1] = scale_values[1]
        scale_matrix[2, 2] = scale_values[2]
    else:
        scale_matrix = np.eye(4) * scale_values[0]
        scale_matrix[3, 3] = 1.0
    
    mesh.apply_transform(scale_matrix)
    return mesh


@lru_cache(maxsize=None)
def _load_part_mesh_trimesh(asset_file: str) -> "trimesh.Trimesh":
    """Load a part mesh as a trimesh object.
    
    Tries multiple methods:
    1. Uses yourdfpy to properly parse URDF and load mesh
    2. Falls back to using open3d loader (which works) and converts to trimesh
    3. Last resort: direct trimesh.load() on the mesh file
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required for mesh-based contact detection. Install it with: pip install trimesh")
    
    mesh_path, scale_values = _parse_urdf_mesh(asset_file)
    urdf_path = (ASSETS_ROOT / asset_file).resolve()
    
    # Method 1: Try using yourdfpy to properly parse URDF and load mesh
    if YOURDFPY_AVAILABLE and urdf_path.is_file():
        try:
            robot = yourdfpy.URDF.load(str(urdf_path))
            scene = robot.get_visual_scene()
            
            # Extract meshes from the scene
            if isinstance(scene, trimesh.Scene):
                geometries = list(scene.geometry.values())
                if geometries:
                    # Combine all geometries into a single mesh
                    mesh = trimesh.util.concatenate(geometries)
                    if isinstance(mesh, trimesh.Trimesh) and not mesh.is_empty:
                        mesh = _apply_scale_trimesh(mesh, scale_values)
                        mesh.remove_unreferenced_vertices()
                        return mesh
        except Exception as e:
            # Fall through to next method
            pass
    
    # Method 2: Use open3d loader (which works) and convert to trimesh
    try:
        o3d_mesh = _load_part_mesh(asset_file)
        if not o3d_mesh.is_empty():
            # Convert open3d mesh to trimesh
            # Note: scaling is already applied in _load_part_mesh via _apply_scale_o3d
            mesh = _open3d_to_trimesh(o3d_mesh)
            mesh.remove_unreferenced_vertices()
            return mesh
    except Exception as e:
        # Fall through to last resort
        pass
    
    # Method 3: Last resort - direct trimesh.load() on the mesh file
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh file not found at {mesh_path}")
    
    mesh = trimesh.load(str(mesh_path), force='mesh')
    if isinstance(mesh, trimesh.Scene):
        # If it's a scene, get the first mesh or concatenate all
        geometries = list(mesh.geometry.values())
        if not geometries:
            raise ValueError(f"No geometries found in {mesh_path}")
        if len(geometries) == 1:
            mesh = geometries[0]
        else:
            # Concatenate multiple geometries
            mesh = trimesh.util.concatenate(geometries)
    
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected Trimesh object, got {type(mesh)}")
    
    if mesh.is_empty:
        raise ValueError(f"Mesh at {mesh_path} is empty")
    
    mesh = _apply_scale_trimesh(mesh, scale_values)
    mesh.remove_unreferenced_vertices()
    return mesh


@lru_cache(maxsize=1)
def _load_gripper_mesh_trimesh() -> "trimesh.Trimesh":
    """Load the gripper mesh as a trimesh object."""
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required for mesh-based contact detection. Install it with: pip install trimesh")
    
    gripper_path = GRIPPER_MESH_PATH.resolve()
    if not gripper_path.is_file():
        raise FileNotFoundError(
            "Gripper mesh not found. Generate it with 'generate_panda_gripper_mesh.py' "
            f"and place it at {gripper_path}"
        )
    
    mesh = trimesh.load(str(gripper_path))
    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        if not geometries:
            raise ValueError(f"No geometries found in {gripper_path}")
        mesh = geometries[0]
    
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected Trimesh object, got {type(mesh)}")
    
    if mesh.is_empty:
        raise ValueError(f"Gripper mesh at {gripper_path} is empty")
    
    mesh.remove_unreferenced_vertices()
    return mesh


def _trimesh_to_open3d(trimesh_mesh: "trimesh.Trimesh"):
    """Convert a trimesh mesh to an open3d mesh."""
    o3d = _ensure_open3d()
    
    # Create open3d mesh from vertices and faces
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
    
    # Compute normals for better visualization
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.compute_triangle_normals()
    
    return o3d_mesh


def _open3d_to_trimesh(o3d_mesh) -> "trimesh.Trimesh":
    """Convert an open3d mesh to a trimesh mesh."""
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required for mesh conversion. Install it with: pip install trimesh")
    
    # Extract vertices and triangles from open3d mesh
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    
    # Create trimesh mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    return mesh


def _render_contact_visualization(
    gripper_mesh: "trimesh.Trimesh",
    part_mesh: "trimesh.Trimesh",
    gripper_pose: np.ndarray,
    part_pose: np.ndarray,
    contact_tolerance: float,
    min_distance: float,
    has_contact: bool,
    window_title: str = "Contact Visualization",
    observation_images: Optional[List[np.ndarray]] = None,
):
    """
    Render visualization of gripper and part meshes showing their transformed positions and contact status.
    
    Args:
        gripper_mesh: Gripper mesh in its local frame
        part_mesh: Part mesh in its local frame
        gripper_pose: 7D pose [x, y, z, qx, qy, qz, qw] of gripper in robot frame
        part_pose: 7D pose [x, y, z, qx, qy, qz, qw] of part in robot frame
        contact_tolerance: Distance threshold used for contact detection
        min_distance: Minimum distance between meshes
        has_contact: Whether contact was detected
        window_title: Title for the visualization window
        observation_images: Optional list of camera images from the observation to display before mesh visualization
    """
    o3d = _ensure_open3d()
    
    # Convert poses to transformation matrices
    gripper_T = T.pose2mat(gripper_pose)
    part_T = T.pose2mat(part_pose)
    
    # Transform meshes to world coordinates
    gripper_transformed = gripper_mesh.copy()
    gripper_transformed.apply_transform(gripper_T)
    
    part_transformed = part_mesh.copy()
    part_transformed.apply_transform(part_T)
    
    # Convert to open3d meshes
    gripper_o3d = _trimesh_to_open3d(gripper_transformed)
    part_o3d = _trimesh_to_open3d(part_transformed)
    
    # Color meshes: gripper in blue, part in orange/yellow
    # If contact detected, make part red
    gripper_color = np.array([0.3, 0.5, 0.9])  # Blue
    if has_contact:
        part_color = np.array([0.9, 0.2, 0.2])  # Red when in contact
    else:
        part_color = np.array([0.9, 0.7, 0.3])  # Orange/yellow otherwise
    
    gripper_o3d.paint_uniform_color(gripper_color)
    part_o3d.paint_uniform_color(part_color)
    
    # Create visualization geometries
    geometries = [gripper_o3d, part_o3d]
    
    # Add coordinate frames at mesh origins
    gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.05, origin=gripper_T[:3, 3]
    )
    part_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.05, origin=part_T[:3, 3]
    )
    geometries.extend([gripper_frame, part_frame])
    
    # Add wireframe bounding boxes to better see overlap
    gripper_bbox = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        gripper_o3d.get_axis_aligned_bounding_box()
    )
    gripper_bbox.paint_uniform_color([0.2, 0.4, 0.8])  # Darker blue
    geometries.append(gripper_bbox)
    
    part_bbox = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        part_o3d.get_axis_aligned_bounding_box()
    )
    part_bbox.paint_uniform_color([0.8, 0.6, 0.2])  # Darker orange
    geometries.append(part_bbox)
    
    # Create text label (as a small sphere with text annotation)
    # Note: open3d doesn't have built-in text rendering, so we'll add info to title
    info_text = (
        f"Contact: {'YES' if has_contact else 'NO'} | "
        f"Distance: {min_distance*1000:.2f}mm | "
        f"Tolerance: {contact_tolerance*1000:.2f}mm"
    )
    
    print(f"\n{window_title}")
    print(f"  {info_text}")
    
    # Show observation images first if available
    if observation_images:
        try:
            import cv2  # type: ignore[import]
            
            print(f"\nDisplaying observation images...")
            print("  Press any key in the image window to continue to mesh visualization")
            
            # Stack images horizontally
            if len(observation_images) > 0:
                # Ensure images are in the right format (H, W, C) and uint8
                processed_images = []
                for img in observation_images:
                    img = np.asarray(img)
                    # Handle channel-first format
                    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
                        img = np.moveaxis(img, 0, -1)
                    # Normalize if needed
                    if img.dtype != np.uint8:
                        if img.max() <= 1.0:
                            img = (img * 255.0).astype(np.uint8)
                        else:
                            img = np.clip(img, 0, 255).astype(np.uint8)
                    processed_images.append(img)
                
                # Stack images horizontally
                if len(processed_images) == 1:
                    stacked_img = processed_images[0]
                else:
                    stacked_img = np.hstack(processed_images)
                
                # Add text overlay with contact info
                display_img = stacked_img.copy()
                text_y = 30
                cv2.putText(
                    display_img,
                    f"Contact: {'YES' if has_contact else 'NO'} | Distance: {min_distance*1000:.2f}mm",
                    (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0) if has_contact else (0, 0, 255),
                    2,
                )
                text_y += 30
                cv2.putText(
                    display_img,
                    f"Tolerance: {contact_tolerance*1000:.2f}mm | Press any key to continue",
                    (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                
                # Display image
                cv2.imshow("Observation Images", display_img)
                cv2.waitKey(0)  # Wait for key press
                cv2.destroyWindow("Observation Images")
        except ImportError:
            print("  Warning: cv2 not available, skipping image display")
        except Exception as e:
            print(f"  Warning: Failed to display images: {e}")
    
    # Visualize meshes
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"{window_title} - {info_text}",
        width=1280,
        height=720,
    )


def _detect_contact_mesh(
    gripper_mesh: "trimesh.Trimesh",
    part_mesh: "trimesh.Trimesh",
    gripper_pose: np.ndarray,
    part_pose: np.ndarray,
    contact_tolerance: float,
    visualize: bool = False,
    observation_images: Optional[List[np.ndarray]] = None,
) -> Tuple[bool, float]:
    """
    Detect contact between gripper and part using mesh-based collision detection.
    
    Args:
        gripper_mesh: Gripper mesh in its local frame
        part_mesh: Part mesh in its local frame
        gripper_pose: 7D pose [x, y, z, qx, qy, qz, qw] of gripper in robot frame
        part_pose: 7D pose [x, y, z, qx, qy, qz, qw] of part in robot frame
        contact_tolerance: Distance threshold (meters) for considering contact
        visualize: If True, render visualization of the meshes
    
    Returns:
        Tuple of (has_contact, min_distance)
    """
    # Convert poses to transformation matrices
    gripper_T = T.pose2mat(gripper_pose)
    part_T = T.pose2mat(part_pose)
    
    # Transform meshes to world coordinates
    gripper_transformed = gripper_mesh.copy()
    gripper_transformed.apply_transform(gripper_T)
    
    part_transformed = part_mesh.copy()
    part_transformed.apply_transform(part_T)
    
    # Create collision manager for efficient distance computation
    manager = trimesh.collision.CollisionManager()
    manager.add_object("gripper", gripper_transformed)
    
    # Compute minimum distance between meshes
    min_distance = manager.min_distance_single(part_transformed)
    
    has_contact = min_distance < contact_tolerance
    
    # Visualize if requested
    if visualize:
        _render_contact_visualization(
            gripper_mesh,
            part_mesh,
            gripper_pose,
            part_pose,
            contact_tolerance,
            float(min_distance),
            has_contact,
            window_title="Mesh Contact Detection",
            observation_images=observation_images,
        )
    
    return has_contact, float(min_distance)


def _apply_scale_o3d(mesh, scale_values: Optional[List[float]]):
    if not scale_values:
        return

    if len(scale_values) == 1:
        mesh.scale(scale_values[0], center=(0.0, 0.0, 0.0))
        return

    transform = np.eye(4)
    if len(scale_values) == 3:
        transform[0, 0], transform[1, 1], transform[2, 2] = scale_values
    else:
        transform[0, 0] = transform[1, 1] = transform[2, 2] = scale_values[0]
    mesh.transform(transform)


@lru_cache(maxsize=None)
def _load_part_mesh(asset_file: str):
    o3d = _ensure_open3d()
    mesh_path, scale_values = _parse_urdf_mesh(asset_file)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh file not found at {mesh_path}")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise ValueError(f"Mesh at {mesh_path} is empty")

    _apply_scale_o3d(mesh, scale_values)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


@lru_cache(maxsize=1)
def _load_gripper_mesh():
    o3d = _ensure_open3d()
    gripper_path = GRIPPER_MESH_PATH.resolve()
    if not gripper_path.is_file():
        raise FileNotFoundError(
            "Gripper mesh not found. Generate it with 'generate_panda_gripper_mesh.py' "
            f"and place it at {gripper_path}"
        )

    mesh = o3d.io.read_triangle_mesh(str(gripper_path))
    if mesh.is_empty():
        raise ValueError(f"Gripper mesh at {gripper_path} is empty")

    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def _mesh_with_transform(base_mesh, transform: np.ndarray, color: Optional[Tuple[float, float, float]] = None):
    mesh = copy.deepcopy(base_mesh)
    mesh.transform(transform)
    if color is not None:
        mesh.paint_uniform_color(color)
    return mesh


def _render_from_summary(
    summary: Dict[str, Dict[str, Dict[str, List[float]]]],
    furniture_filter: Optional[str],
    part_filter: Optional[str],
    samples: int,
    spread_mult: float,
):
    if furniture_filter is not None and furniture_filter not in summary:
        raise ValueError(f"Requested furniture '{furniture_filter}' not found in summary")

    _ = (samples, spread_mult)  # kept for CLI compatibility

    furnitures = [furniture_filter] if furniture_filter else summary.keys()

    for furniture_name in furnitures:
        parts_stats = summary[furniture_name]
        if not parts_stats:
            print(f"No stats available for {furniture_name}, skipping render")
            continue

        parts = (
            [part_filter]
            if part_filter
            else sorted(parts_stats.keys())
        )

        o3d = _ensure_open3d()
        geometries = []
        display_idx = 0

        gripper_mesh = None
        try:
            gripper_mesh = _load_gripper_mesh()
        except Exception as exc:
            print(f"Warning: unable to load gripper mesh: {exc}")

        furniture_conf = config["furniture"].get(furniture_name, {})

        for part_name in parts:
            if part_name not in parts_stats:
                print(f"Part '{part_name}' not in summary for {furniture_name}, skipping")
                continue

            stats = parts_stats[part_name]
            mean_rel_pos = np.array(stats["mean_relative_position"], dtype=np.float64)
            mean_rel_quat = _quat_normalize(np.array(stats["mean_relative_quaternion"], dtype=np.float64))

            rel_T = _pose_vec_to_mat(mean_rel_pos, mean_rel_quat)
            rel_T_inv = np.linalg.inv(rel_T)

            part_conf = furniture_conf.get(part_name)
            if not isinstance(part_conf, dict) or "asset_file" not in part_conf:
                print(f"Part config missing asset file for '{part_name}', skipping mesh rendering")
                continue

            try:
                part_mesh = _load_part_mesh(part_conf["asset_file"])
            except Exception as exc:
                print(f"Warning: unable to load mesh for part '{part_name}': {exc}")
                part_mesh = None

            # Use the original/oriented part pose (identity transform)
            part_T = np.eye(4)
            offset = np.array([0.5 * display_idx, 0.0, 0.0])
            part_T_display = part_T.copy()
            part_T_display[:3, 3] = offset

            ee_T_display = part_T_display @ rel_T_inv

            if part_mesh is not None:
                geometries.append(
                    _mesh_with_transform(
                        part_mesh,
                        part_T_display,
                        color=(0.9, 0.7, 0.3),
                    )
                )

            if gripper_mesh is not None:
                geometries.append(
                    _mesh_with_transform(
                        gripper_mesh,
                        ee_T_display,
                        color=(0.5, 0.7, 0.9),
                    )
                )

            geometries.append(
                o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05).translate(offset)
            )

            display_idx += 1

        if not geometries:
            print(f"No geometries to render for {furniture_name}")
            continue

        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Sampled grasps for {furniture_name}",
        )


def _detect_grasp_labels(
    observations: List[Dict],
    part_names: List[str],
    thresholds: Dict[str, float],
    robot_from_april: np.ndarray,
    furniture_name: str,
) -> List[List[str]]:
    """Detect grasp labels using mesh-based contact detection."""
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required for mesh-based contact detection. Install it with: pip install trimesh")
    
    num_parts = len(part_names)
    labels: List[List[str]] = [[] for _ in range(len(observations))]
    consecutive_counters = {idx: 0 for idx in range(num_parts)}
    
    # Load gripper mesh once
    gripper_mesh = _load_gripper_mesh_trimesh()
    
    # Load part meshes once
    furniture_conf = config["furniture"].get(furniture_name, {})
    part_meshes = {}
    for part_name in part_names:
        part_conf = furniture_conf.get(part_name)
        if not isinstance(part_conf, dict) or "asset_file" not in part_conf:
            continue
        try:
            part_meshes[part_name] = _load_part_mesh_trimesh(part_conf["asset_file"])
        except Exception:
            continue

    for step_idx, obs in enumerate(tqdm(observations, desc="Processing observations")):
        robot_state = obs.get("robot_state")
        parts_poses = obs.get("parts_poses")
        if robot_state is None or parts_poses is None:
            continue

        ee_pos = _to_numpy(robot_state.get("ee_pos"))
        ee_quat = _to_numpy(robot_state.get("ee_quat"))
        gripper_width = float(np.asarray(robot_state.get("gripper_width")))
        ee_pose_vec = np.concatenate([ee_pos, ee_quat])

        parts_array = _to_numpy(parts_poses).reshape(num_parts, 7)

        # Check if gripper is closed (quick filter)
        gripper_closed = gripper_width < thresholds["gripper"]

        for part_idx in range(num_parts):
            part_name = part_names[part_idx]
            
            # Skip if mesh not available
            if part_name not in part_meshes:
                consecutive_counters[part_idx] = 0
                continue
            
            part_pose_vec = parts_array[part_idx]
            part_pose_mat_april = T.pose2mat(part_pose_vec)
            part_pose_mat_robot = robot_from_april @ part_pose_mat_april
            
            # Convert part pose matrix to pose vector in robot frame
            part_pos_robot, part_quat_robot = T.mat2pose(part_pose_mat_robot)
            part_pose_vec_robot = np.concatenate([part_pos_robot, part_quat_robot])
            
            # Use mesh-based contact detection
            if gripper_closed:
                has_contact, _ = _detect_contact_mesh(
                    gripper_mesh,
                    part_meshes[part_name],
                    ee_pose_vec,
                    part_pose_vec_robot,
                    thresholds["contact_tolerance"],
                    visualize=False,
                    observation_images=[obs["color_image1"], obs["color_image2"]],
                )
            else:
                has_contact = False

            if has_contact:
                consecutive_counters[part_idx] += 1
            else:
                consecutive_counters[part_idx] = 0

            if consecutive_counters[part_idx] >= thresholds["min_steps"]:
                if part_name not in labels[step_idx]:
                    labels[step_idx].append(part_name)

    return labels


def _prepare_rgb_frames(obs: Dict) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for key in [
        "color_image1",
        "color_image2",
        "color_image3",
        "image1",
        "image2",
        "image3",
    ]:
        if key not in obs:
            continue
        img = np.asarray(obs[key])
        if img.ndim != 3:
            continue
        if img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
            img = np.moveaxis(img, 0, -1)
        if img.shape[-1] not in (3, 4):
            continue
        if img.dtype != np.uint8:
            img = img.astype(np.float32)
            if img.max() <= 1.0:
                img *= 255.0
            img = np.clip(img, 0.0, 255.0).astype(np.uint8)
        frames.append(img)
    return frames


def _render_demonstration_video(
    observations: List[Dict],
    grasp_labels: List[List[str]],
    window_title: str,
    play_speed_hz: float = 30.0,
):
    import cv2  # type: ignore[import]

    wait_ms = max(1, int(1000.0 / max(play_speed_hz, 1e-3)))

    for step_idx, obs in enumerate(observations):
        frames = _prepare_rgb_frames(obs)
        if not frames:
            continue

        bgr_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
        frame_bgr = np.hstack(bgr_frames)

        label_parts = grasp_labels[step_idx]
        label_text = ", ".join(label_parts) if label_parts else "none"
        display = np.ascontiguousarray(frame_bgr)
        cv2.putText(
            display,
            f"grasp: {label_text}",
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 255, 0),
            thickness=2,
        )
        cv2.putText(
            display,
            f"frame: {step_idx}",
            org=(10, 60),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 255, 255),
            thickness=2,
        )

        cv2.imshow(window_title, display)
        key = cv2.waitKey(wait_ms)
        if key in (27, ord("q")):
            break

    cv2.destroyWindow(window_title)


def main():
    args = parse_args()
    dataset_root = Path(args.dataset).expanduser().resolve()
    annotation_files = list(_iter_annotation_files(dataset_root))

    if not annotation_files:
        raise FileNotFoundError(f"No .pkl files found under {dataset_root}")

    def _make_part_store():
        return {
            "relative_pos": [],
            "relative_quat": [],
            "ee_pos": [],
            "ee_quat": [],
            "part_pos": [],
            "part_quat": [],
        }

    aggregate_store: Dict[str, Dict[str, Dict[str, List[np.ndarray]]]] = defaultdict(
        lambda: defaultdict(_make_part_store)
    )

    for idx, annotation_path in enumerate(annotation_files):
        print(f"Processing {idx+1} / {len(annotation_files)}: {annotation_path}")
        with open(annotation_path, "rb") as f:
            data = pickle.load(f)

        furniture_name = args.furniture or data.get("furniture")
        if furniture_name is None:
            raise ValueError(
                f"Furniture name not provided via --furniture or dataset metadata for {annotation_path}"
            )

        furniture = furniture_factory(furniture_name)
        part_names = [part.name for part in furniture.parts]
        thresholds = _prepare_thresholds(furniture_name, args)
        # Environment defines: april_to_robot_mat = base_tag_from_robot_mat (line 643-644)
        # This transforms FROM april TO robot: april_to_robot_mat @ pose_april = pose_robot
        # So robot_from_april should be base_tag_from_robot_mat directly
        robot_from_april = config["robot"]["tag_base_from_robot_base"]

        observations = data.get("observations", [])

        if args.render_video and observations:
            grasp_labels = _detect_grasp_labels(
                observations,
                part_names,
                thresholds,
                robot_from_april,
                furniture_name,
            )
            window_title = f"{furniture_name} - {annotation_path.stem}" if hasattr(annotation_path, "stem") else str(annotation_path)
            _render_demonstration_video(
                observations,
                grasp_labels,
                window_title=window_title,
            )

        _analyze_observations(
            furniture_name,
            observations,
            part_names,
            thresholds,
            robot_from_april,
            aggregate_store,
            visualize_contact=args.visualize_contact,
            visualize_contact_limit=args.visualize_contact_limit,
        )

    summary = _summarize(aggregate_store)

    if args.output is not None:
        # Ensure output is a directory
        output_dir = args.output.expanduser().resolve()
        if output_dir.is_file():
            raise ValueError(f"Output path must be a directory, not a file: {output_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary.json
        summary_path = output_dir / "grasp_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to {summary_path}")
        
        # Convert aggregate_store to JSON-serializable format
        def convert_aggregate_store(store):
            """Convert numpy arrays in aggregate_store to lists for JSON serialization."""
            json_store = {}
            for furniture_name, parts_data in store.items():
                json_store[furniture_name] = {}
                for part_name, samples in parts_data.items():
                    json_store[furniture_name][part_name] = {
                        "relative_pos": [arr.tolist() for arr in samples["relative_pos"]],
                        "relative_quat": [arr.tolist() for arr in samples["relative_quat"]],
                        "ee_pos": [arr.tolist() for arr in samples["ee_pos"]],
                        "ee_quat": [arr.tolist() for arr in samples["ee_quat"]],
                        "part_pos": [arr.tolist() for arr in samples["part_pos"]],
                        "part_quat": [arr.tolist() for arr in samples["part_quat"]],
                    }
            return json_store
        
        # Save aggregate_store.json
        aggregate_store_path = output_dir / "grasps.json"
        json_aggregate_store = convert_aggregate_store(aggregate_store)
        with open(aggregate_store_path, "w", encoding="utf-8") as f:
            json.dump(json_aggregate_store, f, indent=2)
        print(f"Saved aggregate_store to {aggregate_store_path}")

    if args.render:
        render_furniture = args.render_furniture or args.furniture
        _render_from_summary(
            summary,
            render_furniture,
            args.render_part,
            samples=args.render_samples,
            spread_mult=args.render_spread_mult,
        )


if __name__ == "__main__":
    main()