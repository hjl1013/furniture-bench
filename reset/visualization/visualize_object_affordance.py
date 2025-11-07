"""
Interactive 3D visualization for object affordance trajectories.

Given the JSON output from `extract_object_affordance.py`, this script visualizes a single trajectory for a pair of parts.
- The first part name is treated as the fixed/base frame (placed at the origin).
- The second part is animated using the saved relative poses (pose of target in base frame).
- If the saved pair order is reversed, the script inverts the saved poses before animating.

The visualization uses Open3D and is interactive: while the animation plays, you can rotate/zoom/pan the view.

Usage example:
python reset/visualization/visualize_object_affordance.py \
  --json out/object_affordance_trajectories.json \
  --furniture my_furniture \
  --pair base_part target_part \
  --sample 0 --fps 30
"""
from pathlib import Path
import argparse
import json
import time
from typing import List, Tuple

import numpy as np

# Try to import project helpers from extract_grasp
try:
    from reset.extract_from_demo.extract_grasp import _trimesh_to_open3d, _ensure_open3d, _load_part_mesh_trimesh
    from reset.extract_from_demo.extract_grasp import TRIMESH_AVAILABLE
except Exception:
    try:
        from reset.extract_from_demo.extract_grasp import _trimesh_to_open3d, _ensure_open3d, _load_part_mesh_trimesh
        from reset.extract_from_demo.extract_grasp import TRIMESH_AVAILABLE
    except Exception:
        _trimesh_to_open3d = None
        _ensure_open3d = None
        _load_part_mesh_trimesh = None
        TRIMESH_AVAILABLE = False

from furniture_bench.config import config
from furniture_bench.furniture import furniture_factory
from furniture_bench.utils import transform as T


def _invert_pose_vec(pose: List[float]) -> List[float]:
    arr = np.asarray(pose, dtype=np.float32)
    mat = T.pose2mat(arr)
    inv = np.linalg.inv(mat)
    pos, quat = T.mat2pose(inv)
    return np.concatenate([pos, quat]).tolist()


def _apply_transform_to_vertices(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    # vertices: (N,3), transform: 4x4
    homo = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)], axis=1)
    transformed = (transform @ homo.T).T[:, :3]
    return transformed


def visualize_trajectory(
    json_path: Path,
    furniture_name: str,
    base_part: str,
    target_part: str,
    sample_idx: int = 0,
    fps: int = 30,
    show_trace: bool = True,
):
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if furniture_name not in data:
        raise KeyError(f"Furniture '{furniture_name}' not found in JSON. Available: {list(data.keys())}")

    furniture_data = data[furniture_name]
    pair_key_forward = f"{base_part}__{target_part}"
    pair_key_backward = f"{target_part}__{base_part}"

    reversed_order = False
    if pair_key_forward in furniture_data:
        trajectories = furniture_data[pair_key_forward]
        reversed_order = False
    elif pair_key_backward in furniture_data:
        trajectories = furniture_data[pair_key_backward]
        reversed_order = True
    else:
        raise KeyError(f"Pair not found for furniture '{furniture_name}'. Available pairs: {list(furniture_data.keys())}")

    if not trajectories:
        raise ValueError(f"No trajectories recorded for pair '{base_part}', '{target_part}'")

    if sample_idx < 0 or sample_idx >= len(trajectories):
        raise IndexError(f"sample_idx out of range (0..{len(trajectories)-1})")

    traj = trajectories[sample_idx]

    # If recorded order is reversed (stored poses are pose of base in target frame), invert each pose
    if reversed_order:
        traj = [_invert_pose_vec(p) for p in traj]

    # Convert to numpy array (N,7)
    traj_arr = np.asarray(traj, dtype=np.float64)

    # Load meshes for visualization
    furniture_conf = config["furniture"].get(furniture_name, {})
    if not isinstance(furniture_conf, dict):
        raise KeyError(f"No furniture config for {furniture_name}")

    base_conf = furniture_conf.get(base_part)
    target_conf = furniture_conf.get(target_part)
    if base_conf is None or "asset_file" not in base_conf:
        raise KeyError(f"Base part '{base_part}' has no asset_file in config")
    if target_conf is None or "asset_file" not in target_conf:
        raise KeyError(f"Target part '{target_part}' has no asset_file in config")

    if not TRIMESH_AVAILABLE or _load_part_mesh_trimesh is None or _trimesh_to_open3d is None or _ensure_open3d is None:
        raise ImportError("Visualization requires the project's extract_grasp helpers and Open3D/trimesh. Ensure dependencies are installed.")

    o3d = _ensure_open3d()

    # Load trimesh meshes - these are in each part's local coordinate frame
    mesh_base_tm = _load_part_mesh_trimesh(base_conf["asset_file"])  # trimesh.Trimesh
    mesh_target_tm = _load_part_mesh_trimesh(target_conf["asset_file"])  # trimesh.Trimesh

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
    mesh_base_o3d.paint_uniform_color([0.8, 0.7, 0.3])  # orange
    mesh_target_o3d.paint_uniform_color([0.2, 0.5, 0.9])  # blue

    # Ensure base part is positioned at origin in its own coordinate frame
    # Base part defines the coordinate frame, so it stays fixed
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

    # Optionally set a nicer camera view (leave interactive)
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True

    # Animation loop
    frame_dt = 1.0 / max(1, fps)
    prev_time = time.time()

    # For each frame: apply the relative pose transform directly to original target vertices
    # The relative pose (rel_pose) represents the pose of target part in base part's coordinate frame
    # We transform target vertices from target's local frame to base frame using this relative pose
    # IMPORTANT: Always use target_vertices_orig (original vertices in target's local frame) 
    #            and apply rel_mat transform - never accumulate transforms
    points_accum = []
    for idx in range(traj_arr.shape[0]):
        # Get relative pose: pose of target part in base part's coordinate frame
        rel_pose = traj_arr[idx]
        
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
            # rebuild line set
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

        # maintain playback speed but keep UI responsive
        elapsed = time.time() - prev_time
        sleep = max(0.0, frame_dt - elapsed)
        time.sleep(sleep)
        prev_time = time.time()

    print("Animation finished — window remains open for inspection. Close it to exit.")
    # keep window open until closed by user
    while True:
        try:
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.02)
        except KeyboardInterrupt:
            break


def parse_args():
    p = argparse.ArgumentParser(description="Visualize object affordance trajectory (interactive 3D)")
    p.add_argument("--json", required=True, type=Path, help="Path to object_affordance_trajectories.json")
    p.add_argument("--furniture", required=True, help="Furniture name as stored in JSON")
    p.add_argument("--pair", required=True, nargs=2, help="Two part names: base target")
    p.add_argument("--sample", type=int, default=0, help="Index of trajectory sample to visualize")
    p.add_argument("--fps", type=float, default=30.0, help="Playback frames per second")
    p.add_argument("--no-trace", dest="trace", action="store_false", help="Do not show trajectory trace")
    return p.parse_args()


def main():
    args = parse_args()
    base, target = args.pair[0], args.pair[1]
    visualize_trajectory(args.json, args.furniture, base, target, sample_idx=args.sample, fps=args.fps, show_trace=args.trace)


if __name__ == '__main__':
    main()
