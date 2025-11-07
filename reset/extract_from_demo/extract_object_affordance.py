"""
Extract relative trajectories between parts when collision persists for a minimum duration.

Behavior:
1. Detect collision between all pairs of parts using mesh-based distance (trimesh CollisionManager).
2. When a collision for a pair starts, record the relative poses between the two parts (pose of B in A frame) for each frame while collision persists.
3. When collision ends, if the recorded trajectory length >= min_consecutive_steps, keep/save it; otherwise discard.

Saves results as JSON: a mapping furniture -> pair_name -> list of trajectories, where each trajectory
is a list of 7-element pose vectors [x,y,z,qx,qy,qz,qw].

This file intentionally borrows mesh loading code from the project's extract_grasp helper if available.
"""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import argparse
import json
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

# Project imports
from furniture_bench.config import config
from furniture_bench.furniture import furniture_factory
from furniture_bench.utils import transform as T

# Try to reuse the mesh-loading helper from extract_grasp if available
try:
    from .extract_grasp import _load_part_mesh_trimesh, TRIMESH_AVAILABLE
except Exception:
    # Fallback: try import from package path
    try:
        from reset.extract_from_demo.extract_grasp import _load_part_mesh_trimesh, TRIMESH_AVAILABLE
    except Exception:
        _load_part_mesh_trimesh = None
        TRIMESH_AVAILABLE = False

try:
    import trimesh
except Exception:
    trimesh = None
    TRIMESH_AVAILABLE = False


def _iter_annotation_files(dataset_path: Path):
    if dataset_path.is_file():
        yield dataset_path
        return

    if dataset_path.suffix == ".pkl":
        yield dataset_path
        return

    for pkl_path in sorted(dataset_path.rglob("*.pkl")):
        if pkl_path.is_file():
            yield pkl_path


def _pose_vec_to_mat(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    pose_vec = np.concatenate([pos, quat])
    return T.pose2mat(pose_vec)


def _compute_relative_pose_between_parts(a_pose_vec: np.ndarray, b_pose_vec: np.ndarray) -> np.ndarray:
    """Return relative pose of B in A's frame as 7-dim vector [x,y,z,qx,qy,qz,qw]."""
    a_mat = T.pose2mat(a_pose_vec)
    b_mat = T.pose2mat(b_pose_vec)
    rel = np.linalg.inv(a_mat) @ b_mat
    pos, quat = T.mat2pose(rel)
    return np.concatenate([pos, quat])


def _process_annotations(
    annotation_paths: List[Path],
    furniture_override: Optional[str],
    contact_tolerance: float,
    min_consecutive_steps: int,
    output_dir: Optional[Path],
    visualize: bool = False,
):
    if not TRIMESH_AVAILABLE or trimesh is None or _load_part_mesh_trimesh is None:
        raise ImportError(
            "Mesh-based collision detection requires trimesh and the project's mesh loader. Ensure trimesh is installed and extract_grasp is importable."
        )

    aggregate: Dict[str, Dict[str, List[List[List[float]]]]] = defaultdict(lambda: defaultdict(list))

    for idx, ann_path in enumerate(annotation_paths):
        print(f"Processing {idx+1} / {len(annotation_paths)}: {ann_path}")
        with open(ann_path, "rb") as f:
            data = pickle.load(f)

        furniture_name = furniture_override or data.get("furniture")
        if furniture_name is None:
            print(f"Skipping {ann_path} - furniture not found and no override provided")
            continue

        furniture = furniture_factory(furniture_name)
        part_names = [p.name for p in furniture.parts]
        num_parts = len(part_names)

        # Load meshes for all parts (may skip some if not found)
        furniture_conf = config["furniture"].get(furniture_name, {})
        part_meshes = {}
        for part_name in part_names:
            part_conf = furniture_conf.get(part_name)
            if not isinstance(part_conf, dict) or "asset_file" not in part_conf:
                print(f"Warning: missing asset file for part '{part_name}', skipping mesh load")
                continue
            try:
                part_meshes[part_name] = _load_part_mesh_trimesh(part_conf["asset_file"])
            except Exception as e:
                print(f"Warning: failed to load mesh for part '{part_name}': {e}")

        if not part_meshes:
            print(f"No meshes available for furniture {furniture_name}, skipping file {ann_path}")
            continue

        observations = data.get("observations", [])
        robot_from_april = config["robot"]["tag_base_from_robot_base"]

        # Prepare per-pair state trackers
        # keys are (i,j) indices with i<j
        pair_state = {}
        for i in range(num_parts):
            for j in range(i + 1, num_parts):
                pair_state[(i, j)] = {
                    "in_contact": False,
                    "counter": 0,
                    "buffer": [],  # list of relative poses while contact
                }

        # iterate frames
        for obs in tqdm(observations, desc="Frames"):
            parts_poses = obs.get("parts_poses")
            if parts_poses is None:
                continue

            parts_array = np.asarray(parts_poses).reshape(num_parts, 7)

            # For each pair compute min distance via CollisionManager
            for i in range(num_parts):
                for j in range(i + 1, num_parts):
                    name_i = part_names[i]
                    name_j = part_names[j]

                    if name_i not in part_meshes or name_j not in part_meshes:
                        # Cannot detect collisions without both meshes
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
                        # In some versions or for degenerate meshes, min_distance_single may fail.
                        # Fall back to AABB distance as a conservative proxy.
                        try:
                            aabb_i = mesh_i.bounds  # (min, max)
                            aabb_j = mesh_j.bounds
                            # compute center distance minus extents
                            center_i = aabb_i.mean(axis=0)
                            center_j = aabb_j.mean(axis=0)
                            min_dist = float(np.linalg.norm(center_i - center_j))
                        except Exception:
                            min_dist = float(np.inf)

                    has_contact = min_dist < contact_tolerance

                    state = pair_state[(i, j)]

                    if has_contact:
                        # record relative pose of j in i's frame
                        rel_pose = _compute_relative_pose_between_parts(part_i_pose_april, part_j_pose_april)
                        state["buffer"].append(rel_pose.tolist())
                        state["counter"] += 1
                        state["in_contact"] = True
                    else:
                        # contact ended (or not present)
                        if state["in_contact"]:
                            # evaluate buffer
                            if state["counter"] >= min_consecutive_steps:
                                pair_key = f"{name_i}__{name_j}"
                                aggregate[furniture_name][pair_key].append(state["buffer"].copy())
                            # reset
                            state["buffer"].clear()
                            state["counter"] = 0
                            state["in_contact"] = False
                        else:
                            # still not in contact; nothing to do
                            pass

        # After finishing frames, flush active contacts
        for (i, j), state in pair_state.items():
            if state["in_contact"] and state["counter"] >= min_consecutive_steps:
                name_i = part_names[i]
                name_j = part_names[j]
                pair_key = f"{name_i}__{name_j}"
                aggregate[furniture_name][pair_key].append(state["buffer"].copy())

    # Save results if requested
    if output_dir is not None:
        out = Path(output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / "object_affordance_trajectories.json"
        # Convert numpy types into native python
        def _make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, float):
                return float(obj)
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            return obj

        serializable = {}
        for furn, pairs in aggregate.items():
            serializable[furn] = {}
            for pair_name, trajs in pairs.items():
                # each traj is list of list of floats; ensure lists
                serializable[furn][pair_name] = [[list(map(float, pose)) for pose in traj] for traj in trajs]

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved trajectories to {out_path}")

    return aggregate


def parse_args():
    parser = argparse.ArgumentParser(description="Extract object affordance trajectories from demonstrations")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory or single .pkl file")
    parser.add_argument("--furniture", default=None, help="Override furniture name (otherwise from file)")
    parser.add_argument("--contact-tolerance", type=float, default=0.005, help="Distance threshold (m) to consider contact. Default: 0.005")
    parser.add_argument("--min-consecutive-steps", type=int, default=3, help="Minimum consecutive steps of contact to keep a trajectory")
    parser.add_argument("--output", type=Path, default=None, help="Directory to save results JSON")
    parser.add_argument("--visualize", action="store_true", help="(Not implemented) show quick visuals for collisions")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset).expanduser().resolve()
    annotation_files = list(_iter_annotation_files(dataset_root))
    if not annotation_files:
        raise FileNotFoundError(f"No .pkl files found under {dataset_root}")

    aggregate = _process_annotations(
        annotation_files,
        args.furniture,
        args.contact_tolerance,
        args.min_consecutive_steps,
        args.output,
        visualize=args.visualize,
    )

    # Print brief summary
    for furn, pairs in aggregate.items():
        print(f"\nFurniture: {furn}")
        for pair_name, trajs in pairs.items():
            print(f"  {pair_name}: {len(trajs)} trajectory(ies)")


if __name__ == "__main__":
    main()
