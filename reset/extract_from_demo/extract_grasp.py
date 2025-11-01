import argparse
import copy
import json
import pickle
import xml.etree.ElementTree as ET
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from furniture_bench.config import config
from furniture_bench.furniture import furniture_factory
import furniture_bench.utils.transform as T
from furniture_bench.utils.averageQuaternions import averageQuaternions


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
        "distance": args.distance_threshold,
        "velocity_diff": args.velocity_difference_threshold,
        "min_part_speed": args.min_part_speed,
        "min_steps": args.min_consecutive_steps,
    }


def _analyze_observations(
    furniture_name: str,
    observations: List[Dict],
    part_names: List[str],
    thresholds: Dict[str, float],
    robot_from_april: np.ndarray,
    aggregate_store: Dict[str, Dict[str, Dict[str, List[np.ndarray]]]],
):
    num_parts = len(part_names)
    prev_part_positions = {idx: None for idx in range(num_parts)}
    prev_ee_pos = None
    consecutive_counters = {idx: 0 for idx in range(num_parts)}

    for obs in observations:
        robot_state = obs.get("robot_state")
        parts_poses = obs.get("parts_poses")
        if robot_state is None or parts_poses is None:
            continue

        ee_pos = _to_numpy(robot_state.get("ee_pos"))
        ee_quat = _to_numpy(robot_state.get("ee_quat"))
        gripper_width = float(np.asarray(robot_state.get("gripper_width")))
        ee_pose_vec = np.concatenate([ee_pos, ee_quat])

        parts_array = _to_numpy(parts_poses).reshape(num_parts, 7)

        ee_speed = 0.0 if prev_ee_pos is None else float(np.linalg.norm(ee_pos - prev_ee_pos))

        for part_idx in range(num_parts):
            part_pose_vec = parts_array[part_idx]
            part_pose_mat_april = T.pose2mat(part_pose_vec)
            part_pose_mat_robot = robot_from_april @ part_pose_mat_april
            part_pos_robot = part_pose_mat_robot[:3, 3]

            part_speed = 0.0
            if prev_part_positions[part_idx] is not None:
                part_speed = float(np.linalg.norm(part_pos_robot - prev_part_positions[part_idx]))

            distance = float(np.linalg.norm(part_pos_robot - ee_pos))

            gripper_closed = gripper_width < thresholds["gripper"]
            close_enough = distance < thresholds["distance"]
            velocity_similar = abs(part_speed - ee_speed) < thresholds["velocity_diff"] or part_speed > thresholds["min_part_speed"]

            if gripper_closed and close_enough and velocity_similar:
                consecutive_counters[part_idx] += 1
            else:
                consecutive_counters[part_idx] = 0

            if consecutive_counters[part_idx] >= thresholds["min_steps"]:
                rel_pos, rel_quat = _compute_relative_pose(ee_pose_vec, part_pose_mat_robot)

                part_store = aggregate_store[furniture_name][part_names[part_idx]]
                part_store["relative_pos"].append(rel_pos)
                part_store["relative_quat"].append(rel_quat)
                part_store["ee_pos"].append(ee_pos)
                part_store["ee_quat"].append(ee_quat)
                part_store["part_pos"].append(part_pos_robot)
                part_store["part_quat"].append(part_pose_vec[3:])

            prev_part_positions[part_idx] = part_pos_robot

        prev_ee_pos = ee_pos


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
    parser = argparse.ArgumentParser(description="Extract grasp pose statistics from demonstrations")
    parser.add_argument("--dataset", required=True, help="Path to a dataset directory or a single pickle file")
    parser.add_argument("--furniture", default=None, help="Override furniture name (otherwise taken from dataset)")
    parser.add_argument("--distance-threshold", type=float, default=0.6, help="Maximum distance (m) between EE and part to consider a grasp")
    parser.add_argument("--gripper-close-threshold", type=float, default=None, help="Absolute threshold (m) for gripper width when considered closed")
    parser.add_argument("--gripper-close-ratio", type=float, default=0.6, help="Ratio of max gripper width used when threshold is not provided")
    parser.add_argument("--velocity-difference-threshold", type=float, default=0.02, help="Maximum difference between EE and part speeds when detecting a grasp")
    parser.add_argument("--min-part-speed", type=float, default=0.005, help="Minimum part speed (m) to treat as moving together with the EE")
    parser.add_argument("--min-consecutive-steps", type=int, default=1, help="Number of consecutive frames satisfying heuristics before logging a grasp")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save the statistics as JSON")
    parser.add_argument("--render", action="store_true", help="Render sampled grasp poses based on computed statistics")
    parser.add_argument("--render-video", action="store_true", help="Render video of the demonstrations")
    parser.add_argument("--render-part", default=None, help="Name of a specific part to render (defaults to all)")
    parser.add_argument("--render-furniture", default=None, help="Furniture name to render (defaults to stats furniture)")
    parser.add_argument("--render-samples", type=int, default=5, help="Number of random samples to visualize per part")
    parser.add_argument("--render-spread-mult", type=float, default=1.0, help="Multiplier on positional std when sampling part poses")
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
) -> List[List[str]]:
    num_parts = len(part_names)
    labels: List[List[str]] = [[] for _ in range(len(observations))]

    prev_part_positions = {idx: None for idx in range(num_parts)}
    prev_ee_pos = None
    consecutive_counters = {idx: 0 for idx in range(num_parts)}

    for step_idx, obs in enumerate(observations):
        robot_state = obs.get("robot_state")
        parts_poses = obs.get("parts_poses")
        if robot_state is None or parts_poses is None:
            continue

        ee_pos = _to_numpy(robot_state.get("ee_pos"))
        gripper_width = float(np.asarray(robot_state.get("gripper_width")))

        parts_array = _to_numpy(parts_poses).reshape(num_parts, 7)

        ee_speed = 0.0 if prev_ee_pos is None else float(np.linalg.norm(ee_pos - prev_ee_pos))

        for part_idx in range(num_parts):
            part_pose_vec = parts_array[part_idx]
            part_pose_mat_april = T.pose2mat(part_pose_vec)
            part_pose_mat_robot = robot_from_april @ part_pose_mat_april
            part_pos_robot = part_pose_mat_robot[:3, 3]

            part_speed = 0.0
            if prev_part_positions[part_idx] is not None:
                part_speed = float(np.linalg.norm(part_pos_robot - prev_part_positions[part_idx]))

            distance = float(np.linalg.norm(part_pos_robot - ee_pos))

            gripper_closed = gripper_width < thresholds["gripper"]
            close_enough = distance < thresholds["distance"]
            velocity_similar = (
                abs(part_speed - ee_speed) < thresholds["velocity_diff"]
                or part_speed > thresholds["min_part_speed"]
            )

            if gripper_closed and close_enough and velocity_similar:
                consecutive_counters[part_idx] += 1
            else:
                consecutive_counters[part_idx] = 0

            if consecutive_counters[part_idx] >= thresholds["min_steps"]:
                if part_names[part_idx] not in labels[step_idx]:
                    labels[step_idx].append(part_names[part_idx])

            prev_part_positions[part_idx] = part_pos_robot

        prev_ee_pos = ee_pos

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

    for annotation_path in annotation_files:
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
        robot_from_april = np.linalg.inv(config["robot"]["tag_base_from_robot_base"])

        observations = data.get("observations", [])

        if args.render_video and observations:
            grasp_labels = _detect_grasp_labels(
                observations,
                part_names,
                thresholds,
                robot_from_april,
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
        )

    summary = _summarize(aggregate_store)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to {args.output}")

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