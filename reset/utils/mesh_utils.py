"""Mesh loading and manipulation utility functions."""
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

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


def _ensure_open3d():
    """Ensure open3d module is loaded."""
    global _OPEN3D_MODULE
    if _OPEN3D_MODULE is None:
        try:
            import open3d as o3d  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Rendering meshes requires the 'open3d' package. Install it or omit --render."
            ) from exc
        _OPEN3D_MODULE = o3d
    return _OPEN3D_MODULE


def parse_urdf_mesh(asset_file: str) -> Tuple[Path, Optional[List[float]]]:
    """Parse URDF file to extract mesh path and scale values."""
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


def apply_scale_trimesh(mesh: "trimesh.Trimesh", scale_values: Optional[List[float]]) -> "trimesh.Trimesh":
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


def apply_scale_o3d(mesh, scale_values: Optional[List[float]]):
    """Apply scale to an open3d mesh."""
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


def trimesh_to_open3d(trimesh_mesh: "trimesh.Trimesh"):
    """Convert a trimesh mesh to an open3d mesh."""
    o3d = _ensure_open3d()
    
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
    
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.compute_triangle_normals()
    
    return o3d_mesh


def open3d_to_trimesh(o3d_mesh) -> "trimesh.Trimesh":
    """Convert an open3d mesh to a trimesh mesh."""
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required for mesh conversion. Install it with: pip install trimesh")
    
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    return mesh


@lru_cache(maxsize=None)
def load_part_mesh_trimesh(asset_file: str) -> "trimesh.Trimesh":
    """
    Load a part mesh as a trimesh object.
    
    Tries multiple methods:
    1. Uses yourdfpy to properly parse URDF and load mesh
    2. Falls back to using open3d loader (which works) and converts to trimesh
    3. Last resort: direct trimesh.load() on the mesh file
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required for mesh-based contact detection. Install it with: pip install trimesh")
    
    mesh_path, scale_values = parse_urdf_mesh(asset_file)
    urdf_path = (ASSETS_ROOT / asset_file).resolve()
    
    # Method 1: Try using yourdfpy to properly parse URDF and load mesh
    if YOURDFPY_AVAILABLE and urdf_path.is_file():
        try:
            robot = yourdfpy.URDF.load(str(urdf_path))
            scene = robot.get_visual_scene()
            
            if isinstance(scene, trimesh.Scene):
                geometries = list(scene.geometry.values())
                if geometries:
                    mesh = trimesh.util.concatenate(geometries)
                    if isinstance(mesh, trimesh.Trimesh) and not mesh.is_empty:
                        mesh = apply_scale_trimesh(mesh, scale_values)
                        mesh.remove_unreferenced_vertices()
                        return mesh
        except Exception:
            pass
    
    # Method 2: Use open3d loader and convert to trimesh
    try:
        o3d_mesh = load_part_mesh(asset_file)
        if not o3d_mesh.is_empty():
            mesh = open3d_to_trimesh(o3d_mesh)
            mesh.remove_unreferenced_vertices()
            return mesh
    except Exception:
        pass
    
    # Method 3: Last resort - direct trimesh.load()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh file not found at {mesh_path}")
    
    mesh = trimesh.load(str(mesh_path), force='mesh')
    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        if not geometries:
            raise ValueError(f"No geometries found in {mesh_path}")
        if len(geometries) == 1:
            mesh = geometries[0]
        else:
            mesh = trimesh.util.concatenate(geometries)
    
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected Trimesh object, got {type(mesh)}")
    
    if mesh.is_empty:
        raise ValueError(f"Mesh at {mesh_path} is empty")
    
    mesh = apply_scale_trimesh(mesh, scale_values)
    mesh.remove_unreferenced_vertices()
    return mesh


@lru_cache(maxsize=None)
def load_part_mesh(asset_file: str):
    """Load a part mesh as an open3d mesh."""
    o3d = _ensure_open3d()
    mesh_path, scale_values = parse_urdf_mesh(asset_file)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh file not found at {mesh_path}")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise ValueError(f"Mesh at {mesh_path} is empty")

    apply_scale_o3d(mesh, scale_values)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


@lru_cache(maxsize=1)
def load_gripper_mesh_trimesh() -> "trimesh.Trimesh":
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


@lru_cache(maxsize=1)
def load_gripper_mesh():
    """Load the gripper mesh as an open3d mesh."""
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


def compute_mesh_distance(
    mesh_a: "trimesh.Trimesh",
    mesh_b: "trimesh.Trimesh",
) -> float:
    """
    Compute minimum distance between two meshes.
    
    Args:
        mesh_a: First mesh
        mesh_b: Second mesh
    
    Returns:
        Minimum distance between meshes, or inf if computation fails
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required for mesh distance computation")
    
    try:
        manager = trimesh.collision.CollisionManager()
        manager.add_object("a", mesh_a)
        min_dist = manager.min_distance_single(mesh_b)
        return float(min_dist)
    except Exception:
        # Fallback to proximity-based distance
        try:
            closest_points, distances, _ = trimesh.proximity.closest_point(mesh_a, mesh_b.vertices)
            if len(distances) > 0:
                return float(np.min(distances))
            else:
                return float(np.inf)
        except Exception:
            # Last resort: use AABB center distance
            try:
                aabb_a = mesh_a.bounds
                aabb_b = mesh_b.bounds
                center_a = aabb_a.mean(axis=0)
                center_b = aabb_b.mean(axis=0)
                return float(np.linalg.norm(center_a - center_b))
            except Exception:
                return float(np.inf)

