"""
Helper utilities for viser-based 3D visualization.

This module provides functions to convert Open3D and trimesh geometries to viser format
and manage viser visualization sessions.
"""
import numpy as np
import socket
from typing import Optional, Tuple, List
from pathlib import Path

try:
    import viser
    VISER_AVAILABLE = True
except ImportError:
    VISER_AVAILABLE = False
    viser = None


def _ensure_viser():
    """Ensure viser is available."""
    if not VISER_AVAILABLE:
        raise ImportError("viser is required for viser visualization. Install with: pip install viser")
    return viser


def _mat_to_wxyz(transform: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert 4x4 transformation matrix to wxyz quaternion."""
    from furniture_bench.utils import transform as T
    quat = T.mat2quat(transform[:3, :3])
    # Convert from (x, y, z, w) to (w, x, y, z)
    return (quat[3], quat[0], quat[1], quat[2])


def _trimesh_to_viser_mesh_data(trimesh_mesh, color: Optional[Tuple[float, float, float]] = None):
    """
    Convert a trimesh mesh to data format suitable for viser.
    
    Args:
        trimesh_mesh: trimesh.Trimesh object
        color: Optional RGB color tuple (0-1 range)
    
    Returns:
        Dictionary with vertices, faces, and optionally colors
    """
    vertices = np.asarray(trimesh_mesh.vertices, dtype=np.float64)
    faces = np.asarray(trimesh_mesh.faces, dtype=np.int32)
    
    result = {
        "vertices": vertices,
        "faces": faces,
    }
    
    # Handle colors
    if color is not None:
        # Apply uniform color to all vertices
        num_vertices = len(vertices)
        colors = np.tile(np.array(color, dtype=np.float32), (num_vertices, 1))
        result["vertex_colors"] = colors
    elif hasattr(trimesh_mesh.visual, 'vertex_colors') and trimesh_mesh.visual.vertex_colors is not None:
        # Use mesh vertex colors if available
        vertex_colors = np.asarray(trimesh_mesh.visual.vertex_colors, dtype=np.float32)
        if vertex_colors.shape[1] >= 3:
            result["vertex_colors"] = vertex_colors[:, :3] / 255.0  # Normalize to 0-1
    
    return result


def _open3d_to_viser_mesh_data(o3d_mesh, color: Optional[Tuple[float, float, float]] = None):
    """
    Convert an Open3D mesh to data format suitable for viser.
    
    Args:
        o3d_mesh: open3d.geometry.TriangleMesh object
        color: Optional RGB color tuple (0-1 range)
    
    Returns:
        Dictionary with vertices, faces, and optionally colors
    """
    vertices = np.asarray(o3d_mesh.vertices, dtype=np.float64)
    faces = np.asarray(o3d_mesh.triangles, dtype=np.int32)
    
    result = {
        "vertices": vertices,
        "faces": faces,
    }
    
    # Handle colors
    if color is not None:
        # Apply uniform color to all vertices
        num_vertices = len(vertices)
        colors = np.tile(np.array(color, dtype=np.float32), (num_vertices, 1))
        result["vertex_colors"] = colors
    elif o3d_mesh.has_vertex_colors():
        # Use mesh vertex colors if available
        vertex_colors = np.asarray(o3d_mesh.vertex_colors, dtype=np.float32)
        result["vertex_colors"] = vertex_colors
    
    return result


def add_mesh_to_viser_scene(
    server,
    path: str,
    mesh_data: dict,
    transform: Optional[np.ndarray] = None,
):
    """
    Add a mesh to viser scene.
    
    Args:
        server: viser.ViserServer instance
        path: Unique path/name for the mesh in the scene
        mesh_data: Dictionary with 'vertices', 'faces', and optionally 'vertex_colors'
        transform: Optional 4x4 transformation matrix
    """
    import tempfile
    import os
    
    vertices = mesh_data["vertices"].copy()
    faces = mesh_data["faces"].copy()
    
    # Apply transform if provided
    if transform is not None:
        # Convert to homogeneous coordinates
        homo_vertices = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1)
        transformed_homo = (transform @ homo_vertices.T).T
        vertices = transformed_homo[:, :3]
    
    # Create a trimesh object
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Apply color if available
        if "vertex_colors" in mesh_data:
            vertex_colors = mesh_data["vertex_colors"]
            if len(vertex_colors) > 0:
                # Convert to uint8 if needed
                if vertex_colors.dtype != np.uint8:
                    vertex_colors_uint8 = (np.clip(vertex_colors, 0, 1) * 255).astype(np.uint8)
                else:
                    vertex_colors_uint8 = vertex_colors
                mesh.visual.vertex_colors = vertex_colors_uint8
        
        # Use viser's add_mesh_trimesh method (preferred - works directly with trimesh)
        mesh_added = False
        
        if hasattr(server.scene, 'add_mesh_trimesh'):
            try:
                server.scene.add_mesh_trimesh(
                    path,
                    mesh=mesh,
                )
                mesh_added = True
            except Exception as e:
                # If add_mesh_trimesh fails, try add_mesh_simple
                pass
        
        # Fallback to add_mesh_simple (requires file path)
        if not mesh_added and hasattr(server.scene, 'add_mesh_simple'):
            try:
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                mesh.export(tmp_path)
                
                abs_tmp_path = os.path.abspath(tmp_path)
                server.scene.add_mesh_simple(
                    path,
                    file_path=abs_tmp_path,
                )
                mesh_added = True
            except Exception as e:
                pass
        
        if not mesh_added:
            raise RuntimeError(
                f"Could not add mesh to viser scene. "
                f"Tried add_mesh_trimesh and add_mesh_simple but both failed. "
                f"Make sure viser is properly installed."
            )
            
    except ImportError:
        raise ImportError("trimesh is required for viser mesh visualization. Install with: pip install trimesh")
    except Exception as e:
        raise RuntimeError(f"Failed to add mesh to viser scene: {e}")


def update_mesh_in_viser_scene(
    server,
    path: str,
    mesh_data: dict,
    transform: Optional[np.ndarray] = None,
):
    """
    Update an existing mesh in viser scene.
    
    Args:
        server: viser.ViserServer instance
        path: Path/name of the mesh to update
        mesh_data: Dictionary with 'vertices', 'faces', and optionally 'vertex_colors'
        transform: Optional 4x4 transformation matrix
    """
    # Remove old mesh and add new one
    try:
        server.scene.remove(path)
    except Exception:
        pass
    add_mesh_to_viser_scene(server, path, mesh_data, transform)


def add_frame_to_viser_scene(
    server,
    path: str,
    transform: np.ndarray,
    size: float = 0.05,
    show_axes: bool = True,
):
    """
    Add a coordinate frame to viser scene.
    
    Args:
        server: viser.ViserServer instance
        path: Unique path/name for the frame
        transform: 4x4 transformation matrix
        size: Size of the frame axes (not directly used, kept for compatibility)
        show_axes: Whether to show axes
    """
    position = tuple(transform[:3, 3])
    wxyz = _mat_to_wxyz(transform)
    
    server.scene.add_frame(
        path,
        wxyz=wxyz,
        position=position,
        show_axes=show_axes,
    )


def update_frame_in_viser_scene(
    server,
    path: str,
    transform: np.ndarray,
):
    """
    Update an existing coordinate frame in viser scene.
    
    Args:
        server: viser.ViserServer instance
        path: Path/name of the frame to update
        transform: 4x4 transformation matrix
    """
    frame = server.scene[path]
    if frame is not None:
        frame.position = tuple(transform[:3, 3])
        frame.wxyz = _mat_to_wxyz(transform)


def add_line_set_to_viser_scene(
    server,
    path: str,
    points: np.ndarray,
    lines: Optional[np.ndarray] = None,
    color: Optional[Tuple[float, float, float]] = None,
):
    """
    Add a line set to viser scene.
    
    Args:
        server: viser.ViserServer instance
        path: Unique path/name for the line set
        points: Nx3 array of points
        lines: Optional Mx2 array of line indices (if None, creates lines connecting consecutive points)
        color: Optional RGB color tuple (0-1 range)
    """
    if len(points) == 0:
        return
    
    if lines is None:
        # Create lines connecting consecutive points
        if len(points) < 2:
            return
        lines = np.array([[i, i + 1] for i in range(len(points) - 1)], dtype=np.int32)
    
    if len(lines) == 0:
        return
    
    # Viser's add_line_segments expects points as (N, 2, 3) array
    # where N is the number of line segments, each defined by two 3D points
    line_segments = points[lines]  # Shape: (M, 2, 3) where M is number of lines
    
    # Prepare colors if provided
    colors = None
    if color is not None:
        # Create color array: (N, 2, 3) where each endpoint has the same color
        # viser expects one color per endpoint, so we need to duplicate the color for both endpoints
        color_array = np.array(color, dtype=np.float32)  # Shape: (3,)
        # Reshape to (1, 1, 3) then tile to (N, 2, 3)
        colors = np.tile(color_array.reshape(1, 1, 3), (len(line_segments), 2, 1))
    
    # Use viser's add_line_segments method
    try:
        if hasattr(server.scene, 'add_line_segments'):
            if colors is not None:
                server.scene.add_line_segments(
                    name=path,
                    points=line_segments,
                    colors=colors,
                )
            else:
                server.scene.add_line_segments(
                    name=path,
                    points=line_segments,
                )
        else:
            # Fallback: if add_line_segments doesn't exist, try alternative methods
            raise AttributeError("add_line_segments not available")
    except Exception as e:
        raise RuntimeError(f"Failed to add line segments to viser scene: {e}")


def update_line_set_in_viser_scene(
    server,
    path: str,
    points: np.ndarray,
    lines: Optional[np.ndarray] = None,
    color: Optional[Tuple[float, float, float]] = None,
):
    """
    Update an existing line set in viser scene.
    
    Args:
        server: viser.ViserServer instance
        path: Path/name of the line set to update
        points: Nx3 array of points
        lines: Optional Mx2 array of line indices
        color: Optional RGB color tuple (0-1 range)
    """
    # Remove old line set and add new one
    try:
        server.scene.remove(path)
    except Exception:
        pass
    add_line_set_to_viser_scene(server, path, points, lines, color)


def _find_available_port(start_port: int = 8080, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    for i in range(max_attempts):
        port = start_port + i
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Could not find an available port starting from {start_port}")


def create_viser_server(
    title: str = "Viser Visualization",
    port: Optional[int] = None,
) -> "viser.ViserServer":
    """
    Create and configure a viser server.
    
    Args:
        title: Window/title for the visualization (not directly used by viser, kept for compatibility)
        port: Optional port number (default: auto-assign starting from 8080)
    
    Returns:
        viser.ViserServer instance
    """
    _ensure_viser()
    
    # If no port specified, find an available one
    if port is None:
        port = _find_available_port(start_port=8080)
    
    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+z")
    
    # Add grid for reference
    server.scene.add_grid(
        "/grid",
        width=2.0,
        height=2.0,
        position=(0.0, 0.0, 0.0),
    )
    
    return server

