#!/usr/bin/env python3
"""Visualize a part mesh after loading to debug mesh loading issues."""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from furniture_bench.config import config
from reset.extract_from_demo.extract_grasp import (
    _load_part_mesh_trimesh,
    _parse_urdf_mesh,
    _ensure_open3d,
    _trimesh_to_open3d,
)

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("ERROR: trimesh is not available. Install with: pip install trimesh")
    sys.exit(1)


def print_mesh_info(mesh, title="Mesh"):
    """Print detailed information about a mesh."""
    print(f"\n{'='*60}")
    print(f"{title} Information")
    print(f"{'='*60}")
    print(f"Type: {type(mesh)}")
    print(f"Is empty: {mesh.is_empty}")
    print(f"Number of vertices: {len(mesh.vertices)}")
    print(f"Number of faces: {len(mesh.faces)}")
    
    if len(mesh.vertices) > 0:
        print(f"\nVertex statistics:")
        print(f"  Min: {mesh.vertices.min(axis=0)}")
        print(f"  Max: {mesh.vertices.max(axis=0)}")
        print(f"  Mean: {mesh.vertices.mean(axis=0)}")
        print(f"  Std: {mesh.vertices.std(axis=0)}")
        
        # Bounding box
        bounds = mesh.bounds
        print(f"\nBounding box:")
        print(f"  Min corner: {bounds[0]}")
        print(f"  Max corner: {bounds[1]}")
        print(f"  Extents: {bounds[1] - bounds[0]}")
        print(f"  Center: {mesh.centroid}")
        
        # Volume and area
        if mesh.is_volume:
            print(f"  Volume: {mesh.volume:.6f}")
        print(f"  Surface area: {mesh.area:.6f}")
        
        # Check for degenerate cases
        if np.allclose(bounds[1] - bounds[0], 0, atol=1e-6):
            print(f"\n⚠️  WARNING: Mesh appears to be a point (zero extents)")
        elif np.any(np.abs(bounds[1] - bounds[0]) < 1e-6):
            print(f"\n⚠️  WARNING: Mesh appears to be flat/degenerate in some dimension")
        
        # Check scale
        extents = bounds[1] - bounds[0]
        max_extent = np.max(extents)
        min_extent = np.min(extents[extents > 1e-6])
        if max_extent > 0:
            aspect_ratio = max_extent / min_extent if min_extent > 0 else float('inf')
            print(f"  Aspect ratio (max/min): {aspect_ratio:.2f}")
            if aspect_ratio > 1000:
                print(f"  ⚠️  WARNING: Very high aspect ratio - mesh might be flat")


def visualize_mesh(mesh, title="Mesh Visualization", color=None, use_viser=False):
    """Visualize a trimesh using open3d or viser."""
    if color is None:
        color = np.array([0.7, 0.7, 0.9])  # Light blue
    color_tuple = tuple(color[:3])
    
    if use_viser:
        from reset.visualization.viser_utils import (
            create_viser_server,
            add_mesh_to_viser_scene,
            add_frame_to_viser_scene,
            _trimesh_to_viser_mesh_data,
            VISER_AVAILABLE,
        )
        if not VISER_AVAILABLE:
            raise ImportError("viser is required for viser visualization. Install with: pip install viser")
        
        import furniture_bench.utils.transform as T
        
        server = create_viser_server(title=title)
        
        # Add mesh
        mesh_data = _trimesh_to_viser_mesh_data(mesh, color=color_tuple)
        add_mesh_to_viser_scene(server, "/mesh", mesh_data)
        
        # Add coordinate frame at origin
        identity_transform = np.eye(4)
        add_frame_to_viser_scene(server, "/frame", identity_transform, size=0.1)
        
        # Add bounding box as lines
        bounds = mesh.bounds
        if len(bounds) == 2:
            bbox_corners = np.array([
                [bounds[0][0], bounds[0][1], bounds[0][2]],
                [bounds[1][0], bounds[0][1], bounds[0][2]],
                [bounds[1][0], bounds[1][1], bounds[0][2]],
                [bounds[0][0], bounds[1][1], bounds[0][2]],
                [bounds[0][0], bounds[0][1], bounds[1][2]],
                [bounds[1][0], bounds[0][1], bounds[1][2]],
                [bounds[1][0], bounds[1][1], bounds[1][2]],
                [bounds[0][0], bounds[1][1], bounds[1][2]],
            ])
            bbox_lines = np.array([
                [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # top face
                [0, 4], [1, 5], [2, 6], [3, 7],  # vertical edges
            ])
            from reset.visualization.viser_utils import add_line_set_to_viser_scene
            add_line_set_to_viser_scene(server, "/bbox", bbox_corners, bbox_lines, color=(1.0, 0.0, 0.0))
        
        print(f"\nVisualizing {title} with viser...")
        print("Open your browser to the URL shown above to view the visualization.")
        print("Press Ctrl+C to exit.")
        
        try:
            import time
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nExiting...")
    else:
        o3d = _ensure_open3d()
        
        # Convert to open3d
        o3d_mesh = _trimesh_to_open3d(mesh)
        
        # Set color
        o3d_mesh.paint_uniform_color(color)
        
        # Create coordinate frame at origin
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        
        # Create bounding box wireframe
        bbox = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
            o3d_mesh.get_axis_aligned_bounding_box()
        )
        bbox.paint_uniform_color([1.0, 0.0, 0.0])  # Red
        
        # Visualize
        geometries = [o3d_mesh, coord_frame, bbox]
        
        print(f"\nVisualizing {title}...")
        print("Controls:")
        print("  - Mouse: Rotate (left drag), Pan (right drag), Zoom (scroll)")
        print("  - Press 'Q' or close window to exit")
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name=title,
            width=1280,
            height=720,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a part mesh after loading to debug issues"
    )
    parser.add_argument(
        "--furniture",
        required=True,
        help="Furniture name (e.g., 'square_table', 'desk', 'cabinet')"
    )
    parser.add_argument(
        "--part",
        required=True,
        help="Part name (e.g., 'square_table_top', 'desk_top', 'cabinet_body')"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Don't open visualization window, just print info"
    )
    parser.add_argument(
        "--use-viser",
        action="store_true",
        help="Use viser for visualization instead of Open3D"
    )
    args = parser.parse_args()
    
    # Get furniture config
    furniture_conf = config["furniture"].get(args.furniture)
    if furniture_conf is None:
        print(f"ERROR: Furniture '{args.furniture}' not found in config")
        print(f"Available furniture: {list(config['furniture'].keys())}")
        sys.exit(1)
    
    # Get part config
    part_conf = furniture_conf.get(args.part)
    if part_conf is None:
        print(f"ERROR: Part '{args.part}' not found in furniture '{args.furniture}'")
        print(f"Available parts: {list(furniture_conf.keys())}")
        sys.exit(1)
    
    if "asset_file" not in part_conf:
        print(f"ERROR: Part '{args.part}' missing 'asset_file' in config")
        sys.exit(1)
    
    asset_file = part_conf["asset_file"]
    print(f"Loading mesh for:")
    print(f"  Furniture: {args.furniture}")
    print(f"  Part: {args.part}")
    print(f"  Asset file: {asset_file}")
    
    # Parse URDF to get mesh path
    try:
        mesh_path, scale_values = _parse_urdf_mesh(asset_file)
        print(f"\nParsed URDF:")
        print(f"  Mesh path: {mesh_path}")
        print(f"  Mesh exists: {mesh_path.is_file()}")
        print(f"  Mesh file size: {mesh_path.stat().st_size if mesh_path.is_file() else 'N/A'} bytes")
        print(f"  Scale values: {scale_values}")
        
        # Also try to check what's actually in the URDF
        from furniture_bench.config import config as fb_config
        ASSETS_ROOT = Path(__file__).resolve().parents[2] / "furniture_bench" / "assets"
        urdf_path = (ASSETS_ROOT / asset_file).resolve()
        if urdf_path.is_file():
            import xml.etree.ElementTree as ET
            tree = ET.parse(urdf_path)
            mesh_elements = tree.findall(".//mesh")
            print(f"\nURDF contains {len(mesh_elements)} <mesh> element(s):")
            for i, elem in enumerate(mesh_elements):
                filename = elem.get("filename")
                scale = elem.get("scale")
                print(f"  Mesh {i+1}: filename='{filename}', scale='{scale}'")
    except Exception as e:
        print(f"ERROR parsing URDF: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Load raw mesh first (before scaling)
    print(f"\n{'='*60}")
    print("Loading raw mesh (before scaling)...")
    print(f"{'='*60}")
    try:
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Mesh file not found at {mesh_path}")
        
        # Try loading with different options
        print(f"Attempting to load with trimesh.load()...")
        raw_mesh = trimesh.load(str(mesh_path), force='mesh')
        
        if isinstance(raw_mesh, trimesh.Scene):
            print(f"  Loaded as Scene, extracting geometries...")
            geometries = list(raw_mesh.geometry.values())
            if not geometries:
                raise ValueError(f"No geometries found in {mesh_path}")
            print(f"  Found {len(geometries)} geometry(ies), using first one")
            raw_mesh = geometries[0]
        
        if not isinstance(raw_mesh, trimesh.Trimesh):
            raise ValueError(f"Expected Trimesh object, got {type(raw_mesh)}")
        
        print_mesh_info(raw_mesh, "Raw Mesh (before scaling)")
    except Exception as e:
        print(f"ERROR loading raw mesh: {e}")
        print(f"\nTrying alternative loading method...")
        try:
            # Try loading as OBJ directly
            if mesh_path.suffix.lower() == '.obj':
                raw_mesh = trimesh.load(str(mesh_path), file_type='obj', process=False)
                if isinstance(raw_mesh, trimesh.Scene):
                    geometries = list(raw_mesh.geometry.values())
                    raw_mesh = geometries[0] if geometries else None
                if raw_mesh is not None:
                    print(f"Successfully loaded using OBJ loader")
                    print_mesh_info(raw_mesh, "Raw Mesh (OBJ loader)")
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Load processed mesh (after scaling)
    print(f"\n{'='*60}")
    print("Loading processed mesh (after scaling)...")
    print(f"{'='*60}")
    try:
        processed_mesh = _load_part_mesh_trimesh(asset_file)
        print_mesh_info(processed_mesh, "Processed Mesh (after scaling)")
        
        # Compare
        if len(raw_mesh.vertices) == len(processed_mesh.vertices):
            vertex_diff = np.abs(raw_mesh.vertices - processed_mesh.vertices)
            max_diff = vertex_diff.max()
            print(f"\nVertex comparison (raw vs processed):")
            print(f"  Max vertex difference: {max_diff:.6f}")
            if max_diff > 1e-6:
                print(f"  Vertices changed (likely due to scaling)")
            else:
                print(f"  Vertices unchanged")
    except Exception as e:
        print(f"ERROR loading processed mesh: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Visualize
    if not args.no_visualize:
        print(f"\n{'='*60}")
        print("Visualization")
        print(f"{'='*60}")
        try:
            visualize_mesh(processed_mesh, title=f"{args.furniture} - {args.part}", use_viser=args.use_viser)
        except Exception as e:
            print(f"ERROR during visualization: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

