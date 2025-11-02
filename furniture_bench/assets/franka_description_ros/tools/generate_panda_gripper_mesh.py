#!/usr/bin/env python3
"""Generate a simple mesh for the Franka Panda gripper fingers.

This script recreates the Panda gripper collision geometry using the
box primitives defined in `franka_panda.urdf`.  The resulting mesh can be
exported to formats such as STL, OBJ, or PLY.  By default, the output
represents the gripper in a fully closed configuration; you can provide a
prismatic joint offset to model partially opened fingers.

Example:

    ./generate_panda_gripper_mesh.py \
        --output panda_gripper_collision.stl \
        --finger-offset 0.01 \
        --hand-mesh ../meshes/collision/hand.stl \
        --origin-translation 0.01 0.02 -0.005
"""

from __future__ import annotations

import argparse
import math
import pathlib
from typing import Iterable, Optional

import numpy as np
import trimesh


# Dimensions and transforms taken from the Panda URDF collision definition.
FINGER_COLLISION_BOXES = (
    {
        "size": (22e-3, 15e-3, 20e-3),
        "origin": (0.0, 18.5e-3, 11e-3),
        "rpy": (0.0, 0.0, 0.0),
    },
    {
        "size": (22e-3, 8.8e-3, 3.8e-3),
        "origin": (0.0, 6.8e-3, 2.2e-3),
        "rpy": (0.0, 0.0, 0.0),
    },
    {
        "size": (17.5e-3, 7e-3, 23.5e-3),
        "origin": (0.0, 15.9e-3, 28.35e-3),
        "rpy": (math.radians(30.0), 0.0, 0.0),
    },
    {
        "size": (17.5e-3, 15.2e-3, 18.5e-3),
        "origin": (0.0, 7.58e-3, 45.25e-3),
        "rpy": (0.0, 0.0, 0.0),
    },
)


def _box_mesh(extents: Iterable[float], origin: Iterable[float], rpy: Iterable[float]) -> trimesh.Trimesh:
    """Create a box mesh transformed by origin and roll-pitch-yaw."""

    mesh = trimesh.creation.box(extents=np.asarray(extents, dtype=float))

    transform = trimesh.transformations.euler_matrix(*rpy)
    transform[:3, 3] = np.asarray(origin, dtype=float)

    mesh.apply_transform(transform)
    return mesh


def _build_finger(sign: float, joint_offset: float) -> trimesh.Trimesh:
    """Assemble the finger mesh for the given side.

    Args:
        sign: +1 for the left finger, -1 for the right finger.
        joint_offset: prismatic joint displacement in metres.
    """

    components = []
    for primitive in FINGER_COLLISION_BOXES:
        origin = np.asarray(primitive["origin"], dtype=float).copy()
        origin[1] *= sign

        roll, pitch, yaw = primitive["rpy"]
        rpy = (roll * sign, pitch, yaw)

        components.append(_box_mesh(primitive["size"], origin, rpy))

    finger = trimesh.util.concatenate(components)
    finger.apply_translation((0.0, sign * joint_offset, 0.0))
    return finger


def _load_hand_mesh(path: pathlib.Path) -> trimesh.Trimesh:
    hand_mesh = trimesh.load(path, force="mesh")
    if hand_mesh.is_empty:
        raise ValueError(f"Hand mesh at '{path}' did not contain geometry")
    return hand_mesh


def build_gripper_mesh(
    joint_offset: float = 0.04,
    hand_mesh_path: Optional[pathlib.Path] = None,
    origin_translation: Optional[Iterable[float]] = None,
) -> trimesh.Trimesh:
    """Create a combined mesh for the Panda gripper.

    Args:
        joint_offset: Translation of each finger along its prismatic joint axis.
            Must be between 0 and 0.04 metres for the stock Panda gripper.
        hand_mesh_path: Optional path to an existing base hand mesh to merge.
        origin_translation: Optional translation vector [x, y, z] to shift the mesh origin.
            Defaults to None (no translation).

    Returns:
        A `trimesh.Trimesh` describing the gripper geometry in the `panda_hand`
        link frame (with optional origin translation applied).
    """

    if joint_offset < 0.0:
        raise ValueError("Finger joint offset cannot be negative")

    left_finger = _build_finger(sign=+1.0, joint_offset=joint_offset)
    right_finger = _build_finger(sign=-1.0, joint_offset=joint_offset)

    meshes = [left_finger, right_finger]

    if hand_mesh_path is not None:
        meshes.append(_load_hand_mesh(hand_mesh_path))

    gripper_mesh = trimesh.util.concatenate(meshes)

    # Apply origin translation if provided
    if origin_translation is not None:
        translation = np.asarray(origin_translation, dtype=float)
        if translation.shape != (3,):
            raise ValueError(f"Origin translation must be a 3D vector [x, y, z], got shape {translation.shape}")
        gripper_mesh.apply_translation(translation)

    return gripper_mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Panda gripper mesh from URDF collision boxes.")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("panda_gripper_collision.stl"),
        help="Destination mesh file. Format is inferred from the extension.",
    )
    parser.add_argument(
        "--finger-offset",
        type=float,
        default=0.02,
        help="Prismatic joint displacement for each finger in metres (0.0 – 0.04).",
    )
    parser.add_argument(
        "--hand-mesh",
        type=pathlib.Path,
        help="Optional path to an existing Panda hand mesh to merge with the generated fingers.",
    )
    parser.add_argument(
        "--origin-translation",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Translation vector [x, y, z] in metres to shift the mesh origin. "
             "Example: --origin-translation 0.01 0.02 -0.005",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.finger_offset > 0.04:
        raise ValueError("Finger offset exceeds physical gripper limit of 0.04 m")

    output_path: pathlib.Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hand_mesh_path: Optional[pathlib.Path] = args.hand_mesh
    if hand_mesh_path is not None:
        hand_mesh_path = hand_mesh_path.expanduser().resolve()

    gripper_mesh = build_gripper_mesh(
        args.finger_offset,
        hand_mesh_path,
        origin_translation=args.origin_translation,
    )
    gripper_mesh.export(output_path)

    origin_info = ""
    if args.origin_translation is not None:
        origin_info = f" (origin translated by {args.origin_translation})"
    print(f"Exported Panda gripper mesh to {output_path}{origin_info}")


if __name__ == "__main__":
    main()

