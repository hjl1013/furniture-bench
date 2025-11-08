"""Pose and quaternion utility functions."""
from typing import Tuple

import numpy as np

from furniture_bench.utils import transform as T


def compute_relative_pose(ee_pose: np.ndarray, part_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute relative pose of part in gripper frame.
    
    Args:
        ee_pose: End-effector pose vector [x,y,z,qx,qy,qz,qw] or matrix (4x4)
        part_pose: Part pose vector [x,y,z,qx,qy,qz,qw] or matrix (4x4)
    
    Returns:
        Tuple of (relative_position, relative_quaternion)
    """
    if ee_pose.shape == (7,):
        ee_mat = T.pose2mat(ee_pose)
    else:
        ee_mat = ee_pose
    
    if part_pose.shape == (7,):
        part_mat = T.pose2mat(part_pose)
    else:
        part_mat = part_pose
    
    rel_mat = np.linalg.inv(ee_mat) @ part_mat
    rel_pos = rel_mat[:3, 3]
    rel_quat = T.mat2quat(rel_mat[:3, :3])
    return rel_pos, rel_quat


def compute_relative_pose_between_parts(a_pose_vec: np.ndarray, b_pose_vec: np.ndarray) -> np.ndarray:
    """
    Compute relative pose of B in A's frame.
    
    Args:
        a_pose_vec: Part A pose vector [x,y,z,qx,qy,qz,qw]
        b_pose_vec: Part B pose vector [x,y,z,qx,qy,qz,qw]
    
    Returns:
        Relative pose vector [x,y,z,qx,qy,qz,qw] of B in A's frame
    """
    a_mat = T.pose2mat(a_pose_vec)
    b_mat = T.pose2mat(b_pose_vec)
    rel = np.linalg.inv(a_mat) @ b_mat
    pos, quat = T.mat2pose(rel)
    return np.concatenate([pos, quat])


def pose_vec_to_mat(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Convert position and quaternion to 4x4 transformation matrix."""
    pose_vec = np.concatenate([pos, quat])
    return T.pose2mat(pose_vec)


def quat_normalize(quat: np.ndarray) -> np.ndarray:
    """Normalize quaternion."""
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return quat
    return quat / norm

