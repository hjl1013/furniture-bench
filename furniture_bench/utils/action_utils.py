from typing import Literal

import numpy as np

import furniture_bench.utils.transform as T


ActionRotRepr = Literal["quat", "axis", "rot_6d"]

_EPS = 1e-8


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < _EPS:
        return v.copy()
    return v / norm


def _rot6d_to_matrix(rot_6d: np.ndarray) -> np.ndarray:
    a1 = _normalize(rot_6d[:3])
    a2 = rot_6d[3:6] - np.dot(a1, rot_6d[3:6]) * a1
    a2 = _normalize(a2)
    a3 = np.cross(a1, a2)
    return np.stack([a1, a2, a3], axis=-1)


def _matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([matrix[:, 0], matrix[:, 1]])


def _quat_to_repr(quat: np.ndarray, repr_type: ActionRotRepr) -> np.ndarray:
    if repr_type == "quat":
        return quat
    if repr_type == "axis":
        return T.quat2axisangle(quat)
    if repr_type == "rot_6d":
        rot_mat = T.quat2mat(quat)
        return _matrix_to_rot6d(rot_mat)
    raise ValueError(f"Unsupported rotation representation: {repr_type}")


def _repr_to_quat(value: np.ndarray, repr_type: ActionRotRepr) -> np.ndarray:
    if repr_type == "quat":
        quat = value
    elif repr_type == "axis":
        quat = T.axisangle2quat(value)
    elif repr_type == "rot_6d":
        rot_mat = _rot6d_to_matrix(value)
        quat = T.mat2quat(rot_mat)
    else:
        raise ValueError(f"Unsupported rotation representation: {repr_type}")
    quat = quat / np.linalg.norm(quat)
    if quat[3] < 0:
        quat = -quat
    return quat


def delta_to_absolute(
    action: np.ndarray,
    ee_pos: np.ndarray,
    ee_quat: np.ndarray,
    repr_type: ActionRotRepr,
) -> np.ndarray:
    """Convert relative (delta) action to absolute pose action."""
    action = np.asarray(action).copy()
    abs_pos = ee_pos + action[:3]
    delta_quat = _repr_to_quat(action[3:-1], repr_type)
    abs_quat = T.quat_multiply(ee_quat, delta_quat)
    abs_quat /= np.linalg.norm(abs_quat)
    if abs_quat[3] < 0:
        abs_quat = -abs_quat
    abs_rot = _quat_to_repr(abs_quat, repr_type)
    return np.concatenate([abs_pos, abs_rot, np.array([action[-1]])])


def absolute_to_delta(
    action: np.ndarray,
    ee_pos: np.ndarray,
    ee_quat: np.ndarray,
    repr_type: ActionRotRepr,
) -> np.ndarray:
    """Convert absolute pose action to relative (delta) action."""
    action = np.asarray(action).copy()
    target_pos = action[:3]
    target_rot_repr = action[3:-1]
    target_quat = _repr_to_quat(target_rot_repr, repr_type)
    delta_pos = target_pos - ee_pos
    inv_quat = T.quat_inverse(ee_quat)
    delta_quat = T.quat_multiply(inv_quat, target_quat)
    delta_quat /= np.linalg.norm(delta_quat)
    if delta_quat[3] < 0:
        delta_quat = -delta_quat
    delta_rot = _quat_to_repr(delta_quat, repr_type)
    return np.concatenate([delta_pos, delta_rot, np.array([action[-1]])])


def absolute_from_observation(
    action: np.ndarray, observation: dict, repr_type: ActionRotRepr
) -> np.ndarray:
    """Convert relative action to absolute using observation dict."""
    ee_pos = observation["robot_state"]["ee_pos"]
    ee_quat = observation["robot_state"]["ee_quat"]
    return delta_to_absolute(action, ee_pos, ee_quat, repr_type)


