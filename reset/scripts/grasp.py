import numpy as np

def get_grasp_eef_pose(object_name: str, pose: np.ndarray, mode: int = 0):
    """
    Grasp end-effector pose prediction
    Args:
        object_name: name of the object to grasp
        pose: pose of the object to grasp
        mode: mode of the grasp prediction (there are multiple ways to grasp the object)
    Returns:
        grasp_eef_pose: grasp end-effector pose of the object
    """
    pass