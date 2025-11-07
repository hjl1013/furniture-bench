import pickle
from pathlib import Path
import argparse
import os

import furniture_bench

import gym
import cv2
import torch
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--furniture", default="cabinet")
    parser.add_argument(
        "--dataset", help="Dataset path to replay (data directory or pickle)"
    )
    parser.add_argument(
        "--scripted", action="store_true", help="Execute hard-coded assembly script."
    )
    parser.add_argument("--no-action", action="store_true")
    parser.add_argument("--random-action", action="store_true")
    parser.add_argument(
        "--input-device",
        help="Device to control the robot.",
        choices=["keyboard", "oculus", "keyboard-oculus"],
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--init-assembled",
        action="store_true",
        help="Initialize the environment with the assembled furniture.",
    )
    parser.add_argument(
        "--save-camera-input",
        action="store_true",
        help="Save camera input of the simulator at the beginning of the episode.",
    )
    parser.add_argument(
        "--record", action="store_true", help="Record the video of the simulator."
    )
    parser.add_argument(
        "--high-res",
        action="store_true",
        help="Use high resolution images for the camera input.",
    )
    parser.add_argument(
        "--randomness",
        default="low",
        help="Randomness level of the environment.",
    )
    parser.add_argument(
        "--high-random-idx",
        default=0,
        type=int,
        help="The index of high_randomness.",
    )
    parser.add_argument(
        "--env-id",
        default="FurnitureSim-v0",
        help="Environment id of FurnitureSim",
    )
    parser.add_argument(
        "--replay-path", type=str, help="Path to the saved data to replay action."
    )

    parser.add_argument(
        "--act-rot-repr",
        type=str,
        help="Rotation representation for action space.",
        choices=["quat", "axis", "rot_6d"],
        default="quat",
    )

    parser.add_argument(
        "--compute-device-id",
        type=int,
        default=0,
        help="GPU device ID used for simulation.",
    )

    parser.add_argument(
        "--graphics-device-id",
        type=int,
        default=0,
        help="GPU device ID used for rendering.",
    )
    parser.add_argument(
        "--use-viser",
        action="store_true",
        help="Use viser for visualization instead of Open3D"
    )

    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args()

    # Create FurnitureSim environment.
    env = gym.make(
        args.env_id,
        furniture=args.furniture,
        num_envs=args.num_envs,
        resize_img=not args.high_res,
        init_assembled=args.init_assembled,
        record=args.record,
        headless=args.headless,
        save_camera_input=args.save_camera_input,
        randomness=args.randomness,
        high_random_idx=args.high_random_idx,
        act_rot_repr=args.act_rot_repr,
        compute_device_id=args.compute_device_id,
        graphics_device_id=args.graphics_device_id,
        use_viser=args.use_viser,
    )

    # dummy reset
    ob = env.reset()
    done = False

    data_paths = Path(args.dataset).glob("*")

    for data_path in data_paths:
        annotation_path = list(data_path.glob("*.pkl"))[0]

        # Initialize FurnitureSim to the initial state in data
        with open(annotation_path, "rb") as f:
            data = pickle.load(f)
            initial_state = [{
                "robot_state": data['observations'][0]['robot_state'],
                "parts_poses": data['observations'][0]['parts_poses'],
            }]
            ob = env.reset_to(initial_state)

        def action_tensor(ac):
            if isinstance(ac, (list, np.ndarray)):
                return torch.tensor(ac).float().to(env.device)

            ac = ac.clone()
            if len(ac.shape) == 1:
                ac = ac[None]
            return ac.tile(args.num_envs, 1).float().to(env.device)

        import time
        display_time = 1
        start_time = time.time()
        while time.time() - start_time < display_time:
            if args.act_rot_repr == "quat":
                ac = action_tensor([0, 0, 0, 0, 0, 0, 1, -1])
            else:
                ac = action_tensor([0, 0, 0, 0, 0, 0, -1])
            ob, rew, done, _ = env.step(ac)