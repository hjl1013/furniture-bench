"""Instantiate FurnitureSim-v0 and test various functionalities."""

import argparse
import pickle

import furniture_bench

import gym
import cv2
import torch
import numpy as np

from furniture_bench.utils.action_utils import absolute_to_delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--furniture", default="square_table")
    parser.add_argument(
        "--file-path", help="Demo path to replay (data directory or pickle)"
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
        help="Use Viser for rendering.",
    )

    parser.add_argument(
        "--reverse-action",
        action="store_true",
        help="Reverse the action trajectory.",
    )
    parser.add_argument(
        "--absolute-action",
        action="store_true",
        help="Treat replayed actions as absolute end-effector targets.",
    )

    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args()

    # Set seed.
    import random
    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    # Initialize FurnitureSim.
    ob = env.reset()
    done = False

    def action_tensor(ac):
        if isinstance(ac, (list, np.ndarray)):
            return torch.tensor(ac).float().to(env.device)

        ac = ac.clone()
        if len(ac.shape) == 1:
            ac = ac[None]
        return ac.tile(args.num_envs, 1).float().to(env.device)

    def _to_numpy(array):
        if isinstance(array, torch.Tensor):
            return array.detach().cpu().numpy().squeeze()
        return np.asarray(array).squeeze()

    def to_delta_action(abs_action, robot_state):
        ee_pos = _to_numpy(robot_state["ee_pos"])
        ee_quat = _to_numpy(robot_state["ee_quat"])
        return absolute_to_delta(_to_numpy(abs_action), ee_pos, ee_quat, args.act_rot_repr)

    def prepare_action(action, robot_state):
        if args.absolute_action:
            delta = to_delta_action(action, robot_state)
            return action_tensor(delta)
        return action_tensor(action)

    # Rollout one episode with a selected policy:
    if args.input_device is not None:
        # Teleoperation.
        device_interface = furniture_bench.device.make(args.input_device)

        while not done:
            action, _ = device_interface.get_action()
            action = action_tensor(action)
            ob, rew, done, _ = env.step(action)

    elif args.no_action or args.init_assembled:
        # Execute 0 actions.
        while True:
            if args.act_rot_repr == "quat":
                ac = action_tensor([0, 0, 0, 0, 0, 0, 1, -1])
            else:
                ac = action_tensor([0, 0, 0, 0, 0, 0, -1])
            ob, rew, done, _ = env.step(ac)
    elif args.random_action:
        # Execute randomly sampled actions.
        import tqdm

        pbar = tqdm.tqdm()
        while True:
            ac = action_tensor(env.action_space.sample())
            ob, rew, done, _ = env.step(ac)
            pbar.update(args.num_envs)

    elif args.file_path is not None:
        # Play actions in the demo.
        with open(args.file_path, "rb") as f:
            data = pickle.load(f)
        for ac in data["actions"]:
            ac_tensor = prepare_action(ac, ob["robot_state"])
            ob, rew, done, _ = env.step(ac_tensor)
    elif args.scripted:
        # Execute hard-coded assembly script.
        while not done:
            action, skill_complete = env.get_assembly_action()
            action = action_tensor(action)
            ob, rew, done, _ = env.step(action)
    elif args.replay_path and args.reverse_action:
        # Replay the trajectory in reverse.
        with open(args.replay_path, "rb") as f:
            data = pickle.load(f)
        env.reset_to([data["observations"][-1]])  # reset to the last observation.
        ob = env.get_observation()
        for ac in reversed(data["actions"]):
            ac_tensor = prepare_action(ac, ob["robot_state"])
            ob, rew, done, _ = env.step(ac_tensor)
    elif args.replay_path:
        # Replay the trajectory.
        with open(args.replay_path, "rb") as f:
            data = pickle.load(f)
        env.reset_to([data["observations"][0]])  # reset to the first observation.
        ob = env.get_observation()
        for ac in data["actions"]:
            ac_tensor = prepare_action(ac, ob["robot_state"])
            ob, rew, done, _ = env.step(ac_tensor)
    else:
        raise ValueError(f"No action specified")

    print("done")


if __name__ == "__main__":
    main()
