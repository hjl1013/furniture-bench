#!/usr/bin/env python3
"""
Test script to visualize initial state using get_initial_state from get_info.py.

This script loads an initial state using the get_initial_state function and displays it
in the FurnitureSim environment.
"""
import argparse
import time
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import gym
import furniture_bench
from reset.scripts.get_info import get_initial_state
import torch

def main():
    parser = argparse.ArgumentParser(
        description="Test visualization of initial state using get_initial_state"
    )
    parser.add_argument(
        "--furniture",
        default="cabinet",
        help="Furniture type to visualize"
    )
    parser.add_argument(
        "--initial-state-path",
        type=str,
        default=None,
        help="Optional path to initial state file (defaults to extracted_info/initial_state.pkl)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no GUI)"
    )
    parser.add_argument(
        "--display-time",
        type=float,
        default=5.0,
        help="Time in seconds to display the initial state (default: 5.0)"
    )
    parser.add_argument(
        "--env-id",
        default="FurnitureSim-v0",
        help="Environment id of FurnitureSim"
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
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--high-res",
        action="store_true",
        help="Use high resolution images for the camera input."
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
        "--use-viser",
        action="store_true",
        help="Use viser for visualization instead of Open3D"
    )
    
    args = parser.parse_args()

    print("=" * 60)
    print("Testing get_initial_state function")
    print("=" * 60)
    
    # Load initial state using get_initial_state
    print(f"\nLoading initial state...")
    try:
        initial_state_path = Path(args.initial_state_path) if args.initial_state_path else None
        initial_state = get_initial_state(initial_state_path=initial_state_path)
        print(f"Successfully loaded initial state")
        print(f"  - Has robot_state: {'robot_state' in initial_state}")
        print(f"  - Has parts_poses: {'parts_poses' in initial_state}")
        if 'parts_poses' in initial_state:
            parts_poses = np.asarray(initial_state['parts_poses'])
            print(f"  - Number of parts: {len(parts_poses)}")
    except Exception as e:
        print(f"ERROR: Failed to load initial state: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Create FurnitureSim environment
    print(f"\nCreating FurnitureSim environment (furniture={args.furniture})...")
    try:
        env = gym.make(
            args.env_id,
            furniture=args.furniture,
            num_envs=args.num_envs,
            resize_img=not args.high_res,
            init_assembled=False,
            record=False,
            headless=args.headless,
            save_camera_input=False,
            randomness=args.randomness,
            high_random_idx=args.high_random_idx,
            act_rot_repr=args.act_rot_repr,
            compute_device_id=args.compute_device_id,
            graphics_device_id=args.graphics_device_id,
            use_viser=args.use_viser,
        )
        ob = env.reset()
        print("Environment created successfully")
    except Exception as e:
        print(f"ERROR: Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Reset environment to the initial state
    print(f"\nResetting environment to initial state...")
    try:
        # env.reset_to expects a list of initial states
        initial_state_list = [initial_state]
        obs = env.reset_to(initial_state_list)
        print("Environment reset successfully")
    except Exception as e:
        print(f"ERROR: Failed to reset environment: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Display the initial state
    print(f"\nDisplaying initial state for {args.display_time} seconds...")
    print("Press Ctrl+C to exit early")
    
    def action_tensor(ac):
        """Convert action to tensor format."""
        if isinstance(ac, (list, np.ndarray)):
            return torch.tensor(ac).float().to(env.device)
        ac = ac.clone()
        if len(ac.shape) == 1:
            ac = ac[None]
        return ac.tile(args.num_envs, 1).float().to(env.device)

    # Zero action (no movement, gripper closed)
    if args.act_rot_repr == "quat":
        zero_action = [0, 0, 0, 0, 0, 0, 1, -1]  # [x, y, z, qx, qy, qz, qw, gripper]
    else:
        zero_action = [0, 0, 0, 0, 0, 0, -1]  # [x, y, z, rx, ry, rz, gripper]

    start_time = time.time()
    try:
        while time.time() - start_time < args.display_time:
            ac = action_tensor(zero_action)
            obs, rew, done, info = env.step(ac)
            
            # Small sleep to avoid busy waiting
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nERROR during display: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())

