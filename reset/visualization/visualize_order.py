"""
Visualize manipulation order from saved JSON file.

Reads a manipulation_order.json file and renders a video with action labels overlayed
on the demonstration frames.
"""
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Try to import helper functions from extract_grasp
try:
    from reset.extract_from_demo.extract_grasp import _prepare_rgb_frames  # type: ignore[import]
except Exception:
    try:
        from extract_grasp import _prepare_rgb_frames  # type: ignore[import]
    except Exception:
        _prepare_rgb_frames = None


def _iter_annotation_files(dataset_path: Path):
    """Iterate over annotation files."""
    if dataset_path.is_file():
        yield dataset_path
        return

    if dataset_path.suffix == ".pkl":
        yield dataset_path
        return

    for pkl_path in sorted(dataset_path.rglob("*.pkl")):
        if pkl_path.is_file():
            yield pkl_path


def _render_action_video(
    observations: List[Dict],
    actions: List[Dict],
    window_title: str,
    play_speed_hz: float = 30.0,
):
    """
    Render demonstration video with action labels overlayed.
    
    Args:
        observations: List of observation dictionaries
        actions: List of action dictionaries with start_step and end_step
        window_title: Title for the video window
        play_speed_hz: Playback speed in Hz
    """
    if _prepare_rgb_frames is None:
        raise ImportError("_prepare_rgb_frames not available from extract_grasp")
    
    try:
        import cv2  # type: ignore[import]
    except ImportError:
        raise ImportError("cv2 is required for video rendering. Install with: pip install opencv-python")
    
    wait_ms = max(1, int(1000.0 / max(play_speed_hz, 1e-3)))
    
    # Build a mapping from frame index to active action
    frame_to_action = {}
    for action in actions:
        start_step = action.get("start_step", 0)
        end_step = action.get("end_step", len(observations) - 1)
        for step_idx in range(start_step, end_step + 1):
            if step_idx not in frame_to_action:
                frame_to_action[step_idx] = []
            frame_to_action[step_idx].append(action)
    
    for step_idx, obs in enumerate(observations):
        frames = _prepare_rgb_frames(obs)
        if not frames:
            continue
        
        bgr_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
        frame_bgr = np.hstack(bgr_frames)
        
        # Get active actions for this frame
        active_actions = frame_to_action.get(step_idx, [])
        
        display = np.ascontiguousarray(frame_bgr)
        
        # Display frame number
        cv2.putText(
            display,
            f"frame: {step_idx}",
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=(0, 255, 255),
            thickness=2,
        )
        
        # Display active actions
        y_offset = 60
        if active_actions:
            for action_idx, action in enumerate(active_actions):
                action_type = action["type"]
                
                if action_type == "MOVE":
                    action_text = f"MOVE: {action['grasped_part']}"
                    color = (0, 255, 0)  # Green
                elif action_type == "INTERACT":
                    action_text = f"INTERACT: {action['target_part']} <-> {action['base_part']}"
                    color = (0, 165, 255)  # Orange
                else:
                    action_text = f"{action_type}"
                    color = (255, 255, 255)  # White
                
                cv2.putText(
                    display,
                    action_text,
                    org=(10, y_offset),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=color,
                    thickness=2,
                )
                y_offset += 30
        else:
            cv2.putText(
                display,
                "action: none",
                org=(10, y_offset),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(128, 128, 128),
                thickness=2,
            )
        
        cv2.imshow(window_title, display)
        key = cv2.waitKey(wait_ms)
        if key in (27, ord("q")):
            break
    
    cv2.destroyWindow(window_title)


def find_dataset_file(order_file_path: Path) -> Optional[Path]:
    """
    Try to find the dataset file associated with the order file.
    
    Looks for .pkl files in the same directory or parent directories.
    """
    # Check same directory
    parent_dir = order_file_path.parent
    pkl_files = list(parent_dir.glob("*.pkl"))
    if pkl_files:
        # Return the first one (or could be smarter about matching)
        return pkl_files[0]
    
    # Check parent directories
    for parent in parent_dir.parents:
        pkl_files = list(parent.glob("*.pkl"))
        if pkl_files:
            return pkl_files[0]
    
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize manipulation order from saved JSON file"
    )
    parser.add_argument(
        "--order-file",
        type=Path,
        required=True,
        help="Path to manipulation_order.json file",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to dataset directory or .pkl file (auto-detected if not provided)",
    )
    parser.add_argument(
        "--play-speed-hz",
        type=float,
        default=30.0,
        help="Playback speed for rendered video in Hz (default: 30.0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load order file
    order_file_path = Path(args.order_file).expanduser().resolve()
    if not order_file_path.is_file():
        raise FileNotFoundError(f"Order file not found: {order_file_path}")
    
    print(f"Loading order file: {order_file_path}")
    with open(order_file_path, "r", encoding="utf-8") as f:
        order_data = json.load(f)
    
    furniture_name = order_data.get("furniture")
    actions = order_data.get("actions", [])
    num_actions = order_data.get("num_actions", len(actions))
    
    print(f"Furniture: {furniture_name}")
    print(f"Number of actions: {num_actions}")
    
    # Find dataset file
    if args.dataset is not None:
        dataset_path = Path(args.dataset).expanduser().resolve()
    else:
        # Try to auto-detect
        dataset_path = find_dataset_file(order_file_path)
        if dataset_path is None:
            raise FileNotFoundError(
                "Could not find dataset file. Please specify --dataset or ensure "
                "a .pkl file exists in the same directory as the order file."
            )
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Load dataset
    if dataset_path.is_file() and dataset_path.suffix == ".pkl":
        annotation_files = [dataset_path]
    else:
        annotation_files = list(_iter_annotation_files(dataset_path))
    
    if not annotation_files:
        raise FileNotFoundError(f"No .pkl files found at {dataset_path}")
    
    # Load first annotation file
    annotation_path = annotation_files[0]
    print(f"Using annotation file: {annotation_path}")
    
    with open(annotation_path, "rb") as f:
        data = pickle.load(f)
    
    observations = data.get("observations", [])
    if not observations:
        raise ValueError("No observations found in dataset")
    
    print(f"Loaded {len(observations)} observations")
    
    # Print action summary
    print("\nActions:")
    for i, action in enumerate(actions):
        action_type = action["type"]
        if action_type == "MOVE":
            print(
                f"  {i+1}. MOVE: {action['grasped_part']} "
                f"(steps {action.get('start_step', '?')}-{action.get('end_step', '?')})"
            )
        elif action_type == "INTERACT":
            print(
                f"  {i+1}. INTERACT: {action['target_part']} with {action['base_part']} "
                f"(steps {action.get('start_step', '?')}-{action.get('end_step', '?')})"
            )
    
    # Render video
    window_title = f"{furniture_name} - Manipulation Order"
    if hasattr(annotation_path, "stem"):
        window_title += f" - {annotation_path.stem}"
    
    print(f"\nRendering video with action labels...")
    print("Press 'q' or ESC to quit the video")
    
    _render_action_video(
        observations,
        actions,
        window_title=window_title,
        play_speed_hz=args.play_speed_hz,
    )
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

