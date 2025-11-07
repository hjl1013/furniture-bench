"""
Extract initial states from demonstration data files.

This script processes .pkl files and extracts the initial state (first observation)
from each file, saving them as a list to initial_state.pkl in the specified directory.
"""
import pickle
from pathlib import Path
import argparse
from typing import List, Dict
import numpy as np


def _iter_annotation_files(dataset_path: Path):
    """Iterate over all .pkl annotation files in the dataset path."""
    if dataset_path.is_file():
        yield dataset_path
        return

    if dataset_path.suffix == ".pkl":
        yield dataset_path
        return

    for pkl_path in sorted(dataset_path.rglob("*.pkl")):
        if pkl_path.is_file():
            yield pkl_path


def extract_initial_states(
    annotation_paths: List[Path],
) -> List[Dict]:
    """
    Extract initial states from annotation files.
    
    Args:
        annotation_paths: List of paths to .pkl annotation files
    
    Returns:
        List of initial state dictionaries, each containing:
        - robot_state: Dictionary with robot state information
        - parts_poses: Array of part poses (in AprilTag frame)
    """
    initial_states = []
    
    for annotation_path in annotation_paths:
        try:
            with open(annotation_path, "rb") as f:
                data = pickle.load(f)
            
            observations = data.get("observations", [])
            if not observations:
                print(f"Warning: No observations found in {annotation_path}")
                continue
            
            # Extract initial state from first observation
            first_obs = observations[0]
            robot_state = first_obs.get("robot_state")
            parts_poses = first_obs.get("parts_poses")
            
            if robot_state is None or parts_poses is None:
                print(f"Warning: Missing robot_state or parts_poses in {annotation_path}")
                continue
            
            initial_state = {
                "robot_state": robot_state,
                "parts_poses": parts_poses,
            }
            initial_states.append(initial_state)
            
        except Exception as e:
            print(f"Error processing {annotation_path}: {e}")
            continue
    
    return initial_states


def main():
    parser = argparse.ArgumentParser(
        description="Extract initial states from demonstration data files"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory containing .pkl files or a single .pkl file"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to a single .pkl file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory path to save initial_state.pkl"
    )
    args = parser.parse_args()

    # Determine input path
    if args.data_dir is not None:
        input_path = Path(args.data_dir)
    elif args.data_path is not None:
        input_path = Path(args.data_path)
    else:
        raise ValueError("Either --data-dir or --data-path must be specified.")

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # Get all annotation files
    annotation_paths = list(_iter_annotation_files(input_path))
    if not annotation_paths:
        raise FileNotFoundError(f"No .pkl files found in {input_path}")

    print(f"Found {len(annotation_paths)} annotation file(s)")
    
    # Extract initial states
    initial_states = extract_initial_states(
        annotation_paths
    )
    
    if not initial_states:
        raise ValueError("No initial states could be extracted from the files")
    
    print(f"Extracted {len(initial_states)} initial state(s)")
    
    # Save to output directory
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle file (standard format for data)
    output_path = output_dir / "initial_state.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(initial_states, f)
    
    print(f"Saved {len(initial_states)} initial state(s) to {output_path}")


if __name__ == "__main__":
    main()