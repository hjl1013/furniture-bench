import pickle
from pathlib import Path
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--use-perception", action="store_true")
    args = parser.parse_args()

    if args.data_dir is not None:
        data_dir = Path(args.data_dir)
        file = os.path.join(args.data_dir, data_dir.name + ".pkl")
    elif args.data_path is not None:
        file = args.data_path
    else:
        raise ValueError("Either data_dir or data_path must be specified.")

    with open(file, "rb") as f:
        data = pickle.load(f)
        print(data['observations'][0].keys())
        print(data['observations'][0]['parts_poses'])

    if args.use_perception:
        raise NotImplementedError("Perception is not supported yet.")
    else:
        states = {}