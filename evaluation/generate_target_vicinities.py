# generate_target_vicinities.py

import os
import re
import argparse
from config.generate_test_config import compute_target_vicinity
import json

def parse_rotation_from_path(path):
    match = re.search(r"rotation_\d+_pitch(\d+)_yaw(\d+)_roll(\d+)", path)
    if not match:
        raise ValueError(f"Invalid rotation folder name: {path}")
    pitch, yaw, roll = map(int, match.groups())
    return pitch, yaw, roll

def generate_for_all(root_dir: str, grade: str = "0"):
    for rotation in os.listdir(root_dir):
        rot_path = os.path.join(root_dir, rotation)
        if not os.path.isdir(rot_path):
            continue

        try:
            pitch, yaw, roll = parse_rotation_from_path(rotation)
        except ValueError as e:
            print(f"[WARN] {e}")
            continue

        for dist_folder in os.listdir(rot_path):
            if not dist_folder.endswith("m"):
                continue
            dist = int(dist_folder.replace("m", ""))
            full_path = os.path.join(rot_path, dist_folder)

            vicinity = compute_target_vicinity(dist, grade, (pitch, yaw, roll))
            out_path = os.path.join(full_path, "target_vicinity.json")
            with open(out_path, 'w') as f:
                json.dump(vicinity, f, indent=2)
            print(f"[OK] Wrote: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/multidomain_test")
    parser.add_argument("--grade", default="0")
    args = parser.parse_args()

    generate_for_all(args.root, args.grade)
