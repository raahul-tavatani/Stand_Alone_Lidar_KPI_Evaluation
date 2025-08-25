# evaluation/generate_target_vicinities.py

import os
import re
import json
import argparse
from typing import Tuple, List
from config.generate_test_config import compute_target_vicinity, compute_control_volume

_ROT_RE = re.compile(r"rotation_\d+_pitch(\d+)_yaw(\d+)_roll(\d+)", re.IGNORECASE)

def _is_distance_folder(name: str) -> bool:
    return bool(re.match(r"^\s*\d+\s*m\s*$", name.replace("\\", "/")))

def _parse_rotation_from_name(name: str) -> Tuple[float, float, float]:
    m = _ROT_RE.search(name)
    if not m:
        raise ValueError(f"Invalid rotation folder name: {name}")
    pitch, yaw, roll = map(float, m.groups())
    return pitch, yaw, roll

def _list_immediate_subdirs(path: str) -> List[str]:
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def _write_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def generate_for_all(
    root_dir: str,
    grade: str = "A",
    default_rotation: Tuple[float, float, float] = (90.0, 0.0, 0.0),
    frame_rate_hz: float = 10.0,
):
    """
    Generates target_vicinity.json (and control_volume.json) for:
      (A) rotation-first layout: <root>/rotation_.../<distance>m/
      (B) flat layout:           <root>/<distance>m/

    For flat layout we use default_rotation (pitch,yaw,roll) for the target plane.
    """
    root_subdirs = _list_immediate_subdirs(root_dir)

    # Decide layout
    has_rotation = any(_ROT_RE.search(d) for d in root_subdirs)
    has_flat_dist = any(_is_distance_folder(d) for d in root_subdirs)

    if has_rotation:
        # (A) rotation-first layout
        for rot_name in root_subdirs:
            rot_path = os.path.join(root_dir, rot_name)
            if not _ROT_RE.search(rot_name):
                continue
            pitch, yaw, roll = _parse_rotation_from_name(rot_name)

            for sub in _list_immediate_subdirs(rot_path):
                if not _is_distance_folder(sub):
                    continue
                dist_m = int(re.findall(r"(\d+)", sub)[0])
                dist_path = os.path.join(rot_path, sub)

                vic = compute_target_vicinity(dist_m, grade, (pitch, yaw, roll))
                _write_json(os.path.join(dist_path, "target_vicinity.json"), vic)

                ctl = compute_control_volume(
                    target_distance_m=dist_m,
                    target_vicinity_corners_xyz=vic.get("corner_points_xyz"),
                    frame_rate_hz=frame_rate_hz,
                )
                _write_json(os.path.join(dist_path, "control_volume.json"), ctl)

                print(f"[OK] Wrote: {os.path.join(dist_path, 'target_vicinity.json')}")
                print(f"[OK] Wrote: {os.path.join(dist_path, 'control_volume.json')}")

    elif has_flat_dist:
        # (B) flat layout: distances directly in root
        pitch, yaw, roll = default_rotation
        for sub in root_subdirs:
            if not _is_distance_folder(sub):
                continue
            dist_m = int(re.findall(r"(\d+)", sub)[0])
            dist_path = os.path.join(root_dir, sub)

            vic = compute_target_vicinity(dist_m, grade, (pitch, yaw, roll))
            _write_json(os.path.join(dist_path, "target_vicinity.json"), vic)

            ctl = compute_control_volume(
                target_distance_m=dist_m,
                target_vicinity_corners_xyz=vic.get("corner_points_xyz"),
                frame_rate_hz=frame_rate_hz,
            )
            _write_json(os.path.join(dist_path, "control_volume.json"), ctl)

            print(f"[OK] Wrote: {os.path.join(dist_path, 'target_vicinity.json')}")
            print(f"[OK] Wrote: {os.path.join(dist_path, 'control_volume.json')}")

    else:
        print(f"[WARN] No rotation folders or distance folders found under: {root_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--grade", default="A")
    parser.add_argument("--frame_rate_hz", type=float, default=10.0)
    args = parser.parse_args()
    generate_for_all(args.root, grade=args.grade, frame_rate_hz=args.frame_rate_hz)
