# evaluation/run_evaluation.py

import os
import json
import re
from utils.lidar_utils import generate_cluster_map_from_folder
from utils.kpi_utils import evaluate_kpis

def find_distance_folders(root_dir):
    """Return list of (rotation_folder, distance_folder, distance_meters)"""
    rotation_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                        if os.path.isdir(os.path.join(root_dir, d))]

    distance_folders = []
    for rot_folder in rotation_folders:
        for sub in os.listdir(rot_folder):
            full = os.path.join(rot_folder, sub)
            if os.path.isdir(full) and sub.endswith("m"):
                try:
                    dist = int(sub.replace("m", ""))
                    distance_folders.append((rot_folder, full, dist))
                except ValueError:
                    continue
    return distance_folders


def count_pcd_files(folder_path):
    """Count how many frame_XXX.pcd files exist in a folder"""
    return len([
        f for f in os.listdir(folder_path)
        if re.match(r"frame_\d+\.pcd", f)
    ])


def run_all_evaluations(root_dir="outputs/multidomain_test"):
    print(f"[INFO] Starting KPI evaluation in: {root_dir}")
    results = []
    distance_folders = find_distance_folders(root_dir)

    if not distance_folders:
        print(f"[WARN] No distance folders found in: {root_dir}")
        return

    for rot_path, dist_path, distance in sorted(distance_folders):
        rotation_name = os.path.basename(rot_path)
        print(f"\n[DEBUG] Processing → {rotation_name} @ {distance}m")

        try:
            cluster_map = generate_cluster_map_from_folder(dist_path)
            total_frames = count_pcd_files(dist_path)

            if total_frames == 0:
                print(f"[SKIP] No frames in {dist_path}")
                continue

            kpis = evaluate_kpis(
                cluster_map,
                target_distance=distance,
                total_frames=total_frames
            )
            result = {
                "rotation": rotation_name,
                "distance_m": distance,
                "frame_count": total_frames,
                **kpis
            }
            results.append(result)
            print(f"[OK] Evaluated: {rotation_name} @ {distance}m ({total_frames} frames)")
        except Exception as e:
            print(f"[ERROR] Skipped {rotation_name} @ {distance}m → {e}")

    # Save result JSON
    out_path = os.path.join(root_dir, "kpi_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[INFO] ✅ KPI evaluation complete.")
    print(f"[INFO] 📝 Results saved to: {out_path}")
    print(f"[INFO] 🧪 Total test cases evaluated: {len(results)}")
