# utils/lidar_utils.py

import numpy as np
import open3d as o3d
import json
import os
import pandas as pd
from utils.cluster_utils import track_clusters_data_driven


def save_pointcloud(lidar_data, filename):
    points = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape((-1, 4))
    xyz = points[:, :3]
    intensity = np.clip(points[:, 3] / 255.0, 0.0, 1.0)
    color = np.tile(intensity.reshape(-1, 1), (1, 3))
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(color)

    if not filename.endswith('.pcd'):
        filename += '.pcd'

    try:
        o3d.io.write_point_cloud(filename, pcd)
        print(f"[INFO] Saved point cloud with {len(points)} points → {filename}")
    except Exception as e:
        print(f"[ERROR] Could not save PCD: {e}")


def extract_points_in_target_vicinity(pcd_path: str, vicinity_json_path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    with open(vicinity_json_path, 'r') as f:
        vicinity = json.load(f)

    cp = vicinity.get("corner_points_xyz", {})
    if not isinstance(cp, dict) or len(cp) < 4:
        raise ValueError(f"[ERROR] Invalid corner_points_xyz in {vicinity_json_path}")

    # enforce deterministic key order and validate each corner (must be 3D)
    corners_list = []
    for k in sorted(cp.keys()):
        v = cp[k]
        if not (isinstance(v, (list, tuple)) and len(v) == 3):
            raise ValueError(f"[ERROR] Corner '{k}' malformed: {v}")
        corners_list.append([float(v[0]), float(v[1]), float(v[2])])

    corners = np.asarray(corners_list, dtype=float).reshape(-1, 3)
    if corners.shape[0] < 4:
        raise ValueError(f"[ERROR] Need ≥4 corners to define OBB, got {corners.shape[0]} in {vicinity_json_path}")

    obb = o3d.geometry.OrientedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(corners)
    )
    indices = obb.get_point_indices_within_bounding_box(
        o3d.utility.Vector3dVector(points)
    )
    return points[indices]





def generate_cluster_map_from_folder(folder_path: str, save_debug: bool = True):
    """
    Process all PCDs in a folder and return cluster map.
    Also optionally save filtered .pcd and cluster map table.
    """
    vicinity_path = os.path.join(folder_path, "target_vicinity.json")
    if not os.path.exists(vicinity_path):
        raise FileNotFoundError(f"[ERROR] target_vicinity.json not found in {folder_path}")

    import re
    pcd_files = sorted([
        f for f in os.listdir(folder_path)
        if re.match(r"frame_\d+\.pcd", f)
    ])
    if not pcd_files:
        raise RuntimeError(f"[ERROR] No PCD files found in {folder_path}")

    all_points_by_frame = {}
    all_filtered_points = []

    for filename in pcd_files:
        frame_id = int(re.findall(r"\d+", filename)[0])
        pcd_path = os.path.join(folder_path, filename)
        points = extract_points_in_target_vicinity(pcd_path, vicinity_path)

        # shape guard
        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
            print(f"[WARN] Bad points in {filename}: shape={getattr(points, 'shape', None)} → skip")
            continue
        if points.size > 0:
            all_points_by_frame[frame_id] = points.astype(np.float32, copy=False)
            all_filtered_points.append(points.astype(np.float32, copy=False))


    if not all_filtered_points:
        print(f"[WARN] No target-vicinity points found in: {folder_path}")
        return []

    # Save target-vicinity-only PCD (for visualization)
    if save_debug:
        all_points = np.vstack(all_filtered_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        o3d.io.write_point_cloud(os.path.join(folder_path, "target_vicinity_points.pcd"), pcd)

    # Run clustering
    cluster_map = track_clusters_data_driven(all_points_by_frame)
    if save_debug:
        # Save as JSON
        with open(os.path.join(folder_path, "cluster_map.json"), "w") as f:
            json.dump(cluster_map, f, indent=2)

        # Save as CSV
        df = pd.DataFrame(cluster_map)
        df.to_csv(os.path.join(folder_path, "cluster_map.csv"), index=False)

    return cluster_map
