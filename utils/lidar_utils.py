# utils/lidar_utils.py

import numpy as np
import open3d as o3d

def save_pointcloud(lidar_data, filename):
    # Parse raw LiDAR buffer into Nx4 (x, y, z, intensity)
    points = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
    points = points.reshape((-1, 4))

    # Extract XYZ and normalize intensity to [0,1]
    xyz = points[:, :3]
    intensity = np.clip(points[:, 3] / 255.0, 0.0, 1.0)
    color = np.tile(intensity.reshape(-1, 1), (1, 3))  # RGB fake color from intensity

    # Save using Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(color)

    # Auto-add .pcd extension if missing
    if not filename.endswith('.pcd'):
        filename += '.pcd'

    try:
        o3d.io.write_point_cloud(filename, pcd)
        print(f"[INFO] Saved point cloud with {len(points)} points → {filename}")
    except Exception as e:
        print(f"[ERROR] Could not save PCD: {e}")
