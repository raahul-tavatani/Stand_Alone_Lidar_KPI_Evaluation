# visualize_clusters.py

import open3d as o3d
import numpy as np
import json
import argparse
import os


def plot_cluster_centers(cluster_map, radius=0.1, color=[1, 0, 0]):
    """Render clusters as hollow (wireframe) spheres"""
    spheres = []
    for cluster in cluster_map:
        az, el, r = cluster["azimuth"], cluster["elevation"], cluster["range"]
        az_rad, el_rad = np.radians(az), np.radians(el)

        x = r * np.cos(el_rad) * np.cos(az_rad)
        y = r * np.cos(el_rad) * np.sin(az_rad)
        z = r * np.sin(el_rad)

        # Create wireframe sphere (converted to LineSet)
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=6)
        mesh_sphere.translate((x, y, z))
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color(color)
        line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_sphere)
        spheres.append(line_set)

    return spheres


def draw_target_vicinity_box(vicinity_path):
    with open(vicinity_path, 'r') as f:
        data = json.load(f)

    corner_pts = np.array(list(data["corner_points_xyz"].values()))
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(corner_pts))
    obb.color = (0.0, 1.0, 0.0)  # Green
    return obb


def visualize(folder_path):
    pcd_path = os.path.join(folder_path, "target_vicinity_points.pcd")
    cluster_map_path = os.path.join(folder_path, "cluster_map.json")
    vicinity_path = os.path.join(folder_path, "target_vicinity.json")

    # Load point cloud
    pcd = o3d.io.read_point_cloud(pcd_path)

    # Load cluster map
    with open(cluster_map_path, 'r') as f:
        cluster_map = json.load(f)

    # Cluster centers as wireframe spheres
    spheres = plot_cluster_centers(cluster_map, radius=0.1, color=[1, 0, 0])

    # Target vicinity bounding box
    obb = draw_target_vicinity_box(vicinity_path)

    print(f"[INFO] Visualizing {len(spheres)} clusters on point cloud with target vicinity box...")
    o3d.visualization.draw_geometries([pcd, obb, *spheres])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize cluster centers on filtered PCD")
    parser.add_argument("--folder", required=True,
                        help="Path to folder with target_vicinity_points.pcd, cluster_map.json, target_vicinity.json")
    args = parser.parse_args()

    visualize(args.folder)
