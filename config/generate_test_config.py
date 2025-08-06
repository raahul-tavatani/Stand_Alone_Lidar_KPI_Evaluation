import os
import math
import json


def deg_to_rad(deg):
    return math.radians(deg)


def point_in_spherical(r, azimuth_rad, elevation_rad):
    """Convert spherical (r, azimuth, elevation) to Cartesian (x, y, z)"""
    x = r * math.cos(elevation_rad) * math.cos(azimuth_rad)
    y = r * math.cos(elevation_rad) * math.sin(azimuth_rad)
    z = r * math.sin(elevation_rad)
    return [x, y, z]




def rotate_point(point, pitch_deg, yaw_deg, roll_deg):
    """Rotate a 3D point using pitch, yaw, roll (in degrees)."""
    
    pitch = math.radians(90 - pitch_deg)
    yaw = math.radians( yaw_deg)
    roll = math.radians(roll_deg)

    # Rotation matrices
    Rx = [
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ]

    Ry = [
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ]

    Rz = [
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ]

    # Apply R = Rz * Ry * Rx
    def mat_mult(m, v):
        return [
            m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2],
            m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2],
            m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2],
        ]

    rotated = mat_mult(Rx, point)
    rotated = mat_mult(Ry, rotated)
    rotated = mat_mult(Rz, rotated)
    return rotated


def compute_target_vicinity(target_distance_m: float, grade: str, rotation: tuple = (0, 0, 0)):
    """
    Generate 8 corner points of the target vicinity as a frustum-like volume,
    shifted downward by 0.25 meters in Z, and rotated using the specified pitch, yaw, roll.
    """
    grade_tolerances = {
        "0": {"delta_r": 0.10, "delta_ang_deg": 0.0},
        "A": {"delta_r": 0.02, "delta_ang_deg": 0.2},
        "B": {"delta_r": 0.05, "delta_ang_deg": 0.5},
        "C": {"delta_r": 0.10, "delta_ang_deg": 1.0}
    }

    if grade not in grade_tolerances:
        raise ValueError(f"Unsupported grade: {grade}")

    delta = grade_tolerances[grade]
    delta_r = delta["delta_r"] * target_distance_m
    delta_ang_deg = delta["delta_ang_deg"]

    r_min = target_distance_m - delta_r
    r_max = target_distance_m + delta_r

    # Target is 1x1 m → half-size = 0.5 m
    half_target_size = 0.5

    core_azimuth_deg = math.degrees(math.atan(half_target_size / target_distance_m))
    delta_azimuth_deg = core_azimuth_deg + delta_ang_deg
    delta_elevation_deg = core_azimuth_deg
    delta_elevation_top_deg = delta_elevation_deg + delta_ang_deg

    # Convert degrees to radians
    azi_rad = deg_to_rad(delta_azimuth_deg)
    ele_rad_lower = -deg_to_rad(delta_elevation_deg)
    ele_rad_upper = deg_to_rad(delta_elevation_top_deg)

    # Define 8 corner points of the frustum in sensor frame
    corners = {
        "front_lower_left": point_in_spherical(r_min, -azi_rad, ele_rad_lower),
        "front_lower_right": point_in_spherical(r_min, azi_rad, ele_rad_lower),
        "front_upper_left": point_in_spherical(r_min, -azi_rad, ele_rad_upper),
        "front_upper_right": point_in_spherical(r_min, azi_rad, ele_rad_upper),
        "back_lower_left": point_in_spherical(r_max, -azi_rad, ele_rad_lower),
        "back_lower_right": point_in_spherical(r_max, azi_rad, ele_rad_lower),
        "back_upper_left": point_in_spherical(r_max, -azi_rad, ele_rad_upper),
        "back_upper_right": point_in_spherical(r_max, azi_rad, ele_rad_upper),
    }

    # Apply vertical shift and then rotation
    z_shift = 0
    pitch, yaw, roll = rotation
    center_of_rotation = [target_distance_m, 0.0, 0.0]  # center of rotation (LiDAR center or PCD center)
    for key, point in corners.items():
        point[2] += z_shift  # Apply vertical shift first (not_used)
        translated = [point[i] - center_of_rotation[i] for i in range(3)] # translate to origin
        rotated = rotate_point(translated, pitch, yaw, roll) #rotate about origin
        rotated_back = [rotated[i] + center_of_rotation[i] for i in range(3)] # push pack to the plane origin
        corners[key] = rotated_back
        


    return {
        "target_distance_m": target_distance_m,
        "grade": grade,
        "r_min": round(r_min, 3),
        "r_max": round(r_max, 3),
        "delta_azimuth_deg": round(delta_azimuth_deg, 4),
        "delta_elevation_deg": round(delta_elevation_deg, 4),
        "delta_elevation_top_deg": round(delta_elevation_top_deg, 4),
        "corner_points_xyz": corners
    }


def generate_target_vicinity_configs(output_root: str, grade: str, distances: list, rotation: tuple):
    """
    Generate `target_vicinity.json` files in each distance folder with rotation applied.
    """
    for dist in distances:
        vicinity_data = compute_target_vicinity(dist, grade, rotation)
        dist_folder = os.path.join(output_root, f"{int(dist)}m")
        os.makedirs(dist_folder, exist_ok=True)
        out_path = os.path.join(dist_folder, "target_vicinity.json")
        with open(out_path, 'w') as f:
            json.dump(vicinity_data, f, indent=2)
        print(f"[INFO] Saved target_vicinity.json at {out_path}")


def compute_control_volume(target_distance_m: float):
    """
    Compute control volume as a rectangular cuboid aligned with sensor's Cartesian system.
    """
    length = max(target_distance_m + 3.0, 3.0)
    half_width = 3.0
    height = 10.0

    corners = {
        "front_bottom_left":  [target_distance_m, -half_width, 0],
        "front_bottom_right": [target_distance_m, half_width, 0],
        "front_top_left":     [target_distance_m, -half_width, height],
        "front_top_right":    [target_distance_m, half_width, height],
        "back_bottom_left":   [-3.0, -half_width, 0],
        "back_bottom_right":  [-3.0, half_width, 0],
        "back_top_left":      [-3.0, -half_width, height],
        "back_top_right":     [-3.0, half_width, height],
    }

    return {
        "target_distance_m": target_distance_m,
        "length_m": length,
        "width_m": 2 * half_width,
        "height_m": height,
        "corner_points_xyz": corners
    }


def generate_control_volume_configs(output_root: str, distances: list):
    """
    Generate `control_volume.json` files in each distance folder.
    """
    for dist in distances:
        control_volume_data = compute_control_volume(dist)
        dist_folder = os.path.join(output_root, f"{int(dist)}m")
        os.makedirs(dist_folder, exist_ok=True)
        out_path = os.path.join(dist_folder, "control_volume.json")
        with open(out_path, 'w') as f:
            json.dump(control_volume_data, f, indent=2)
        print(f"[INFO] Saved control_volume.json at {out_path}")
