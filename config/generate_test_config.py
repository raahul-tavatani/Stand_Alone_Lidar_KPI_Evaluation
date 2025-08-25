import os
import math
import json
# add near the top of the file:
from typing import Optional, Dict, List, Any



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


def compute_control_volume(
    target_distance_m: float,
    target_vicinity_corners_xyz: Optional[Dict[str, List[float]]] = None,
    lateral_half_width_m: float = 3.0,
    extra_front_space_m: float = 3.0,
    vertical_margin_up_m: float = 100.0,
    frame_rate_hz: float = 10.0,
) -> Dict[str, Any]:
    """
    Control volume per DIN SAE 91471 (axis-aligned to sensor), with timing info:
      - x: [0, target_distance + extra_front_space]  (forward only, +3 m beyond target)
      - y: [-lateral_half_width, +lateral_half_width]  (≥ 6 m width total)
      - z: [lower edge of target, lower edge + vertical_margin_up]  (no upper restriction → large margin)
    Also embeds frame_rate_hz and frame_dt for KPI-3 (Hz).
    """
    # Timing
    fr = float(frame_rate_hz)
    frame_dt = (1.0 / fr) if fr > 0 else None

    # Longitudinal (sensor at x=0)
    x_min = 0.0
    x_max = float(target_distance_m + extra_front_space_m)

    # Lateral
    y_min = -float(lateral_half_width_m)
    y_max = +float(lateral_half_width_m)

    # Vertical: bottom at target lower edge if available; otherwise 0
    if target_vicinity_corners_xyz:
        zs = [float(v[2]) for v in target_vicinity_corners_xyz.values()]
        z_bottom = float(min(zs))
    else:
        z_bottom = 0.0
    z_top = z_bottom + float(vertical_margin_up_m)

    corners = {
        "front_bottom_left":  [x_max, y_min, z_bottom],
        "front_bottom_right": [x_max, y_max, z_bottom],
        "front_top_left":     [x_max, y_min, z_top],
        "front_top_right":    [x_max, y_max, z_top],
        "back_bottom_left":   [x_min, y_min, z_bottom],
        "back_bottom_right":  [x_min, y_max, z_bottom],
        "back_top_left":      [x_min, y_min, z_top],
        "back_top_right":     [x_min, y_max, z_top],
    }

    return {
        "target_distance_m": float(target_distance_m),
        "width_m": float(2 * lateral_half_width_m),
        "extra_front_space_m": float(extra_front_space_m),
        "z_bottom_m": float(z_bottom),
        "z_top_m": float(z_top),
        "frame_rate_hz": fr,        # used by evaluator to compute KPI-3 in Hz
        "frame_dt": frame_dt,       # evaluator also accepts this directly
        "corner_points_xyz": corners
    }



def generate_control_volume_configs(output_root: str, distances: list, frame_rate_hz: float = 10.0):
    """
    Generate control_volume.json files with timing info.
    """
    for dist in distances:
        dist_folder = os.path.join(output_root, f"{int(dist)}m")
        os.makedirs(dist_folder, exist_ok=True)

        # try to read target_vicinity to set z-bottom properly
        vicinity_path = os.path.join(dist_folder, "target_vicinity.json")
        vicinity_corners = None
        if os.path.exists(vicinity_path):
            with open(vicinity_path, "r") as f:
                v = json.load(f)
            vicinity_corners = v.get("corner_points_xyz")

        control_volume_data = compute_control_volume(
            target_distance_m=dist,
            target_vicinity_corners_xyz=vicinity_corners,
            frame_rate_hz=frame_rate_hz
        )

        out_path = os.path.join(dist_folder, "control_volume.json")
        with open(out_path, 'w') as f:
            json.dump(control_volume_data, f, indent=2)
        print(f"[INFO] Saved control_volume.json at {out_path}")
def compute_target_vicinity_centered(center_xyz, grade, target_distance_m=None):
    """
    Build a frustum-like vicinity around the LOS to center_xyz.
    Used by Angular-Separability: ONLY the back plane gets a vicinity.
    Leaves the original multi-domain vicinity function untouched.
    """
    import math

    cx, cy, cz = map(float, center_xyz)
    r_center = math.sqrt(cx*cx + cy*cy + cz*cz)

    # Use the caller-specified target distance for tolerance, or fall back to center range
    if target_distance_m is None:
        target_distance_m = r_center

    grade_tolerances = {
        "0": {"delta_r": 0.10, "delta_ang_deg": 0.0},
        "A": {"delta_r": 0.02, "delta_ang_deg": 0.2},
        "B": {"delta_r": 0.05, "delta_ang_deg": 0.5},
        "C": {"delta_r": 0.10, "delta_ang_deg": 1.0}
    }
    if grade not in grade_tolerances:
        raise ValueError(f"Unsupported grade: {grade}")

    tol = grade_tolerances[grade]
    delta_r = tol["delta_r"] * float(target_distance_m)
    r_min = max(0.0, r_center - delta_r)
    r_max = r_center + delta_r

    # 1x1 m target angular half-extent at this range (+ grade angular tolerance)
    half_target = 0.5
    core_half_deg = math.degrees(math.atan(half_target / max(r_center, 1e-6)))
    half_az = core_half_deg + tol["delta_ang_deg"]
    half_el_low = core_half_deg
    half_el_up  = core_half_deg + tol["delta_ang_deg"]

    az_center = math.degrees(math.atan2(cy, cx))
    el_center = math.degrees(math.atan2(cz, math.sqrt(cx*cx + cy*cy)))

    def sph_to_xyz(r, az_deg, el_deg):
        az = math.radians(az_deg); el = math.radians(el_deg)
        x = r * math.cos(el) * math.cos(az)
        y = r * math.cos(el) * math.sin(az)
        z = r * math.sin(el)
        return [x, y, z]

    corners = {
        "front_lower_left":  sph_to_xyz(r_min, az_center - half_az, el_center - half_el_low),
        "front_lower_right": sph_to_xyz(r_min, az_center + half_az, el_center - half_el_low),
        "front_upper_left":  sph_to_xyz(r_min, az_center - half_az, el_center + half_el_up),
        "front_upper_right": sph_to_xyz(r_min, az_center + half_az, el_center + half_el_up),
        "back_lower_left":   sph_to_xyz(r_max, az_center - half_az, el_center - half_el_low),
        "back_lower_right":  sph_to_xyz(r_max, az_center + half_az, el_center - half_el_low),
        "back_upper_left":   sph_to_xyz(r_max, az_center - half_az, el_center + half_el_up),
        "back_upper_right":  sph_to_xyz(r_max, az_center + half_az, el_center + half_el_up),
    }

    return {
        "mode": "angular_sep_back_only",
        "center_xyz": [cx, cy, cz],
        "target_distance_m": float(target_distance_m),
        "grade": grade,
        "r_min": round(r_min, 4),
        "r_max": round(r_max, 4),
        "half_azimuth_deg": round(half_az, 6),
        "half_elevation_low_deg": round(half_el_low, 6),
        "half_elevation_up_deg": round(half_el_up, 6),
        "corner_points_xyz": corners
    }
