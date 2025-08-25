# tests/test_radial_sep1.py
import os, time, json, math, threading
from typing import Dict, Any
import carla

from utils.lidar_utils import save_pointcloud
from config.generate_test_config import (
    compute_target_vicinity_centered,   # LoS-based, expects SENSOR-FRAME center
    compute_control_volume,
)

# ================= config =================
SAVE_ROOT           = "./outputs/rs1/scenario_001"
PLANE_BP_ID         = "static.prop.planea"   # your 1×1 m plane
Z_POS               = 1.7                    # LiDAR & planes height (sensor z in world)
PLANE_W             = 1.0
PLANE_H             = 1.0
TILT_DEG            = 45.0                   # rear target tilt about lateral (Y) axis
CAPTURE_FRAMES      = 10
FRAME_DT_S          = 0.1                    # must match world.fixed_delta_seconds
D0_LIST_M           = [10.0, 50.0, 100.0]
GAPS_M              = [0.05, 0.50, 1.00, 1.50, 2.00]  # foremost-point gaps
OVERLAP_MODE        = "angular"              # "angular" or "meters"
OVERLAP_EPS_DEG     = 0.2                    # if OVERLAP_MODE="angular"
OVERLAP_DELTA_M     = 0.04                   # if OVERLAP_MODE="meters"
GRADE               = "0"                    # 0, A, B, C
# ==========================================

def _ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def _find_plane_blueprint(world, bp_id):
    lib = world.get_blueprint_library()
    try:
        return lib.find(bp_id)
    except Exception:
        cands = [b for b in lib if "plane" in b.id.lower()]
        if not cands:
            raise RuntimeError("No plane blueprint found")
        print(f"[WARN] Preferred '{bp_id}' not found; using '{cands[0].id}'")
        return cands[0]

def _spawn(world, bp, location, rotation):
    return world.spawn_actor(bp, carla.Transform(location, rotation))

def _record(world, lidar, out_dir, n_frames, timeout_s=180.0):
    _ensure_dir(out_dir)
    lock = threading.Lock()
    got = {"n": 0}
    done_evt = threading.Event()

    def cb(data):
        with lock:
            if got["n"] >= n_frames: return
            out = os.path.join(out_dir, f"frame_{got['n']:03}.pcd")
            save_pointcloud(data, out)
            got["n"] += 1
            if got["n"] >= n_frames: done_evt.set()

    lidar.listen(cb)
    start = time.time()
    try:
        while not done_evt.is_set():
            world.tick()
            if time.time() - start > timeout_s:
                print(f"[ERROR] Timeout ({got['n']}/{n_frames})")
                break
            time.sleep(0.001)
    finally:
        lidar.stop()
    return got["n"]

# ---------- geometry helpers ----------
def _x_center_for_tilted_plane(d0, g, height_m, tilt_deg):
    # Center x so foremost point sits g behind T1 plane at x=d0
    return d0 + g + 0.5 * height_m * math.sin(math.radians(tilt_deg))

def _y_center_for_overlap(x2c, mode, eps_deg, delta_m):
    # T1 center at y=-0.5 (right edge at y=0). Place T2 slightly into that edge.
    delta = x2c * math.tan(math.radians(eps_deg)) if mode == "angular" else float(delta_m)
    return 0.5 - delta  # so T2 left edge = -delta (slight overlap with T1 right edge=0)

def _euclid(x, y, z): return math.sqrt(x*x + y*y + z*z)

def _az_el_from_xyz(x, y, z):
    az = math.degrees(math.atan2(y, x))
    el = math.degrees(math.atan2(z, math.sqrt(x*x + y*y)))
    return az, el
# --------------------------------------

def _world_to_sensor(pt_world, lidar_loc):
    """Convert world XYZ to sensor frame (PCD frame) by subtracting LiDAR origin."""
    return [pt_world[0] - lidar_loc.x,
            pt_world[1] - lidar_loc.y,
            pt_world[2] - lidar_loc.z]

def _print_vicinity_los_debug(tag: str, center_sensor_xyz, vic):
    """Verify vicinity LoS ≈ target center LoS (in SENSOR frame)."""
    az_gt, el_gt = _az_el_from_xyz(*center_sensor_xyz)
    v = vic.get("corner_points_xyz", {})
    try:
        fctr = [(v["front_lower_left"][i] + v["front_upper_right"][i]) * 0.5 for i in range(3)]
        bctr = [(v["back_lower_left"][i]  + v["back_upper_right"][i])  * 0.5 for i in range(3)]
        cxyz = [0.5*(fctr[i] + bctr[i]) for i in range(3)]
        az_v, el_v = _az_el_from_xyz(*cxyz)
        da = ((az_v - az_gt + 180.0) % 360.0) - 180.0
        de = el_v - el_gt
        print(f"[LOS] {tag}: Δaz={da:.3f}°, Δel={de:.3f}° (expect ~0°)")
    except Exception:
        pass

def _combine_vicinities(v1: Dict[str, Any], v2: Dict[str, Any]) -> Dict[str, Any]:
    """Combine two target vicinities into a single AABB in SENSOR frame."""
    def all_corners(v):
        cp = v.get("corner_points_xyz", {})
        keys = ["front_lower_left","front_lower_right","front_upper_left","front_upper_right",
                "back_lower_left","back_lower_right","back_upper_left","back_upper_right"]
        return [cp[k] for k in keys if k in cp]

    c1 = all_corners(v1)
    c2 = all_corners(v2)
    if not c1 or not c2:
        raise RuntimeError("Missing corner points in vicinities to combine")

    xs = [p[0] for p in (c1+c2)]
    ys = [p[1] for p in (c1+c2)]
    zs = [p[2] for p in (c1+c2)]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)

    # center & distance at box center
    cx, cy, cz = 0.5*(xmin+xmax), 0.5*(ymin+ymax), 0.5*(zmin+zmax)
    target_distance_m = math.sqrt(cx*cx + cy*cy + cz*cz)

    # r_min / r_max from the eight corners of the union box
    def r_of(p): return math.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
    union_corners = [
        [xmin, ymin, zmin], [xmin, ymax, zmin], [xmin, ymin, zmax], [xmin, ymax, zmax],
        [xmax, ymin, zmin], [xmax, ymax, zmin], [xmax, ymin, zmax], [xmax, ymax, zmax],
    ]
    r_min = min(r_of(p) for p in union_corners)
    r_max = max(r_of(p) for p in union_corners)

    # Angular half-widths: conservative superset
    half_az = max(v1.get("half_azimuth_deg", 0.0), v2.get("half_azimuth_deg", 0.0))
    half_el_low = max(v1.get("half_elevation_low_deg", 0.0), v2.get("half_elevation_low_deg", 0.0))
    half_el_up  = max(v1.get("half_elevation_up_deg", 0.0),  v2.get("half_elevation_up_deg", 0.0))

    combined = {
        "mode": v1.get("mode", v2.get("mode", "angular_sep_back_only")),
        "center_xyz": [cx, cy, cz],
        "target_distance_m": target_distance_m,
        "grade": v1.get("grade", v2.get("grade", "0")),
        "r_min": r_min,
        "r_max": r_max,
        "half_azimuth_deg": half_az,
        "half_elevation_low_deg": half_el_low,
        "half_elevation_up_deg": half_el_up,
        "corner_points_xyz": {
            "front_lower_left":  [xmin, ymin, zmin],
            "front_lower_right": [xmin, ymax, zmin],
            "front_upper_left":  [xmin, ymin, zmax],
            "front_upper_right": [xmin, ymax, zmax],
            "back_lower_left":   [xmax, ymin, zmin],
            "back_lower_right":  [xmax, ymax, zmin],
            "back_upper_left":   [xmax, ymin, zmax],
            "back_upper_right":  [xmax, ymax, zmax],
        }
    }
    return combined

def run(client, world, lidar):
    print("[INFO] RS-1 capture started")
    # Make CARLA tick at FRAME_DT_S (synchronous)
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FRAME_DT_S
        world.apply_settings(settings)
        print(f"[INFO] world.fixed_delta_seconds set to {FRAME_DT_S}")
    except Exception:
        pass

    lidar_loc = lidar.get_transform().location  # LiDAR origin in WORLD (0,0,1.7)

    plane_bp = _find_plane_blueprint(world, PLANE_BP_ID)

    # Top-level manifest
    scenario_root = _ensure_dir(SAVE_ROOT)
    with open(os.path.join(scenario_root, "manifest.json"), "w") as f:
        json.dump({
            "test": "radial_separability_1",
            "frame_dt_s": FRAME_DT_S,
            "capture_frames": CAPTURE_FRAMES,
            "d0_list_m": D0_LIST_M,
            "gaps_m": GAPS_M,
            "tilt_deg": TILT_DEG,
            "overlap_mode": OVERLAP_MODE,
            "overlap_eps_deg": OVERLAP_EPS_DEG,
            "overlap_delta_m": OVERLAP_DELTA_M,
            "plane_size_m": [PLANE_W, PLANE_H],
            "grade": GRADE
        }, f, indent=2)

    # Visual plane orientations (rendering only)
    rot_perp = carla.Rotation(pitch=90.0, yaw=0.0, roll=0.0)             # T1 upright
    rot_tilt = carla.Rotation(pitch=90.0 - TILT_DEG, yaw=0.0, roll=0.0)  # T2 tilted about Y

    for d0 in D0_LIST_M:
        d0_dir = _ensure_dir(os.path.join(scenario_root, f"d0_{int(d0)}m"))

        for g in GAPS_M:
            # Compute placements in WORLD
            x2c = _x_center_for_tilted_plane(d0, g, PLANE_H, TILT_DEG)
            y2c = _y_center_for_overlap(x2c, OVERLAP_MODE, OVERLAP_EPS_DEG, OVERLAP_DELTA_M)

            T1_loc = carla.Location(x=lidar_loc.x + d0,  y=lidar_loc.y - 0.5, z=Z_POS)
            T2_loc = carla.Location(x=lidar_loc.x + x2c, y=lidar_loc.y + y2c, z=Z_POS)

            # SENSOR-frame centers (PCD frame)
            c1_s = _world_to_sensor([T1_loc.x, T1_loc.y, T1_loc.z], lidar_loc)  # ≈ [d0, -0.5, 0.0]
            c2_s = _world_to_sensor([T2_loc.x, T2_loc.y, T2_loc.z], lidar_loc)  # z≈0.0
            r1_s = _euclid(*c1_s)
            r2_s = _euclid(*c2_s)

            gap_dir = _ensure_dir(os.path.join(d0_dir, f"gap_{g:.2f}m"))

            # ── (A) Ground truth (store both world & sensor centers)
            gt = {
                "lidar": {"location_world": [lidar_loc.x, lidar_loc.y, lidar_loc.z]},
                "config": {
                    "frame_dt_s": FRAME_DT_S,
                    "frames": CAPTURE_FRAMES,
                    "grade": GRADE,
                    "plane_w_m": PLANE_W,
                    "plane_h_m": PLANE_H,
                    "tilt_deg": TILT_DEG,
                    "overlap_mode": OVERLAP_MODE,
                    "overlap_eps_deg": OVERLAP_EPS_DEG,
                    "overlap_delta_m": OVERLAP_DELTA_M
                },
                "case": {"d0_m": d0, "gap_m": g},
                "targets": [
                    {"name": "T1",
                     "center_xyz_world": [T1_loc.x, T1_loc.y, T1_loc.z],
                     "center_xyz_sensor": c1_s,
                     "size_m": [PLANE_W, PLANE_H], "tilted": False},
                    {"name": "T2",
                     "center_xyz_world": [T2_loc.x, T2_loc.y, T2_loc.z],
                     "center_xyz_sensor": c2_s,
                     "size_m": [PLANE_W, PLANE_H], "tilted": True,
                     "foremost_point_x_m": d0 + g}
                ]
            }
            with open(os.path.join(gap_dir, "gt_targets.json"), "w") as f:
                json.dump(gt, f, indent=2)

            # ── (B) TWO vicinities (build in SENSOR frame using *_sensor centers)
            vic_t1 = compute_target_vicinity_centered(
                center_xyz=c1_s, grade=GRADE, target_distance_m=r1_s
            )
            with open(os.path.join(gap_dir, "target_vicinity_T1.json"), "w") as f:
                json.dump(vic_t1, f, indent=2)

            vic_t2 = compute_target_vicinity_centered(
                center_xyz=c2_s, grade=GRADE, target_distance_m=r2_s
            )
            with open(os.path.join(gap_dir, "target_vicinity_T2.json"), "w") as f:
                json.dump(vic_t2, f, indent=2)

            # ── (B2) Combined vicinity (encloses both T1 and T2)
            vic_both = _combine_vicinities(vic_t1, vic_t2)
            with open(os.path.join(gap_dir, "target_vicinity.json"), "w") as f:
                json.dump(vic_both, f, indent=2)

            # Quick LoS alignment debug (should be ~0° Δaz/Δel)
            _print_vicinity_los_debug("T1", c1_s, vic_t1)
            _print_vicinity_los_debug("T2", c2_s, vic_t2)

            # Control volume in SENSOR frame (x forward from sensor)
            ctrl = compute_control_volume(r2_s)
            with open(os.path.join(gap_dir, "control_volume.json"), "w") as f:
                json.dump(ctrl, f, indent=2)

            print(f"[SPAWN] d0={d0:.2f} | g={g:.2f} | "
                  f"T1_world=({T1_loc.x:.3f},{T1_loc.y:.3f},{T1_loc.z:.3f}) "
                  f"| T2_world=({T2_loc.x:.3f},{T2_loc.y:.3f},{T2_loc.z:.3f}) | r2_sensor={r2_s:.3f} m")

            # ── (C) Spawn & capture → frames live in gap_dir (no 'frames/' folder)
            t1 = t2 = None
            try:
                t1 = _spawn(world, plane_bp, T1_loc, rot_perp)
                t2 = _spawn(world, plane_bp, T2_loc, rot_tilt)
                saved = _record(world, lidar, gap_dir, CAPTURE_FRAMES, timeout_s=300.0)
                if saved < CAPTURE_FRAMES:
                    print(f"[WARN] Saved {saved}/{CAPTURE_FRAMES} frames")
            finally:
                for a in (t1, t2):
                    try: a.destroy()
                    except Exception: pass
                print("[CLEANUP] Targets destroyed")

    print("[SUCCESS] RS-1 capture complete")

# Optional launcher if you want to run this file directly
if __name__ == "__main__":
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Spawn a stationary LiDAR sensor at the origin (world x=y=0, z=Z_POS)
    bp_lib = world.get_blueprint_library()
    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("rotation_frequency", str(int(1.0/FRAME_DT_S)))  # 10 Hz if FRAME_DT_S=0.1

    lidar = world.spawn_actor(
        lidar_bp,
        carla.Transform(carla.Location(x=0.0, y=0.0, z=Z_POS),
                        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))
    )
    try:
        run(client, world, lidar)
    finally:
        try: lidar.destroy()
        except: pass
