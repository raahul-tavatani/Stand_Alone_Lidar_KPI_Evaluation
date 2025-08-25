# tests/test_radial_sep1.py
import os, time, json, math, threading
import carla

from utils.lidar_utils import save_pointcloud
from config.generate_test_config import (
    compute_target_vicinity,
    compute_control_volume,
)

# ================= config (edit as needed) =================
SAVE_ROOT           = "./outputs/rs1/scenario_001"
PLANE_BP_ID         = "static.prop.planea"      # your 1×1 m plane prop id
Z_POS               = 1.7                       # LiDAR & planes height
PLANE_W             = 1.0                       # width  (m)
PLANE_H             = 1.0                       # height (m)
TILT_DEG            = 45.0                      # rear target tilt about lateral (Y) axis
CAPTURE_FRAMES      = 10                        # ~3 s @ 10 Hz
FRAME_DT_S          = 0.1                       # must match world.fixed_delta_seconds
D0_LIST_M           = [10.0, 50.0, 100.0]
GAPS_M              = [0.05, 0.50, 1.00, 1.50, 2.00]  # foremost-point gaps
OVERLAP_MODE        = "angular"                 # "angular" or "meters"
OVERLAP_EPS_DEG     = 0.2                       # if OVERLAP_MODE="angular"
OVERLAP_DELTA_M     = 0.04                      # if OVERLAP_MODE="meters" (≈4 cm)
GRADE               = "0"                       # vicinity grade for cropping/eval
# ===========================================================

def _ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def _find_plane_blueprint(world, bp_id):
    lib = world.get_blueprint_library()
    try:
        return lib.find(bp_id)
    except Exception:
        # fallback: anything that looks like a plane
        cands = [b for b in lib if "plane" in b.id.lower()]
        if not cands:
            raise RuntimeError("No plane blueprint found")
        print(f"[WARN] Preferred '{bp_id}' not found; using '{cands[0].id}'")
        return cands[0]

def _spawn(world, bp, location, rotation):
    return world.spawn_actor(bp, carla.Transform(location, rotation))

def _record(world, lidar, out_dir, n_frames, timeout_s=180.0):
    # NOTE: frames are saved directly into out_dir (no subfolder),
    # because the clusterer expects target_vicinity.json in the same folder.
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
            print(f"[CAPTURE] {os.path.basename(out)} ({got['n']}/{n_frames})")
            if got["n"] >= n_frames:
                done_evt.set()

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

def _x_center_for_tilted_plane(d0, g, height_m, tilt_deg):
    # Ensure foremost point sits g behind T1 plane (at x=d0)
    # Center x = d0 + g + (H/2)*sin(tilt)
    return d0 + g + 0.5 * height_m * math.sin(math.radians(tilt_deg))

def _y_center_for_overlap(x2c, mode, eps_deg, delta_m):
    # T1 center at y=-0.5 (right edge at y=0). Place T2 slightly into that edge.
    if mode == "angular":
        delta = x2c * math.tan(math.radians(eps_deg))  # meters
    else:
        delta = float(delta_m)
    return 0.5 - delta  # so T2 left edge = -delta (slight overlap with T1 right edge=0)

def _write_target_vicinity_json(gap_dir, d0, grade, rotation_deg):
    """Write target_vicinity.json into gap_dir using the same math as your generator."""
    vic = compute_target_vicinity(target_distance_m=d0, grade=grade, rotation=rotation_deg)
    out_path = os.path.join(gap_dir, "target_vicinity.json")
    with open(out_path, "w") as f:
        json.dump(vic, f, indent=2)
    print(f"[INFO] target_vicinity.json written at {out_path}")

def _write_control_volume_json(gap_dir, x_front_m):
    """Write control_volume.json into gap_dir; front plane set beyond the far target."""
    cv = compute_control_volume(target_distance_m=x_front_m)
    out_path = os.path.join(gap_dir, "control_volume.json")
    with open(out_path, "w") as f:
        json.dump(cv, f, indent=2)
    print(f"[INFO] control_volume.json written at {out_path}")

def run(client, world, lidar):
    print("[INFO] RS-1 capture started")
    lidar_loc = lidar.get_transform().location

    # Verify plane BP
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

    # Rotations (CARLA convention used in your tests)
    rot_perp = carla.Rotation(pitch=90.0, yaw=0.0, roll=0.0)             # T1
    rot_tilt = carla.Rotation(pitch=90.0 - TILT_DEG, yaw=0.0, roll=0.0)  # T2 (tilted about Y)

    for d0 in D0_LIST_M:
        d0_dir = _ensure_dir(os.path.join(scenario_root, f"d0_{int(d0)}m"))

        for g in GAPS_M:
            x2c = _x_center_for_tilted_plane(d0, g, PLANE_H, TILT_DEG)
            y2c = _y_center_for_overlap(x2c, OVERLAP_MODE, OVERLAP_EPS_DEG, OVERLAP_DELTA_M)

            # T1 center & rotation
            T1_loc = carla.Location(x=lidar_loc.x + d0, y=lidar_loc.y - 0.5, z=Z_POS)
            T1_rot = rot_perp

            # T2 center & rotation
            T2_loc = carla.Location(x=lidar_loc.x + x2c, y=lidar_loc.y + y2c, z=Z_POS)
            T2_rot = rot_tilt

            gap_dir = _ensure_dir(os.path.join(d0_dir, f"gap_{g:.2f}m"))
            print(f"[SPAWN] d0={d0:.2f} m | g={g:.2f} m | T2 x_c={x2c:.3f} m, y_c={y2c:.3f} m")

            # Minimal GT for downstream eval
            gt = {
                "lidar": {"location": [lidar_loc.x, lidar_loc.y, lidar_loc.z]},
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
                    {
                        "name": "T1",
                        "center_xyz": [T1_loc.x, T1_loc.y, T1_loc.z],
                        "rotation_deg": [T1_rot.pitch, T1_rot.yaw, T1_rot.roll],
                        "size_m": [PLANE_W, PLANE_H],
                        "tilted": False
                    },
                    {
                        "name": "T2",
                        "center_xyz": [T2_loc.x, T2_loc.y, T2_loc.z],
                        "rotation_deg": [T2_rot.pitch, T2_rot.yaw, T2_rot.roll],
                        "size_m": [PLANE_W, PLANE_H],
                        "tilted": True,
                        "foremost_point_x_m": d0 + g
                    }
                ]
            }
            with open(os.path.join(gap_dir, "gt_targets.json"), "w") as f:
                json.dump(gt, f, indent=2)

            # Spawn
            try:
                t1 = _spawn(world, plane_bp, T1_loc, T1_rot)
                t2 = _spawn(world, plane_bp, T2_loc, T2_rot)
                print("[INFO] Targets spawned")
            except Exception as e:
                print("[ERROR] Spawn failed:", e)
                continue

            # Record frames (saved directly into gap_dir)
            try:
                saved = _record(world, lidar, gap_dir, CAPTURE_FRAMES, timeout_s=300.0)
                if saved < CAPTURE_FRAMES:
                    print(f"[WARN] Saved {saved}/{CAPTURE_FRAMES} frames")
            finally:
                for a in (t1, t2):
                    try: a.destroy()
                    except Exception: pass
                print("[CLEANUP] Targets destroyed")

            # === Write the TWO JSONs the clusterer expects in THIS folder ===
            # target_vicinity.json → use front target (d0) with grade=0; this angular gate
            # is wide enough to include both due to ±10% range band and slight az overlap.
            _write_target_vicinity_json(
                gap_dir=gap_dir,
                d0=d0,
                grade=GRADE,
                rotation_deg=(T1_rot.pitch, T1_rot.yaw, T1_rot.roll)
            )

            # control_volume.json → set front plane beyond the far target (x2c + 1 m) to include both
            _write_control_volume_json(gap_dir=gap_dir, x_front_m=x2c + 1.0)

    print("[SUCCESS] RS-1 capture complete")
