# evaluation/run_evaluation.py

import os
import json
import csv
import re
import numpy as np
import open3d as o3d

from utils.lidar_utils import generate_cluster_map_from_folder
from utils.cluster_utils import track_clusters_data_driven
from utils.kpi_utils import evaluate_kpis

# ------------- helpers -------------

def find_distance_folders(root_dir):
    """
    Returns a list of tuples: (rotation_folder_or_root, distance_folder_full_path, distance_m)
    Supports:
      (A) <root>/rotation_.../<distance>m/
      (B) <root>/<distance>m/
    """
    items = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
             if os.path.isdir(os.path.join(root_dir, d))]

    # (B) flat distances directly under root
    flat = []
    for d in items:
        name = os.path.basename(d)
        if re.search(r"^\s*\d+\s*m\s*$", name.replace("\\", "/")):
            m = re.findall(r"(\d+)", name)
            if m:
                flat.append((root_dir, d, int(m[0])))

    if flat:
        return flat

    # (A) rotation-first
    out = []
    for rot_folder in items:
        for sub in os.listdir(rot_folder):
            full = os.path.join(rot_folder, sub)
            if os.path.isdir(full) and re.search(r"\d+\s*m$", sub.replace("\\", "/")):
                m = re.findall(r"(\d+)\s*m$", sub.replace("\\", "/"))
                if m:
                    out.append((rot_folder, full, int(m[0])))
    return out


def count_pcd_files(folder_path):
    return len([f for f in os.listdir(folder_path) if re.match(r"frame_\d+\.pcd", f)])

def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def _maybe_frame_dt(control_json_path):
    try:
        d = _load_json(control_json_path)
        if "frame_dt" in d and d["frame_dt"]:
            return float(d["frame_dt"])
        if "frame_rate_hz" in d and d["frame_rate_hz"]:
            fr = float(d["frame_rate_hz"])
            return 1.0 / fr if fr > 0 else None
    except Exception:
        pass
    return None

def _load_corners(json_path):
    d = _load_json(json_path)
    cp = d.get("corner_points_xyz", {})
    corners = []
    for k in sorted(cp.keys()):
        v = cp[k]
        if isinstance(v, (list, tuple)) and len(v) == 3:
            corners.append([float(v[0]), float(v[1]), float(v[2])])
    if len(corners) < 4:
        raise ValueError(f"Invalid corner_points_xyz in {json_path}")
    return np.asarray(corners, dtype=float).reshape(-1, 3)

def _obb_from_corners(corners_xyz):
    return o3d.geometry.OrientedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(corners_xyz)
    )

def _points_in_obb(points_xyz, obb):
    idx = obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(points_xyz))
    return points_xyz[idx] if len(idx) else np.empty((0, 3), dtype=np.float32)

def _cart2aer(xyz):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x*x + y*y + z*z)
    az = np.degrees(np.arctan2(y, x))
    el = np.degrees(np.arctan2(z, np.sqrt(x*x + y*y)))
    return az, el, r

def _target_extent_deg_from_corners(corners_xyz):
    az, el, _ = _cart2aer(corners_xyz)
    a = np.unwrap(np.radians(az))
    az_span = np.degrees(a.max() - a.min())
    el_span = float(el.max() - el.min())
    return float(az_span), float(el_span)

def _generate_fp_cluster_map(dist_path):
    tgt_json = os.path.join(dist_path, "target_vicinity.json")
    ctl_json = os.path.join(dist_path, "control_volume.json")
    if not os.path.exists(ctl_json):
        return []
    corners_ctl = _load_corners(ctl_json)
    obb_ctl = _obb_from_corners(corners_ctl)
    obb_tgt = None
    if os.path.exists(tgt_json):
        corners_tgt = _load_corners(tgt_json)
        obb_tgt = _obb_from_corners(corners_tgt)

    pcd_files = sorted([f for f in os.listdir(dist_path) if re.match(r"frame_\d+\.pcd", f)])
    all_points_by_frame = {}
    for fname in pcd_files:
        frame_id = int(re.findall(r"\d+", fname)[0])
        pcd = o3d.io.read_point_cloud(os.path.join(dist_path, fname))
        pts = np.asarray(pcd.points, dtype=np.float32)
        if pts.size == 0:
            continue
        inside_ctl = _points_in_obb(pts, obb_ctl)
        if inside_ctl.size == 0:
            continue
        if obb_tgt is None:
            fp_pts = inside_ctl
        else:
            inside_tgt = _points_in_obb(inside_ctl, obb_tgt)
            if inside_tgt.size == 0:
                fp_pts = inside_ctl
            else:
                # set difference via structured view
                a = inside_ctl.view([('', inside_ctl.dtype)] * inside_ctl.shape[1])
                b = inside_tgt.view([('', inside_tgt.dtype)] * inside_tgt.shape[1])
                mask = ~np.isin(a, b)
                fp_pts = inside_ctl[mask.ravel()]
        if fp_pts.size > 0:
            all_points_by_frame[frame_id] = fp_pts

    if not all_points_by_frame:
        return []

    # cluster FP points (angular plane)
    cluster_map_fp = track_clusters_data_driven(all_points_by_frame)
    for c in cluster_map_fp:
        if "range" not in c and "range_mean" in c:
            c["range"] = c["range_mean"]
        if "detections_per_frame" not in c and "detections_per_frame_capped" in c:
            c["detections_per_frame"] = c["detections_per_frame_capped"]
        if "frame_count" not in c:
            c["frame_count"] = len(c.get("frames_seen", []))
    return cluster_map_fp

# ------------- main -------------

def run_all_evaluations(root_dir="outputs/multidomain_test"):
    results = []
    distance_folders = find_distance_folders(root_dir)

    for rot_path, dist_path, distance in sorted(distance_folders):
        rotation_name = os.path.basename(rot_path)
        try:
            tgt_json = os.path.join(dist_path, "target_vicinity.json")
            ctl_json = os.path.join(dist_path, "control_volume.json")
            has_tgt = os.path.exists(tgt_json)
            has_ctl = os.path.exists(ctl_json)

            # detect test type
            if has_tgt and has_ctl:
                test_type = "multi_domain"
            elif has_tgt and not has_ctl:
                test_type = "single_target"
            elif not has_tgt and has_ctl:
                test_type = "far"  # false-alarm
            else:
                test_type = "single_target"  # fallback

            # TP clusters (inside target vicinity) if target exists
            cluster_map_tp = generate_cluster_map_from_folder(dist_path) if has_tgt else []

            # FP clusters (inside control volume but outside target) if control exists
            cluster_map_fp = _generate_fp_cluster_map(dist_path) if has_ctl else []

            # timing & extents
            total_frames = count_pcd_files(dist_path)
            frame_dt = _maybe_frame_dt(ctl_json) if has_ctl else None
            target_extent_deg = None
            if has_tgt:
                az_span_deg, el_span_deg = _target_extent_deg_from_corners(_load_corners(tgt_json))
                target_extent_deg = (az_span_deg, el_span_deg)

            # choose TP set for KPI core: in FAR, TP set is empty by design
            tp_for_eval = cluster_map_tp if has_tgt else []

            # compute KPIs generically
            kpis = evaluate_kpis(
                tp_for_eval,
                target_distance=float(distance),
                total_frames=total_frames,
                gt_azimuth=0.0,
                gt_elevation=0.0,
                frame_dt=frame_dt,
                target_extent_deg=target_extent_deg,
                cluster_map_fp=cluster_map_fp,
                test_type=test_type
            )

            result = {
                "rotation": rotation_name,
                "distance_m": distance,
                "test_type": test_type,
                "frame_count": total_frames,
                "frame_dt_s": frame_dt,
                "target_extent_az_deg": round(target_extent_deg[0], 6) if target_extent_deg else None,
                "target_extent_el_deg": round(target_extent_deg[1], 6) if target_extent_deg else None,
                **kpis
            }
            results.append(result)
            print(f"[OK] Evaluated: {rotation_name} @ {distance}m ({total_frames} frames) [{test_type}]")
        except Exception as e:
            print(f"[ERROR] Skipped: {rotation_name} @ {distance}m → {e}")

    # Save JSON
    json_path = os.path.join(root_dir, "kpi_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[DONE] KPI results saved to: {json_path}")

    # Save CSV
    if results:
        csv_path = os.path.join(root_dir, "kpi_results.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"[DONE] KPI results saved to: {csv_path}")
