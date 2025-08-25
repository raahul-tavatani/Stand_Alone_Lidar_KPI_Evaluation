# evaluation/rs1_grid_tubes.py
"""
RS-1 (Grid Tubes) – DIN SAE 91471 style radial detection profile.

For each az–el cell:
  • Build a constant-radius tube inside target_vicinity OBB.
  • Collect ALL points inside that tube across frames.
  • Convert to radial distances r = ||P|| (from sensor origin).
  • Make an aggregated histogram over r → relative probability curve (PDF-like).
  • Decide separability from the histogram: two strongest peaks with a valley
    whose probability is <= valley_frac * min(peak_probs) and spaced by >= min_sep_m.

Also keeps per-frame diagnostics (points/peaks per frame).

Example (Windows cmd.exe):
  python -m evaluation.rs1_grid_tubes ^
    --root ./outputs/rs1/scenario_001 ^
    --az-step-deg 1.0 --el-step-deg 1.0 ^
    --tube-radius-m 0.08 ^
    --min-sep-m 0.05 --min-peak-points 2 --valley-frac 0.6 ^
    --csv --debug
"""

import os
import re
import json
import math
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import open3d as o3d

from utils.cluster_utils import _cartesian_to_angular, _find_range_peaks

# ─────────────── IO helpers ───────────────

def _is_gap_dir(name: str) -> bool:
    return name.startswith("gap_") and name.endswith("m")

def _is_d0_dir(name: str) -> bool:
    return name.startswith("d0_") and name.endswith("m")

def _find_gap_folders(root_dir: str) -> List[str]:
    out: List[str] = []
    if not os.path.isdir(root_dir): return out
    for d0 in os.listdir(root_dir):
        d0_path = os.path.join(root_dir, d0)
        if not os.path.isdir(d0_path) or not _is_d0_dir(d0): continue
        for gap in os.listdir(d0_path):
            gap_path = os.path.join(d0_path, gap)
            if os.path.isdir(gap_path) and _is_gap_dir(gap): out.append(gap_path)
    return sorted(out)

def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def _read_frame_points(folder: str) -> Dict[int, np.ndarray]:
    """Read frame_XXX.pcd → {frame_id: Nx3 float32} in SENSOR frame."""
    pcd_files = sorted([f for f in os.listdir(folder) if re.match(r"^frame_\d+\.pcd$", f)])
    by_frame: Dict[int, np.ndarray] = {}
    for fname in pcd_files:
        fid = int(re.findall(r"\d+", fname)[0])
        p = os.path.join(folder, fname)
        pcd = o3d.io.read_point_cloud(p)
        pts = np.asarray(pcd.points, dtype=np.float32)
        if pts.size: by_frame[fid] = pts
    return by_frame

def _obb_from_vicinity(json_path: str) -> Tuple[o3d.geometry.OrientedBoundingBox, np.ndarray]:
    """Load target_vicinity.json → OBB and its corner points (SENSOR frame)."""
    d = _load_json(json_path)
    cp = d.get("corner_points_xyz", {})
    if not isinstance(cp, dict) or len(cp) < 4:
        raise ValueError(f"Bad or missing corners in: {json_path}")
    keys = [
        "front_lower_left", "front_lower_right",
        "front_upper_left", "front_upper_right",
        "back_lower_left",  "back_lower_right",
        "back_upper_left",  "back_upper_right",
    ]
    pts = [cp[k] for k in keys if k in cp]
    if not pts: pts = list(cp.values())
    corners = np.asarray(pts, dtype=float).reshape(-1, 3)
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(corners)
    )
    return obb, corners

def _points_in_obb(points_xyz: np.ndarray, obb: o3d.geometry.OrientedBoundingBox) -> np.ndarray:
    idx = obb.get_point_indices_within_bounding_box(
        o3d.utility.Vector3dVector(points_xyz)
    )
    return points_xyz[idx] if len(idx) else np.empty((0, 3), dtype=np.float32)

# ─────────────── angles, wrap, rays ───────────────

def _wrap_deg(a: np.ndarray) -> np.ndarray:
    x = (a + 180.0) % 360.0 - 180.0
    x[x == -180.0] = 180.0
    return x

def _angles_from_xyz(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    az, el, _ = _cartesian_to_angular(xyz.astype(np.float32))
    return az.astype(float), el.astype(float)

def _circular_span_deg(angles_deg: np.ndarray, pad_deg: float = 0.2) -> Tuple[float,float,bool]:
    """Minimal arc [start,end] covering all angles (deg). Returns (start, end, wraps)."""
    if angles_deg.size == 0:
        return -180.0, 180.0, False
    a = _wrap_deg(angles_deg.copy())
    th = np.deg2rad(a)
    c = np.mean(np.cos(th)); s = np.mean(np.sin(th))
    mu = math.degrees(math.atan2(s, c))  # center in [-180,180)
    d = _wrap_deg(a - mu)
    dmin, dmax = float(np.min(d)), float(np.max(d))
    start = float(_wrap_deg(np.array([mu + dmin - pad_deg]))[0])
    end   = float(_wrap_deg(np.array([mu + dmax + pad_deg]))[0])
    wraps = end < start
    return start, end, wraps

def _deg2rad(x): return x * math.pi / 180.0

def _dir_from_az_el_deg(az_deg: float, el_deg: float) -> np.ndarray:
    az = _deg2rad(az_deg); el = _deg2rad(el_deg)
    c = math.cos(el)
    return np.array([c*math.cos(az), c*math.sin(az), math.sin(el)], dtype=float)  # unit

def _ray_obb_intersection(obb: o3d.geometry.OrientedBoundingBox,
                          ray_o: np.ndarray,
                          ray_d: np.ndarray) -> Optional[Tuple[float, float]]:
    """Intersect ray O + t D with OBB. Returns (t_enter, t_exit) if hit in front; else None."""
    C = np.asarray(obb.center, dtype=float)
    R = np.asarray(obb.R if hasattr(obb, "R") else obb.rotation, dtype=float)
    ext = np.asarray(obb.extent if hasattr(obb, "extent") else obb.get_extent(), dtype=float) * 0.5
    ro = R.T @ (ray_o - C)
    rd = R.T @ ray_d
    tmin, tmax = -1e30, 1e30
    eps = 1e-12
    for i in range(3):
        if abs(rd[i]) < eps:
            if ro[i] < -ext[i] or ro[i] > ext[i]: return None
        else:
            t1 = (-ext[i] - ro[i]) / rd[i]
            t2 = ( ext[i] - ro[i]) / rd[i]
            if t1 > t2: t1, t2 = t2, t1
            if t1 > tmin: tmin = t1
            if t2 < tmax: tmax = t2
            if tmax < tmin: return None
    if tmax < max(tmin, 0.0): return None
    return float(max(tmin, 0.0)), float(tmax)

# ─────────────── grid & tubes ───────────────

def _build_grid_centers(az_step_deg: float, el_step_deg: float,
                        az_start: float, az_end: float, az_wraps: bool,
                        el_min: float, el_max: float) -> List[Tuple[float,float]]:
    """Half-open, non-overlapping bins → we only use the centers to define tubes."""
    centers: List[Tuple[float,float]] = []
    el_edges = np.arange(el_min, el_max + 1e-9, el_step_deg, dtype=float)
    if len(el_edges) < 2: el_edges = np.array([el_min, el_max], dtype=float)
    el_centers = (el_edges[:-1] + el_edges[1:]) * 0.5
    def _arc_centers(a0, a1):
        edges = np.arange(a0, a1 + 1e-9, az_step_deg, dtype=float)
        if len(edges) < 2: edges = np.array([a0, a1], dtype=float)
        return (edges[:-1] + edges[1:]) * 0.5
    if not az_wraps:
        az_centers = _arc_centers(az_start, az_end)
    else:
        az_centers = np.concatenate([_arc_centers(az_start, 180.0), _arc_centers(-180.0, az_end)])
    for elc in el_centers:
        for azc in az_centers:
            centers.append((float(((azc + 180.0) % 360.0) - 180.0), float(elc)))
    return centers

def _compute_tube_radius(s_enter: float,
                         el_center_deg: float,
                         az_step_deg: float,
                         el_step_deg: float,
                         kappa: float,
                         tube_radius_min_m: float) -> float:
    """Constant radius so tubes don't touch at near face (widest risk)."""
    elr = _deg2rad(el_center_deg)
    alpha_h = _deg2rad(az_step_deg) * max(1e-6, math.cos(elr))
    alpha_v = _deg2rad(el_step_deg)
    alpha_min = min(alpha_h, alpha_v)
    r = kappa * 0.5 * max(s_enter, 0.5) * alpha_min  # clamp s_enter
    return float(max(r, tube_radius_min_m))

# Local Freedman–Diaconis bin width (clamped)
def _fd_bin_width_local(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    if r.size <= 1:
        return 0.10
    q75, q25 = np.percentile(r, [75, 25])
    iqr = max(q75 - q25, 1e-6)
    w = 2.0 * iqr * (r.size ** (-1/3))
    # slightly finer minimum than default 3 cm
    return float(np.clip(w, 0.01, 0.15))

def _pdf_peaks_and_valley(centers: np.ndarray, probs: np.ndarray, min_sep_m: float):
    """
    Return (peak_indices_sorted_desc, (i0,i1,valley_idx,valley_prob)) for the two strongest peaks
    that are separated by >= min_sep_m. If fewer than 2 valid peaks, valley_info is None.
    """
    n = len(probs)
    if n < 3:
        return [], None

    # local maxima (allow flat tops)
    cand = [i for i in range(1, n-1) if probs[i] >= probs[i-1] and probs[i] >= probs[i+1]]
    if not cand:
        cand = [int(np.argmax(probs))]

    # sort by peak probability (desc) and enforce min separation
    cand = sorted(cand, key=lambda i: probs[i], reverse=True)
    selected: List[int] = []
    for i in cand:
        if all(abs(centers[i] - centers[j]) >= min_sep_m for j in selected):
            selected.append(i)

    if len(selected) >= 2:
        i0, i1 = sorted(selected[:2])
        seg = probs[i0:i1+1]
        valley_local = int(np.argmin(seg))
        valley_idx = i0 + valley_local
        valley_prob = float(probs[valley_idx])
        return selected, (i0, i1, valley_idx, valley_prob)

    return selected, None

# ─────────────── per-gap core ───────────────

def _process_gap_grid_tubes(gap_dir: str,
                            az_step_deg: float,
                            el_step_deg: float,
                            kappa: float,
                            tube_radius_min_m: float,
                            tube_radius_override_m: Optional[float],
                            min_sep_m: float,
                            min_peak_points: int,
                            min_peak_frac: float,
                            valley_frac: float,
                            csv_out: bool,
                            debug: bool):

    tv_path = os.path.join(gap_dir, "target_vicinity.json")
    if not os.path.exists(tv_path):
        print(f"[SKIP] {gap_dir}: target_vicinity.json not found")
        return

    obb, corners = _obb_from_vicinity(tv_path)

    # angular span of OBB (robust wrap handling)
    az_c, el_c = _angles_from_xyz(corners)
    az_start, az_end, az_wraps = _circular_span_deg(az_c, pad_deg=0.2)
    el_min = float(np.min(el_c) - 0.2)
    el_max = float(np.max(el_c) + 0.2)

    # fallback: widen if span is tiny
    az_span = (az_end - az_start) if not az_wraps else (360.0 - ((az_start - az_end) % 360.0))
    if az_span < max(az_step_deg, 0.05):
        th = np.deg2rad(_wrap_deg(az_c))
        c = np.mean(np.cos(th)); s = np.mean(np.sin(th))
        mu = math.degrees(math.atan2(s, c))
        az_start, az_end, az_wraps = _circular_span_deg(np.array([mu-2*az_step_deg, mu+2*az_step_deg]))
    if (el_max - el_min) < max(el_step_deg, 0.05):
        m = float(np.median(el_c))
        el_min = m - 2*el_step_deg; el_max = m + 2*el_step_deg

    if debug:
        print(f"[DBG] {gap_dir}: az_span=({az_start:.2f}°, {az_end:.2f}°), wraps={az_wraps}; "
              f"el_span=({el_min:.2f}°, {el_max:.2f}°)")

    centers = _build_grid_centers(az_step_deg, el_step_deg, az_start, az_end, az_wraps, el_min, el_max)
    if debug:
        print(f"[DBG] centers generated: {len(centers)}")

    # test center ray of each cell against OBB; keep those that intersect
    ray_o = np.zeros(3, dtype=float)
    cells = []
    for azc, elc in centers:
        d = _dir_from_az_el_deg(azc, elc)  # unit
        hit = _ray_obb_intersection(obb, ray_o, d)
        if hit is None: continue
        s_enter, s_exit = hit
        if s_exit <= s_enter + 1e-6: continue
        r_tube = tube_radius_override_m if (tube_radius_override_m and tube_radius_override_m > 0.0) \
                 else _compute_tube_radius(s_enter, elc, az_step_deg, el_step_deg, kappa, tube_radius_min_m)
        cells.append({"az": azc, "el": elc, "dir": d, "s_enter": s_enter, "s_exit": s_exit, "r_tube": float(r_tube)})

    if debug:
        print(f"[DBG] rays intersecting OBB: {len(cells)}")

    if not cells:
        print(f"[SKIP] {gap_dir}: no cell rays intersect the OBB")
        return

    frames = _read_frame_points(gap_dir)
    if not frames:
        print(f"[WARN] {gap_dir}: no frames found")
        return

    all_frame_ids = sorted(frames.keys())

    # per-cell accumulators
    per_cell_frames_seen: List[List[int]] = [[] for _ in range(len(cells))]
    per_cell_num_points: List[Dict[int, int]] = [dict() for _ in range(len(cells))]
    per_cell_num_groups: List[Dict[int, int]] = [dict() for _ in range(len(cells))]
    per_cell_peaks_per_frame: List[Dict[int, List[float]]] = [dict() for _ in range(len(cells))]
    per_cell_peak_counts_per_frame: List[Dict[int, List[int]]] = [dict() for _ in range(len(cells))]

    # aggregated radial distances per cell (across frames)
    per_cell_radial_all: List[List[float]] = [[] for _ in range(len(cells))]

    def _filter_to_obb(pts: np.ndarray) -> np.ndarray:
        return _points_in_obb(pts, obb)

    # MAIN LOOP: for each frame, test ALL points against EACH tube
    for fid, pts in frames.items():
        pts = _filter_to_obb(pts)
        if pts.size == 0: continue

        # Precompute ||P|| and ||P||^2 once per frame
        r2 = np.einsum("ij,ij->i", pts, pts)
        r_all = np.sqrt(r2)

        for cid, c in enumerate(cells):
            d = c["dir"]
            s_enter = c["s_enter"]; s_exit = c["s_exit"]; r_tube = c["r_tube"]

            # axial coordinate along tube axis (from sensor origin)
            s_vals = pts @ d

            # segment gate (to keep only the OBB portion of the tube)
            m_seg = (s_vals >= s_enter) & (s_vals <= s_exit)
            if not np.any(m_seg):
                continue

            # perpendicular distance to axis: sqrt(||P||^2 - s^2)
            s_seg = s_vals[m_seg]
            r2_seg = r2[m_seg]
            r_seg = r_all[m_seg]
            dist2 = r2_seg - s_seg*s_seg
            m_tube = dist2 <= (r_tube * r_tube + 1e-12)
            if not np.any(m_tube):
                continue

            # --- use ALL points in tube cross-section (radial metric) ---
            r_use = r_seg[m_tube].astype(float, copy=False)
            npts = int(r_use.size)
            if npts == 0:
                continue

            # per-frame peaks (diagnostic only)
            peaks_pf = _find_range_peaks(
                r_use,
                min_sep_m=float(min_sep_m),
                min_peak_frac=float(min_peak_frac),
                min_peak_points=int(min_peak_points),
                valley_frac=float(valley_frac)
            )

            per_cell_frames_seen[cid].append(int(fid))
            per_cell_num_points[cid][int(fid)] = npts
            per_cell_num_groups[cid][int(fid)] = len(peaks_pf)
            per_cell_peaks_per_frame[cid][int(fid)] = [float(p[0]) for p in peaks_pf]
            per_cell_peak_counts_per_frame[cid][int(fid)] = [int(p[1]) for p in peaks_pf]

            # accumulate for global (aggregated) radial PDF
            per_cell_radial_all[cid].append(r_use)

    # build CSV rows for ALL cells
    rows_all: List[Dict] = []
    for cid, c in enumerate(cells):
        frames_seen = sorted(set(per_cell_frames_seen[cid]))
        frame_count = len(frames_seen)

        # --- aggregated radial PDF over all frames for this cell ---
        if per_cell_radial_all[cid]:
            r_cat = np.concatenate(per_cell_radial_all[cid]).astype(float, copy=False)
            if r_cat.size >= 2:
                bw = _fd_bin_width_local(r_cat)
                # ensure at least a few bins for stability
                span = max(1e-6, r_cat.max() - r_cat.min())
                nbins = max(5, int(np.ceil(span / bw)) + 1)
                hist, edges = np.histogram(r_cat, bins=nbins)
                centers = 0.5 * (edges[:-1] + edges[1:])
                probs = hist.astype(float)
                total = probs.sum()
                if total > 0:
                    probs /= total  # relative probability

                # --- PDF-based separability (valley between the two strongest peaks) ---
                peaks_idx, valley_info = _pdf_peaks_and_valley(centers, probs, min_sep_m=float(min_sep_m))

                if len(peaks_idx) >= 2 and valley_info is not None:
                    i0, i1, iv, valley_p = valley_info
                    peak0_p, peak1_p = float(probs[i0]), float(probs[i1])
                    ref_p = max(1e-12, min(peak0_p, peak1_p))  # conservative reference
                    valley_ratio = valley_p / ref_p
                    is_sep = 1 if valley_ratio <= float(valley_frac) else 0
                    peak_delta = float(abs(centers[i1] - centers[i0]))
                    peaks_glob_str = "[" + ",".join(f"{centers[i]:.6f}" for i in (i0, i1)) + "]"
                    if debug:
                        print(f"[DBG] cell {cid}: peaks at {centers[i0]:.4f},{centers[i1]:.4f} m; "
                              f"valley_ratio={valley_ratio:.3f} (thr={valley_frac}), sep={is_sep}")
                else:
                    peaks_glob_str = "[]"
                    peak_delta = ""
                    is_sep = 0

                pdf_centers_str = "[" + ",".join(f"{v:.6f}" for v in centers) + "]"
                pdf_probs_str   = "[" + ",".join(f"{p:.6f}" for p in probs) + "]"

            else:
                pdf_centers_str = "[]"; pdf_probs_str = "[]"; peaks_glob_str = "[]"
                peak_delta = ""; is_sep = 0
        else:
            pdf_centers_str = "[]"; pdf_probs_str = "[]"; peaks_glob_str = "[]"
            peak_delta = ""; is_sep = 0

        # per-frame fields
        k_pairs = [f"{f}:{per_cell_num_groups[cid].get(f, 0)}" for f in all_frame_ids]
        k_per_frame_str = ",".join(k_pairs)

        p_pairs = [f"{f}:{per_cell_num_points[cid].get(f, 0)}" for f in all_frame_ids]
        points_per_frame_str = ",".join(p_pairs)

        r_items = []
        multi_frames = []
        for f in sorted(per_cell_peaks_per_frame[cid].keys()):
            peaks = per_cell_peaks_per_frame[cid][f]
            if not peaks: continue
            if len(peaks) >= 2:
                multi_frames.append(f)
            r_items.append(f"{f}:(" + "|".join(f"{p:.6f}" for p in peaks) + ")")
        ranges_per_frame_str = ";".join(r_items)

        # XYZ for peaks beyond the first two (diagnostic)
        xyz_items = []
        d = c["dir"]
        for f in sorted(per_cell_peaks_per_frame[cid].keys()):
            peaks = per_cell_peaks_per_frame[cid][f]
            if len(peaks) <= 2: continue
            extra = sorted(peaks)[2:]
            xyz_list = []
            for r in extra:
                P = d * float(r)  # LOS direction at that range
                xyz_list.append(f"({P[0]:.4f},{P[1]:.4f},{P[2]:.4f})")
            xyz_items.append(f"{f}:" + "[" + "|".join(xyz_list) + "]")
        extra_xyz_per_frame_str = ";".join(xyz_items)

        row = {
            "cell_id": cid,
            "azimuth_deg": round(float(c["az"]), 4),
            "elevation_deg": round(float(c["el"]), 4),
            "tube_radius_m": round(float(c["r_tube"]), 4),
            "s_enter_m": round(float(c["s_enter"]), 4),
            "s_exit_m": round(float(c["s_exit"]), 4),
            "frame_count": frame_count,

            # per-frame diagnostics
            "points_per_frame": points_per_frame_str,
            "k_per_frame": k_per_frame_str,
            "ranges_per_frame": ranges_per_frame_str,
            "multi_range_frames": ",".join(str(f) for f in multi_frames),
            "has_multi_range": 1 if len(multi_frames) > 0 else 0,
            "extra_peak_xyz_per_frame": extra_xyz_per_frame_str,

            # DIN-style radial detection profile (aggregated)
            "pdf_centers": pdf_centers_str,
            "pdf_probs": pdf_probs_str,
            "global_peaks_m": peaks_glob_str,
            "peak_delta_m": peak_delta,
            "is_separable": is_sep,
        }
        rows_all.append(row)

    if csv_out:
        import csv as _csv
        csv_path = os.path.join(gap_dir, "grid_tubes_map.csv")
        header = [
            "cell_id","azimuth_deg","elevation_deg","tube_radius_m",
            "s_enter_m","s_exit_m","frame_count",
            "points_per_frame","k_per_frame","ranges_per_frame",
            "multi_range_frames","has_multi_range","extra_peak_xyz_per_frame",
            "pdf_centers","pdf_probs","global_peaks_m","peak_delta_m","is_separable"
        ]
        with open(csv_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in rows_all:
                w.writerow(r)

    print(f"[OK] {gap_dir}: cells={len(rows_all)} | wrote grid_tubes_map.csv")

# ─────────────── CLI ───────────────

def main():
    ap = argparse.ArgumentParser(description="RS-1 (Grid Tubes): DIN-style radial detection profile (aggregated over frames)")
    ap.add_argument("--root", default="./outputs/rs1/scenario_001", help="Root with d0_*m/gap_*m folders")
    ap.add_argument("--az-step-deg", type=float, default=0.10, help="Azimuth grid step (deg)")
    ap.add_argument("--el-step-deg", type=float, default=0.36, help="Elevation grid step (deg)")
    ap.add_argument("--kappa", type=float, default=0.9, help="Safety factor for per-cell radius from near-face spacing")
    ap.add_argument("--tube-radius-min-m", type=float, default=0.02, help="Minimum tube radius (m)")
    ap.add_argument("--tube-radius-m", type=float, default=None, help="Override: fixed tube radius for all cells (m)")
    ap.add_argument("--min-sep-m", type=float, default=0.05, help="Min separation between peaks (m)")
    ap.add_argument("--min-peak-points", type=int, default=2, help="Min points for a peak (diagnostic per-frame)")
    ap.add_argument("--min-peak-frac", type=float, default=0.0, help="Min per-frame peak fraction (diagnostic)")
    ap.add_argument("--valley-frac", type=float, default=0.7, help="Valley depth threshold on PDF (≤ this × smaller peak)")
    ap.add_argument("--csv", action="store_true", help="Write grid_tubes_map.csv (ALL cells, with per-frame info + radial PDF)")
    ap.add_argument("--debug", action="store_true", help="Verbose spans and counts")
    args = ap.parse_args()

    gaps = _find_gap_folders(args.root)
    if not gaps:
        print(f"[ERROR] No gap folders under: {args.root}")
        return

    for g in gaps:
        _process_gap_grid_tubes(
            gap_dir=g,
            az_step_deg=args.az_step_deg,
            el_step_deg=args.el_step_deg,
            kappa=args.kappa,
            tube_radius_min_m=args.tube_radius_min_m,
            tube_radius_override_m=args.tube_radius_m,
            min_sep_m=args.min_sep_m,
            min_peak_points=args.min_peak_points,
            min_peak_frac=args.min_peak_frac,
            valley_frac=args.valley_frac,
            csv_out=args.csv,
            debug=args.debug
        )

    print("[DONE] RS-1 grid-of-tubes evaluation complete.")

if __name__ == "__main__":
    main()
