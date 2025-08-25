# tools/rs1_viz_grid_tubes.py
"""
Visualize RS-1 grid-of-tubes: OBB + tubes (transparent) + point cloud
(filtered to target vicinity OBB).

Example:
  python tools/rs1_viz_grid_tubes.py ^
    --gap-dir .\outputs\rs1\scenario_001\d0_50m\gap_0.05m ^
    --az-step-deg 0.08 --el-step-deg 0.08 ^
    --kappa 1 --tube-radius-m 0.032 ^
    --merge-all-frames 1 --pcd-voxel 0.02 --max-tubes 2000 --draw-rays 1 ^
    --tube-alpha 0.25 --debug 1
"""

import os, re, math, json, argparse
from typing import Tuple, List, Optional, Dict

import numpy as np
import open3d as o3d

# ------------------------------- IO -------------------------------

def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def _read_frames(gap_dir: str) -> Dict[int, np.ndarray]:
    files = sorted([f for f in os.listdir(gap_dir) if re.match(r"^frame_\d+\.pcd$", f)])
    out: Dict[int, np.ndarray] = {}
    for f in files:
        fid = int(re.findall(r"\d+", f)[0])
        p = os.path.join(gap_dir, f)
        pc = o3d.io.read_point_cloud(p)
        pts = np.asarray(pc.points, dtype=np.float32)
        if pts.size:
            out[fid] = pts
    return out

def _obb_from_vicinity(json_path: str) -> o3d.geometry.OrientedBoundingBox:
    d = _load_json(json_path)
    cp = d.get("corner_points_xyz", {})
    if not isinstance(cp, dict) or len(cp) < 4:
        raise RuntimeError(f"Bad corner_points_xyz in {json_path}")
    keys = [
        "front_lower_left", "front_lower_right",
        "front_upper_left", "front_upper_right",
        "back_lower_left",  "back_lower_right",
        "back_upper_left",  "back_upper_right",
    ]
    pts = [cp[k] for k in keys if k in cp]
    if not pts:
        pts = list(cp.values())
    corners = np.asarray(pts, dtype=float).reshape(-1, 3)
    return o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(corners))

def _points_in_obb(points_xyz: np.ndarray, obb: o3d.geometry.OrientedBoundingBox) -> np.ndarray:
    if points_xyz.size == 0:
        return points_xyz
    idx = obb.get_point_indices_within_bounding_box(
        o3d.utility.Vector3dVector(points_xyz)
    )
    return points_xyz[idx] if len(idx) else np.empty((0, 3), dtype=np.float32)

def obb_lineset_from_json(json_path: str, color=(1, 0, 0)) -> o3d.geometry.LineSet:
    data = _load_json(json_path)
    cp = data.get("corner_points_xyz", {})
    required = [
        "front_lower_left", "front_lower_right",
        "front_upper_left", "front_upper_right",
        "back_lower_left",  "back_lower_right",
        "back_upper_left",  "back_upper_right",
    ]
    for k in required:
        if k not in cp:
            raise RuntimeError(f"Missing corner '{k}' in {json_path}")

    pts = np.array([
        cp["front_lower_left"],
        cp["front_lower_right"],
        cp["front_upper_left"],
        cp["front_upper_right"],
        cp["back_lower_left"],
        cp["back_lower_right"],
        cp["back_upper_left"],
        cp["back_upper_right"],
    ], dtype=float)

    lines = np.array([
        [0,1], [1,3], [3,2], [2,0],  # front
        [4,5], [5,7], [7,6], [6,4],  # back
        [0,4], [1,5], [2,6], [3,7],  # sides
    ], dtype=np.int32)

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines)
    )
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.array(color, float), (lines.shape[0], 1)))
    return ls

# ------------------------- angles & rays --------------------------

def _cartesian_to_angular(points_xyz: np.ndarray):
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    r = np.sqrt(x*x + y*y + z*z)
    az = np.degrees(np.arctan2(y, x))
    el = np.degrees(np.arctan2(z, np.sqrt(x*x + y*y)))
    return az, el, r

def _wrap_deg(a):
    x = (a + 180.0) % 360.0 - 180.0
    if np.isscalar(x):
        return 180.0 if x == -180.0 else x
    x[x == -180.0] = 180.0
    return x

def _circular_span_deg(angles_deg: np.ndarray, pad_deg: float = 0.2) -> Tuple[float,float,bool]:
    if angles_deg.size == 0:
        return -180.0, 180.0, False
    a = _wrap_deg(angles_deg.copy())
    th = np.deg2rad(a)
    c, s = np.mean(np.cos(th)), np.mean(np.sin(th))
    mu = math.degrees(math.atan2(s, c))
    d = _wrap_deg(a - mu)
    dmin, dmax = float(np.min(d)), float(np.max(d))
    start = _wrap_deg(mu + dmin - pad_deg)
    end   = _wrap_deg(mu + dmax + pad_deg)
    wraps = end < start
    return float(start), float(end), bool(wraps)

def _deg2rad(x): return x * math.pi / 180.0

def _dir_from_az_el_deg(az_deg: float, el_deg: float) -> np.ndarray:
    az = _deg2rad(az_deg); el = _deg2rad(el_deg)
    c = math.cos(el)
    return np.array([c*math.cos(az), c*math.sin(az), math.sin(el)], dtype=float)

def _ray_obb_intersection(obb: o3d.geometry.OrientedBoundingBox,
                          ray_o: np.ndarray,
                          ray_d: np.ndarray) -> Optional[Tuple[float, float]]:
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

# --------------------------- grid & tubes -------------------------

def _build_grid_centers(az_step_deg: float, el_step_deg: float,
                        az_start: float, az_end: float, az_wraps: bool,
                        el_min: float, el_max: float) -> List[Tuple[float,float]]:
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
    elr = _deg2rad(el_center_deg)
    alpha_h = _deg2rad(az_step_deg) * max(1e-6, math.cos(elr))
    alpha_v = _deg2rad(el_step_deg)
    alpha_min = min(alpha_h, alpha_v)
    r = kappa * 0.5 * max(s_enter, 0.5) * alpha_min
    return float(max(r, tube_radius_min_m))

# ----------------------------- meshes -----------------------------

def _rotation_matrix_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        if c > 0.0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(a[0]) > 0.9: axis = np.array([0.0, 1.0, 0.0])
        v = np.cross(a, axis); v = v / (np.linalg.norm(v) + 1e-12)
        return -np.eye(3) + 2*np.outer(v, v)
    s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]], dtype=float)
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2 + 1e-12))
    return R

def make_cylinder_along_segment(p0: np.ndarray, p1: np.ndarray, radius: float, resolution: int = 24) -> o3d.geometry.TriangleMesh:
    p0 = np.asarray(p0, dtype=float); p1 = np.asarray(p1, dtype=float)
    v = p1 - p0
    L = float(np.linalg.norm(v))
    if L <= 1e-9:
        L = 1e-6
        v = np.array([0, 0, 1.0], dtype=float)
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=L, resolution=resolution, split=1)
    cyl.compute_vertex_normals()
    z = np.array([0.0, 0.0, 1.0], dtype=float)
    d = v / (np.linalg.norm(v) + 1e-12)
    R = _rotation_matrix_from_a_to_b(z, d)
    cyl.rotate(R, center=np.array([0.0, 0.0, 0.0]))
    center = 0.5 * (p0 + p1)
    cyl.translate(center)
    return cyl

def make_ray_line(p0: np.ndarray, p1: np.ndarray, color=(0,0,0)) -> o3d.geometry.LineSet:
    pts = o3d.utility.Vector3dVector(np.vstack([p0, p1]))
    lines = o3d.utility.Vector2iVector(np.array([[0,1]], dtype=np.int32))
    ls = o3d.geometry.LineSet(points=pts, lines=lines)
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.array(color, dtype=float).reshape(1,3), (1,1)))
    return ls

def obb_lineset(obb: o3d.geometry.OrientedBoundingBox, color=(1,0,0)) -> o3d.geometry.LineSet:
    pts = np.asarray(obb.get_box_points(), dtype=float)
    lines = np.array([
        [0,1],[1,3],[3,2],[2,0],
        [4,5],[5,7],[7,6],[6,4],
        [0,4],[1,5],[2,6],[3,7],
    ], dtype=np.int32)
    ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(pts),
                              lines=o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.array(color, dtype=float).reshape(1,3), (lines.shape[0],1)))
    return ls

# ----------------------------- main -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-dir", required=True)
    ap.add_argument("--az-step-deg", type=float, default=1.0)
    ap.add_argument("--el-step-deg", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=0.9)
    ap.add_argument("--tube-radius-min-m", type=float, default=0.02)
    ap.add_argument("--tube-radius-m", type=float, default=0.0, help="Override fixed radius for all tubes; 0=auto per-cell")
    ap.add_argument("--tube-alpha", type=float, default=0.25, help="Transparency for tubes (0=fully transparent, 1=opaque)")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--merge-all-frames", type=int, default=0)
    ap.add_argument("--pcd-voxel", type=float, default=0.0)
    ap.add_argument("--max-tubes", type=int, default=500)
    ap.add_argument("--draw-rays", type=int, default=0)
    ap.add_argument("--debug", type=int, default=0)
    args = ap.parse_args()

    tv_path = os.path.join(args.gap_dir, "target_vicinity.json")
    if not os.path.exists(tv_path):
        raise FileNotFoundError(f"target_vicinity.json not found in {args.gap_dir}")

    obb = _obb_from_vicinity(tv_path)

    # angular span from OBB corners
    corners = np.asarray(obb.get_box_points(), dtype=float)
    az_c, el_c, _ = _cartesian_to_angular(corners.astype(np.float32))
    az_start, az_end, az_wraps = _circular_span_deg(az_c, pad_deg=0.2)
    el_min = float(np.min(el_c) - 0.2)
    el_max = float(np.max(el_c) + 0.2)
    az_span = (az_end - az_start) if not az_wraps else (360.0 - ((az_start - az_end) % 360.0))
    if az_span < max(args.az_step_deg, 0.05):
        mu = float(np.degrees(np.arctan2(np.mean(np.sin(np.deg2rad(az_c))), np.mean(np.cos(np.deg2rad(az_c))))))
        az_start, az_end, az_wraps = _circular_span_deg(np.array([mu-2*args.az_step_deg, mu+2*args.az_step_deg]))
    if (el_max - el_min) < max(args.el_step_deg, 0.05):
        m = float(np.median(el_c))
        el_min = m - 2*args.el_step_deg
        el_max = m + 2*args.el_step_deg

    if args.debug:
        print(f"[DBG] az_span=({az_start:.2f},{az_end:.2f}) wraps={az_wraps} | el_span=({el_min:.2f},{el_max:.2f})")

    centers = _build_grid_centers(args.az_step_deg, args.el_step_deg, az_start, az_end, az_wraps, el_min, el_max)

    ray_o = np.zeros(3, dtype=float)
    cells = []
    for azc, elc in centers:
        d = _dir_from_az_el_deg(azc, elc)
        hit = _ray_obb_intersection(obb, ray_o, d)
        if hit is None: continue
        s_enter, s_exit = hit
        if s_exit <= s_enter + 1e-6: continue
        r_tube = args.tube_radius_m if args.tube_radius_m > 0.0 else _compute_tube_radius(
            s_enter, elc, args.az_step_deg, args.el_step_deg, args.kappa, args.tube_radius_min_m
        )
        cells.append({"az": azc, "el": elc, "dir": d, "s_enter": s_enter, "s_exit": s_exit, "r_tube": float(r_tube)})
    if args.debug:
        print(f"[DBG] cells intersecting OBB: {len(cells)}")

    # Point cloud (one frame or merged) — FILTERED TO OBB
    frames = _read_frames(args.gap_dir)
    if not frames:
        raise RuntimeError("No frame_XXX.pcd found")

    if args.merge_all_frames:
        P = np.vstack([v for v in frames.values()]).astype(np.float32, copy=False)
    else:
        if args.frame not in frames:
            raise KeyError(f"Frame {args.frame} not found; available: {sorted(frames.keys())[:10]}...")
        P = frames[args.frame]

    total_before = int(P.shape[0])
    P = _points_in_obb(P, obb)
    total_after = int(P.shape[0])
    if args.debug:
        print(f"[DBG] points in PCD: {total_before:,} -> within OBB: {total_after:,}")

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    if args.pcd_voxel > 0 and total_after > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(args.pcd_voxel))
    pcd.paint_uniform_color([0.6, 0.6, 0.6])

    # Build geometries with materials for the modern renderer (transparency)
    geos = []

    # Point cloud (default material is fine)
    geos.append({"name": "pcd", "geometry": pcd})

    # OBB lines
    obb_ls = obb_lineset_from_json(tv_path, color=(1, 0, 0))
    # use unlitLine material to make sure line color/width is respected
    mat_line = o3d.visualization.rendering.MaterialRecord()
    mat_line.shader = "unlitLine"
    mat_line.line_width = 1.0
    geos.append({"name": "obb", "geometry": obb_ls, "material": mat_line})

    # World axes
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geos.append({"name": "axes", "geometry": axes})

    # Tubes (transparent)
    tubes_to_draw = cells if args.max_tubes <= 0 else cells[:args.max_tubes]
    mat_tube = o3d.visualization.rendering.MaterialRecord()
    # Transparent, lit shader:
    mat_tube.shader = "defaultLitTransparency"
    # RGBA (A < 1 enables transparency)
    mat_tube.base_color = (0.2, 0.6, 1.0, float(np.clip(args.tube_alpha, 0.0, 1.0)))
    mat_tube.base_roughness = 0.7
    mat_tube.base_metallic = 0.0

    for i, c in enumerate(tubes_to_draw):
        d = c["dir"]; s0 = c["s_enter"]; s1 = c["s_exit"]; r = c["r_tube"]
        p0 = ray_o + d * s0
        p1 = ray_o + d * s1
        cyl = make_cylinder_along_segment(p0, p1, radius=r, resolution=24)
        geos.append({"name": f"tube_{i}", "geometry": cyl, "material": mat_tube})

        if args.draw_rays:
            ray_ls = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(np.vstack([p0, p1])),
                lines=o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
            )
            mat_ray = o3d.visualization.rendering.MaterialRecord()
            mat_ray.shader = "unlitLine"
            mat_ray.line_width = 1.0
            geos.append({"name": f"ray_{i}", "geometry": ray_ls, "material": mat_ray})

    # Use modern draw API (supports transparency). Fallback to legacy if unavailable.
    if hasattr(o3d.visualization, "draw"):
        try:
            # Newer Open3D versions accept the list as the first positional arg
            o3d.visualization.draw(
                geos,  # pass as positional, not "geometries="
                show_skybox=False,
                title="RS-1 Grid Tubes",
                width=1280,
                height=800,
            )
        except TypeError:
            # Some older versions have a stricter signature; try minimal form
            o3d.visualization.draw(geos)
    else:
        print("[WARN] Open3D 'draw' API not found; falling back to legacy viewer (no transparency).")
        legacy_geoms = [g["geometry"] for g in geos]
        o3d.visualization.draw_geometries(legacy_geoms)


if __name__ == "__main__":
    main()
