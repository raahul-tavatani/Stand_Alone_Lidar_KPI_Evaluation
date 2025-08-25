#!/usr/bin/env python3
# viz_clusters.py
# Visualize angular clustering (azimuth/elevation) with circular clusters.

import os, re, json, argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from collections import defaultdict

# ----------------------------- IO & Filtering -----------------------------

def _load_vicinity_corners(vicinity_json_path: str) -> np.ndarray:
    with open(vicinity_json_path, "r") as f:
        vic = json.load(f)
    cp = vic.get("corner_points_xyz", {})
    if not isinstance(cp, dict) or len(cp) < 4:
        raise ValueError(f"Invalid corner_points_xyz in {vicinity_json_path}")
    corners = []
    for k in sorted(cp.keys()):
        v = cp[k]
        if not (isinstance(v, (list, tuple)) and len(v) == 3):
            raise ValueError(f"Corner '{k}' malformed: {v}")
        corners.append([float(v[0]), float(v[1]), float(v[2])])
    corners = np.asarray(corners, dtype=float).reshape(-1, 3)
    if corners.shape[0] < 4:
        raise ValueError("Need ≥4 corners to build OBB")
    return corners

def _extract_points_in_target_vicinity(pcd_path: str, corners_xyz: np.ndarray) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    if points.size == 0:
        return points
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(corners_xyz)
    )
    idx = obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(points))
    return points[idx]

def _load_folder_points(folder: str, subsample: int = 0):
    """
    Returns:
      frame_ids: (N,) int
      points:    (N,3) float32
      frames_list: sorted unique frame ids
    """
    vjson = os.path.join(folder, "target_vicinity.json")
    if not os.path.exists(vjson):
        raise FileNotFoundError(f"target_vicinity.json not found in {folder}")

    corners = _load_vicinity_corners(vjson)

    pcd_files = sorted([f for f in os.listdir(folder) if re.match(r"frame_\d+\.pcd", f)])
    if not pcd_files:
        raise RuntimeError(f"No frame_*.pcd in {folder}")

    pts_all, fids_all = [], []
    for fname in pcd_files:
        f_id = int(re.findall(r"\d+", fname)[0])
        pts = _extract_points_in_target_vicinity(os.path.join(folder, fname), corners)
        if isinstance(pts, np.ndarray) and pts.ndim == 2 and pts.shape[1] == 3 and pts.size > 0:
            pts_all.append(pts.astype(np.float32, copy=False))
            fids_all.append(np.full(pts.shape[0], f_id, dtype=np.int32))

    if not pts_all:
        raise RuntimeError(f"No points inside target vicinity for {folder}")

    points = np.vstack(pts_all)
    frame_ids = np.concatenate(fids_all)

    if subsample and points.shape[0] > subsample:
        sel = np.random.choice(points.shape[0], subsample, replace=False)
        points = points[sel]
        frame_ids = frame_ids[sel]

    frames_list = sorted(list(set(frame_ids.tolist())))
    return frame_ids, points, frames_list

# ----------------------------- Angular helpers -----------------------------

def cartesian_to_angular(xyz: np.ndarray):
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    r = np.sqrt(x*x + y*y + z*z)
    az = np.degrees(np.arctan2(y, x))
    el = np.degrees(np.arctan2(z, np.sqrt(x*x + y*y)))
    return az.astype(np.float32), el.astype(np.float32), r.astype(np.float32)

def median_step_deg(az: np.ndarray, el: np.ndarray, frame_ids: np.ndarray, subsample_per_frame=4000):
    steps = []
    for f in np.unique(frame_ids):
        idx = np.where(frame_ids == f)[0]
        if idx.size < 3:
            continue
        if idx.size > subsample_per_frame:
            idx = np.random.choice(idx, subsample_per_frame, replace=False)
        Aaz, Ael = az[idx], el[idx]
        A = np.vstack([Aaz, Ael]).T
        for i in range(A.shape[0]):
            da = (A[:,0] - A[i,0] + 180.0) % 360.0 - 180.0
            de =  A[:,1] - A[i,1]
            d  = np.hypot(da, de)
            d[i] = np.inf
            steps.append(d.min())
    return float(np.median(steps)) if steps else 0.1

# ----------------------------- Clustering (fast) -----------------------------

def cluster_angular(az, el, frame_ids, rc_deg=None):
    """
    Returns:
      centers:  (K,2) [az, el]
      labels:   (N,) cluster id for each point
      info:     list of dicts per cluster (frames_seen, sample_size, etc.)
      rc_deg:   the cluster radius actually used
    """
    if rc_deg is None:
        step = median_step_deg(az, el, frame_ids)
        rc_deg = 0.5 * step if step > 0 else 0.05

    cell = rc_deg
    nbin_az = int(np.ceil(360.0 / cell))
    def bin_az(a):  return ((np.floor((a + 180.0)/cell)).astype(np.int32)) % nbin_az
    def bin_el(e):  return np.floor((e + 90.0)/cell).astype(np.int32)

    baz = bin_az(az)
    bel = bin_el(el)

    grid = defaultdict(list)   # (baz, bel) -> [cluster_ids]
    centers_az, centers_el = [], []
    members = []               # list[list point indices]
    per_frame_counts = []      # list[dict frame_id -> count]
    labels = -np.ones(az.shape[0], dtype=np.int32)

    for i in range(az.shape[0]):
        cx, cy = baz[i], bel[i]
        best_j, best_d = -1, 1e9
        # local 3x3 neighborhood
        for dx in (-1, 0, 1):
            gx = (cx + dx) % nbin_az
            for dy in (-1, 0, 1):
                gy = cy + dy
                if (gx, gy) not in grid:
                    continue
                for j in grid[(gx, gy)]:
                    da = (az[i] - centers_az[j] + 180.0) % 360.0 - 180.0
                    de =  el[i] - centers_el[j]
                    d  = (da*da + de*de) ** 0.5
                    if d < best_d:
                        best_d, best_j = d, j
        if best_j != -1 and best_d <= rc_deg:
            members[best_j].append(i)
            labels[i] = best_j
            f = int(frame_ids[i])
            per_frame_counts[best_j][f] = per_frame_counts[best_j].get(f, 0) + 1
        else:
            j = len(centers_az)
            centers_az.append(float(az[i])); centers_el.append(float(el[i]))
            members.append([i])
            labels[i] = j
            per_frame_counts.append({int(frame_ids[i]): 1})
            grid[(cx, cy)].append(j)

    # recentre once (mean on unit circle for az)
    for j in range(len(centers_az)):
        idx = np.asarray(members[j], dtype=np.int32)
        if idx.size < 2: 
            continue
        az_rad = np.radians(az[idx])
        cx = float(np.cos(az_rad).mean()); sx = float(np.sin(az_rad).mean())
        centers_az[j] = np.degrees(np.arctan2(sx, cx))
        centers_el[j] = float(el[idx].mean())

    centers = np.vstack([centers_az, centers_el]).T if centers_az else np.zeros((0,2), dtype=float)

    # build per-cluster info
    info = []
    for j in range(len(centers_az)):
        idx = np.asarray(members[j], dtype=np.int32)
        frames_seen = sorted(set(int(frame_ids[k]) for k in idx))
        info.append({
            "cluster_id": j,
            "center_az": float(centers_az[j]),
            "center_el": float(centers_el[j]),
            "frames_seen": frames_seen,
            "frame_count": len(frames_seen),
            "sample_size": int(idx.size),
        })
    return centers, labels, info, rc_deg

# ----------------------------- Plotting -----------------------------

def plot_clusters(az, el, labels, centers, rc_deg, out_path=None, show=False, title=None):
    K = centers.shape[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal', adjustable='box')  # keep circles circular in deg
    try:
        ax.set_box_aspect(1)                  # square box if supported
    except Exception:
        pass

    sc = ax.scatter(az, el, c=labels, s=5, alpha=0.75, cmap="tab20", edgecolors="none")
    ax.scatter(centers[:,0], centers[:,1], s=50, marker="x", linewidths=1.5, color="k", label="cluster centers")

    # Draw radius circles (as Ellipses with equal width/height in data coords)
    for j in range(K):
        ax.add_patch(Ellipse(
            (centers[j,0], centers[j,1]),
            width=2*rc_deg, height=2*rc_deg,
            fill=False, linewidth=0.8, alpha=0.7,
            transform=ax.transData
        ))

    ax.set_xlabel("Azimuth [deg]")
    ax.set_ylabel("Elevation [deg]")
    ax.set_title(title or f"Angular Clusters (rc = {rc_deg:.4f}°)")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=160)
        print(f"[OK] Saved plot → {out_path}")
    if show:
        plt.show()
    plt.close(fig)

def plot_by_frame(az, el, frame_ids, out_path=None, show=False, title=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal', adjustable='box')
    try:
        ax.set_box_aspect(1)
    except Exception:
        pass
    sc = ax.scatter(az, el, c=frame_ids, s=5, alpha=0.8, cmap="viridis", edgecolors="none")
    ax.set_xlabel("Azimuth [deg]")
    ax.set_ylabel("Elevation [deg]")
    ax.set_title(title or "Points colored by frame id")
    ax.grid(True, linestyle="--", alpha=0.25)
    cb = fig.colorbar(sc, ax=ax, label="frame id")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160)
        print(f"[OK] Saved plot → {out_path}")
    if show:
        plt.show()
    plt.close(fig)

# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Visualize angular clustering (DIN SAE style)")
    ap.add_argument("--folder", required=True, help="Folder with frame_*.pcd and target_vicinity.json")
    ap.add_argument("--rc_deg", type=float, default=None, help="Cluster radius in degrees (default: auto ~ 0.5*median step)")
    ap.add_argument("--subsample", type=int, default=0, help="Randomly subsample total points before clustering")
    ap.add_argument("--save", default=None, help="Path to save cluster plot (PNG)")
    ap.add_argument("--save_frames", default=None, help="Path to save by-frame plot (PNG)")
    ap.add_argument("--show", action="store_true", help="Show interactive windows")
    args = ap.parse_args()

    frame_ids, points, frames_list = _load_folder_points(args.folder, subsample=args.subsample)
    az, el, rng = cartesian_to_angular(points)
    centers, labels, info, rc = cluster_angular(az, el, frame_ids, rc_deg=args.rc_deg)

    print(f"[INFO] Auto rc_deg={rc:.4f}° (override with --rc_deg)")
    print(f"[INFO] Points: {len(az)} | Clusters: {centers.shape[0]} | Frames in slice: {len(frames_list)}")
    singles = sum(1 for c in info if c["frame_count"] == 1)
    print(f"[INFO] Singleton clusters (frame_count=1): {singles}")

    title = f"Angular Clusters • {os.path.basename(os.path.normpath(args.folder))} • rc={rc:.4f}°"
    plot_clusters(az, el, labels, centers, rc, out_path=args.save, show=args.show, title=title)

    if args.save_frames or args.show:
        title2 = f"By-frame view • {os.path.basename(os.path.normpath(args.folder))}"
        plot_by_frame(az, el, frame_ids, out_path=args.save_frames, show=args.show, title=title2)

if __name__ == "__main__":
    main()
