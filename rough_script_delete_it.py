#!/usr/bin/env python3
# viz_clusters.py
# Visualize DIN SAE 91471-style angular clustering (azimuth/elevation).

import os, re, json, argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from collections import defaultdict

# ----------------------------- IO & Filtering ---------# We'll create a DataFrame from the user's pasted KPI table (3 rotations x 10 distances).
# Then we'll make a few plots and save them as PNGs, and also show the table back.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import tan, radians
from caas_jupyter_tools import display_dataframe_to_user

# Assemble the data: (rotation, distance, frame_dt_s, targ_az, targ_el, KPI1, KPI2, KPI3, KPI8, KPI9_az_std, KPI9_el_std, KPI10_az_abs, KPI10_el_abs, KPI10_mag_abs)
rows = [
    # rotation_1_pitch90_yaw0_roll0
    ("rotation_1_pitch90_yaw0_roll0", 10, 0.1, 6.12481, 5.92481, 0.552409, 240.187337, 5.524094, 0.008526, 0.014088, 0.015973, 0.011695, 0.126631, 0.127326),
    ("rotation_1_pitch90_yaw0_roll0", 20, 0.1, 3.264192, 3.064192, 0.55126, 218.253875, 5.512597, 0.004268, 0.016613, 0.017988, 0.014299, 0.123787, 0.124813),
    ("rotation_1_pitch90_yaw0_roll0", 30, 0.1, 2.309683, 2.109683, 0.560773, 185.728774, 5.607735, 0.002615, 0.018397, 0.01538, 0.015794, 0.069577, 0.072519),
    ("rotation_1_pitch90_yaw0_roll0", 40, 0.1, 1.83232, 1.63232, 0.542153, 183.220469, 5.421533, 0.002164, 0.021871, 0.012286, 0.015839, 0.116491, 0.118221),
    ("rotation_1_pitch90_yaw0_roll0", 50, 0.1, 1.545877, 1.345877, 0.553211, 157.169115, 5.53211, 0.001538, 0.018344, 0.01313, 0.015204, 0.063719, 0.06612),
    ("rotation_1_pitch90_yaw0_roll0", 60, 0.1, 1.354908, 1.154908, 0.567033, 174.463974, 5.67033, 0.001588, 0.010943, 0.020001, 0.008498, 0.072788, 0.07365),
    ("rotation_1_pitch90_yaw0_roll0", 70, 0.1, 1.218497, 1.018497, 0.531818, 124.089871, 5.318182, 0.001066, 0.008166, 0.012294, 0.006579, 0.126665, 0.126897),
    ("rotation_1_pitch90_yaw0_roll0", 80, 0.1, 1.116188, 0.916188, 0.514493, 134.945128, 5.144928, 0.00107, 0.016104, 0.015123, 0.013436, 0.128242, 0.129194),
    ("rotation_1_pitch90_yaw0_roll0", 90, 0.1, 1.036613, 0.836613, 0.514876, 139.522388, 5.14876, 0.001047, 0.010218, 0.013611, 0.008517, 0.11725, 0.11764),
    ("rotation_1_pitch90_yaw0_roll0", 100, 0.1, 0.972953, 0.772953, 0.538182, 73.133729, 5.381818, 0.000389, 0.02935, 0.0, 0.023344, 0.0635, 0.069051),
    # rotation_2_pitch90_yaw60_roll0
    ("rotation_2_pitch90_yaw60_roll0", 10, 0.1, 5.052414, 6.270441, 0.552054, 136.80184, 5.520535, -0.022756, 0.013305, 0.024063, 0.115828, 0.03312, 0.122038),
    ("rotation_2_pitch90_yaw60_roll0", 20, 0.1, 3.6213, 3.172024, 0.551087, 96.109875, 5.51087, -0.008699, 0.014013, 0.01982, 0.024726, 0.133085, 0.135931),
    ("rotation_2_pitch90_yaw60_roll0", 30, 0.1, 3.142764, 2.168003, 0.549237, 67.366032, 5.492375, -0.010301, 0.012582, 0.021555, 0.01583, 0.069517, 0.072126),
    ("rotation_2_pitch90_yaw60_roll0", 40, 0.1, 2.903363, 1.671338, 0.548913, 56.877882, 5.48913, -0.011557, 0.012332, 0.02542, 0.012726, 0.116498, 0.117837),
    ("rotation_2_pitch90_yaw60_roll0", 50, 0.1, 2.7597, 1.375042, 0.563636, 43.481676, 5.636364, -0.015622, 0.010491, 0.01391, 0.01412, 0.068607, 0.070689),
    ("rotation_2_pitch90_yaw60_roll0", 60, 0.1, 2.663919, 1.178218, 0.564964, 43.648948, 5.649635, -0.0163, 0.012321, 0.021465, 0.012636, 0.063087, 0.065598),
    ("rotation_2_pitch90_yaw60_roll0", 70, 0.1, 2.595502, 1.037977, 0.55375, 29.69484, 5.5375, -0.020876, 0.00538, 0.019868, 0.010698, 0.120027, 0.120614),
    ("rotation_2_pitch90_yaw60_roll0", 80, 0.1, 2.54419, 0.932984, 0.519118, 28.64739, 5.191176, -0.021096, 0.009839, 0.026289, 0.010721, 0.131361, 0.132052),
    ("rotation_2_pitch90_yaw60_roll0", 90, 0.1, 2.504281, 0.851435, 0.5, 29.077494, 5.0, -0.000458, 0.009432, 0.019087, 0.007442, 0.125557, 0.125862),
    ("rotation_2_pitch90_yaw60_roll0", 100, 0.1, 2.472353, 0.786267, 0.546429, 14.403815, 5.464286, -0.031, 0.017648, 0.0, 0.016412, 0.0635, 0.066375),
    # rotation_3_pitch60_yaw0_roll0
    ("rotation_3_pitch60_yaw0_roll0", 10, 0.1, 6.296438, 6.269271, 0.549796, 192.860753, 5.497964, -0.003634, 0.016625, 0.00952, 0.013403, 0.131979, 0.133),
    ("rotation_3_pitch60_yaw0_roll0", 20, 0.1, 3.314, 3.795592, 0.549869, 151.606785, 5.498689, 0.016865, 0.016241, 0.00884, 0.016597, 0.059779, 0.062887),
    ("rotation_3_pitch60_yaw0_roll0", 30, 0.1, 2.335169, 2.969923, 0.560573, 130.924923, 5.605727, 0.020572, 0.020561, 0.019051, 0.015939, 0.056568, 0.060266),
    ("rotation_3_pitch60_yaw0_roll0", 40, 0.1, 1.848679, 2.556963, 0.564706, 86.312607, 5.647059, 0.026639, 0.01318, 0.007004, 0.011871, 0.061246, 0.062833),
    ("rotation_3_pitch60_yaw0_roll0", 50, 0.1, 1.557729, 2.309156, 0.557492, 90.908121, 5.574924, 0.034336, 0.013507, 0.017539, 0.010406, 0.064352, 0.06592),
    ("rotation_3_pitch60_yaw0_roll0", 60, 0.1, 1.364156, 2.143942, 0.55, 62.229228, 5.5, -0.071381, 0.016119, 0.016552, 0.011567, 0.120422, 0.121423),
    ("rotation_3_pitch60_yaw0_roll0", 70, 0.1, 1.226083, 2.025927, 0.54359, 62.803051, 5.435897, -0.082562, 0.015771, 0.015406, 0.013542, 0.119159, 0.120289),
    ("rotation_3_pitch60_yaw0_roll0", 80, 0.1, 1.122635, 1.937414, 0.536232, 63.447999, 5.362319, -0.095281, 0.01462, 0.018804, 0.011569, 0.119858, 0.120699),
    ("rotation_3_pitch60_yaw0_roll0", 90, 0.1, 1.042238, 1.86857, 0.52459, 31.322283, 5.245902, 0.058228, 0.018405, 0.0, 0.016237, 0.0635, 0.066135),
    ("rotation_3_pitch60_yaw0_roll0", 100, 0.1, 0.977961, 1.813494, 0.492727, 31.011676, 4.927273, 0.064476, 0.033442, 0.0, 0.027366, 0.0635, 0.070737),
]

cols = ["rotation","distance_m","frame_dt_s","target_extent_az_deg","target_extent_el_deg","KPI_1_TP_Prob","KPI_2_Angular_Cluster_Density_deg^-2","KPI_3_Revisit_Rate","KPI_8_Radial_Accuracy_m","KPI_9_az_std_deg","KPI_9_el_std_deg","KPI_10_az_abs_deg","KPI_10_el_abs_deg","KPI_10_mag_abs_deg"]
df = pd.DataFrame(rows, columns=cols)

# Derived fields
df["frame_rate_hz"] = 1.0/df["frame_dt_s"]
df["KPI_3_from_KPI_1"] = df["KPI_1_TP_Prob"] * df["frame_rate_hz"]
df["KPI_3_diff"] = df["KPI_3_Revisit_Rate"] - df["KPI_3_from_KPI_1"]

# Show the dataframe to the user
display_dataframe_to_user("KPI summary (parsed from your table)", df)

# --- Plot 1: KPI-1 vs distance (lines per rotation) ---
plt.figure()
for rot, g in df.groupby("rotation"):
    g2 = g.sort_values("distance_m")
    plt.plot(g2["distance_m"], g2["KPI_1_TP_Prob"], marker="o", label=rot)
plt.xlabel("Distance (m)")
plt.ylabel("KPI-1: TP Probability (per frame)")
plt.title("KPI-1 vs Distance")
plt.legend()
plt.tight_layout()
plt.savefig("/mnt/data/kpi1_vs_distance.png", dpi=160)
plt.close()

# --- Plot 2: KPI-3 vs distance (lines per rotation) ---
plt.figure()
for rot, g in df.groupby("rotation"):
    g2 = g.sort_values("distance_m")
    plt.plot(g2["distance_m"], g2["KPI_3_Revisit_Rate"], marker="o", label=rot)
plt.xlabel("Distance (m)")
plt.ylabel("KPI-3: Revisit Rate (Hz)")
plt.title("KPI-3 vs Distance")
plt.legend()
plt.tight_layout()
plt.savefig("/mnt/data/kpi3_vs_distance.png", dpi=160)
plt.close()

# --- Plot 3: KPI-2 (Angular Cluster Density) vs distance ---
plt.figure()
for rot, g in df.groupby("rotation"):
    g2 = g.sort_values("distance_m")
    plt.plot(g2["distance_m"], g2["KPI_2_Angular_Cluster_Density_deg^-2"], marker="o", label=rot)
plt.xlabel("Distance (m)")
plt.ylabel("KPI-2: Angular Cluster Density (deg$^{-2}$)")
plt.title("KPI-2 vs Distance (check radius consistency)")
plt.legend()
plt.tight_layout()
plt.savefig("/mnt/data/kpi2_vs_distance.png", dpi=160)
plt.close()

# --- Plot 4: KPI-8 (Radial Accuracy) vs distance ---
plt.figure()
for rot, g in df.groupby("rotation"):
    g2 = g.sort_values("distance_m")
    plt.plot(g2["distance_m"], g2["KPI_8_Radial_Accuracy_m"], marker="o", label=rot)
plt.xlabel("Distance (m)")
plt.ylabel("KPI-8: Radial Accuracy (m)")
plt.title("KPI-8 vs Distance")
plt.legend()
plt.tight_layout()
plt.savefig("/mnt/data/kpi8_vs_distance.png", dpi=160)
plt.close()

# --- Plot 5: KPI-10 magnitude abs deg vs distance ---
plt.figure()
for rot, g in df.groupby("rotation"):
    g2 = g.sort_values("distance_m")
    plt.plot(g2["distance_m"], g2["KPI_10_mag_abs_deg"], marker="o", label=rot)
plt.xlabel("Distance (m)")
plt.ylabel("KPI-10: Angular Accuracy |mag| (deg)")
plt.title("KPI-10 (magnitude) vs Distance")
plt.legend()
plt.tight_layout()
plt.savefig("/mnt/data/kpi10_mag_vs_distance.png", dpi=160)
plt.close()

# --- Plot 6: KPI-9 az/el std vs distance (averaged for simplicity) ---
plt.figure()
for rot, g in df.groupby("rotation"):
    g2 = g.sort_values("distance_m")
    plt.plot(g2["distance_m"], g2["KPI_9_az_std_deg"], marker="o", label=f"{rot} az std")
    plt.plot(g2["distance_m"], g2["KPI_9_el_std_deg"], marker="x", label=f"{rot} el std")
plt.xlabel("Distance (m)")
plt.ylabel("KPI-9: Angular Precision (std deg)")
plt.title("KPI-9 (az & el std) vs Distance")
plt.legend()
plt.tight_layout()
plt.savefig("/mnt/data/kpi9_std_vs_distance.png", dpi=160)
plt.close()

# Save a quick textual summary file
summary_lines = []
summary_lines.append(f"Rows: {len(df)}")
summary_lines.append(f"KPI-3 minus KPI-1*frame_rate -> mean diff: {df['KPI_3_diff'].mean():.6f}, max abs diff: {df['KPI_3_diff'].abs().max():.6f}")
for rot, g in df.groupby("rotation"):
    g2 = g.sort_values("distance_m")
    summary_lines.append(f"\n[{rot}]")
    summary_lines.append(f"  KPI-1 mean: {g2['KPI_1_TP_Prob'].mean():.4f}, std: {g2['KPI_1_TP_Prob'].std():.4f}")
    summary_lines.append(f"  KPI-2 mean: {g2['KPI_2_Angular_Cluster_Density_deg^-2'].mean():.2f}")
    summary_lines.append(f"  KPI-8 mean (m): {g2['KPI_8_Radial_Accuracy_m'].mean():.4f}")
    summary_lines.append(f"  KPI-10 |mag| mean (deg): {g2['KPI_10_mag_abs_deg'].mean():.4f}")

with open("/mnt/data/kpi_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))

# Return paths of images to the user by printing them (the assistant will link them)
print("Saved plots:")
print("/mnt/data/kpi1_vs_distance.png")
print("/mnt/data/kpi3_vs_distance.png")
print("/mnt/data/kpi2_vs_distance.png")
print("/mnt/data/kpi8_vs_distance.png")
print("/mnt/data/kpi10_mag_vs_distance.png")
print("/mnt/data/kpi9_std_vs_distance.png")
print("Summary text: /mnt/data/kpi_summary.txt")
--------------------

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
      frames_list: sorted unique frame ids (for info)
    """
    vjson = os.path.join(folder, "target_vicinity.json")
    if not os.path.exists(vjson):
        raise FileNotFoundError(f"target_vicinity.json not found in {folder}")

    corners = _load_vicinity_corners(vjson)

    # Prefer already filtered file if present
    pref_pcd = os.path.join(folder, "target_vicinity_points.pcd")
    points_list = []
    frame_ids_list = []

    if os.path.exists(pref_pcd):
        # All frames concatenated without per-frame IDs—fallback: parse frame_*.pcd anyway
        pass  # we'll still iterate frames for correct frame IDs

    pcd_files = sorted([f for f in os.listdir(folder) if re.match(r"frame_\d+\.pcd", f)])
    if not pcd_files:
        raise RuntimeError(f"No frame_*.pcd in {folder}")

    for fname in pcd_files:
        f_id = int(re.findall(r"\d+", fname)[0])
        pts = _extract_points_in_target_vicinity(os.path.join(folder, fname), corners)
        if isinstance(pts, np.ndarray) and pts.ndim == 2 and pts.shape[1] == 3 and pts.size > 0:
            points_list.append(pts.astype(np.float32, copy=False))
            frame_ids_list.append(np.full(pts.shape[0], f_id, dtype=np.int32))

    if not points_list:
        raise RuntimeError(f"No points inside target vicinity for {folder}")

    points = np.vstack(points_list)
    frame_ids = np.concatenate(frame_ids_list)

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
    sc = ax.scatter(az, el, c=labels, s=5, alpha=0.75, cmap="tab20", edgecolors="none")
    ax.scatter(centers[:,0], centers[:,1], s=50, marker="x", linewidths=1.5, color="k", label="cluster centers")

    # radius circles
    for j in range(K):
        circ = Circle((centers[j,0], centers[j,1]), rc_deg, fill=False, linewidth=0.6, alpha=0.6)
        ax.add_patch(circ)

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
    # Optional second view: color by frame id
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(az, el, c=frame_ids, s=5, alpha=0.75, cmap="viridis", edgecolors="none")
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
    ap = argparse.ArgumentParser(description="Visualize angular clustering (DIN SAE 91471 style)")
    ap.add_argument("--folder", required=True, help="Folder containing frame_*.pcd and target_vicinity.json")
    ap.add_argument("--rc_deg", type=float, default=None, help="Cluster radius in degrees (default: auto ~ 0.5*median step)")
    ap.add_argument("--subsample", type=int, default=0, help="Randomly subsample total points to this many before clustering")
    ap.add_argument("--save", default=None, help="Path to save cluster plot (PNG).")
    ap.add_argument("--save_frames", default=None, help="Optional: save 'by-frame' plot (PNG).")
    ap.add_argument("--show", action="store_true", help="Show interactive windows")
    args = ap.parse_args()

    frame_ids, points, frames_list = _load_folder_points(args.folder, subsample=args.subsample)
    az, el, rng = cartesian_to_angular(points)
    centers, labels, info, rc = cluster_angular(az, el, frame_ids, rc_deg=args.rc_deg)

    print(f"[INFO] Auto rc_deg={rc:.4f}° (override with --rc_deg)")
    print(f"[INFO] Points: {len(az)} | Clusters: {centers.shape[0]} | Frames in slice: {len(frames_list)}")
    # Quick summary
    singles = sum(1 for c in info if c["frame_count"] == 1)
    print(f"[INFO] Singleton clusters (frame_count=1): {singles}")

    title = f"Angular Clusters • {os.path.basename(os.path.normpath(args.folder))} • rc={rc:.4f}°"
    plot_clusters(az, el, labels, centers, rc, out_path=args.save, show=args.show, title=title)

    if args.save_frames or args.show:
        title2 = f"By-frame view • {os.path.basename(os.path.normpath(args.folder))}"
        plot_by_frame(az, el, frame_ids, out_path=args.save_frames, show=args.show, title=title2)

if __name__ == "__main__":
    main()
