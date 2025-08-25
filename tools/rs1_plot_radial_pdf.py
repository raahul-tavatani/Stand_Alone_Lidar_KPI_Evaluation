# tools/rs1_plot_radial_pdf.py
"""
Plot a DIN/SAE-style radial PDF (relative detection probability vs range)
for one tube (cell) produced by evaluation/rs1_grid_tubes.py.

- By default, reads pdf_centers/pdf_probs from grid_tubes_map.csv and plots them.
- If those fields are empty or you pass --recompute, it reconstructs the radial
  distribution from raw PCDs using the tube geometry stored in the same CSV row.

Usage (Windows cmd.exe):
  python tools/rs1_plot_radial_pdf.py ^
    --gap-dir .\outputs\rs1\scenario_001\d0_10m\gap_0.05m ^
    --cell-id 33 ^
    --save-png radial_pdf_cell33.png

Optional:
  --csv-name grid_tubes_map.csv
  --recompute              # force recompute from frames (needs open3d)
  --min-sep-m 0.05 --min-peak-points 2 --min-peak-frac 0.0 --valley-frac 0.7
"""

import os
import re
import csv
import json
import math
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# Make local project modules importable when running as a plain script
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Optional (only needed if --recompute or CSV lacks PDF)
try:
    import open3d as o3d
except Exception:
    o3d = None

# We reuse your peak finder
from utils.cluster_utils import _find_range_peaks


# ----------------------------- CSV helpers -----------------------------

def _parse_float_list(s: str) -> List[float]:
    if s is None:
        return []
    s = s.strip()
    if not s or s == "[]":
        return []
    if s[0] == "[" and s[-1] == "]":
        s = s[1:-1]
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    out: List[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            pass
    return out


def _load_row_for_cell(csv_path: str, cell_id: int) -> Dict:
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            # robust to string/int cell_id
            cid = int(row["cell_id"])
            if cid == cell_id:
                return row
    raise KeyError(f"cell_id={cell_id} not found in {csv_path}")


# --------------------------- Recompute PDF -----------------------------

def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

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
    return o3d.geometry.OrientedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(corners)
    )

def _read_frames(gap_dir: str) -> Dict[int, np.ndarray]:
    files = sorted([f for f in os.listdir(gap_dir) if re.match(r"^frame_\d+\.pcd$", f)])
    out = {}
    for f in files:
        fid = int(re.findall(r"\d+", f)[0])
        path = os.path.join(gap_dir, f)
        pc = o3d.io.read_point_cloud(path)
        pts = np.asarray(pc.points, dtype=np.float32)
        if pts.size:
            out[fid] = pts
    return out

def _deg2rad(x): return x * math.pi / 180.0

def _dir_from_az_el_deg(az_deg: float, el_deg: float) -> np.ndarray:
    az = _deg2rad(az_deg); el = _deg2rad(el_deg)
    c = math.cos(el)
    return np.array([c*math.cos(az), c*math.sin(az), math.sin(el)], dtype=float)  # unit

def _fd_bin_width_local(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    if r.size <= 1:
        return 0.10
    q75, q25 = np.percentile(r, [75, 25])
    iqr = max(q75 - q25, 1e-6)
    w = 2.0 * iqr * (r.size ** (-1/3))
    return float(np.clip(w, 0.01, 0.15))


def recompute_pdf_from_frames(
    gap_dir: str,
    az_deg: float,
    el_deg: float,
    s_enter: float,
    s_exit: float,
    tube_radius: float,
    min_sep_m: float,
    min_peak_points: int,
    min_peak_frac: float,
    valley_frac: float,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float,int]]]:
    """
    Rebuild aggregated radial PDF for a single tube, and return:
      centers (m), probs (unit sum), peaks (list of (center,count))
    """
    if o3d is None:
        raise RuntimeError("open3d not available; install it or omit --recompute.")

    tv_path = os.path.join(gap_dir, "target_vicinity.json")
    if not os.path.exists(tv_path):
        raise FileNotFoundError(f"target_vicinity.json not found in {gap_dir}")

    obb = _obb_from_vicinity(tv_path)
    frames = _read_frames(gap_dir)
    if not frames:
        raise RuntimeError(f"No frame_*.pcd found in {gap_dir}")

    d = _dir_from_az_el_deg(float(az_deg), float(el_deg))  # unit
    r_accum: List[np.ndarray] = []

    for fid, pts in frames.items():
        # OBB clip (fast)
        idx = obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(pts))
        if not idx:
            continue
        P = pts[np.asarray(idx, dtype=int)]

        # axial coordinate
        s_vals = P @ d
        m_seg = (s_vals >= s_enter) & (s_vals <= s_exit)
        if not np.any(m_seg):
            continue
        Pseg = P[m_seg]
        s_seg = s_vals[m_seg]

        # radial distance to axis for tube check
        r2 = np.einsum("ij,ij->i", Pseg, Pseg)
        dist2_axis = r2 - s_seg*s_seg
        m_tube = dist2_axis <= (tube_radius * tube_radius + 1e-12)
        if not np.any(m_tube):
            continue

        # use ALL points (DIN/SAE radial metric): r = ||P||
        r_use = np.sqrt(r2[m_tube])
        if r_use.size:
            r_accum.append(r_use.astype(float, copy=False))

    if not r_accum:
        return np.array([]), np.array([]), []

    r_cat = np.concatenate(r_accum).astype(float, copy=False)
    bw = _fd_bin_width_local(r_cat)
    nbins = int(np.ceil((r_cat.max() - r_cat.min()) / bw)) + 1
    hist, edges = np.histogram(r_cat, bins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    probs = hist.astype(float)
    if probs.sum() > 0:
        probs /= probs.sum()

    peaks = _find_range_peaks(
        r_cat,
        min_sep_m=float(min_sep_m),
        min_peak_frac=float(min_peak_frac),
        min_peak_points=int(min_peak_points),
        valley_frac=float(valley_frac)
    )
    return centers, probs, peaks


# ------------------------------- Plotting -------------------------------

def _plot_pdf(centers: np.ndarray,
              probs: np.ndarray,
              cell_id: int,
              peaks: Optional[List[Tuple[float,int]]] = None,
              title_extra: str = ""):
    plt.figure(figsize=(8.5, 4.8))
    plt.plot(centers, probs, marker="o", linewidth=1)
    plt.xlabel("Range r (m)")
    plt.ylabel("Relative probability")
    ttl = f"RS-1 Radial PDF — cell {cell_id}"
    if title_extra:
        ttl += f"  {title_extra}"
    plt.title(ttl)
    plt.grid(alpha=0.3)

    if peaks:
        # sort by support descending for display
        peaks_sorted = sorted(peaks, key=lambda t: -t[1])
        for i, (pm, cnt) in enumerate(peaks_sorted[:4]):  # annotate up to 4
            plt.axvline(pm, linestyle="--", alpha=0.5)
            plt.text(pm, max(probs)*0.9 - i*0.08*max(probs),
                     f"peak {i+1}: {pm:.3f} m ({cnt})",
                     rotation=90, va="top", ha="center")

        if len(peaks_sorted) >= 2:
            delta = abs(peaks_sorted[0][0] - peaks_sorted[1][0])
            plt.text(0.02, 0.95, f"Δ between two strongest peaks ≈ {delta:.3f} m",
                     transform=plt.gca().transAxes, fontsize=9,
                     bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    plt.tight_layout()


# --------------------------------- CLI ----------------------------------

def main():
    ap = argparse.ArgumentParser(description="Plot radial PDF for one tube (cell)")
    ap.add_argument("--gap-dir", required=True, help="Path to .../d0_*m/gap_*m")
    ap.add_argument("--cell-id", type=int, required=True, help="Cell (tube) id to plot")
    ap.add_argument("--csv-name", default="grid_tubes_map.csv", help="CSV file in gap-dir")
    ap.add_argument("--save-png", default=None, help="If set, save plot to this path instead of showing")
    ap.add_argument("--recompute", action="store_true", help="Recompute PDF from frames using tube geometry in CSV row")

    # Peak-finding thresholds (used only when recomputing)
    ap.add_argument("--min-sep-m", type=float, default=0.05)
    ap.add_argument("--min-peak-points", type=int, default=2)
    ap.add_argument("--min-peak-frac", type=float, default=0.0)
    ap.add_argument("--valley-frac", type=float, default=0.7)

    args = ap.parse_args()

    csv_path = os.path.join(args.gap_dir, args.csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")

    row = _load_row_for_cell(csv_path, args.cell_id)

    # Prefer CSV-provided PDF
    centers_csv = _parse_float_list(row.get("pdf_centers", ""))
    probs_csv   = _parse_float_list(row.get("pdf_probs", ""))

    # Peaks from CSV (if present)
    peaks_csv = _parse_float_list(row.get("global_peaks_m", ""))
    peaks_for_annot = [(float(p), -1) for p in peaks_csv] if peaks_csv else None

    title_extra = ""

    if (not centers_csv or not probs_csv) or args.recompute:
        # Fallback / forced recompute from frames
        if o3d is None:
            raise RuntimeError("open3d is required for --recompute (pip install open3d)")
        az = float(row["azimuth_deg"])
        el = float(row["elevation_deg"])
        s_enter = float(row["s_enter_m"])
        s_exit  = float(row["s_exit_m"])
        r_tube  = float(row["tube_radius_m"])

        centers, probs, peaks = recompute_pdf_from_frames(
            gap_dir=args.gap_dir,
            az_deg=az, el_deg=el,
            s_enter=s_enter, s_exit=s_exit,
            tube_radius=r_tube,
            min_sep_m=args.min_sep_m,
            min_peak_points=args.min_peak_points,
            min_peak_frac=args.min_peak_frac,
            valley_frac=args.valley_frac
        )
        if centers.size == 0:
            raise RuntimeError("No points fell inside tube; cannot make PDF.")
        centers_csv = centers.tolist()
        probs_csv   = probs.tolist()
        peaks_for_annot = peaks
        title_extra = "(recomputed)"
    else:
        title_extra = "(from CSV)"

    _plot_pdf(
        centers=np.asarray(centers_csv, dtype=float),
        probs=np.asarray(probs_csv, dtype=float),
        cell_id=int(row["cell_id"]),
        peaks=peaks_for_annot,
        title_extra=title_extra
    )

    if args.save_png:
        out = args.save_png
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        plt.savefig(out, dpi=150)
        print(f"[OK] saved: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
