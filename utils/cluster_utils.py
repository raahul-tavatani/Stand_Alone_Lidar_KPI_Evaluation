# utils/cluster_utils.py
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

def _cartesian_to_angular(points_xyz: np.ndarray):
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    r = np.sqrt(x*x + y*y + z*z)
    az = np.degrees(np.arctan2(y, x))
    el = np.degrees(np.arctan2(z, np.sqrt(x*x + y*y)))
    return az.astype(np.float32), el.astype(np.float32), r.astype(np.float32)

def _median_step_deg(az, el, frame_ids, subsample=4000) -> float:
    steps = []
    for f in np.unique(frame_ids):
        idx = np.where(frame_ids == f)[0]
        if idx.size < 3:
            continue
        if idx.size > subsample:
            idx = np.random.choice(idx, subsample, replace=False)
        Aaz, Ael = az[idx], el[idx]
        A = np.vstack([Aaz, Ael]).T
        for i in range(A.shape[0]):
            da = (A[:, 0] - A[i, 0] + 180.0) % 360.0 - 180.0
            de =  A[:, 1] - A[i, 1]
            d  = np.hypot(da, de)
            d[i] = np.inf
            steps.append(d.min())
    return float(np.median(steps)) if steps else 0.1

def _fd_bin_width(r: np.ndarray) -> float:
    if r.size <= 1:
        return 0.10
    q75, q25 = np.percentile(r, [75, 25])
    iqr = max(q75 - q25, 1e-6)
    w = 2.0 * iqr * (r.size ** (-1/3))
    return float(np.clip(w, 0.03, 0.15))

def _smooth_hist(y: np.ndarray, passes: int = 2) -> np.ndarray:
    k = np.array([1.0, 2.0, 1.0], dtype=float)
    k /= k.sum()
    out = y.astype(float)
    for _ in range(max(0, passes)):
        out = np.convolve(out, k, mode="same")
    return out

def _find_range_peaks(r: np.ndarray,
                      min_sep_m: Optional[float] = None,
                      min_peak_frac: float = 0.03,
                      min_peak_points: int = 15,
                      valley_frac: float = 0.35) -> List[Tuple[float, int]]:
    r = r[np.isfinite(r)]
    n = r.size
    if n == 0:
        return []
    if n == 1:
        return [(float(r[0]), 1)]

    w = _fd_bin_width(r)
    if min_sep_m is None:
        min_sep_m = max(0.12, 2.0 * w)

    nbins = int(np.ceil((r.max() - r.min()) / w)) + 1
    if nbins < 2:
        return [(float(np.mean(r)), int(n))]

    hist, edges = np.histogram(r, bins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sm = _smooth_hist(hist, passes=2)

    thr = max(int(np.ceil(min_peak_frac * n)), int(min_peak_points))
    candidates = []
    for i in range(1, len(sm) - 1):
        if sm[i] >= sm[i-1] and sm[i] >= sm[i+1] and sm[i] >= thr:
            candidates.append((i, centers[i], sm[i]))

    if not candidates:
        idx = int(np.argmax(sm))
        idx = max(0, min(idx, len(centers) - 1))   # clamp to valid center index
        return [(float(centers[idx]), int(n))]

    candidates.sort(key=lambda t: -t[2])

    chosen = []
    used = np.zeros(len(sm), dtype=bool)
    for idx, c_center, c_height in candidates:
        if used[idx]:
            continue
        ok = True
        for jdx, _, _ in chosen:
            if abs(centers[idx] - centers[jdx]) < min_sep_m:
                ok = False
                break
        if not ok:
            continue
        left = idx - 1
        while left > 0 and sm[left] <= sm[left+1]:
            left -= 1
        right = idx + 1
        while right < len(sm) - 1 and sm[right] <= sm[right-1]:
            right += 1
        # valley check against any already chosen neighbor
        for jdx, _, jh in chosen:
            a, b = (jdx, idx) if jdx < idx else (idx, jdx)
            if b - a <= 1:
                continue
            valley = np.min(sm[a:b+1])
            if valley > valley_frac * min(sm[a], sm[b]):
                ok = False
                break
        if not ok:
            continue
        chosen.append((idx, c_center, c_height))
        used[idx] = True

    if not chosen:
        chosen = [(candidates[0][0], candidates[0][1], candidates[0][2])]

    chosen.sort(key=lambda t: t[0])
    boundaries = []
    for k in range(len(chosen) - 1):
        i1 = chosen[k][0]; i2 = chosen[k+1][0]
        if i2 - i1 <= 1:
            split_edge = 0.5 * (edges[i1+1] + edges[i2])
        else:
            seg = sm[i1:i2+1]
            valley_rel = i1 + int(np.argmin(seg))
            split_edge = edges[valley_rel+1]
        boundaries.append(split_edge)

    counts = []
    for k, (idx, c_center, _) in enumerate(chosen):
        if len(chosen) == 1:
            c = int(hist.sum())
        else:
            if k == 0:
                left_edge = edges[0]
                right_edge = boundaries[0]
            elif k == len(chosen) - 1:
                left_edge = boundaries[-1]
                right_edge = edges[-1]
            else:
                left_edge = boundaries[k-1]
                right_edge = boundaries[k]
        if len(chosen) == 1:
            mask = (r >= edges[0]) & (r <= edges[-1])
        else:
            mask = (r >= left_edge) & (r < right_edge)
        c = int(np.count_nonzero(mask))
        counts.append(c)

    peaks = [(float(chosen[i][1]), int(counts[i])) for i in range(len(chosen))]
    return peaks

def track_clusters_data_driven(
    all_points_by_frame: Dict[int, np.ndarray],
    rc_deg: Optional[float] = None,
    frame_dt: Optional[float] = None,
    min_sep_m: Optional[float] = None,
    min_peak_frac: float = 0.03,
    min_peak_points: int = 15,
    valley_frac: float = 0.35,
) -> List[Dict[str, Any]]:

    azs, els, rngs, fids = [], [], [], []
    for f, pts in all_points_by_frame.items():
        if not isinstance(pts, np.ndarray) or pts.ndim != 2 or pts.shape[1] != 3 or pts.size == 0:
            continue
        a, e, r = _cartesian_to_angular(pts)
        azs.append(a); els.append(e); rngs.append(r)
        fids.append(np.full(len(a), int(f), dtype=np.int32))
    if not azs:
        return []

    az = np.concatenate(azs); el = np.concatenate(els); rr = np.concatenate(rngs); frame_ids = np.concatenate(fids)

    if rc_deg is None:
        step = _median_step_deg(az, el, frame_ids)
        rc_deg = 0.5 * step if step > 0 else 0.05
    if rc_deg > 0.1:
        print(f"[WARN] rc_deg={rc_deg:.3f}° > 0.1° — sensor sampling may be too coarse for fine separability.")

    cell = rc_deg
    nbin_az = int(np.ceil(360.0 / cell)) if cell > 0 else 36000
    def _bin_az(a): return (np.floor((a + 180.0)/cell).astype(np.int32)) % nbin_az
    def _bin_el(e): return np.floor((e + 90.0)/cell).astype(np.int32)

    baz = _bin_az(az); bel = _bin_el(el)
    grid = defaultdict(list)
    centers_az: List[float] = []
    centers_el: List[float] = []
    members: List[List[int]]  = []

    for i in range(az.shape[0]):
        cx, cy = baz[i], bel[i]
        best_j, best_d = -1, 1e9
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
        else:
            j = len(centers_az)
            centers_az.append(float(az[i])); centers_el.append(float(el[i]))
            members.append([i])
            grid[(cx, cy)].append(j)

    for j in range(len(centers_az)):
        idx = np.asarray(members[j], dtype=np.int32)
        if idx.size < 2:
            continue
        az_rad = np.radians(az[idx])
        cx = float(np.cos(az_rad).mean()); sx = float(np.sin(az_rad).mean())
        centers_az[j] = np.degrees(np.arctan2(sx, cx))
        centers_el[j] = float(el[idx].mean())

    all_frames_sorted = sorted(set(int(f) for f in frame_ids))
    total_duration = (len(all_frames_sorted) * frame_dt) if frame_dt else None

    cluster_map: List[Dict[str, Any]] = []
    for j in range(len(centers_az)):
        idx = np.asarray(members[j], dtype=np.int32)
        if idx.size == 0:
            continue

        frames_seen = sorted(set(int(frame_ids[k]) for k in idx))
        num_points_per_frame: Dict[int, int] = {}
        num_range_groups_per_frame: Dict[int, int] = {}
        range_peaks_per_frame: Dict[int, List[float]] = {}
        peak_counts_per_frame: Dict[int, List[int]] = {}

        two_hits = 0
        for f in frames_seen:
            m = (frame_ids[idx] == f)
            r_f = rr[idx][m]
            npts = int(r_f.size)
            num_points_per_frame[f] = npts

            peaks = _find_range_peaks(r_f, min_sep_m=min_sep_m,
                                      min_peak_frac=min_peak_frac,
                                      min_peak_points=min_peak_points,
                                      valley_frac=valley_frac)
            num_range_groups_per_frame[f] = len(peaks)
            range_peaks_per_frame[f] = [float(p[0]) for p in peaks]
            peak_counts_per_frame[f] = [int(p[1]) for p in peaks]
            if len(peaks) >= 2:
                two_hits += 1

        two_range_fraction = (two_hits / len(frames_seen)) if frames_seen else 0.0
        dets_total_1pf = sum(1 for f in frames_seen if num_points_per_frame[f] > 0)
        rr_hz = (dets_total_1pf / total_duration) if total_duration else None

        cluster_map.append({
            "cluster_id": j,
            "azimuth": round(centers_az[j], 4),
            "elevation": round(centers_el[j], 4),
            "range_mean": round(float(rr[idx].mean()), 3),
            "frames_seen": frames_seen,
            "frame_count": len(frames_seen),
            "sample_size": int(idx.size),
            "num_points_per_frame": num_points_per_frame,
            "num_range_groups_per_frame": num_range_groups_per_frame,
            "range_peaks_per_frame": range_peaks_per_frame,
            "peak_counts_per_frame": peak_counts_per_frame,
            "two_range_fraction": round(float(two_range_fraction), 3),
            "revisit_rate_hz": rr_hz,
        })

    return cluster_map
