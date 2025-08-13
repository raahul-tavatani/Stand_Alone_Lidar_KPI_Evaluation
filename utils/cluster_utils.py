import numpy as np
from collections import defaultdict

def _cartesian_to_angular(points_xyz: np.ndarray):
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    r = np.sqrt(x*x + y*y + z*z)
    az = np.degrees(np.arctan2(y, x))
    el = np.degrees(np.arctan2(z, np.sqrt(x*x + y*y)))  # stable near horizon
    return az.astype(np.float32), el.astype(np.float32), r.astype(np.float32)

def _median_step_deg(az, el, frame_ids, subsample=4000):
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
            da = (A[:,0] - A[i,0] + 180.0) % 360.0 - 180.0
            de =  A[:,1] - A[i,1]
            d = np.hypot(da, de)
            d[i] = np.inf
            steps.append(d.min())
    return float(np.median(steps)) if steps else 0.1

def track_clusters_data_driven(all_points_by_frame: dict, rc_deg=None, frame_dt=None):
    """Spec-conformant angular clustering with grid-hash NN (near O(N)). Returns a list of cluster dicts."""
    # flatten frames with guards
    azs, els, rngs, fids = [], [], [], []
    for f, pts in all_points_by_frame.items():
        if not isinstance(pts, np.ndarray) or pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
            continue
        a, e, r = _cartesian_to_angular(pts)
        azs.append(a); els.append(e); rngs.append(r)
        fids.append(np.full(len(a), int(f), dtype=np.int32))
    if not azs:
        return []

    az = np.concatenate(azs); el = np.concatenate(els); rr = np.concatenate(rngs); frame_ids = np.concatenate(fids)

    # choose radius ~ half the sampling step (aim: ~1 det/frame per cluster)
    if rc_deg is None:
        step = _median_step_deg(az, el, frame_ids)
        rc_deg = 0.5 * step if step > 0 else 0.05
    # optional: warn if >0.1 deg (per spec guidance)
    if rc_deg > 0.1:
        print(f"[WARN] rc_deg={rc_deg:.3f}° > 0.1° — sensor sampling may be too coarse for long durations")

    # grid-hash over (az, el)
    cell = rc_deg
    nbin_az = int(np.ceil(360.0 / cell))
    def _bin_az(a):  # wrap
        return ((np.floor((a + 180.0)/cell)).astype(np.int32)) % nbin_az
    def _bin_el(e):
        return np.floor((e + 90.0)/cell).astype(np.int32)

    baz = _bin_az(az); bel = _bin_el(el)
    grid = defaultdict(list)          # (baz, bel) -> [cluster_ids]
    centers_az, centers_el = [], []
    members = []                      # list of index lists
    per_frame_counts = []             # list of dict frame->count

    for i in range(az.shape[0]):
        cx, cy = baz[i], bel[i]
        best_j, best_d = -1, 1e9
        # search 3x3 neighborhood
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
            f = int(frame_ids[i])
            per_frame_counts[best_j][f] = per_frame_counts[best_j].get(f, 0) + 1
        else:
            j = len(centers_az)
            centers_az.append(float(az[i])); centers_el.append(float(el[i]))
            members.append([i]); per_frame_counts.append({int(frame_ids[i]): 1})
            grid[(cx, cy)].append(j)

    # recentre once (mean on unit circle for az, arithmetic for el)
    for j in range(len(centers_az)):
        idx = np.asarray(members[j], dtype=np.int32)
        if idx.size < 2:
            continue
        az_rad = np.radians(az[idx])
        cx = float(np.cos(az_rad).mean()); sx = float(np.sin(az_rad).mean())
        centers_az[j] = np.degrees(np.arctan2(sx, cx))
        centers_el[j] = float(el[idx].mean())

    # build output (cap to 1 det/frame for revisit metrics if frame_dt given)
    all_frames_sorted = sorted(set(int(f) for f in frame_ids))
    total_duration = (len(all_frames_sorted) * frame_dt) if frame_dt else None

    cluster_map = []
    for j in range(len(centers_az)):
        idx = np.asarray(members[j], dtype=np.int32)
        frames_seen = sorted(set(int(frame_ids[k]) for k in idx))
        dets_capped = {f: 1 if per_frame_counts[j].get(f, 0) > 0 else 0 for f in frames_seen}
        dets_total_1pf = sum(dets_capped.values())
        rr_hz = (dets_total_1pf / total_duration) if total_duration else None

        cluster_map.append({
            "cluster_id": j,
            "azimuth": round(centers_az[j], 4),
            "elevation": round(centers_el[j], 4),
            "range": round(float(rr[idx].mean()), 3),
            "range_mean": round(float(rr[idx].mean()), 3), 
            "frames_seen": frames_seen,
            "frame_count": len(frames_seen),
            "sample_size": int(idx.size),
            "detections_per_frame": dets_capped,
            "revisit_rate_hz": rr_hz,
        })
    return cluster_map
