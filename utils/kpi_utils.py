# utils/kpi_utils.py

import numpy as np
from math import erf, sqrt, log

# --------- small helpers ---------

def _circ_diff_deg(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return d

def _circ_mean_deg(values_deg):
    vals = np.asarray(values_deg, dtype=float)
    ang = np.deg2rad(vals)
    c = np.cos(ang).mean()
    s = np.sin(ang).mean()
    return np.rad2deg(np.arctan2(s, c))

def _norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def _gauss_intersections(mu1, s1, mu2, s2):
    if s1 <= 0 or s2 <= 0:
        return []
    A = 1.0/(2*s1*s1) - 1.0/(2*s2*s2)
    B = -mu1/(s1*s1) + mu2/(s2*s2)
    C = (mu1*mu1)/(2*s1*s1) - (mu2*mu2)/(2*s2*s2) - log(max(s2,1e-12)/max(s1,1e-12))
    if abs(A) < 1e-12:
        if abs(B) < 1e-12:
            return []
        return [-C / B]
    disc = B*B - 4*A*C
    if disc < 0:
        return []
    rdisc = sqrt(max(0.0, disc))
    x1 = (-B - rdisc)/(2*A)
    x2 = (-B + rdisc)/(2*A)
    return [x1, x2] if x1 <= x2 else [x2, x1]

def _bayes_accuracy_two_normals(mu1, s1, mu2, s2):
    """KPI 11 helper: max accuracy (equal priors) for 1D Gaussians."""
    if s1 <= 0 or s2 <= 0:
        return float("nan")
    xs = _gauss_intersections(mu1, s1, mu2, s2)

    def P(mu, s, a, b):
        return max(0.0, _norm_cdf((b - mu)/s) - _norm_cdf((a - mu)/s))

    if not xs:  # no intersection, near-perfect separation
        d = abs(mu1 - mu2)/sqrt(s1*s1 + s2*s2)
        return float(0.5*(1 + erf(d/sqrt(2))))

    segs = [-np.inf] + xs + [np.inf]
    acc = 0.0
    for a, b in zip(segs[:-1], segs[1:]):
        m = 0.0 if (not np.isfinite(a) or not np.isfinite(b)) else 0.5*(a+b)
        p1 = (1.0/(s1*sqrt(2*np.pi))) * np.exp(-0.5*((m-mu1)/s1)**2)
        p2 = (1.0/(s2*sqrt(2*np.pi))) * np.exp(-0.5*((m-mu2)/s2)**2)
        if p1 >= p2:
            acc += 0.5 * P(mu1, s1, a, b)
        else:
            acc += 0.5 * P(mu2, s2, a, b)
    return float(acc)

# --------- unified KPI evaluator ---------

def evaluate_kpis(
    cluster_map,                  # TP clusters (or all, if single-target)
    target_distance: float,
    total_frames: int,
    gt_azimuth: float = 0.0,
    gt_elevation: float = 0.0,
    *,
    # Optional inputs (pass when available)
    frame_dt: float = None,                 # seconds per frame (for KPI 3 in Hz)
    target_extent_deg: tuple = None,        # (az_span_deg, el_span_deg); prefer real from vicinity corners
    target_size_m: tuple = (1.0, 1.0),      # used only if extent not given (fallback approx)
    cluster_map_fp: list = None,            # FP clusters (outside vicinity, inside control volume)
    fp_detections: list = None,             # list of (az,el) FP detections (preferred for KPI 5)
    surface_feature_samples: dict = None,   # for KPI 11; dict like {"A":[...], "B":[...]} e.g., intensity
    test_type: str = "auto",                # "auto" | "single_target" | "multi_domain" | "far"
):
    """
    Computes KPIs 1–5, 7–11 generically. Also KPI 6 (FAR) if test_type == "far".
    - Works with your new cluster schema:
        'azimuth','elevation','range' (or 'range_mean'),
        'frames_seen','frame_count','sample_size','detections_per_frame'
    - Missing pieces gracefully become None.
    """
    if not cluster_map or total_frames <= 0:
        return {}

    # --- normalize TP cluster fields ---
    az = np.array([c.get("azimuth", c.get("az")) for c in cluster_map], dtype=float)
    el = np.array([c.get("elevation", c.get("el")) for c in cluster_map], dtype=float)
    r  = np.array([c.get("range", c.get("range_mean")) for c in cluster_map], dtype=float)
    ss = np.array([int(c.get("sample_size", 0)) for c in cluster_map], dtype=int)

    def _cap_dpf(c):
        dpf = c.get("detections_per_frame", c.get("detections_per_frame_capped"))
        if isinstance(dpf, dict):
            return {int(k): int(v > 0) for k, v in dpf.items()}
        fs = c.get("frames_seen", [])
        return {int(f): 1 for f in fs}

    dpf_tp = [ _cap_dpf(c) for c in cluster_map ]

    # ---------- KPI 1: TP detection probability ----------
    pd_per_cluster = np.array([sum(d.values())/total_frames for d in dpf_tp], dtype=float)
    KPI_1 = float(np.round(pd_per_cluster.mean(), 6))

    # ---------- KPI 2: Angular cluster density (deg^-2) ----------
    if target_extent_deg is not None:
        az_span_deg, el_span_deg = target_extent_deg
    else:
        W, H = target_size_m
        half_az = np.degrees(np.arctan((W/2.0)/max(target_distance, 1e-6)))
        half_el = np.degrees(np.arctan((H/2.0)/max(target_distance, 1e-6)))
        az_span_deg, el_span_deg = 2*half_az, 2*half_el
    area_deg2 = max(az_span_deg, 0) * max(el_span_deg, 0)
    KPI_2 = float(np.round((len(cluster_map)/area_deg2) if area_deg2 > 0 else np.nan, 6))

    # ---------- KPI 3: Revisit rate ----------
    rr_counts = np.array([sum(d.values()) for d in dpf_tp], dtype=float)
    if frame_dt and frame_dt > 0:
        duration_s = total_frames * frame_dt
        KPI_3 = float(np.round((rr_counts/duration_s).mean(), 6))
    else:
        KPI_3 = float(np.round((rr_counts/total_frames).mean(), 6))  # per-frame normalized

    # ---------- KPI 4: False-positive detection rate ----------
    # Needs FP clusters (inside control volume, outside target).
    if cluster_map_fp:
        ss_fp = np.array([int(c.get("sample_size", 0)) for c in cluster_map_fp], dtype=int)
        total_det = int(ss.sum()) + int(ss_fp.sum())
        KPI_4 = float(np.round((ss_fp.sum()/total_det) if total_det > 0 else np.nan, 6))
    else:
        KPI_4 = None

    # ---------- KPI 5: False-positive detection deviation (deg) ----------
    if cluster_map_fp or fp_detections:
        if fp_detections and len(fp_detections) > 0:
            az_fp = np.array([a for a, _ in fp_detections], dtype=float)
            el_fp = np.array([e for _, e in fp_detections], dtype=float)
        else:
            az_fp = np.array([c.get("azimuth", c.get("az")) for c in cluster_map_fp], dtype=float)
            el_fp = np.array([c.get("elevation", c.get("el")) for c in cluster_map_fp], dtype=float)
        half_az = 0.5 * az_span_deg
        half_el = 0.5 * el_span_deg
        d_az = np.maximum(0.0, np.maximum(az_fp - half_az, -half_az - az_fp))
        d_el = np.maximum(0.0, np.maximum(el_fp - half_el, -half_el - el_fp))
        KPI_5 = {
            "azimuth_mean_dev_deg": float(np.round(np.mean(d_az), 6)) if d_az.size > 0 else None,
            "elevation_mean_dev_deg": float(np.round(np.mean(d_el), 6)) if d_el.size > 0 else None,
        }
    else:
        KPI_5 = None

    # ---------- KPI 7: Radial precision (weighted) ----------
    KPI_7 = None
    if all(("range_samples" in c and isinstance(c["range_samples"], (list, tuple)) and len(c["range_samples"]) > 1)
           for c in cluster_map):
        stds = np.array([np.std(c["range_samples"], ddof=1) for c in cluster_map], dtype=float)
        w = ss / max(ss.sum(), 1)
        KPI_7 = float(np.round(np.sum(w * stds), 6))

    # ---------- KPI 8: Radial accuracy (weighted mean error) ----------
    w = ss / max(ss.sum(), 1)
    KPI_8 = float(np.round(np.sum(w * (r - float(target_distance))), 6))

    # ---------- KPI 9: Angular precision ----------
    frame_ids = sorted({f for d in dpf_tp for f in d.keys()})
    frame_avg_az, frame_avg_el = [], []
    for f in frame_ids:
        present = [i for i, d in enumerate(dpf_tp) if d.get(f, 0) > 0]
        if not present:
            continue
        frame_avg_az.append(_circ_mean_deg(az[present]))
        frame_avg_el.append(float(np.mean(el[present])))
    if len(frame_avg_az) >= 2:
        KPI_9 = {
            "azimuth_std_deg": float(np.round(np.std(frame_avg_az, ddof=1), 6)),
            "elevation_std_deg": float(np.round(np.std(frame_avg_el, ddof=1), 6)),
        }
    else:
        KPI_9 = {"azimuth_std_deg": None, "elevation_std_deg": None}

    # ---------- KPI 10: Angular accuracy ----------
    az_abs, el_abs, mag_abs = [], [], []
    for az_f, el_f in zip(frame_avg_az, frame_avg_el):
        da = abs(_circ_diff_deg(az_f, gt_azimuth))
        de = abs(el_f - gt_elevation)
        az_abs.append(da); el_abs.append(de); mag_abs.append(sqrt(da*da + de*de))
    KPI_10 = {
        "azimuth_abs_mean_deg": float(np.round(np.mean(az_abs), 6)) if az_abs else None,
        "elevation_abs_mean_deg": float(np.round(np.mean(el_abs), 6)) if el_abs else None,
        "magnitude_abs_mean_deg": float(np.round(np.mean(mag_abs), 6)) if mag_abs else None,
    }

    # ---------- KPI 11: Surface differentiation accuracy ----------
    KPI_11 = None
    if surface_feature_samples and len(surface_feature_samples) >= 2:
        keys = list(surface_feature_samples.keys())[:2]
        s1 = np.asarray(surface_feature_samples[keys[0]], dtype=float)
        s2 = np.asarray(surface_feature_samples[keys[1]], dtype=float)
        if s1.size > 1 and s2.size > 1:
            mu1, mu2 = float(np.mean(s1)), float(np.mean(s2))
            sd1, sd2 = float(np.std(s1, ddof=1)), float(np.std(s2, ddof=1))
            KPI_11 = float(np.round(_bayes_accuracy_two_normals(mu1, sd1, mu2, sd2), 6))

    # ---------- KPI 6: False alarm rate (only in FAR tests) ----------
    KPI_6 = None
    if (test_type or "auto").lower() in ("far", "false_alarm"):
        # In FAR: everything in control volume is FP. Need measurement duration.
        if frame_dt and frame_dt > 0:
            duration_s = total_frames * frame_dt
            n_fp_clusters = len(cluster_map_fp) if cluster_map_fp else 0
            KPI_6 = float(np.round(n_fp_clusters / duration_s, 6))
        else:
            KPI_6 = None

    return {
        "KPI_1_TP_Prob": KPI_1,
        "KPI_2_Angular_Cluster_Density_deg^-2": KPI_2,
        "KPI_3_Revisit_Rate": KPI_3,                    # Hz if frame_dt provided, else per-frame
        "KPI_4_False_Positive_Detection_Rate": KPI_4,   # None if no FP inputs
        "KPI_5_False_Positive_Detection_Deviation_deg": KPI_5,  # None if no FP inputs
        "KPI_6_False_Alarm_Rate_Hz": KPI_6,             # Only for FAR tests
        "KPI_7_Radial_Precision_m": KPI_7,              # None without range samples
        "KPI_8_Radial_Accuracy_m": KPI_8,
        "KPI_9_Angular_Precision_std_deg": KPI_9,
        "KPI_10_Angular_Accuracy_abs_deg": KPI_10,
        "KPI_11_Surface_Differentiation_Accuracy": KPI_11,
    }
