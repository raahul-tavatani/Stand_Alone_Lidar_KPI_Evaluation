# utils/kpi_utils.py

import numpy as np

def evaluate_kpis(cluster_map: list, target_distance: float, total_frames: int, gt_azimuth=0.0, gt_elevation=0.0):
    """
    Evaluate KPIs 1, 2, 3, 7, 8, 9, 10 from a cluster map.
    Assumes all clusters are true-positive (filtered inside vicinity).
    """
    if not cluster_map:
        print("[WARN] No clusters to evaluate.")
        return {}

    # Convert cluster_map to np arrays for processing
    az = np.array([c["azimuth"] for c in cluster_map])
    el = np.array([c["elevation"] for c in cluster_map])
    r  = np.array([c["range"] for c in cluster_map])
    sample_sizes = np.array([c["sample_size"] for c in cluster_map])
    frame_counts = np.array([c["frame_count"] for c in cluster_map])

    total_detections = sum(sample_sizes)
    if total_detections == 0:
        print("[WARN] No detections.")
        return {}

    # KPI 1: True-Positive Detection Probability
    tpdp_per_cluster = frame_counts / total_frames
    KPI_1 = round(np.mean(tpdp_per_cluster), 4)

    # KPI 2: Angular Cluster Density (per deg²)
    # Area in deg² = target extent (±θ, ±φ). Assume 1m×1m target at distance d.
    target_angle = np.degrees(np.arctan(0.5 / target_distance))
    area_deg2 = (2 * target_angle) ** 2
    KPI_2 = round(len(cluster_map) / area_deg2, 4)

    # KPI 3: Revisit Rate = mean detections per cluster / total_frames
    revisit_rate = sample_sizes / total_frames
    KPI_3 = round(np.mean(revisit_rate), 4)

    # KPI 7: Radial Precision = weighted stddev of radial distances
    stddevs = np.array([0.0] * len(r))  # assume 0 per cluster (no individual points)
    # can't compute per-cluster stddev without raw points, placeholder 0
    KPI_7 = None  # mark as not computable now

    # KPI 8: Radial Accuracy = weighted mean(r - GT)
    radial_errors = r - target_distance
    weights = sample_sizes / total_detections
    KPI_8 = round(np.sum(weights * radial_errors), 4)

    # KPI 9: Angular Precision = stddev of cluster centers over frames
    KPI_9 = {
        "azimuth_std": round(np.std(az), 4),
        "elevation_std": round(np.std(el), 4)
    }

    # KPI 10: Angular Accuracy = mean offset from ground-truth
    angular_errors = np.sqrt((az - gt_azimuth)**2 + (el - gt_elevation)**2)
    KPI_10 = round(np.sum(weights * angular_errors), 4)

    return {
        "KPI_1_TP_Prob": KPI_1,
        "KPI_2_Angular_Cluster_Density": KPI_2,
        "KPI_3_Revisit_Rate": KPI_3,
        "KPI_7_Radial_Precision": KPI_7,
        "KPI_8_Radial_Accuracy": KPI_8,
        "KPI_9_Angular_Precision": KPI_9,
        "KPI_10_Angular_Accuracy": KPI_10
    }
