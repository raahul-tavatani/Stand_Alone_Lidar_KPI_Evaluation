# utils/radial_separability_kpi.py
"""
Radial Separability KPI aggregator.

Reads all grid_tubes_map.csv files under:
  <root>/d0_*m/gap_*m/grid_tubes_map.csv

Outputs an Excel with:
  • summary_by_gap: totals and separable rates per (d0_m, gap_m)
  • separable_cases: detailed rows for cells where is_separable == 1
  • all_cells: all rows with parsed helpers

Usage:
  python -m utils.radial_separability_kpi ^
    --root ./outputs/rs1/scenario_001 ^
    --out  ./outputs/rs1/scenario_001/radial_separability_kpi.xlsx
"""

import os
import re
import ast
import argparse
import numpy as np
import pandas as pd

def _parse_float_suffix_m(name: str) -> float:
    """
    Extract trailing meters value from folder like 'd0_10m' or 'gap_0.05m'.
    Returns np.nan on failure.
    """
    m = re.search(r"([-\d\.]+)m$", name)
    if not m: 
        return float("nan")
    try:
        return float(m.group(1))
    except Exception:
        return float("nan")

def _safe_literal_list(s: str):
    """
    Convert strings like "[10.046585,10.048379]" to a Python list of floats.
    Returns [] if unparseable.
    """
    if not isinstance(s, str) or not s:
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            # coerce to float, ignore non-numerics
            out = []
            for x in v:
                try: out.append(float(x))
                except: pass
            return out
        return []
    except Exception:
        # fallback: regex floats
        fs = re.findall(r"[-+]?\d*\.\d+|\d+", s)
        try:
            return [float(x) for x in fs]
        except Exception:
            return []

def _coerce_numeric(series, default=np.nan):
    try:
        return pd.to_numeric(series, errors="coerce")
    except Exception:
        return pd.Series([default]*len(series))

def _collect_csvs(root: str):
    rows = []
    paths = []
    for d0 in sorted(os.listdir(root)):
        d0_path = os.path.join(root, d0)
        if not (os.path.isdir(d0_path) and d0.startswith("d0_") and d0.endswith("m")):
            continue
        for gap in sorted(os.listdir(d0_path)):
            gap_path = os.path.join(d0_path, gap)
            if not (os.path.isdir(gap_path) and gap.startswith("gap_") and gap.endswith("m")):
                continue
            csv_path = os.path.join(gap_path, "grid_tubes_map.csv")
            if os.path.isfile(csv_path):
                paths.append((d0, gap, csv_path))
    return paths

def build_kpi(root: str, out_xlsx: str, debug: bool=False):
    paths = _collect_csvs(root)
    if not paths:
        raise FileNotFoundError(f"No grid_tubes_map.csv found under: {root}")

    df_all_list = []
    for d0_name, gap_name, csv_path in paths:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            if debug: 
                print(f"[WARN] Failed to read {csv_path}: {e}")
            continue

        d0_m  = _parse_float_suffix_m(d0_name)
        gap_m = _parse_float_suffix_m(gap_name)

        # Parse/Coerce fields we rely on
        df["is_separable"] = _coerce_numeric(df.get("is_separable", 0)).fillna(0).astype(int)
        df["peak_delta_m"] = _coerce_numeric(df.get("peak_delta_m", np.nan))
        df["azimuth_deg"]  = _coerce_numeric(df.get("azimuth_deg", np.nan))
        df["elevation_deg"]= _coerce_numeric(df.get("elevation_deg", np.nan))
        df["tube_radius_m"]= _coerce_numeric(df.get("tube_radius_m", np.nan))
        df["s_enter_m"]    = _coerce_numeric(df.get("s_enter_m", np.nan))
        df["s_exit_m"]     = _coerce_numeric(df.get("s_exit_m", np.nan))

        # Extract first two global peaks if present
        peaks = df.get("global_peaks_m", None)
        if peaks is None:
            peak1 = [np.nan]*len(df)
            peak2 = [np.nan]*len(df)
        else:
            peak1, peak2 = [], []
            for s in peaks:
                vals = _safe_literal_list(s)
                if len(vals) >= 2:
                    peak1.append(float(vals[0]))
                    peak2.append(float(vals[1]))
                elif len(vals) == 1:
                    peak1.append(float(vals[0]))
                    peak2.append(np.nan)
                else:
                    peak1.append(np.nan)
                    peak2.append(np.nan)
        df["peak1_m"] = peak1
        df["peak2_m"] = peak2

        # annotate
        df["d0_dir"]  = d0_name
        df["gap_dir"] = gap_name
        df["d0_m"]    = d0_m
        df["gap_m"]   = gap_m

        df_all_list.append(df)

        if debug:
            print(f"[OK] {csv_path}: rows={len(df)}, separable={int((df['is_separable']==1).sum())}")

    if not df_all_list:
        raise RuntimeError("No valid CSVs could be read.")

    df_all = pd.concat(df_all_list, ignore_index=True)

    # Detailed sheet: only separable cases
    df_sep = df_all[df_all["is_separable"] == 1].copy()
    sep_cols = [
        "d0_m","gap_m","d0_dir","gap_dir","cell_id",
        "azimuth_deg","elevation_deg","tube_radius_m",
        "s_enter_m","s_exit_m","frame_count",
        "peak1_m","peak2_m","peak_delta_m",
        "multi_range_frames","has_multi_range"
    ]
    sep_cols = [c for c in sep_cols if c in df_sep.columns]
    df_sep = df_sep.loc[:, sep_cols].sort_values(["d0_m","gap_m","cell_id"], na_position="last")

    # Summary sheet by (d0,gap)
    g_all = df_all.groupby(["d0_m","gap_m"], dropna=False).size().rename("n_cells")
    g_sep = df_sep.groupby(["d0_m","gap_m"], dropna=False).size().rename("n_separable")
    summary = pd.concat([g_all, g_sep], axis=1).fillna(0).reset_index()
    summary["n_cells"] = summary["n_cells"].astype(int)
    summary["n_separable"] = summary["n_separable"].astype(int)
    summary["separable_rate"] = np.where(summary["n_cells"]>0,
                                         summary["n_separable"]/summary["n_cells"], np.nan)
    summary = summary.sort_values(["d0_m","gap_m"], na_position="last")

    # Write Excel
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
    try:
        with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as xw:
            summary.to_excel(xw, index=False, sheet_name="summary_by_gap")
            df_sep.to_excel(xw, index=False, sheet_name="separable_cases")
            # all cells (optionally large, but helpful)
            df_all.to_excel(xw, index=False, sheet_name="all_cells")
    except Exception:
        # Fallback to default engine (openpyxl)
        with pd.ExcelWriter(out_xlsx) as xw:
            summary.to_excel(xw, index=False, sheet_name="summary_by_gap")
            df_sep.to_excel(xw, index=False, sheet_name="separable_cases")
            df_all.to_excel(xw, index=False, sheet_name="all_cells")

    print(f"[DONE] Wrote KPI Excel → {out_xlsx}")
    print(f"  Gaps processed: {len(paths)} | total cells: {len(df_all)} | separable: {len(df_sep)}")

def main():
    ap = argparse.ArgumentParser(description="Aggregate radial separability KPI into an Excel report.")
    ap.add_argument("--root", required=True, help="Root with d0_*m/gap_*m folders that contain grid_tubes_map.csv")
    ap.add_argument("--out",  required=True, help="Output Excel path (*.xlsx)")
    ap.add_argument("--debug", action="store_true", help="Verbose logging")
    args = ap.parse_args()

    build_kpi(args.root, args.out, debug=args.debug)

if __name__ == "__main__":
    main()
