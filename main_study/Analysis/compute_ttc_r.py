#!/usr/bin/env python3
"""
compute_ttc_at_response.py — V2P Study | True TTC-at-Response
================================================================
The log field "TTC_at_response" (in trials.csv/participants.csv, from
run_sim.py) is actually a PRE-alarm walking-speed extrapolation (kerb→
midpoint markers), not a TTC measured at the moment of actual response —
see chat for the full derivation. This script computes the DV that was
actually intended: the vehicle-kinematic TTC (logged per-frame as
"TTC_secondary", currently unused) at the real reaction timestamp
detected from video (rt_results.csv).

REQUIRES: rt_results.csv must contain 'alarm_sim_s' — only present if
analyze_rt_video.py has been re-run with the alarm_sim_s patch (see chat).

METHOD, per trial with a valid rt_s:
    sim_reaction_time = alarm_sim_s + rt_s
    -> look up the raw per-trial CARLA log (same file analyze_logs.py reads)
    -> find the frame whose 'timestamp' is closest to sim_reaction_time
    -> read that frame's TTC_secondary  ==  the true TTC-at-response

The original walking-speed-based value is kept, unchanged, as a separate
column (renamed here to ttc_baseline_approach for clarity) — nothing is
overwritten, both are reported.

USAGE:
    python compute_ttc_at_response.py --rt_results rt_results.csv \
        --logs_dir ../logs --trials_csv results/trials.csv \
        --out rt_results_with_true_ttc.csv
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


def parse_header(line: str) -> dict:
    result = {}
    for token in line.lstrip("# ").split():
        if ":" in token:
            k, v = token.split(":", 1)
            result[k.lower()] = v
    return result


def find_trial_log(logs_dir: str, pid_short: str, mode: str, run_id: str) -> str | None:
    """Locate the raw per-trial CSV (same naming as DataLogger: {mode}_{run}.csv)."""
    candidates = [
        os.path.join(logs_dir, pid_short, f"{mode}_{run_id}.csv"),
    ]
    # Fallback: glob in case of naming drift, matched by header content instead.
    for path in candidates:
        if os.path.exists(path):
            return path

    folder = os.path.join(logs_dir, pid_short)
    if os.path.isdir(folder):
        for fpath in glob.glob(os.path.join(folder, "*.csv")):
            try:
                with open(fpath, encoding="utf-8") as f:
                    first = f.readline()
            except OSError:
                continue
            if not first.startswith("#"):
                continue
            h = parse_header(first)
            if h.get("mode") == mode and h.get("run") == run_id:
                return fpath
    return None


def ttc_at_sim_time(log_path: str, sim_time: float) -> tuple[float | None, str]:
    """
    Read the per-frame section of a raw trial log, return the TTC_secondary
    value at the frame closest to sim_time.
    Returns (ttc_value_or_None, status_note).
    """
    try:
        with open(log_path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as e:
        return None, f"log_read_error: {e}"

    # Skip the '#' metadata header line, parse the rest as CSV.
    data_lines = [l for l in lines if not l.startswith("#")]
    if len(data_lines) < 2:
        return None, "no_frame_data"

    from io import StringIO
    df = pd.read_csv(StringIO("".join(data_lines)))
    if df.empty or "timestamp" not in df.columns:
        return None, "malformed_frame_data"

    df = df[df["timestamp"].notna()]
    if df.empty:
        return None, "no_valid_timestamps"

    idx = (df["timestamp"] - sim_time).abs().idxmin()
    row = df.loc[idx]

    closest_dt = abs(row["timestamp"] - sim_time)
    if closest_dt > 0.5:
        return None, f"no_frame_within_0.5s (closest={closest_dt:.2f}s away)"

    ttc_raw = row.get("TTC_secondary", "N/A")
    if str(ttc_raw) == "N/A" or pd.isna(ttc_raw):
        return None, "ttc_secondary_NA_at_that_frame"

    try:
        return float(ttc_raw), ""
    except (TypeError, ValueError):
        return None, "ttc_secondary_unparseable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rt_results", default="rt_results.csv")
    ap.add_argument("--logs_dir", default="logs",
                     help="Root folder containing per-participant raw log subfolders")
    ap.add_argument("--trials_csv", default=None,
                     help="Optional: trials.csv from analyze_logs.py, to merge in "
                          "the original walking-speed-based ttc_at_response for "
                          "side-by-side comparison (renamed ttc_baseline_approach)")
    ap.add_argument("--out", default="rt_results_with_true_ttc.csv")
    args = ap.parse_args()

    rt = pd.read_csv(args.rt_results)
    if "alarm_sim_s" not in rt.columns:
        raise SystemExit(
            "[ERROR] 'alarm_sim_s' column missing from rt_results.csv — "
            "re-run analyze_rt_video.py with the alarm_sim_s patch first (see chat)."
        )

    results = []
    n_ok = n_fail = 0
    for _, row in rt.iterrows():
        if pd.isna(row.get("rt_s")) or pd.isna(row.get("alarm_sim_s")):
            results.append({"ttc_at_actual_response": None, "ttc_lookup_status": "no_rt"})
            continue

        pid_short = row["participant_id"]  # already "P1" style from analyze_rt_video.py
        mode = "a" if row["condition"].lower() == "adaptive" else "b"
        run_id = row["trial_id"]

        log_path = find_trial_log(args.logs_dir, pid_short, mode, run_id)
        if log_path is None:
            results.append({"ttc_at_actual_response": None,
                             "ttc_lookup_status": "raw_log_not_found"})
            n_fail += 1
            continue

        sim_reaction_time = row["alarm_sim_s"] + row["rt_s"]
        ttc_val, status = ttc_at_sim_time(log_path, sim_reaction_time)

        results.append({"ttc_at_actual_response": ttc_val,
                         "ttc_lookup_status": status if ttc_val is None else "ok"})
        if ttc_val is not None:
            n_ok += 1
        else:
            n_fail += 1

    out_df = pd.concat([rt.reset_index(drop=True), pd.DataFrame(results)], axis=1)

    if args.trials_csv:
        trials = pd.read_csv(args.trials_csv)
        trials["pid_norm"] = trials["pid"].astype(str)
        trials["mode_norm"] = trials["mode"]
        base = trials[trials["trial_type"] == "hit"][
            ["pid", "mode", "run_id", "ttc_at_response"]
        ].rename(columns={"ttc_at_response": "ttc_baseline_approach"})
        out_df["pid_num"] = out_df["participant_id"].str.lstrip("Pp").astype(int)
        out_df["mode_norm"] = out_df["condition"].str.lower().map(
            {"adaptive": "a", "baseline": "b"})
        out_df = out_df.merge(
            base, left_on=["pid_num", "mode_norm", "trial_id"],
            right_on=["pid", "mode", "run_id"], how="left"
        ).drop(columns=["pid", "mode", "run_id", "pid_num", "mode_norm"])

    if os.path.exists("participant_profile.csv"):
        profile = pd.read_csv("participant_profile.csv")
        profile["pid_num"] = profile["PID"].astype(int)
        out_df["pid_num"] = out_df["participant_id"].str.lstrip("Pp").astype(int)
        out_df = out_df.merge(
            profile[["pid_num", "Walker_speed", "Walking_style"]],
            on="pid_num", how="left"
        ).drop(columns=["pid_num"])
    else:
        print("[WARN] participant_profile.csv not found — Walker_speed/Walking_style skipped")

    out_df.to_csv(args.out, index=False)

    print(f"[SUMMARY] {n_ok} trials with a resolved ttc_at_actual_response, "
          f"{n_fail} failed lookups")
    print(rt.groupby("condition").size() if "condition" in rt.columns else "")
    print(f"[SAVE] {args.out}")


if __name__ == "__main__":
    main()