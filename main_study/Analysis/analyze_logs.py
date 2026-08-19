"""
V2P Warning Strategy — Data Analysis Script
============================================
Masterarbeit | UX Design & Human Factors

Usage:
    python analyze.py [logs_dir]

    logs_dir defaults to './logs' (same folder as run_sim.py).

Outputs (written to ./results/):
    trials.csv          — one row per valid trial
    participants.csv    — per-participant means, both modes
    stats.txt           — statistical test results
    plots/              — figures (PDF + PNG)

Primary DV:
    TTC_at_response — time (s) remaining until pedestrian reaches the
                      danger zone, estimated from walking speed at marker X.
                      Higher = more safety margin kept.

Trial types used per analysis:
    hit   (t1–t5) — primary comparison: TTC_at_response, collision rate
    stop  (c1–c2) — catch trials: false-stop rate (did f-alarm cause halt?)
    safe  (s1–s2) — filler: excluded from statistical tests
"""

import os
import re
import sys
import csv
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)

# ─── CONSTANTS ────────────────────────────────────────────
TRIAL_TYPE = {
    "t1": "hit", "t2": "hit", "t3": "hit", "t4": "hit", "t5": "hit",
    "c1": "stop", "c2": "stop",
    "s1": "safe", "s2": "safe",
}

MODE_LABELS = {"a": "Adaptive", "b": "Baseline"}


# ─────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────
def parse_header(line: str) -> dict:
    """
    Extract key–value pairs from the CSV comment header, e.g.:
    # PID:1 Mode:a Run:t1 Outcome:collision T_A:4.99 ...
    Returns a dict with all keys lower-cased.
    """
    result = {}
    for token in line.lstrip("# ").split():
        if ":" in token:
            k, v = token.split(":", 1)
            result[k.lower()] = v
    return result

def load_participant_profile(path="participant_profile.csv"):
    if not os.path.exists(path):
        print(f"[WARN] {path} not found — Walker_speed/Walking_style columns will be empty")
        return {}
    df = pd.read_csv(path)
    return {int(row["PID"]): (row["Walker_speed"], row["Walking_style"])
            for _, row in df.iterrows()}

def load_all_logs(logs_dir: str = "./logs", profile: dict = None) -> pd.DataFrame:
    """
    Recursively scan logs_dir for CSV files, parse each header, and
    build a trial-level DataFrame (one row per scenario run).
    """
    records = []
    log_path = Path(logs_dir)

    if not log_path.exists():
        print(f"[ERROR] Log directory not found: {logs_dir}")
        sys.exit(1)

    csv_files = sorted(f for f in log_path.rglob("*.csv") if f.name != "run_order.csv")
    if not csv_files:
        print(f"[ERROR] No CSV files found in {logs_dir}")
        sys.exit(1)

    print(f"[LOAD] Found {len(csv_files)} log files.")

    for fpath in csv_files:
        with open(fpath, "r", newline="") as f:
            lines = f.readlines()

        if not lines or not lines[0].startswith("#"):
            print(f"  [SKIP] No header: {fpath.name}")
            continue

        h = parse_header(lines[0])

        # Parse numeric / None fields
        def to_float(v):
            try:
                return float(v) if v not in ("None", "N/A", None) else None
            except (ValueError, TypeError):
                return None

        run_id = h.get("run", "")
        pid_int = int(h.get("pid", 0) or 0)
        walker_speed, walking_style = (profile or {}).get(pid_int, (None, None))
        record = {
            "file":              str(fpath),
            "pid":               h.get("pid"),
            "mode":              h.get("mode"),
            "run_id":            run_id,
            "trial_type":        TRIAL_TYPE.get(run_id, "unknown"),
            "outcome":           h.get("outcome"),
            "t_a":               to_float(h.get("t_a")),
            "t_b":               to_float(h.get("t_b")),
            "t_c":               to_float(h.get("t_c")),
            "ttc_at_response":   to_float(h.get("ttc_at_response")),
            "early_warn_t":      to_float(h.get("earlywarnt")),
            "full_warn_t":       to_float(h.get("fullwarnt")),
            "timing_error":      to_float(h.get("timingerror")),
            "walker_speed":      walker_speed,
            "walking_style":     walking_style,
        }

        # Derived fields
        if record["t_a"] is not None and record["t_b"] is not None:
            record["elapsed_ab"] = round(record["t_b"] - record["t_a"], 4)
        else:
            record["elapsed_ab"] = None

        # Response: C was pressed before vehicle arrived
        record["responded"]     = record["t_c"] is not None
        record["collision"]     = record["outcome"] == "collision"
        record["response_stop"] = record["outcome"] == "response_stop"
        record["safe_stop"]     = record["outcome"] == "safe_stop"

        # Time from X press until C press (reaction latency proxy)
        if record["t_b"] is not None and record["t_c"] is not None:
            record["t_b_to_c"] = round(record["t_c"] - record["t_b"], 4)
        else:
            record["t_b_to_c"] = None

        records.append(record)

    df = pd.DataFrame(records)
    print(f"[LOAD] {len(df)} trials loaded from "
          f"{df['pid'].nunique()} participant(s).\n")
    return df


# ─────────────────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove incomplete or aborted trials.
    Excluded: outcome == 'quit', trial_type == 'unknown'.
    """
    n_before = len(df)
    excluded_quit = df[df["outcome"] == "quit"]
    excluded_unknown = df[df["trial_type"] == "unknown"]
    if not excluded_quit.empty:
        print(f"[CLEAN] Excluded (outcome=quit): "
              f"{excluded_quit[['pid','mode','run_id']].to_dict('records')}")
    if not excluded_unknown.empty:
        print(f"[CLEAN] Excluded (trial_type=unknown): "
              f"{excluded_unknown[['pid','mode','run_id']].to_dict('records')}")
    df = df[df["outcome"] != "quit"].copy()
    df = df[df["trial_type"] != "unknown"].copy()

    # Flag trials where TTC_at_response is missing (sensor/keypress issue)
    missing_ttc = df["ttc_at_response"].isna()
    if missing_ttc.sum() > 0:
        print(f"[WARN] {missing_ttc.sum()} trials with missing TTC_at_response "
              f"(excluded from TTC analyses).")

    print(f"[CLEAN] {n_before - len(df)} trials excluded. "
          f"{len(df)} remaining.\n")
    return df


# ─────────────────────────────────────────────────────────
# 3. ANNOTATIONS
# ─────────────────────────────────────────────────────────
def load_annotations(path: str = "./trial_annotations.csv") -> pd.DataFrame:
    """
    Load post-hoc annotations from trial_annotations.csv.
    Returns empty DataFrame if file not found.
    """
    if not os.path.exists(path):
        print(f"[WARN] No annotation file found at '{path}' — using logged outcomes only.")
        return pd.DataFrame()

    # encoding='utf-8-sig' strips Excel BOM; sep=None auto-detects , vs ; (German Excel)
    ann = pd.read_csv(path, dtype=str, encoding="utf-8-sig", sep=None, engine="python").fillna("")
    ann.columns = [c.lower().strip() for c in ann.columns]
    ann = ann.rename(columns={"run": "run_id"})

    # ── Detect Excel "single-column" mangling ──────────────────────────────
    # When Excel opens a CSV as one column and the user types data into cells,
    # each row gets saved as a single quoted string in the first column.
    # Symptom: pid contains commas AND mode/run_id are empty.
    if ("pid" in ann.columns and "mode" in ann.columns
            and ann["pid"].str.contains(",").any()
            and (ann["mode"] == "").all()):
        print("[INFO] Detected Excel single-column format — re-parsing rows.")
        col_names = ["pid", "mode", "run_id", "logged_outcome",
                     "actual_outcome", "stopped_at_alarm", "looked_at_screen", "note"]
        parsed = ann["pid"].str.strip().str.strip('"').str.split(",", n=7, expand=True)
        parsed.columns = col_names[:len(parsed.columns)]
        # Strip quotes and whitespace from each cell
        for c in parsed.columns:
            parsed[c] = parsed[c].str.strip().str.strip('"')
        ann = parsed

    # Strip whitespace from key columns to avoid silent merge failures
    for col in ["pid", "mode", "run_id"]:
        if col in ann.columns:
            ann[col] = ann[col].str.strip()

    # ── Diagnostics — remove once merge is confirmed working ──
    print("[DEBUG] ann columns :", ann.columns.tolist())
    print("[DEBUG] ann keys (first 3 rows):")
    for _, row in ann.head(3).iterrows():
        print(f"         pid={repr(row.get('pid','?'))}  "
              f"mode={repr(row.get('mode','?'))}  "
              f"run_id={repr(row.get('run_id','?'))}")

    # Migrate legacy Looked_at_Screen → Gaze_general
    LEGACY_GAZE = {"screen_glanced":"phone","screen_looked":"phone",
                   "traffic_glanced":"traffic","traffic_looked":"traffic",
                   "both":"both","glanced":"phone","looked":"phone",
                   "focused":"phone","yes":"phone"}
    if "looked_at_screen" in ann.columns and "gaze_general" not in ann.columns:
        ann["gaze_general"] = ann["looked_at_screen"].str.lower().map(LEGACY_GAZE).fillna("")

    keep = ["pid", "mode", "run_id", "actual_outcome", "stopped_at_alarm",
            "gaze_general", "gaze_alarm",
            "walker_speed", "walking_style", "alarm_reaction"]
    for opt in ["note", "looked_at_screen"]:   # keep old col for compat if present
        if opt in ann.columns:
            keep.append(opt)

    ann = ann[[c for c in keep if c in ann.columns]]
    print(f"[LOAD] Annotations loaded: {len(ann)} rows.\n")
    return ann


# ─────────────────────────────────────────────────────────
# 4. SUMMARY TABLES
# ─────────────────────────────────────────────────────────
def participant_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-participant, per-mode means for hit trials.
    Returns a wide-format DataFrame with one row per participant.
    """
    hit = df[df["trial_type"] == "hit"].copy()

    agg = hit.groupby(["pid", "mode"]).agg(
        ttc_mean      = ("ttc_at_response", "mean"),
        ttc_sd        = ("ttc_at_response", "std"),
        n_trials      = ("run_id",          "count"),
        n_collisions  = ("collision",        "sum"),
        n_responses   = ("response_stop",    "sum"),
    ).reset_index()

    agg["collision_rate"] = agg["n_collisions"] / agg["n_trials"]
    agg["response_rate"]  = agg["n_responses"]  / agg["n_trials"]
    agg["mode_label"]     = agg["mode"].map(MODE_LABELS)

    return agg


def catch_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-participant, per-mode summary for catch (stop) trials.
    False-stop rate = proportion of catch trials where participant pressed C
    in response to the f-alarm (adaptive only).
    """
    catch = df[df["trial_type"] == "stop"].copy()
    if catch.empty:
        return pd.DataFrame()

    agg = catch.groupby(["pid", "mode"]).agg(
        n_trials    = ("run_id",           "count"),
        n_responded = ("stopped_at_alarm", "sum"),     # war: ("responded", "sum")
        ttc_mean    = ("ttc_at_response",  "mean"),
    ).reset_index()

    agg["false_stop_rate"] = agg["n_responded"] / agg["n_trials"]
    agg["mode_label"]      = agg["mode"].map(MODE_LABELS)
    return agg


# ─────────────────────────────────────────────────────────
# 5. STATISTICAL TESTS
# ─────────────────────────────────────────────────────────
def cohen_d(a, b):
    """Cohen's d for two paired arrays."""
    diff = np.array(a) - np.array(b)
    return diff.mean() / (diff.std(ddof=1) + 1e-9)


def run_statistics(df: pd.DataFrame) -> str:
    """
    Paired comparisons: Adaptive vs Baseline.
    Uses Wilcoxon signed-rank test (non-parametric, appropriate for small N).
    Falls back to paired t-test if N >= 10.
    Returns a formatted text report.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("STATISTICAL ANALYSIS — V2P WARNING STRATEGY")
    lines.append("=" * 60)

    hit = df[(df["trial_type"] == "hit") & df["ttc_at_response"].notna()].copy()

    # Build paired arrays (one mean per participant per mode)
    paired = (
        hit.groupby(["pid", "mode"])["ttc_at_response"]
        .mean()
        .unstack("mode")
        .dropna()
    )

    if paired.empty or "a" not in paired.columns or "b" not in paired.columns:
        lines.append("\n[!] Not enough paired data for statistical tests.")
        return "\n".join(lines)

    a_vals = paired["a"].values   # adaptive
    b_vals = paired["b"].values   # baseline
    n      = len(paired)

    lines.append(f"\nN (complete pairs) = {n}")
    lines.append(f"\n{'':30s}  {'Adaptive':>10}  {'Baseline':>10}")
    lines.append(f"{'TTC_at_response mean (s)':30s}  "
                 f"{a_vals.mean():>10.3f}  {b_vals.mean():>10.3f}")
    lines.append(f"{'TTC_at_response SD (s)':30s}  "
                 f"{a_vals.std():>10.3f}  {b_vals.std():>10.3f}")

    lines.append("\n[SUPERSEDED] This is the pre-alarm walking-speed approximation "
                 "(ttc_baseline_approach), NOT the confirmatory H1a test. The true "
                 "kinematic TTC-R (from compute_ttc_r.py + run_h1_h4_analysis.py) is "
                 "the result reported in the thesis for H1a — see chat/methodology doc.")
    lines.append("── ttc_baseline_approach (hit trials, descriptive only) ──")

    if n >= 10:
        t_stat, p_val = stats.ttest_rel(a_vals, b_vals)
        test_name = "Paired t-test"
        stat_label = f"t({n-1}) = {t_stat:.3f}"
    else:
        # Wilcoxon preferred for small N
        try:
            t_stat, p_val = stats.wilcoxon(a_vals, b_vals)
            test_name = "Wilcoxon signed-rank"
            stat_label = f"W = {t_stat:.3f}"
        except Exception:
            lines.append("  [!] Wilcoxon failed (likely zero differences). "
                         "Increase N.")
            t_stat, p_val = stats.ttest_rel(a_vals, b_vals)
            test_name = "Paired t-test (fallback)"
            stat_label = f"t({n-1}) = {t_stat:.3f}"

    d = cohen_d(a_vals, b_vals)
    sig = "***" if p_val < .001 else "**" if p_val < .01 else "*" if p_val < .05 else "n.s."

    lines.append(f"  {test_name}: {stat_label}, p = {p_val:.4f} {sig}")
    lines.append(f"  Cohen's d = {d:.3f}  "
                 f"({'large' if abs(d)>.8 else 'medium' if abs(d)>.5 else 'small'})")
    lines.append(f"  Direction: Adaptive {'>' if a_vals.mean() > b_vals.mean() else '<'} "
                 f"Baseline (delta = {a_vals.mean()-b_vals.mean():+.3f}s)")

    # ── Collision rate ────────────────────────────────────
    lines.append("\n── Collision rate (hit trials) ────────────────────")
    coll = (
        hit.groupby(["pid", "mode"])["collision"]
        .mean()
        .unstack("mode")
        .dropna()
    )
    if "a" in coll.columns and "b" in coll.columns:
        lines.append(f"  Adaptive : {coll['a'].mean():.1%} +/- {coll['a'].std():.1%}")
        lines.append(f"  Baseline : {coll['b'].mean():.1%} +/- {coll['b'].std():.1%}")
        n_a_coll = hit[hit["mode"] == "a"]["collision"].sum()
        n_b_coll = hit[hit["mode"] == "b"]["collision"].sum()
        lines.append(f"  Total collisions — Adaptive: {n_a_coll}, "
                     f"Baseline: {n_b_coll}")

    # ── False-stop rate (catch trials) ───────────────────
    catch = df[df["trial_type"] == "stop"].copy()
    if not catch.empty:
        lines.append("\n── False-stop rate (catch trials, adaptive only) ──")
        a_catch = catch[catch["mode"] == "a"]
        if not a_catch.empty:
            fsr = a_catch.groupby("pid")["stopped_at_alarm"].mean()
            lines.append(f"  Mean false-stop rate (adaptive): "
                         f"{fsr.mean():.1%} +/- {fsr.std():.1%}")
            lines.append(f"  (= proportion of catch trials where researcher "
                         f"pressed C after f-alarm)")

    # ── Manipulation Check: TimingError ────────────────────
    te = df["timing_error"].dropna()
    if len(te) > 0:
        lines.append(f"\n── Manipulation Check: TimingError (nominal TTC targets vs. actual) ──")
        lines.append(f"  N = {len(te)}   Mean = {te.mean():.3f}s   SD = {te.std():.3f}s")
        lines.append(f"  |TimingError| > 0.5s: {(te.abs() > 0.5).sum()} von {len(te)} Trials "
                    f"({(te.abs() > 0.5).mean():.1%})")
        lines.append(f"  |TimingError| > 1.0s: {(te.abs() > 1.0).sum()} von {len(te)} Trials "
                    f"({(te.abs() > 1.0).mean():.1%})")

    lines.append("\n" + "=" * 60)
    lines.append("Significance: * p<.05  ** p<.01  *** p<.001  n.s. = not significant")
    lines.append("=" * 60)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────────────────────
def plot_results(df: pd.DataFrame, out_dir: Path):
    """Generate and save all figures."""
    hit = df[(df["trial_type"] == "hit") & df["ttc_at_response"].notna()]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("V2P Warning Strategy — Results", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors = {"Adaptive": "#2196F3", "Baseline": "#FF5722"}

    # ── Plot 1: TTC_at_response — group means ─────────────
    ax1 = fig.add_subplot(gs[0, 0])
    paired = (
        hit.groupby(["pid", "mode"])["ttc_at_response"]
        .mean()
        .reset_index()
    )
    paired["mode_label"] = paired["mode"].map(MODE_LABELS)

    for pid, grp in paired.groupby("pid"):
        grp_sorted = grp.sort_values("mode_label")
        ax1.plot(grp_sorted["mode_label"].values,
                 grp_sorted["ttc_at_response"].values,
                 color="gray", alpha=0.4, linewidth=1, marker="o",
                 markersize=4)

    for mode_label, grp in paired.groupby("mode_label"):
        ax1.scatter([mode_label] * len(grp),
                    grp["ttc_at_response"],
                    color=colors[mode_label], s=60, zorder=3, alpha=0.7)
        ax1.plot(mode_label, grp["ttc_at_response"].mean(),
                 marker="D", color=colors[mode_label],
                 markersize=10, zorder=4)

    ax1.set_title("TTC at response\n(hit trials, per participant)")
    ax1.set_ylabel("TTC_at_response (s)")
    ax1.set_xlabel("")
    ax1.grid(axis="y", alpha=0.3)

    # ── Plot 2: TTC_at_response — per scenario ────────────
    ax2 = fig.add_subplot(gs[0, 1])
    scenario_ttc = (
        hit.groupby(["run_id", "mode"])["ttc_at_response"]
        .mean()
        .reset_index()
    )
    scenario_ttc["mode_label"] = scenario_ttc["mode"].map(MODE_LABELS)
    run_ids = sorted(hit["run_id"].unique())
    x = np.arange(len(run_ids))
    width = 0.35

    for i, (mode, label) in enumerate([("a", "Adaptive"), ("b", "Baseline")]):
        vals = [
            scenario_ttc.loc[
                (scenario_ttc["run_id"] == r) & (scenario_ttc["mode"] == mode),
                "ttc_at_response"
            ].values[0]
            if len(scenario_ttc.loc[
                (scenario_ttc["run_id"] == r) & (scenario_ttc["mode"] == mode)
            ]) > 0 else np.nan
            for r in run_ids
        ]
        ax2.bar(x + i * width - width / 2, vals, width,
                label=label, color=colors[label], alpha=0.8)

    ax2.set_title("TTC at response\nby scenario")
    ax2.set_ylabel("Mean TTC_at_response (s)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(run_ids)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # ── Plot 3: Collision rate ────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    coll_rate = (
        hit.groupby(["pid", "mode"])["collision"]
        .mean()
        .reset_index()
    )
    coll_rate["mode_label"] = coll_rate["mode"].map(MODE_LABELS)

    for mode_label, grp in coll_rate.groupby("mode_label"):
        mean_val = grp["collision"].mean()
        ax3.bar(mode_label, mean_val,
                color=colors[mode_label], alpha=0.8,
                yerr=grp["collision"].std(), capsize=5)

    ax3.set_title("Collision rate\n(hit trials)")
    ax3.set_ylabel("Proportion")
    ax3.set_ylim(0, 1)
    ax3.grid(axis="y", alpha=0.3)

    # ── Plot 4: TTC_at_response distribution (violin) ─────
    ax4 = fig.add_subplot(gs[1, 0:2])
    ttc_a = hit[hit["mode"] == "a"]["ttc_at_response"].dropna().values
    ttc_b = hit[hit["mode"] == "b"]["ttc_at_response"].dropna().values

    parts = ax4.violinplot([ttc_a, ttc_b], positions=[0, 1],
                           showmedians=True, showmeans=False)
    for pc, color in zip(parts["bodies"], ["#2196F3", "#FF5722"]):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(["Adaptive", "Baseline"])
    ax4.set_title("TTC_at_response distribution (hit trials)")
    ax4.set_ylabel("TTC_at_response (s)")
    ax4.axhline(2.5, color="red", linestyle="--", linewidth=0.8,
                label="TTC_FULL_WARN (2.5 s)")
    ax4.axhline(5.0, color="orange", linestyle="--", linewidth=0.8,
                label="TTC_EARLY_WARN (5.0 s)")
    ax4.legend(fontsize=8)
    ax4.grid(axis="y", alpha=0.3)

    # ── Plot 5: Catch trial false-stop rate ───────────────
    ax5 = fig.add_subplot(gs[1, 2])
    catch = df[df["trial_type"] == "stop"]
    if not catch.empty:
        fsr = (
            catch.groupby(["pid", "mode"])["stopped_at_alarm"]
            .mean()
            .reset_index()
        )
        fsr["mode_label"] = fsr["mode"].map(MODE_LABELS)
        for mode_label, grp in fsr.groupby("mode_label"):
            mean_val = grp["stopped_at_alarm"].mean()
            ax5.bar(mode_label, mean_val,
                    color=colors.get(mode_label, "gray"), alpha=0.8,
                    yerr=grp["stopped_at_alarm"].std() if len(grp) > 1 else 0,
                    capsize=5)
        ax5.set_title("False-stop rate\n(catch trials)")
        ax5.set_ylabel("Proportion pressing C")
        ax5.set_ylim(0, 1)
        ax5.grid(axis="y", alpha=0.3)
    else:
        ax5.text(0.5, 0.5, "No catch trials", ha="center", va="center",
                 transform=ax5.transAxes, color="gray")
        ax5.set_title("False-stop rate\n(catch trials)")

    plt.savefig(out_dir / "results.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "results.png", dpi=150, bbox_inches="tight")
    print(f"[PLOT] Saved → {out_dir / 'results.pdf'}")
    plt.close()


# ─────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────
def main():
    logs_dir = sys.argv[1] if len(sys.argv) > 1 else "./logs"

    out_dir = Path("./results")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    # Load and clean
    profile = load_participant_profile()
    df = load_all_logs(logs_dir, profile)
    df = clean(df)

    # ── Merge annotations ─────────────────────────────────
    ann = load_annotations()
    if not ann.empty:
        # ── Diagnostics — remove once merge is confirmed working ──
        print("[DEBUG] df keys (first 3 rows):")
        for _, row in df[["pid", "mode", "run_id"]].head(3).iterrows():
            print(f"         pid={repr(row['pid'])}  "
                  f"mode={repr(row['mode'])}  "
                  f"run_id={repr(row['run_id'])}")

        df = df.merge(ann, on=["pid", "mode", "run_id"], how="left")

        print("[DEBUG] actual_outcome sample:", df["actual_outcome"].value_counts().to_dict())

        mask = df["actual_outcome"].notna() & (df["actual_outcome"] != "")
        df["effective_outcome"] = df["actual_outcome"].where(mask, df["outcome"])
        # Raw values for detailed breakdown
        df["stopped_at_alarm_raw"] = df["stopped_at_alarm"].str.lower().fillna("")
        df["gaze_general_raw"]     = df.get("gaze_general", pd.Series("", index=df.index)).str.lower().fillna("")
        df["gaze_alarm_raw"]       = df.get("gaze_alarm",   pd.Series("", index=df.index)).str.lower().fillna("")
        # Boolean helpers
        df["stopped_at_alarm"]  = df["stopped_at_alarm_raw"].isin(["f", "vha", "both", "yes"])
        df["looked_at_screen"]  = df["gaze_general_raw"].isin(["phone", "both"]) | \
                                   df["gaze_alarm_raw"].isin(["glanced", "looked"])
        df["looked_at_traffic"] = df["gaze_general_raw"].isin(["traffic", "both"])
    else:
        df["effective_outcome"]    = df["outcome"]
        df["stopped_at_alarm"]     = None
        df["stopped_at_alarm_raw"] = None
        df["looked_at_screen"]     = None
        df["looked_at_traffic"]    = None
        df["gaze_general_raw"]     = None
        df["gaze_alarm_raw"]       = None
        df["note"]                 = None

    # Recalculate boolean flags from effective_outcome
    df["collision"]          = df["effective_outcome"] == "collision"
    df["response_stop"]      = df["effective_outcome"] == "response_stop"
    df["response_run"]       = df["effective_outcome"] == "response_run"
    df["response_run_back"]  = df["effective_outcome"] == "response_run_back"
    df["not_in_time"]        = df["effective_outcome"] == "not_in_time"
    df["safe_stop"]          = df["effective_outcome"] == "safe_stop"
    df["avoided"]            = (df["response_stop"] | df["response_run"]
                                | df["response_run_back"] | df["not_in_time"])

    # ── Trial-level export ────────────────────────────────
    trial_cols = [
        "pid", "mode", "run_id", "trial_type",
        "outcome", "effective_outcome",
        "ttc_at_response", "elapsed_ab", "t_c", "t_b_to_c",
        "early_warn_t", "full_warn_t", "timing_error",
        "responded", "collision", "response_stop", "not_in_time", "avoided", "safe_stop",
        "stopped_at_alarm", "stopped_at_alarm_raw",
        "looked_at_screen", "looked_at_traffic",
        "gaze_general_raw", "gaze_alarm_raw", "note",
        "walker_speed", "walking_style",
    ]
    df[trial_cols].to_csv(out_dir / "trials.csv", index=False)
    print(f"[SAVE] {out_dir / 'trials.csv'}")

    # ── Participant summary ───────────────────────────────
    p_summary = participant_summary(df)
    p_summary.to_csv(out_dir / "participants.csv", index=False)
    print(f"[SAVE] {out_dir / 'participants.csv'}")

    # ── Catch trial summary ───────────────────────────────
    c_summary = catch_summary(df)
    if not c_summary.empty:
        c_summary.to_csv(out_dir / "catch_trials.csv", index=False)
        print(f"[SAVE] {out_dir / 'catch_trials.csv'}")

    # ── Statistics ────────────────────────────────────────
    stats_text = run_statistics(df)
    print("\n" + stats_text)
    with open(out_dir / "stats.txt", "w", encoding="utf-8") as f:
        f.write(stats_text)
    print(f"\n[SAVE] {out_dir / 'stats.txt'}")

    # ── Plots ─────────────────────────────────────────────
    plot_results(df, out_dir / "plots")

    # ── Quick console overview ────────────────────────────
    print("\n── Trial counts ──────────────────────────────────")
    print(df.groupby(["trial_type", "mode", "effective_outcome"]).size()
            .rename("n")
            .to_string())

    print("\n── Mean TTC_at_response (hit trials) ─────────────")
    hit = df[(df["trial_type"] == "hit") & df["ttc_at_response"].notna()]
    print(hit.groupby("mode")["ttc_at_response"]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": "Mean (s)", "std": "SD", "count": "N"})
            .rename(index=MODE_LABELS)
            .round(3)
            .to_string())

    print("\n── Alarm behaviour (hit trials, adaptive) ────────")
    hit_a = df[(df["trial_type"] == "hit") & (df["mode"] == "a")]
    if "stopped_at_alarm_raw" in df.columns and hit_a["stopped_at_alarm_raw"].notna().any():
        raw_a = hit_a["stopped_at_alarm_raw"]
        print(f"  Stopped (any alarm) : {hit_a['stopped_at_alarm'].sum()}/{len(hit_a)}")
        print(f"    f only           : {(raw_a == 'f').sum()}")
        print(f"    vha only         : {(raw_a == 'vha').sum()}")
        print(f"    both             : {(raw_a == 'both').sum()}")
        print(f"    not stopped      : {(raw_a == 'no').sum()}")
        gg = hit_a["gaze_general_raw"]
        ga = hit_a["gaze_alarm_raw"]
        print(f"  Gaze – general:")
        print(f"    mainly phone     : {(gg == 'phone').sum()}")
        print(f"    mainly traffic   : {(gg == 'traffic').sum()}")
        print(f"    both             : {(gg == 'both').sum()}")
        print(f"    neither          : {(gg == 'neither').sum()}")
        print(f"  Gaze – alarm reaction:")
        print(f"    looked           : {(ga == 'looked').sum()}")
        print(f"    glanced          : {(ga == 'glanced').sum()}")
        print(f"    no               : {(ga == 'no').sum()}")

    print("\n── Alarm behaviour (hit trials, baseline) ────────")
    hit_b = df[(df["trial_type"] == "hit") & (df["mode"] == "b")]
    if "stopped_at_alarm_raw" in df.columns and hit_b["stopped_at_alarm_raw"].notna().any():
        raw_b  = hit_b["stopped_at_alarm_raw"]
        ggb = hit_b["gaze_general_raw"]
        gab = hit_b["gaze_alarm_raw"]
        print(f"  Stopped at alarm    : {hit_b['stopped_at_alarm'].sum()}/{len(hit_b)}")
        print(f"  Gaze – general:")
        print(f"    mainly phone     : {(ggb == 'phone').sum()}")
        print(f"    mainly traffic   : {(ggb == 'traffic').sum()}")
        print(f"    both             : {(ggb == 'both').sum()}")
        print(f"    neither          : {(ggb == 'neither').sum()}")
        print(f"  Gaze – alarm reaction:")
        print(f"    looked           : {(gab == 'looked').sum()}")
        print(f"    glanced          : {(gab == 'glanced').sum()}")
        print(f"    no               : {(gab == 'no').sum()}")

    print()


if __name__ == "__main__":
    main()