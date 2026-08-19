#!/usr/bin/env python3
"""
run_h1_h4_analysis.py — V2P Main Study | Master Analysis (H1-H4)
==================================================================
Merges all processed data sources into one consistent dataset and runs
the paired-comparison tests for H1-H4.

INPUT FILES (same folder, or pass --data_dir):
    trials.csv                — from analyze_logs.py   (TTC-R, effective_outcome, catch data)
    participants.csv          — from analyze_logs.py   (per-pid/mode TTC summary)
    rt_results.csv             — from analyze_rt_video.py (video-based RT)
                                 NOTE: run the annotation-merge patch first (see chat),
                                 otherwise 'outcome' in this file is unreliable and
                                 this script will re-derive it from trial_annotations.csv
                                 as a safety net (see correct_rt_outcome()).
    trial_annotations.csv      — post-hoc video annotation (ground truth)
    scored_summary.csv         — TLX / Van der Laan / TiA per pid x condition
    scored_demographics.csv    — demographics (descriptive only)

EXCLUSIONS:
    PID 4 & 15 (P4, P15) are dropped from RT and TTC-R (H1) only — adaptive-block video
    is missing, so the pair is incomplete for behavioral DVs. P3 is dropped due to too many timing errors in most of the trials (alarm too late). P3, P4 and P15 are KEPT in
    H2/H4 (questionnaire data unaffected). Change EXCLUDE_FROM_H1 to adjust.

OUTPUT:
    ./results_master/master_trial_level.csv        (long format, all trials)
    ./results_master/master_participant_level.csv   (wide, one row per pid)
    ./results_master/h1_h4_stats.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

EXCLUDE_FROM_H1 = {3, 4, 15}    # P3: Too many timing errors.
                                # P4: Adaptive video missing entirely.
                                # P15: Adaptive video has known playback/recording
                                # issues (see chat) — all 5 target trials failed
                                # RT detection, consistent with a video problem
                                # rather than genuinely undetectable reactions.
                                # PIDs to drop from RT / TTC-R paired analyses only.
EXCLUDE_FROM_TTC = {3, 4, 15}   # P3: variable walking speed invalidates marker-based TTC-R;
                                # also a video-frame gap risks alarm/trial mis-sync for the
                                # whole block (see chat). Set to set() to include P3 anyway.

# Outcomes that count as a genuine behavioral reaction for RT purposes.
# NOTE: 'response_run'/'response_run_back' are only reliably timed once
# find_body_reaction() is extended beyond pure velocity-drop detection
# (see patch discussed in chat). Until then their RT values may be
# systematically missing even though the trial is correctly classified.
RT_ELIGIBLE_OUTCOMES = {"response_stop", "response_run", "response_run_back"}
# Extended set (point 5, chat): also try response_continue_walk / not_in_time.
# Reported SEPARATELY, never merged into the strict set above.
RT_ELIGIBLE_OUTCOMES_EXTENDED = RT_ELIGIBLE_OUTCOMES | {"response_continue_walk", "not_in_time"}

MIN_TRIALS_PER_CONDITION = 3  # min valid RT trials/condition before a person-mean counts
                               # as stable (see chat) — below this, only used in the
                               # sensitivity pass, never in the main analysis

DATA_QUALITY_NOTES = {
    1: "Older run_sim.py version: no repeat alarm was sent after C was pressed "
       "(changed later; false negatives preferred over false positives). "
       "Document as a protocol-version change in Limitations; first-alarm RT/TTC-R "
       "for P1 remain comparable, multi-alarm/catch-trial dynamics may not be.",
    3: "Variable walking speed across most trials (breaks the fixed-marker TTC-R "
       "assumption) + a missing trial on video for the first block, which risks a "
       "sync-offset in the greedy alarm-to-trial matcher for the rest of that block. "
       "Excluded from TTC-R (see EXCLUDE_FROM_TTC) pending manual re-check of the sync.",
}

CONDITION_TO_MODE = {"adaptive": "a", "baseline": "b", "fixed": "b", "fixed baseline": "b",
                     "adaptive system": "a"}
MODE_TO_CONDITION = {"a": "Adaptive", "b": "Baseline"}


# ─────────────────────────────────────────────────────────
# NORMALIZATION HELPERS
# ─────────────────────────────────────────────────────────
def normalize_pid(x) -> int:
    """'P01' / 'P1' / 1 / '1' -> 1 (int)."""
    s = str(x).strip()
    s = s.lstrip("Pp").lstrip("0") or "0"
    return int(s)


def normalize_mode(x) -> str:
    """Any condition/mode spelling -> 'a' or 'b'."""
    s = str(x).strip().lower()
    if s in ("a", "b"):
        return s
    if s in CONDITION_TO_MODE:
        return CONDITION_TO_MODE[s]
    raise ValueError(f"Unrecognized condition/mode value: {x!r}")


# ─────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────
def load_trials(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "trials.csv")
    df["pid"] = df["pid"].apply(normalize_pid)
    df["mode"] = df["mode"].apply(normalize_mode)
    return df


def load_annotations(data_dir: Path) -> pd.DataFrame:
    ann = pd.read_csv(data_dir / "trial_annotations.csv")
    ann.columns = [c.lower().strip() for c in ann.columns]
    ann = ann.rename(columns={"run": "run_id"})
    ann["pid"] = ann["pid"].apply(normalize_pid)
    ann["mode"] = ann["mode"].apply(normalize_mode)
    return ann


def correct_rt_outcome(rt: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
    """
    Safety net: overwrite rt_results 'outcome' with the annotated Actual_Outcome
    whenever available. This reproduces the fix described in chat in case
    analyze_rt_video.py has not been re-run with the annotation-merge patch yet.
    """
    ann_small = ann[["pid", "mode", "run_id", "actual_outcome"]].copy()
    rt = rt.merge(ann_small, left_on=["pid", "mode", "trial_id"],
                  right_on=["pid", "mode", "run_id"], how="left")
    mask = rt["actual_outcome"].notna() & (rt["actual_outcome"] != "")
    rt["effective_outcome"] = np.where(mask, rt["actual_outcome"], rt["outcome"])
    return rt.drop(columns=["run_id"])


def load_rt_results(data_dir: Path, ann: pd.DataFrame) -> pd.DataFrame:
    # Prefer the file with true kinematic TTC-R (compute_ttc_r.py output) if present.
    ttc_path = data_dir / "rt_results_with_true_ttc.csv"
    src = ttc_path if ttc_path.exists() else data_dir / "rt_results.csv"
    rt = pd.read_csv(src)
    rt["pid"] = rt["participant_id"].apply(normalize_pid)
    rt["mode"] = rt["condition"].apply(normalize_mode)
    rt = correct_rt_outcome(rt, ann)
    if "ttc_at_actual_response" not in rt.columns:
        rt["ttc_at_actual_response"] = np.nan
        rt["ttc_lookup_status"] = "not_computed"
    # Trials whose reaction lies outside the logged simulation window are almost
    # certainly post-trial movement (see chat) — exclude from RT too, not just TTC-R.
    suspect = rt["ttc_lookup_status"].astype(str).str.startswith("no_frame_within")
    if suspect.any():
        rt.loc[suspect, "rt_s"] = np.nan
        rt.loc[suspect, "notes"] = "rt_likely_post_trial_movement"
    return rt


def load_summary(data_dir: Path) -> pd.DataFrame:
    ss = pd.read_csv(data_dir / "scored_summary.csv")
    ss["pid"] = ss["participant_id"].apply(normalize_pid)
    ss["mode"] = ss["condition"].apply(normalize_mode)
    return ss


def load_demographics(data_dir: Path) -> pd.DataFrame:
    dg = pd.read_csv(data_dir / "scored_demographics.csv")
    dg["pid"] = dg["participant_id"].apply(normalize_pid)
    return dg


# ─────────────────────────────────────────────────────────
# H1 — RT & TTC-R
# ─────────────────────────────────────────────────────────
def paired_test(a_vals, b_vals, label: str) -> tuple[str, dict]:
    n = len(a_vals)
    lines = [f"\n── {label} (N complete pairs = {n}) " + "─" * max(1, 40 - len(label))]
    result = {"label": label, "n": n, "p": np.nan, "test": None}
    if n < 3:
        lines.append("  [!] N too small for inferential test.")
        return "\n".join(lines), result

    lines.append(f"  Adaptive : M={np.mean(a_vals):.3f}  SD={np.std(a_vals, ddof=1):.3f}")
    lines.append(f"  Baseline : M={np.mean(b_vals):.3f}  SD={np.std(b_vals, ddof=1):.3f}")

    diff = np.array(a_vals) - np.array(b_vals)

    # Normality of DIFFERENCE scores decides the test, not N alone.
    shapiro_p = None
    if n >= 3:
        _, shapiro_p = stats.shapiro(diff)
        normal = shapiro_p > .05
    else:
        normal = False

    if normal:
        stat, p = stats.ttest_rel(a_vals, b_vals)
        test_name, stat_label = "Paired t-test", f"t({n-1}) = {stat:.3f}"
    else:
        try:
            stat, p = stats.wilcoxon(a_vals, b_vals)
            test_name, stat_label = "Wilcoxon signed-rank", f"W = {stat:.3f}"
        except ValueError:
            stat, p = stats.ttest_rel(a_vals, b_vals)
            test_name, stat_label = "Paired t-test (Wilcoxon fallback failed)", f"t({n-1}) = {stat:.3f}"

    d = diff.mean() / (diff.std(ddof=1) + 1e-9)
    sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
    shapiro_note = (f"Shapiro-Wilk on differences: p={shapiro_p:.3f} "
                     f"({'normal' if normal else 'non-normal -> nonparametric used'})"
                     if shapiro_p is not None else "")
    lines.append(f"  [{shapiro_note}]")
    lines.append(f"  {test_name}: {stat_label}, p = {p:.4f} {sig}")
    lines.append(f"  Cohen's d = {d:.3f}   Delta (Adaptive-Baseline) = {diff.mean():+.3f}")

    result.update({"p": p, "test": test_name, "d": d})
    return "\n".join(lines), result


def mcnemar_test(a_binary, b_binary, label: str) -> tuple[str, dict]:
    """Paired binary comparison (e.g. collision yes/no per participant-block)."""
    n = len(a_binary)
    lines = [f"\n── {label} (McNemar, paired binary, N = {n}) " + "─" * 10]
    result = {"label": label, "n": n, "p": np.nan, "test": "McNemar"}

    a_binary = np.asarray(a_binary).astype(bool)
    b_binary = np.asarray(b_binary).astype(bool)
    both      = np.sum(a_binary & b_binary)
    a_only    = np.sum(a_binary & ~b_binary)
    b_only    = np.sum(~a_binary & b_binary)
    neither   = np.sum(~a_binary & ~b_binary)
    table = [[both, a_only], [b_only, neither]]

    lines.append(f"  Adaptive rate : {a_binary.mean():.1%}   Baseline rate : {b_binary.mean():.1%}")
    lines.append(f"  Discordant pairs: Adaptive-only={a_only}, Baseline-only={b_only} "
                 f"(McNemar only uses these — concordant pairs [{both}+{neither}] "
                 "carry no information about the difference)")

    if a_only + b_only < 6:
        lines.append("  [!] Fewer than 6 discordant pairs — McNemar unreliable here, "
                      "treat as descriptive only.")
        return "\n".join(lines), result

    res = mcnemar(table, exact=(a_only + b_only < 25), correction=True)
    p = res.pvalue
    sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
    lines.append(f"  McNemar: statistic={res.statistic:.3f}, p = {p:.4f} {sig}")
    result["p"] = p
    return "\n".join(lines), result


def descriptive_table(ss: pd.DataFrame, columns: dict, title: str) -> str:
    """Descriptive M/SD per condition, no inferential test — for subscales
    reported for completeness/transparency but not part of confirmatory
    hypothesis testing (avoids the appearance of cherry-picking a subscale
    post-hoc; makes clear the tested subscale was the pre-specified one)."""
    out = ["\n" + "=" * 65, title, "=" * 65,
           "[Descriptive only — not part of confirmatory H1-H4 testing / "
           "multiple-comparison correction]"]
    out.append(f"  {'Subscale':38s} {'Adaptive M':>12s} {'(SD)':>8s} "
               f"{'Baseline M':>12s} {'(SD)':>8s}   n")
    for label, col in columns.items():
        wide = ss.groupby(["pid", "mode"])[col].mean().unstack("mode").dropna()
        a, b = wide["a"], wide["b"]
        out.append(f"  {label:38s} {a.mean():12.3f} {a.std():8.3f} "
                   f"{b.mean():12.3f} {b.std():8.3f}   {len(wide)}")
    return "\n".join(out)


def analyze_h1(trials: pd.DataFrame, rt: pd.DataFrame) -> tuple[str, pd.DataFrame, list]:
    out = ["=" * 65, "H1 — Reaction Time (RT) & Time-to-Collision-at-Response (TTC-R)",
           "=" * 65]
    test_results = []

    trials_h1 = trials[~trials["pid"].isin(EXCLUDE_FROM_H1)]
    rt_h1 = rt[~rt["pid"].isin(EXCLUDE_FROM_H1)]
    if EXCLUDE_FROM_H1:
        out.append(f"[EXCLUDED from H1 only] PID(s): {sorted(EXCLUDE_FROM_H1)} "
                    "(incomplete video -> RT/TTC-R pair unavailable)")
    for pid, note in DATA_QUALITY_NOTES.items():
        out.append(f"[DATA QUALITY NOTE] P{pid}: {note}")

    # --- H1a TTC-R: TRUE kinematic TTC-at-response (compute_ttc_r.py) — primary ---
    rt_ttc = rt_h1[rt_h1["ttc_lookup_status"] == "ok"]
    ttc_counts = rt_ttc.groupby(["pid", "mode"])["ttc_at_actual_response"].count().unstack("mode")
    ttc_means  = rt_ttc.groupby(["pid", "mode"])["ttc_at_actual_response"].mean().unstack("mode")
    ttc_enough = (ttc_counts.get("a", pd.Series(dtype=float)).fillna(0) >= MIN_TRIALS_PER_CONDITION) & \
                 (ttc_counts.get("b", pd.Series(dtype=float)).fillna(0) >= MIN_TRIALS_PER_CONDITION)
    ttc_paired = ttc_means[ttc_enough].dropna()
    if len(ttc_paired) > 0:
        txt, res = paired_test(ttc_paired["a"].values, ttc_paired["b"].values,
                                f"H1a TTC-R (s) — true kinematic, >= {MIN_TRIALS_PER_CONDITION} trials/cond.")
    else:
        txt, res = "\n  [H1a TTC-R] Not enough trials yet to test.", {"label": "H1a TTC-R", "n": 0, "p": np.nan}
    out.append(txt); test_results.append(res)
    out.append(f"  [INFO] {ttc_enough.sum()} of {len(ttc_counts)} people have >= "
               f"{MIN_TRIALS_PER_CONDITION} resolved ttc_at_actual_response trials/condition. "
               f"Requires rt_results_with_true_ttc.csv (compute_ttc_r.py output).")

    # --- Old walking-speed-based value — descriptive only, NOT the confirmatory H1a test ---
    trials_ttc = trials_h1[~trials_h1["pid"].isin(EXCLUDE_FROM_TTC)]
    if EXCLUDE_FROM_TTC:
        out.append(f"[EXCLUDED from baseline-approach descriptive only] PID(s): {sorted(EXCLUDE_FROM_TTC)}")
    hit = trials_ttc[(trials_ttc["trial_type"] == "hit") & trials_ttc["ttc_at_response"].notna()]
    baseline_appr = hit.groupby(["pid", "mode"])["ttc_at_response"].mean().unstack("mode").dropna()
    out.append(f"\n  [DESCRIPTIVE ONLY, not H1a] ttc_baseline_approach (pre-alarm walking-speed "
               f"extrapolation): Adaptive M={baseline_appr['a'].mean():.3f}  "
               f"Baseline M={baseline_appr['b'].mean():.3f}  (N={len(baseline_appr)})")

    # --- H1b RT: strict (main) + extended-outcome + full-sensitivity ---
    def _rt_test(eligible_outcomes: set, label: str, apply_min_trials: bool):
        rv = rt_h1[rt_h1["effective_outcome"].isin(eligible_outcomes)]
        counts = rv.groupby(["pid", "mode"])["rt_s"].count().unstack("mode")
        means  = rv.groupby(["pid", "mode"])["rt_s"].mean().unstack("mode")
        if apply_min_trials:
            ok = (counts.get("a", pd.Series(dtype=float)).fillna(0) >= MIN_TRIALS_PER_CONDITION) & \
                 (counts.get("b", pd.Series(dtype=float)).fillna(0) >= MIN_TRIALS_PER_CONDITION)
            paired = means[ok].dropna()
        else:
            paired = means.dropna()
        txt_, res_ = paired_test(paired["a"].values, paired["b"].values, label)
        return txt_, res_, paired, counts

    txt, res, rt_paired, rt_counts = _rt_test(
        RT_ELIGIBLE_OUTCOMES, f"H1b RT — main (stop/run/run_back, >= {MIN_TRIALS_PER_CONDITION} trials/cond.)",
        apply_min_trials=True)
    out.append(txt); test_results.append(res)

    txt_sens, res_sens, _, _ = _rt_test(
        RT_ELIGIBLE_OUTCOMES, "H1b RT — sensitivity (ALL people, incl. <3 trials/cond.)",
        apply_min_trials=False)
    out.append(txt_sens)
    out.append("  [WARNING] Includes people with only 1-2 trials -> noisier person means. "
               "Robustness check only, not in the multiple-comparison correction below.")

    txt_ext, res_ext, _, _ = _rt_test(
        RT_ELIGIBLE_OUTCOMES_EXTENDED,
        f"H1b RT — extended outcomes (+ continue_walk/not_in_time, >= {MIN_TRIALS_PER_CONDITION} trials/cond.)",
        apply_min_trials=True)
    out.append(txt_ext)
    out.append("  [NOTE] Point 5 (chat): reported separately from the main RT test, not merged — "
               "response_continue_walk/not_in_time reflect a different response quality than "
               "an evasive stop/run/run_back.")

    # --- Collision rate (McNemar, paired binary, effective_outcome) ---
    hit_all = trials[trials["trial_type"] == "hit"]
    coll_paired = hit_all.groupby(["pid", "mode"])["collision"].mean().unstack("mode").dropna()
    coll_bin = (coll_paired > 0).astype(int)  # participant-level: had >=1 collision in that block?
    txt, res = mcnemar_test(coll_bin["a"].values, coll_bin["b"].values,
                             "H1c Collision occurrence (>=1 collision per block)")
    out.append(txt); test_results.append(res)
    out.append(f"  [Descriptive, trial-level] Adaptive: {coll_paired['a'].mean():.1%}   "
               f"Baseline: {coll_paired['b'].mean():.1%}")

    # --- False-stop rate (catch trials, adaptive only — see rationale below) ---
    catch = trials[trials["trial_type"] == "stop"]
    fsr_a = catch[catch["mode"] == "a"].groupby("pid")["stopped_at_alarm"].mean()
    out.append(f"\n── False-stop rate (catch trials, Adaptive only, N={len(fsr_a)}) "
               "──────────────")
    out.append(f"  Adaptive : {fsr_a.mean():.1%} +/- {fsr_a.std():.1%}")
    out.append("  [NOTE] Baseline is intentionally NOT reported here: the fixed-baseline "
               "system never issues any alarm (f or vha) on catch trials, so a 'false "
               "stop' is structurally undefined for Baseline (0/0), not a meaningful "
               "0% comparison point. Report Adaptive's 33.3% as a standalone descriptive "
               "statistic, not a paired contrast (not included in multiple-comparison "
               "correction below, since there is no comparison being made).")

    merged = ttc_paired.add_suffix("_ttc").join(rt_paired.add_suffix("_rt"), how="outer")
    return "\n".join(out), merged.reset_index(), test_results


# NOTE (point 4, chat): Horizontal gaze/head-turn detection is a planned addition to
# analyze_rt_video.py's find_body_reaction() — most participants oriented toward the
# projection on their right (left side of frame), not straight up, so the existing
# vertical-only gaze_up channel misses this. To be added as a new "gaze_side" channel
# once agreed; existing gaze_up (vertical) stays active in parallel, not replaced.


# ─────────────────────────────────────────────────────────
# H2 / H4 — Questionnaires
# ─────────────────────────────────────────────────────────
def analyze_questionnaires(ss: pd.DataFrame) -> tuple[str, list]:
    out = ["\n" + "=" * 65,
           "H2 — Cognitive Load (NASA-TLX)  /  H3 — Acceptance (Van der Laan)  /  "
           "H4 — Trust (TiA)", "=" * 65]
    test_results = []

    metrics = {
        "H2  tlx_weighted_score (cognitive load, lower=better for Adaptive)":
            "tlx_weighted_score",
        "H3  vdl_usefulness (acceptance, higher=better for Adaptive)": "vdl_usefulness",
        "H3  vdl_satisfying (acceptance, higher=better for Adaptive)": "vdl_satisfying",
        "H4  tia_trust_in_automation_core (trust, higher=better for Adaptive)":
            "tia_trust_in_automation_core",
    }
    for label, col in metrics.items():
        wide = ss.groupby(["pid", "mode"])[col].mean().unstack("mode").dropna()
        txt, res = paired_test(wide["a"].values, wide["b"].values, label)
        out.append(txt); test_results.append(res)

    return "\n".join(out), test_results


def multiplicity_report(family_a: list, family_b: list) -> str:
    out = ["\n" + "=" * 65, "MULTIPLE-COMPARISON CORRECTION (Holm-Bonferroni, "
           "corrected WITHIN each pre-specified hypothesis family — see chat "
           "for rationale: family boundaries follow H1 vs H2-H4 wording, not "
           "post-hoc grouping)", "=" * 65]

    def _block(label, results):
        lines = [f"\n  Family: {label} (m={len(results)})"]
        labels_ = [r["label"] for r in results]
        pvals_  = [r["p"] for r in results]
        valid   = [i for i, p in enumerate(pvals_) if not np.isnan(p)]
        if len(valid) < 2:
            lines.append("    Not enough valid tests to correct.")
            return lines
        reject, p_adj, _, _ = multipletests([pvals_[i] for i in valid],
                                              alpha=0.05, method="holm")
        lines.append(f"    {'Test':50s} {'raw p':>8s} {'holm p':>8s}  sig")
        for idx, r, pa in zip(valid, reject, p_adj):
            sig = "*" if r else "n.s."
            lines.append(f"    {labels_[idx]:50s} {pvals_[idx]:8.4f} {pa:8.4f}  {sig}")
        return lines

    out += _block("A — Behavioral (H1: TTC-R, RT)", family_a)
    out += _block("B — Self-report (H2-H4: TLX, VdL x2, TiA)", family_b)
    out.append("\n  [NOTE] Collision occurrence (McNemar) and false-stop rate are "
               "reported with UNCORRECTED p-values as exploratory/secondary safety "
               "indicators — H1 as worded names only RT and TTC-R as confirmatory DVs.")
    return "\n".join(out)


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".", help="Folder containing all input CSVs")
    parser.add_argument("--out_dir", default="./results_master")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    trials = load_trials(data_dir)
    ann = load_annotations(data_dir)
    rt = load_rt_results(data_dir, ann)
    ss = load_summary(data_dir)
    dg = load_demographics(data_dir)

    report = []
    report.append("Significance level: alpha = .05, two-tailed, for all inferential "
                   "tests below (paired t-test / Wilcoxon signed-rank / McNemar). "
                   "Effect sizes reported as Cohen's d for paired samples "
                   "(mean difference / SD of differences).\n")
    h1_text, h1_wide, h1_results = analyze_h1(trials, rt)
    report.append(h1_text)
    q_text, q_results = analyze_questionnaires(ss)
    report.append(q_text)

    tia_all_subscales = {
        "TiA Reliability/Competence":       "tia_reliability_competence",
        "TiA Understandability":            "tia_understandability_predictability",
        "TiA Familiarity":                  "tia_familiarity",
        "TiA Intention of Developers":      "tia_intention_of_developers",
        "TiA Propensity to Trust":          "tia_propensity_to_trust",
        "TiA Trust in Automation (core)":   "tia_trust_in_automation_core",
    }
    report.append(descriptive_table(
        ss, tia_all_subscales,
        "TiA — ALL SIX SUBSCALES (descriptive, for Ch. 6.1 completeness)"))
    report.append("  [Note] Only 'Trust in Automation (core)' (last row) is the "
                   "pre-specified H4 DV and appears again above with its inferential "
                   "test. The other five are reported here for transparency only.")
    # h1_results = [TTC-R, RT, Collision(McNemar)] in that order (see analyze_h1)
    family_a = h1_results[:2]       # TTC-R, RT — matches H1 wording exactly
    family_b = q_results            # TLX, VdL-usefulness, VdL-satisfying, TiA
    report.append(multiplicity_report(family_a, family_b))
    report.append(f"\n\nN demographics on file: {dg['pid'].nunique()}")

    (out_dir / "h1_h4_stats.txt").write_text("\n".join(report), encoding="utf-8")

    trials.to_csv(out_dir / "master_trial_level.csv", index=False)
    h1_wide.to_csv(out_dir / "master_participant_level_h1.csv", index=False)

    plot_h1_h4_results(h1_wide, ss, out_dir)
    plot_tlx_weights(ss, out_dir)

    print("\n".join(report))
    print(f"\n[SAVE] {out_dir/'h1_h4_stats.txt'}")
    print(f"[SAVE] {out_dir/'master_trial_level.csv'}")
    print(f"[SAVE] {out_dir/'master_participant_level_h1.csv'}")


def plot_h1_h4_results(h1_wide: pd.DataFrame, ss: pd.DataFrame, out_dir: Path):
    """
    Paired-comparison plots (Adaptive vs Baseline) for all confirmatory
    measures, saved to out_dir/plots/. Mirrors the style already used in
    analyze_logs.py's plot_results() for visual consistency across the
    thesis's figures.
    """
    import matplotlib.pyplot as plt
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    colors = {"Adaptive": "#2196F3", "Baseline": "#FF5722"}

    def _paired_plot(ax, a_vals, b_vals, title, ylabel):
        pairs = list(zip(a_vals, b_vals))
        for a, b in pairs:
            ax.plot(["Adaptive", "Baseline"], [a, b],
                    color="gray", alpha=0.35, linewidth=1, marker="o", markersize=4)
        ax.scatter(["Adaptive"] * len(a_vals), a_vals,
                    color=colors["Adaptive"], s=55, zorder=3, alpha=0.8)
        ax.scatter(["Baseline"] * len(b_vals), b_vals,
                    color=colors["Baseline"], s=55, zorder=3, alpha=0.8)
        ax.plot("Adaptive", np.mean(a_vals), marker="D", color=colors["Adaptive"],
                 markersize=11, zorder=4, markeredgecolor="black")
        ax.plot("Baseline", np.mean(b_vals), marker="D", color=colors["Baseline"],
                 markersize=11, zorder=4, markeredgecolor="black")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)

    # ── H1: TTC-R and RT (from the already-merged h1_wide) ────────────
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.suptitle("H1 — Behavioral Measures", fontsize=13, fontweight="bold")
    if "a_ttc" in h1_wide.columns and "b_ttc" in h1_wide.columns:
        sub = h1_wide[["a_ttc", "b_ttc"]].dropna()
        _paired_plot(axes[0], sub["a_ttc"].values, sub["b_ttc"].values,
                     f"TTC-R (s)  N={len(sub)}", "TTC-R (s)")
    if "a_rt" in h1_wide.columns and "b_rt" in h1_wide.columns:
        sub = h1_wide[["a_rt", "b_rt"]].dropna()
        _paired_plot(axes[1], sub["a_rt"].values, sub["b_rt"].values,
                     f"RT (s)  N={len(sub)}", "RT (s)")
    plt.tight_layout()
    plt.savefig(plots_dir / "h1_behavioral.pdf", bbox_inches="tight")
    plt.savefig(plots_dir / "h1_behavioral.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved -> {plots_dir / 'h1_behavioral.pdf'}")

    # ── H2-H4: Questionnaire measures ──────────────────────────────────
    metrics = {
        "TLX (weighted)": "tlx_weighted_score",
        "VdL Usefulness": "vdl_usefulness",
        "VdL Satisfying": "vdl_satisfying",
        "TiA Trust (core)": "tia_trust_in_automation_core",
    }
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.suptitle("H2-H4 — Self-Report Measures", fontsize=13, fontweight="bold")
    for ax, (label, col) in zip(axes, metrics.items()):
        wide = ss.groupby(["pid", "mode"])[col].mean().unstack("mode").dropna()
        if "a" in wide.columns and "b" in wide.columns:
            _paired_plot(ax, wide["a"].values, wide["b"].values,
                         f"{label}  N={len(wide)}", label)
    plt.tight_layout()
    plt.savefig(plots_dir / "h2_h4_selfreport.pdf", bbox_inches="tight")
    plt.savefig(plots_dir / "h2_h4_selfreport.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved -> {plots_dir / 'h2_h4_selfreport.pdf'}")


def plot_tlx_weights(ss: pd.DataFrame, out_dir: Path):
    """
    Grouped bar chart of the mean NASA-TLX pairwise weight per subscale,
    split by condition (Adaptive vs Baseline) — visualizes which
    dimensions participants treated as most relevant to their perceived
    workload, and whether that weighting differed between conditions.
    """
    import matplotlib.pyplot as plt
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    colors = {"Adaptive": "#2196F3", "Baseline": "#FF5722"}

    weight_cols = [c for c in ss.columns if c.startswith("tlx_weight_")
                   and c != "tlx_weight_sum_check"]
    if not weight_cols:
        print("[PLOT] No tlx_weight_* columns found — skipping TLX weight plot")
        return

    # 'condition' in ss may be "Adaptive System"/"Fixed Baseline" — normalize labels
    label_map = {}
    for c in ss["condition"].unique():
        cl = str(c).lower()
        if "adapt" in cl:
            label_map[c] = "Adaptive"
        elif "baseline" in cl or "fixed" in cl:
            label_map[c] = "Baseline"
    ss = ss.copy()
    ss["condition_label"] = ss["condition"].map(label_map).fillna(ss["condition"])

    means = ss.groupby("condition_label")[weight_cols].mean()
    # Order dimensions by overall (pooled) mean, descending — matches the
    # earlier chat analysis (Temporal > Mental > Performance > Effort >
    # Frustration > Physical)
    pooled_order = ss[weight_cols].mean().sort_values(ascending=False).index.tolist()
    dim_labels = [c.replace("tlx_weight_", "").title() for c in pooled_order]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(pooled_order))
    width = 0.35
    for i, cond in enumerate(["Adaptive", "Baseline"]):
        if cond not in means.index:
            continue
        vals = [means.loc[cond, c] for c in pooled_order]
        ax.bar(x + i * width - width / 2, vals, width,
               label=cond, color=colors[cond], alpha=0.85)

    ax.set_title("NASA-TLX — Mean Pairwise Weight per Subscale", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean weight (0–5)")
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "tlx_weights.pdf", bbox_inches="tight")
    plt.savefig(plots_dir / "tlx_weights.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved -> {plots_dir / 'tlx_weights.pdf'}")


if __name__ == "__main__":
    main()