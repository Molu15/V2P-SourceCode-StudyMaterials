"""
============================================================
V2P MAIN STUDY -- UNIFIED QUESTIONNAIRE SCORING SCRIPT
============================================================
Aggregates the JSON files downloaded from the four standalone HTML
questionnaires:
  - tlx_questionnaire.html          ("..._TLX_..._....json")
  - vdl_questionnaire.html          ("..._VdL_..._....json")
  - tia_questionnaire.html          ("..._TiA_..._....json")
  - demographics_questionnaire.html ("..._Demographics_....json")

Each questionnaire already computes its own scores client-side (raw
TLX mean, weighted TLX score, Van der Laan usefulness/satisfying,
Koerber TiA 6 subscale means). This script does NOT recompute those
scores from raw item data -- it reads the precomputed values directly
from each JSON file and joins everything into one row per
participant x condition, plus one demographics row per participant.

This avoids duplicating (and risking re-breaking) the scoring logic
that already lives in the HTML tools; it purely aggregates.

INPUT:
  A folder containing all downloaded JSON files (any mix of TLX, VdL,
  TiA, Demographics; both conditions; multiple participants). Files
  are recognized by their "instrument" field inside the JSON, not by
  filename, so renamed files still work.

USAGE:
  python score_v2p_questionnaires.py --input_dir ./responses --out scored_summary.csv

  Produces two files:
    scored_summary.csv            -- one row per participant x condition
                                      (TLX + VdL + TiA scores joined)
    scored_demographics.csv       -- one row per participant

Requires: pandas (pip install pandas --break-system-packages)
============================================================
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

TIA_SUBSCALE_KEYS = [
    "Reliability/Competence",
    "Understandability/Predictability",
    "Familiarity",
    "Intention of Developers",
    "Propensity to Trust",
    "Trust in Automation (core)",
]


def load_json_files(input_dir: Path):
    """Yield (filename, parsed_json) for every .json file in input_dir."""
    for path in sorted(input_dir.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            yield path.name, data
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: could not read {path.name}: {e}", file=sys.stderr)


def key_for(pid, condition):
    return (pid, condition)

def print_tlx_weight_summary(df_cond: pd.DataFrame):
    weight_cols = [c for c in df_cond.columns if c.startswith("tlx_weight_")
                   and c != "tlx_weight_sum_check"]
    if not weight_cols:
        return
    print(f"\n{'='*55}")
    print(f"  NASA-TLX — MEAN PAIRWISE WEIGHT PER SUBSCALE")
    print(f"  (0-5, how often each dimension 'won' a pairwise comparison;")
    print(f"   pooled across all participants and BOTH conditions)")
    print(f"{'='*55}")
    means = df_cond[weight_cols].mean().sort_values(ascending=False)
    for col, m in means.items():
        label = col.replace("tlx_weight_", "").title()
        print(f"     {label:<20} M={m:.2f}")

    print(f"\n  -- by condition --")
    if "condition" in df_cond.columns:
        by_cond = df_cond.groupby("condition")[weight_cols].mean()
        for cond in by_cond.index:
            print(f"\n     {cond}:")
            row = by_cond.loc[cond].sort_values(ascending=False)
            for col, m in row.items():
                label = col.replace("tlx_weight_", "").title()
                print(f"       {label:<18} M={m:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate V2P Main Study questionnaire JSON exports.")
    parser.add_argument("--input_dir", required=True, help="Folder containing downloaded JSON files")
    parser.add_argument("--out", default="scored_summary.csv", help="Output CSV path for condition-level data")
    parser.add_argument("--out_demographics", default="scored_demographics.csv", help="Output CSV path for demographics")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"ERROR: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # condition_rows[(pid, condition)] accumulates fields from TLX/VdL/TiA files
    condition_rows = defaultdict(dict)
    demographics_rows = {}

    file_count = {"TLX": 0, "VdL": 0, "TiA": 0, "Demographics": 0, "unknown": 0}

    for fname, data in load_json_files(input_dir):
        instrument = data.get("instrument", "")
        pid = data.get("participant_id", "").strip()
        if not pid:
            print(f"WARNING: {fname} has no participant_id, skipping", file=sys.stderr)
            continue

        if instrument.startswith("NASA-TLX"):
            file_count["TLX"] += 1
            condition = data.get("condition")
            k = key_for(pid, condition)
            weights = data.get("weights", {})
            ratings = data.get("ratings", {})
            row = {
                "participant_id": pid,
                "session_date": data.get("session_date"),
                "condition": condition,
                "block_order": data.get("block_order"),
                "tlx_raw_mean": data.get("tlx_raw_mean"),
                "tlx_weighted_score": data.get("tlx_weighted_score"),
                "tlx_weight_sum_check": data.get("weight_sum_check"),
            }
            for dim in ["mental", "physical", "temporal", "performance", "effort", "frustration"]:
                row[f"tlx_weight_{dim}"] = weights.get(dim)
                row[f"tlx_rating_{dim}"] = ratings.get(dim)
            condition_rows[k].update(row)

        elif instrument.startswith("Van der Laan"):
            file_count["VdL"] += 1
            condition = data.get("condition")
            k = key_for(pid, condition)
            condition_rows[k].update({
                "participant_id": pid,
                "session_date": data.get("session_date"),
                "condition": condition,
                "block_order": data.get("block_order"),
                "vdl_usefulness": data.get("vdl_usefulness"),
                "vdl_satisfying": data.get("vdl_satisfying"),
            })

        elif instrument.startswith("Trust in Automation"):
            file_count["TiA"] += 1
            condition = data.get("condition")
            k = key_for(pid, condition)
            subscales = data.get("subscale_means", {})
            row = {
                "participant_id": pid,
                "session_date": data.get("session_date"),
                "condition": condition,
                "block_order": data.get("block_order"),
                "tia_unanswered_item_count": data.get("unanswered_item_count"),
            }
            for name in TIA_SUBSCALE_KEYS:
                col = "tia_" + name.lower().replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
                row[col] = subscales.get(name)
            condition_rows[k].update(row)

        elif instrument.startswith("Demographics"):
            file_count["Demographics"] += 1
            demo = data.get("demographics", {})
            screening = data.get("screening", {})
            row = {
                "participant_id": pid,
                "session_date": data.get("session_date"),
            }
            for sk, sv in screening.items():
                row["screen_" + sk] = sv
            for dk, dv in demo.items():
                row["demo_" + dk] = dv
            demographics_rows[pid] = row

        else:
            file_count["unknown"] += 1
            print(f"WARNING: {fname} has unrecognized instrument '{instrument}', skipping", file=sys.stderr)

    print("Files processed:", file_count)

    # Build condition-level summary
    if condition_rows:
        df_cond = pd.DataFrame(list(condition_rows.values()))
        df_cond = df_cond.sort_values(["participant_id", "condition"]).reset_index(drop=True)
        df_cond.to_csv(args.out, index=False)
        print(f"\nWrote {len(df_cond)} condition-level rows to: {args.out}")

        # Pretty print: one participant block at a time
        score_cols = {
            "TLX": ["tlx_raw_mean", "tlx_weighted_score", "tlx_weight_sum_check"],
            "VdL": ["vdl_usefulness", "vdl_satisfying"],
            "TiA": [
                "tia_reliability_competence",
                "tia_understandability_predictability",
                "tia_familiarity",
                "tia_intention_of_developers",
                "tia_propensity_to_trust",
                "tia_trust_in_automation_core",
                "tia_unanswered_item_count",
            ],
        }

        for pid in df_cond["participant_id"].unique():
            rows = df_cond[df_cond["participant_id"] == pid]
            print(f"\n{'='*55}")
            print(f"  Participant: {pid}")
            print(f"{'='*55}")
            for _, row in rows.iterrows():
                print(f"\n  Condition : {row['condition']}")
                print(f"  Block     : {row['block_order']}")
                print(f"  Date      : {row['session_date']}")
                for instrument, cols in score_cols.items():
                    present = [c for c in cols if c in row.index]
                    if not present:
                        continue
                    print(f"\n  -- {instrument} --")
                    for col in present:
                        label = col.replace("tlx_", "").replace("vdl_", "").replace("tia_", "").replace("_", " ").title()
                        val = row[col]
                        val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
                        print(f"     {label:<40} {val_str}")

        # Sanity check: each participant should have exactly 2 condition rows
        counts = df_cond["participant_id"].value_counts()
        irregular = counts[counts != 2]
        if not irregular.empty:
            print("\nWARNING: the following participant IDs do not have exactly "
                  "2 condition rows (1 per condition) -- check for typos, missing "
                  "files, or incomplete questionnaires:", file=sys.stderr)
            print(irregular.to_string(), file=sys.stderr)

        # Sanity check: flag rows missing any of TLX/VdL/TiA data
        expected_cols = ["tlx_raw_mean", "vdl_usefulness", "tia_trust_in_automation_core"]
        present_cols = [c for c in expected_cols if c in df_cond.columns]
        if present_cols:
            incomplete = df_cond[df_cond[present_cols].isna().any(axis=1)]
            if not incomplete.empty:
                print("\nWARNING: missing questionnaire data for:", file=sys.stderr)
                print(incomplete[["participant_id", "condition"] + present_cols].to_string(index=False), file=sys.stderr)

        # Overall summary: mean per condition across all participants
        numeric_cols = df_cond.select_dtypes(include="number").columns.tolist()
        exclude = ["tlx_weight_sum_check", "tia_unanswered_item_count"]
        summary_cols = [c for c in numeric_cols if c not in exclude]

        if summary_cols:
            print(f"\n{'='*55}")
            print(f"  OVERALL SUMMARY  (N={df_cond['participant_id'].nunique()} participants)")
            print(f"  Mean scores per condition")
            print(f"{'='*55}")
            grouped = df_cond.groupby("condition")[summary_cols].mean()
            col_groups = {
                "TLX": [c for c in summary_cols if c.startswith("tlx_")],
                "Van der Laan": [c for c in summary_cols if c.startswith("vdl_")],
                "Trust in Automation": [c for c in summary_cols if c.startswith("tia_")],
            }
            conditions = grouped.index.tolist()
            for instrument, cols in col_groups.items():
                if not cols:
                    continue
                print(f"\n  -- {instrument} --")
                header = f"     {'Measure':<40}" + "".join(f"  {c:<18}" for c in conditions)
                print(header)
                print("     " + "-" * (40 + 20 * len(conditions)))
                for col in cols:
                    label = col.replace("tlx_","").replace("vdl_","").replace("tia_","").replace("_"," ").title()
                    row_str = f"     {label:<40}"
                    for cond in conditions:
                        val = grouped.loc[cond, col]
                        row_str += f"  {val:<18.2f}"
                    print(row_str)
            print_tlx_weight_summary(df_cond)
    else:
        print("No TLX/VdL/TiA condition-level data found.", file=sys.stderr)

    # Build demographics summary
    if demographics_rows:
        df_demo = pd.DataFrame(list(demographics_rows.values()))
        df_demo = df_demo.sort_values("participant_id").reset_index(drop=True)
        df_demo.to_csv(args.out_demographics, index=False)
        print(f"\n{'='*55}")
        print(f"  Demographics ({len(df_demo)} participant(s))")
        print(f"{'='*55}")
        for _, row in df_demo.iterrows():
            print(f"\n  Participant : {row['participant_id']}  ({row['session_date']})")
            demo_fields = [c for c in row.index if c.startswith("demo_")]
            screen_fields = [c for c in row.index if c.startswith("screen_")]
            print("  Screening   :", ", ".join(
                c.replace("screen_","") for c in screen_fields if row[c] is True
            ) or "none passed")
            for col in demo_fields:
                label = col.replace("demo_", "").replace("_", " ").title()
                print(f"     {label:<30} {row[col]}")
    else:
        print("\nNo demographics data found.", file=sys.stderr)

    # Demographics overall summary
    if demographics_rows:
        print(f"\n{'='*55}")
        print(f"  DEMOGRAPHICS SUMMARY  (N={len(demographics_rows)})")
        print(f"{'='*55}")
        demo_fields = [k for k in list(demographics_rows.values())[0].keys()
                       if k.startswith("demo_")]
        for col in demo_fields:
            label = col.replace("demo_","").replace("_"," ").title()
            vals = [r[col] for r in demographics_rows.values() if r.get(col)]
            if col == "demo_age":
                try:
                    ages = [float(v) for v in vals if v]
                    if ages:
                        print(f"  {label:<25} M={sum(ages)/len(ages):.1f}  "
                              f"range={int(min(ages))}–{int(max(ages))}")
                except ValueError:
                    pass
            else:
                from collections import Counter
                counts_demo = Counter(vals)
                counts_str = "  |  ".join(f"{k}: {v}" for k, v in counts_demo.most_common())
                print(f"  {label:<25} {counts_str}")


if __name__ == "__main__":
    main()