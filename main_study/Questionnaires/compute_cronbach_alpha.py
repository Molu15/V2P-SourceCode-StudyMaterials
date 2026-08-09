#!/usr/bin/env python3
"""
compute_cronbach_alpha.py — V2P Main Study | Reliability Analysis
====================================================================
Reads all raw item-level questionnaire JSON exports (TiA + VdL — TLX is
intentionally excluded, see chat: its 6 dimensions are independent
constructs, not interchangeable indicators of one latent trait, so
Cronbach's alpha is not an appropriate/conventional statistic for it)
and computes Cronbach's alpha per subscale, pooled across both conditions.

Expected filename pattern: {PID}_{Instrument}_{Condition}_{date}.json
  Instrument in {"TiA", "VdL"}   (TLX files are skipped)

USAGE:
    python compute_cronbach_alpha.py --data_dir ./questionnaire_items
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


def cronbach_alpha(item_matrix: np.ndarray) -> float:
    """
    Standard Cronbach's alpha.
    item_matrix: rows = respondents (person x condition), columns = items.
    """
    item_matrix = np.asarray(item_matrix, dtype=float)
    k = item_matrix.shape[1]
    if k < 2:
        return np.nan
    item_vars  = item_matrix.var(axis=0, ddof=1)
    total_var  = item_matrix.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)


def spearman_brown_2item(item_matrix: np.ndarray) -> float:
    """Special case for exactly 2 items — more stable than raw alpha at k=2."""
    if item_matrix.shape[1] != 2:
        return np.nan
    r = np.corrcoef(item_matrix[:, 0], item_matrix[:, 1])[0, 1]
    return (2 * r) / (1 + r)


def load_tia(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = {}  # subscale -> {item_id: normalized_value}
    for item_id, info in d["items"].items():
        rows.setdefault(info["subscale"], {})[item_id] = info["normalized"]
    return {"pid": d["participant_id"], "condition": d["condition"], "subscales": rows}


def load_vdl(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = {}  # dim -> {item_id: score}
    for item_id, info in d["items_scored"].items():
        rows.setdefault(info["dim"], {})[item_id] = info["score"]
    return {"pid": d["participant_id"], "condition": d["condition"], "subscales": rows}


def build_item_matrix(records: list, subscale: str) -> tuple[np.ndarray, list, int]:
    """
    records: list of {"pid":..., "condition":..., "subscales": {subscale: {item_id: val}}}
    Returns (matrix, item_ids, n_dropped) — n_dropped = respondent-occasions
    with missing items for this subscale (excluded listwise).
    """
    item_ids = sorted({iid for r in records for iid in r["subscales"].get(subscale, {})})
    rows = []
    n_dropped = 0
    for r in records:
        vals = r["subscales"].get(subscale, {})
        if all(iid in vals and vals[iid] is not None for iid in item_ids):
            rows.append([vals[iid] for iid in item_ids])
        else:
            n_dropped += 1
    return np.array(rows, dtype=float), item_ids, n_dropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".")
    parser.add_argument("--out_dir", default="./results_reliability",
                         help="Where to save the log file and results CSV")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    log = []  # collect every printed line so it can be saved, not just shown
    def p(msg=""):
        print(msg)
        log.append(msg)

    csv_rows = []  # for a machine-readable results file alongside the log

    tia_records, vdl_records = [], []
    n_tlx_skipped = 0
    for f in sorted(data_dir.glob("*.json")):
        name = f.name
        if "_TiA_" in name:
            tia_records.append(load_tia(f))
        elif "_VdL_" in name:
            vdl_records.append(load_vdl(f))
        elif "_TLX_" in name:
            n_tlx_skipped += 1
        else:
            p(f"[WARN] Unrecognized file, skipped: {name}")

    p(f"[LOAD] {len(tia_records)} TiA files, {len(vdl_records)} VdL files "
      f"({n_tlx_skipped} TLX files skipped by design — see docstring)\n")

    p("=" * 65)
    p("CRONBACH'S ALPHA — pooled across Adaptive + Baseline")
    p("=" * 65)

    p("\n── TiA subscales ──────────────────────────")
    for subscale in ["REL", "UND", "FAM", "DEV", "PROP", "TRU"]:
        mat, items, dropped = build_item_matrix(tia_records, subscale)
        if mat.shape[0] < 3:
            p(f"  {subscale:6s}  n too small ({mat.shape[0]}) — skipped")
            csv_rows.append({"instrument": "TiA", "subscale": subscale, "k_items": len(items),
                              "n": mat.shape[0], "alpha": np.nan, "spearman_brown": np.nan,
                              "n_dropped_incomplete": dropped})
            continue
        a = cronbach_alpha(mat)
        sb = np.nan
        note = ""
        if mat.shape[1] == 2:
            sb = spearman_brown_2item(mat)
            note = f"  (k=2 -> Spearman-Brown = {sb:.3f}, more stable than raw alpha at k=2)"
        p(f"  {subscale:6s}  k={mat.shape[1]}  n={mat.shape[0]}  "
          f"alpha={a:.3f}{note}  [{dropped} occasions dropped: incomplete]")
        csv_rows.append({"instrument": "TiA", "subscale": subscale, "k_items": mat.shape[1],
                          "n": mat.shape[0], "alpha": round(a, 3),
                          "spearman_brown": round(sb, 3) if not np.isnan(sb) else np.nan,
                          "n_dropped_incomplete": dropped})

    p("\n── Van der Laan subscales ─────────────────")
    for dim in ["usefulness", "satisfying"]:
        mat, items, dropped = build_item_matrix(vdl_records, dim)
        if mat.shape[0] < 3:
            p(f"  {dim:12s}  n too small ({mat.shape[0]}) — skipped")
            csv_rows.append({"instrument": "VdL", "subscale": dim, "k_items": len(items),
                              "n": mat.shape[0], "alpha": np.nan, "spearman_brown": np.nan,
                              "n_dropped_incomplete": dropped})
            continue
        a = cronbach_alpha(mat)
        p(f"  {dim:12s}  k={mat.shape[1]}  n={mat.shape[0]}  alpha={a:.3f}  "
          f"[{dropped} occasions dropped: incomplete]")
        csv_rows.append({"instrument": "VdL", "subscale": dim, "k_items": mat.shape[1],
                          "n": mat.shape[0], "alpha": round(a, 3), "spearman_brown": np.nan,
                          "n_dropped_incomplete": dropped})

    p("\n[NOTE] n here = respondent-occasions (person x condition), "
      "e.g. 30 participants x 2 conditions = up to 60, minus any incomplete cases.")

    # ── Save log + results CSV ──────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"cronbach_alpha_log_{timestamp}.txt"
    csv_path = out_dir / f"cronbach_alpha_results_{timestamp}.csv"

    log_path.write_text(
        f"Run at: {datetime.now().isoformat()}\n"
        f"data_dir: {data_dir.resolve()}\n\n" + "\n".join(log),
        encoding="utf-8"
    )
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    p(f"\n[SAVE] Log     -> {log_path}")
    p(f"[SAVE] Results -> {csv_path}")


if __name__ == "__main__":
    main()
