#!/usr/bin/env python3
"""
analyze_videos.py — V2P Study | Batch RT Analysis Wrapper (all-in-one)
=======================================================================
Runs analyze_rt_video.py for every P??_A/B.mp4, automatically picking the
best available sync method per participant/block, in this priority order:

  1. PRECISE (--video_start_abs): if the reference-times file has a row for
     this block AND the reference trial's raw log has T_Start_abs — computes
     video_start_abs from that single reference, then every trial gets its
     own independently precise time (no drift, no audio needed).
  2. GUIDED (--sync_offset + --alarm_audio): if a reference row exists but
     T_Start_abs is not available (older participants) — computes a
     sync_offset anchor, then audio-matches each trial near its expected position.
  3. AUTO-SYNC fallback: if no reference row exists for this block at all —
     falls back to plain audio auto-sync, with a loud warning that this is
     unverified and may be unreliable.

ONE reference-times file covers all participants (first_alarm_approx_time.xlsx):
    columns: PID, condition, (first) trial, approx time, ref_type (f/vha,
    default vha), notes (redid / irrelevant / blank)

Aufruf:
    python analyze_videos.py                    # all participants
    python analyze_videos.py --pid P01          # only P01
    python analyze_videos.py --dry_run          # shows only commands
"""

import argparse
import re
import glob
import os
import subprocess
import sys
from collections import Counter
from datetime import time as dt_time
from pathlib import Path

import pandas as pd

# ─── KONFIGURATION ────────────────────────────────────────────────────────
ALARM_THRESHOLD  = 0.07
REACTION_RATIO   = 0.35
DEFAULT_TRIAL_GAP_S = 25.0
OUTPUT_CSV       = "rt_results.csv"
REFERENCE_XLSX   = "first_alarm_approx_time.xlsx"
ATTEMPTS_CSV     = "attempts.csv"

HERE       = Path(__file__).parent
VIDEOS_DIR = HERE / "videos"
LOGS_DIR   = HERE / "logs"
ALARM_WAV  = VIDEOS_DIR / "alert_tone.wav"
ANALYZE_PY = VIDEOS_DIR / "analyze_rt_video.py"

BLOCK_TO_CONDITION = {"A": "Adaptive", "B": "Baseline"}
COND_TO_MODE = {"adaptive": "a", "baseline": "b", "a": "a", "b": "b"}


# ─────────────────────────────────────────────────────────
# Helpers (shared logic, inlined so this stays a single file)
# ─────────────────────────────────────────────────────────
def parse_header(line: str) -> dict:
    result = {}
    for token in line.lstrip("# ").split():
        if ":" in token:
            k, v = token.split(":", 1)
            result[k.lower()] = v
    return result


def to_float(v):
    try:
        return float(v) if str(v).lower() not in ("none", "n/a", "") else None
    except (TypeError, ValueError):
        return None


def excel_time_to_seconds(t):
    if not isinstance(t, dt_time):
        return None
    return t.hour * 60 + t.minute + t.second / 60


def find_trial_log(logs_dir: Path, mode: str, run_id: str):
    exact = logs_dir / f"{mode}_{run_id}.csv"
    if exact.exists():
        return exact
    if logs_dir.is_dir():
        for fpath in glob.glob(str(logs_dir / "*.csv")):
            try:
                with open(fpath, encoding="utf-8") as f:
                    first = f.readline()
            except OSError:
                continue
            if not first.startswith("#"):
                continue
            h = parse_header(first)
            if h.get("mode") == mode and h.get("run") == run_id:
                return Path(fpath)
    return None


def load_run_order(logs_dir: Path, mode: str, raw: bool = False):
    path = logs_dir / "run_order.csv"
    if not path.exists():
        return None
    prefix = mode.upper() + ("_raw " if raw else " ")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(prefix):
                return [r.strip() for r in line[len(prefix):].split(",")]
    return None


def load_trials_ordered(logs_dir: Path, mode: str):
    trials = []
    for fpath in glob.glob(str(logs_dir / "*.csv")):
        try:
            with open(fpath, encoding="utf-8") as f:
                first_line = f.readline()
        except OSError:
            continue
        if not first_line.startswith("#"):
            continue
        h = parse_header(first_line)
        if h.get("mode", "") != mode:
            continue
        timestamps = []
        try:
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#") or not line.strip():
                        continue
                    try:
                        timestamps.append(float(line.split(",")[0]))
                    except (ValueError, IndexError):
                        continue
        except OSError:
            pass
        t_a = to_float(h.get("t_a"))
        max_ts = max(timestamps) if timestamps else (t_a + 15.0 if t_a else 15.0)
        trials.append({
            "run_id": h.get("run", ""), "t_a": t_a,
            "full_warn_t": to_float(h.get("fullwarnt")),
            "early_warn_t": to_float(h.get("earlywarnt")),
            "t_start_abs": to_float(h.get("t_start_abs")),
            "mtime": os.path.getmtime(fpath),
            "trial_duration": (max_ts - t_a) if t_a is not None else max_ts,
        })
    run_order = load_run_order(logs_dir, mode)
    if run_order:
        idx = {rid: i for i, rid in enumerate(run_order)}
        trials.sort(key=lambda t: idx.get(t["run_id"], len(idx)))
    else:
        trials.sort(key=lambda t: t["mtime"])
    return trials


def compute_sync_offset(logs_dir: Path, mode: str, ref_trial: str, observed_s: float,
                         is_redid: bool, attempts_df: pd.DataFrame, pid_num: int):
    trials = load_trials_ordered(logs_dir, mode)
    ids = [t["run_id"] for t in trials]
    if ref_trial not in ids:
        return None, f"'{ref_trial}' not found in logs"

    run_order_raw = load_run_order(logs_dir, mode, raw=True)
    is_dup = run_order_raw and run_order_raw.count(ref_trial) > 1

    if is_dup or is_redid:
        if not run_order_raw or ref_trial not in run_order_raw:
            return None, f"'{ref_trial}' redo-marked but not found in raw sequence"
        dur_lookup = {t["run_id"]: t["trial_duration"] for t in trials}
        raw_pos = run_order_raw.index(ref_trial)
        cumulative_before = sum(dur_lookup.get(rid, 0.0) + DEFAULT_TRIAL_GAP_S
                                 for rid in run_order_raw[:raw_pos])
        if attempts_df is not None:
            match = attempts_df[(attempts_df["pid"] == pid_num) &
                                 (attempts_df["mode"] == mode) &
                                 (attempts_df["run_id"] == ref_trial) &
                                 (attempts_df["attempt_number"] == 1)]
            if len(match) == 1 and pd.notna(match.iloc[0]["vha_elapsed"]) \
                    and pd.notna(match.iloc[0]["t_a"]) and pd.notna(match.iloc[0]["t_b"]):
                row = match.iloc[0]
                offset = (row["t_b"] - row["t_a"]) + row["vha_elapsed"]
                return (observed_s - offset) - cumulative_before, None
        kept = trials[ids.index(ref_trial)]
        if kept["full_warn_t"] is None or kept["t_a"] is None:
            return None, f"'{ref_trial}' (kept) missing t_a/full_warn_t"
        return (observed_s - (kept["full_warn_t"] - kept["t_a"])) - cumulative_before, None

    pos = ids.index(ref_trial)
    ref = trials[pos]
    if ref["full_warn_t"] is None or ref["t_a"] is None:
        return None, f"'{ref_trial}' missing t_a/full_warn_t"
    cumulative_before = sum(t["trial_duration"] + DEFAULT_TRIAL_GAP_S for t in trials[:pos])
    return (observed_s - (ref["full_warn_t"] - ref["t_a"])) - cumulative_before, None


def compute_video_start_abs(logs_dir: Path, mode: str, ref_trial: str,
                             observed_s: float, ref_type: str):
    log_path = find_trial_log(logs_dir, mode, ref_trial)
    if log_path is None:
        return None, f"log for '{ref_trial}' not found"
    with open(log_path, encoding="utf-8") as f:
        header = parse_header(f.readline())
    t_start_abs = to_float(header.get("t_start_abs"))
    target_warn_t = to_float(header.get("earlywarnt")) if ref_type == "f" \
        else to_float(header.get("fullwarnt"))
    if t_start_abs is None or target_warn_t is None:
        return None, "missing T_Start_abs or WarnT in reference trial's log"
    return t_start_abs + target_warn_t - observed_s, None


# ─────────────────────────────────────────────────────────
# Reference table loading
# ─────────────────────────────────────────────────────────
def get_video_start_abs_from_metadata(video_path: Path):
    """Reads video_start_abs directly from the file's own creation_time
    metadata (creation_time = recording END on this Android/Samsung setup,
    confirmed against known T_Start_abs — see chat). No manual reference
    needed when this works.
    Returns (video_start_abs, None) on success, (None, reason) on failure.
    """
    import subprocess as sp
    import json as _json
    from datetime import datetime, timezone

    try:
        result = sp.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(video_path)],
            capture_output=True, text=True, timeout=15,
        )
        data = _json.loads(result.stdout)
        tags = data.get("format", {}).get("tags", {})
        creation_time_str = tags.get("creation_time")
        duration_s = float(data.get("format", {}).get("duration", 0))
        if not creation_time_str or duration_s <= 0:
            return None, "no creation_time or duration in video metadata"
        creation_dt = datetime.strptime(creation_time_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc)
        creation_unix = creation_dt.timestamp()
        return creation_unix - duration_s, None
    except FileNotFoundError:
        return None, "ffprobe not found on PATH"
    except Exception as e:
        return None, f"ffprobe/metadata error: {e}"


def load_reference_table(path: str):
    """Returns {(pid, mode): [list of anchor dicts]} — supports multiple
    anchor rows per block (one per known trial), parsing an optional
    '(f)'/'(vha)' suffix on the trial column, e.g. 't3 (f)'."""
    if not os.path.exists(path):
        print(f"[INFO] {path} not found — all participants will use plain auto-sync fallback")
        return {}
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    table: dict = {}
    for _, r in df.iterrows():
        pid = str(r["PID"]).strip()
        mode = COND_TO_MODE[str(r["condition"]).strip().lower()]
        raw_trial = str(r["(first) trial"]).strip()
        m = re.match(r"^(t\d+)\s*(?:\(([a-zA-Z]+)\))?$", raw_trial)
        if not m:
            continue
        ref_trial = m.group(1)
        ref_type = (m.group(2) or "vha").lower()
        observed_s = excel_time_to_seconds(r["approx time"])
        notes = str(r.get("notes", "")).strip().lower()
        entry = {"ref_trial": ref_trial, "observed_s": observed_s,
                 "ref_type": ref_type, "notes": notes}
        table.setdefault((pid, mode), []).append(entry)
    return table


# ─────────────────────────────────────────────────────────
# Per-block execution
# ─────────────────────────────────────────────────────────
def compute_direct_video_ta(logs_dir: Path, mode: str, trial_id: str,
                             observed_s: float, ref_type: str):
    """Direct per-trial video_ta from ONE manually observed alarm time for
    THAT SPECIFIC trial — no estimation, no audio matching. Used when
    multiple anchor points per block are available (see chat)."""
    log_path = find_trial_log(logs_dir, mode, trial_id)
    if log_path is None:
        return None, f"log for '{trial_id}' not found"
    with open(log_path, encoding="utf-8") as f:
        h = parse_header(f.readline())
    t_a = to_float(h.get("t_a"))
    target_warn_t = to_float(h.get("earlywarnt")) if ref_type == "f" \
        else to_float(h.get("fullwarnt"))
    if t_a is None or target_warn_t is None:
        return None, f"missing t_a/{'EarlyWarnT' if ref_type=='f' else 'FullWarnT'} in {log_path}"
    return observed_s - (target_warn_t - t_a), None


def run_block(pid: str, block: str, reference_table: dict, attempts_df,
              debug_trial, dry_run: bool) -> bool:
    condition = BLOCK_TO_CONDITION[block]
    mode = block.lower()
    video = VIDEOS_DIR / f"{pid}_{block}.mp4"

    pid_num = int(pid.lstrip("Pp"))
    pid_short = "P" + str(pid_num)
    logs = LOGS_DIR / pid_short
    if not logs.exists():
        logs = LOGS_DIR / pid

    if not video.exists():
        print(f"[SKIP]  {pid}_{block}.mp4 — not found")
        return False
    if not logs.exists():
        print(f"[SKIP]  {pid} logs/ — not found")
        return False

    cmd = [
        sys.executable, str(ANALYZE_PY),
        "--video", str(video), "--logs_dir", str(logs),
        "--pid", pid_short, "--condition", condition,
        "--reaction_ratio", str(REACTION_RATIO),
        "--out", OUTPUT_CSV, "--append",
    ]

    ref_list = reference_table.get((pid_short, mode)) or reference_table.get((pid, mode)) or []
    sync_desc = "AUTO-SYNC (⚠ unverified fallback, no anchor)"


    vsa, meta_err = get_video_start_abs_from_metadata(video)
    if vsa is not None:
        first_trial_log = find_trial_log(logs, mode,
                                          "t1") or find_trial_log(logs, mode, "t2")
        plausible = True
        if first_trial_log is not None:
            with open(first_trial_log, encoding="utf-8") as f:
                h = parse_header(f.readline())
            t_start_abs_check = to_float(h.get("t_start_abs"))
            if t_start_abs_check is None:
                plausible = False
            elif not (0 <= (t_start_abs_check - vsa) <= 1800):
                plausible = False
        else:
            plausible = False
        if plausible:
            cmd += ["--video_start_abs", str(vsa)]
            sync_desc = f"PRECISE (video_start_abs={vsa:.2f}, from the video metadata)"
        else:
            vsa = None

    if vsa is None:
        # Priorität 2: DIREKTE Mehrpunkt-Anker aus der Referenz-Tabelle —
        # ein video_ta PRO dokumentiertem Trial, keine Schätzung nötig.
        per_trial_ta = {}
        for entry in ref_list:
            if entry["notes"] == "irrelevant" or entry["observed_s"] is None:
                continue
            vt, err = compute_direct_video_ta(logs, mode, entry["ref_trial"],
                                               entry["observed_s"], entry["ref_type"])
            if vt is not None:
                per_trial_ta[entry["ref_trial"]] = vt

        if per_trial_ta:
            import json as _json
            cmd += ["--per_trial_ta_json", _json.dumps(per_trial_ta)]
            sync_desc = f"DIRECT MULTI-ANCHOR ({len(per_trial_ta)} Trials directly anchored)"
        else:
            print(f"[WARN] {pid_short}/{condition}: no usable anchors — "
                  f"falling back to auto_sync")
            cmd += ["--auto_sync", "--alarm_audio", str(ALARM_WAV),
                    "--alarm_threshold", str(ALARM_THRESHOLD)]
    if debug_trial:
        cmd += ["--debug_trial", debug_trial]

    print(f"\n{'─'*60}")
    print(f"[RUN]   {pid_short} | block {block} ({condition})  —  {sync_desc}")
    print(f"{'─'*60}")

    if dry_run:
        print("        " + " ".join(cmd))
        return True

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run(cmd, env=env)
    return result.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", nargs="*", default=None)
    ap.add_argument("--debug_trial", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if not ALARM_WAV.exists():
        sys.exit(f"[ERROR] alert_tone.wav not found: {ALARM_WAV}")
    if not ANALYZE_PY.exists():
        sys.exit(f"[ERROR] analyze_rt_video.py not found: {ANALYZE_PY}")

    reference_table = load_reference_table(REFERENCE_XLSX)
    attempts_df = pd.read_csv(ATTEMPTS_CSV) if os.path.exists(ATTEMPTS_CSV) else None
    if attempts_df is not None:
        print(f"[INFO] {ATTEMPTS_CSV} loaded ({len(attempts_df)} attempts) — "
              f"exact redo times available")

    all_videos = sorted(VIDEOS_DIR.glob("P??_[AB].mp4"))
    if not all_videos:
        sys.exit(f"[ERROR] No P??_A/B.mp4 videos in {VIDEOS_DIR}")

    if args.pid:
        pids = set(args.pid)
        all_videos = [v for v in all_videos if v.stem.rsplit("_", 1)[0] in pids]
        if not all_videos:
            sys.exit(f"[ERROR] No videos for: {args.pid}")

    out = Path(OUTPUT_CSV)
    if out.exists() and not args.pid:
        out.unlink()
        print(f"[INFO]  Old {OUTPUT_CSV} deleted\n")

    ok = skip = 0
    for video in all_videos:
        pid, block = video.stem.rsplit("_", 1)
        if run_block(pid, block, reference_table, attempts_df,
                      args.debug_trial, args.dry_run):
            ok += 1
        else:
            skip += 1

    print(f"\n{'═'*60}")
    print(f"Finished: {ok} successful, {skip} skipped/failed")
    if ok:
        print(f"Results: {OUTPUT_CSV}")

    if ok and not args.dry_run:
        try:
            rt = pd.read_csv(OUTPUT_CSV)
            if "heading_norm" in rt.columns:
                flag = rt["heading_norm"].notna() & (rt["heading_norm"] < 0.05) \
                        & (rt["reaction_type"] == "run_back")
                if flag.any():
                    print(f"\n{'─'*60}")
                    print(f"[QC] {flag.sum()} Trial(s) with uncertain baseline direction "
                          f"(heading_norm < 0.05) — manual verification needed:")
                    print(rt[flag][["participant_id", "condition", "trial_id",
                                     "reaction_type", "rt_s"]].to_string(index=False))
        except Exception as e:
            print(f"\n[QC] Could not read {OUTPUT_CSV}: {e}")


if __name__ == "__main__":
    main()