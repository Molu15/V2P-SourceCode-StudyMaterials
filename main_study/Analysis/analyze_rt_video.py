#!/usr/bin/env python3
"""
analyze_rt_video.py — V2P Study | MediaPipe RT Analysis
========================================================

Computes reaction time (RT) from a continuous block video recording.

RT Definition:
    Time from alarm onset (vha preferred, f as fallback) to the first
    sustained drop in the participant's walking velocity — i.e. when they
    slow to ≤35 % of their pre-alarm baseline speed (held for ≥4 frames).

    No floor-marker calibration required. Uses MediaPipe hip landmarks as
    a proxy for body-centre motion.

─── SYNC MODES ───────────────────────────────────────────────────────────────

1. AUTO-SYNC  (P1–P11 — recommended)
       --auto_sync  --alarm_audio alarm.wav
   Detects alarm events in the video audio via template matching (librosa),
   clusters the 3×-repetition hits into one event, then matches the sequence
   to CARLA logs. Derives a per-trial video timestamp for the alarm.

2. ABSOLUTE TIMESTAMP  (P12+, requires T_Start_abs in CARLA logs)
       --video_start_abs 1720010000.0
   Uses the Unix timestamp logged by run_sim.py when the ambient sound
   started, together with the video recording start time.

3. MANUAL  (fallback / debugging)
       --sync_offset 5.2  [--trial_gap_s 25]
   Single offset for the whole block; all inter-trial gaps estimated.

─── USAGE ────────────────────────────────────────────────────────────────────

  # Recommended — auto-sync with template (P1–P11):
  python analyze_rt_video.py \\
      --video P01_a_block.mp4 --logs_dir ../logs/P01 \\
      --pid P01 --condition Adaptive \\
      --auto_sync --alarm_audio alarm.wav \\
      --out rt_results.csv

  # Absolute timestamp (P12+):
  python analyze_rt_video.py \\
      --video P12_a_block.mp4 --logs_dir ../logs/P12 \\
      --pid P12 --condition Adaptive \\
      --video_start_abs 1720010000.0 --out rt_results.csv

  # Second block — append to same CSV:
  python analyze_rt_video.py ... --condition Fixed --out rt_results.csv --append

  # Tune detection sensitivity:
  python analyze_rt_video.py ... --auto_sync --alarm_audio alarm.wav \\
      --alarm_threshold 0.04   # lower = more sensitive
      --reaction_ratio 0.4     # lower = less sensitive to slow-downs

  # Debug: show hip-velocity plot for one trial (requires matplotlib):
  python analyze_rt_video.py ... --debug_trial t3

Dependencies:
    pip install mediapipe opencv-python numpy librosa scipy --break-system-packages
"""

import argparse
import csv
import os
import sys
import glob
import subprocess
import tempfile
from pathlib import Path

import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ─── CONSTANTS ────────────────────────────────────────────
DEFAULT_TRIAL_GAP_S   = 25.0    # inter-trial gap estimate (manual mode)
CLUSTER_WINDOW_S      = 2.0     # alarm repetitions within this window → one event
BASELINE_WINDOW_S     = 2.0     # seconds before alarm used as walking baseline
DEFAULT_REACTION_RATIO = 0.35   # velocity must drop to this fraction of baseline
MIN_REACTION_FRAMES   = 4       # sustained frames required to confirm reaction
SMOOTH_FRAMES         = 7       # rolling-average window for hip velocity
MIN_PLAUSIBLE_RT      = 0.15  # 150ms — unterhalb physiologisch nicht plausibel, wird als Rauschen verworfen

# Alarm timing from UdpOverlayService.kt (SoundPool, alert_tone.wav):
#   PULSE_MS = 400  → each play lasts 400 ms
#   PAUSE_MS = 200  → silence between plays
#   3 plays at t = 0 ms, 600 ms, 1200 ms  →  total span ≈ 1600 ms

TARGET_RUNS   = {"t1", "t2", "t3", "t4", "t5"}
HIT_OUTCOMES  = {"collision", "response_stop", "response_run",
                 "response_run_back", "not_in_time"}
CONDITION_TO_MODE = {"adaptive": "a", "fixed": "b", "baseline": "b"}

# ─── MEDIAPIPE MODEL ──────────────────────────────────────
_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
               "pose_landmarker/pose_landmarker_lite/float16/latest/"
               "pose_landmarker_lite.task")
_MODEL_PATH = Path(__file__).parent / "pose_landmarker_lite.task"

def _ensure_model() -> str:
    if not _MODEL_PATH.exists():
        print(f"[MODEL] Downloading pose_landmarker_lite.task (~7 MB) …")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print(f"[MODEL] Saved → {_MODEL_PATH}")
    return str(_MODEL_PATH)

def _make_detector() -> mp_vision.PoseLandmarker:
    # IMAGE mode: each frame is processed independently — no monotonic
    # timestamp requirement, safe for non-linear video seeks across trials.
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=_ensure_model()),
        running_mode=mp_vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        num_poses=1,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


# ─────────────────────────────────────────────────────────
# LOG LOADING
# ─────────────────────────────────────────────────────────
def parse_header(line: str) -> dict:
    result = {}
    for token in line.lstrip("# ").split():
        if ":" in token:
            k, v = token.split(":", 1)
            result[k.lower()] = v
    return result


def to_float(v) -> float | None:
    try:
        return float(v) if str(v).lower() not in ("none", "n/a", "") else None
    except (TypeError, ValueError):
        return None

def load_run_order_raw(logs_dir: str, mode: str) -> list[str] | None:
    """Returns the RAW (unfiltered) sequence — includes discarded redo
    attempts at their true chronological position, unlike load_run_order()."""
    path = os.path.join(logs_dir, "run_order.csv")
    if not os.path.exists(path):
        return None
    prefix = mode.upper() + "_raw "
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(prefix):
                return [r.strip() for r in line[len(prefix):].split(",")]
    return None

def load_run_order(logs_dir: str, mode: str) -> list[str] | None:
    """
    Reads run_order.csv from logs_dir. Returns the CLEANED (non-_raw)
    sequence for the given mode ('a' -> 'A', 'b' -> 'B') as an ordered
    list of run_ids — redo attempts already resolved (only the last
    attempt kept, at its true chronological position).
    Returns None if the file is missing (caller falls back to mtime sort).
    """
    path = os.path.join(logs_dir, "run_order.csv")
    if not os.path.exists(path):
        print(f"[WARN] run_order.csv not found in {logs_dir} — falling back to mtime sort")
        return None

    target_prefix = mode.upper() + " "   # "A " / "B " — NOT "A_raw"/"B_raw"
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(target_prefix):
                seq_str = line[len(target_prefix):]
                return [r.strip() for r in seq_str.split(",")]
    print(f"[WARN] No '{mode.upper()}' line in run_order.csv — falling back to mtime sort")
    return None

def load_logs(logs_dir: str, pid: str, condition: str) -> list[dict]:
    mode_filter = CONDITION_TO_MODE.get(condition.lower())
    if mode_filter is None:
        sys.exit(f"[ERROR] --condition must be 'Adaptive', 'Baseline', or 'Fixed', got '{condition}'")

    files = glob.glob(os.path.join(logs_dir, "*.csv"))
    if not files:
        sys.exit(f"[ERROR] No CSV files in: {logs_dir}")

    trials = []
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                first_line = f.readline()
        except OSError:
            continue
        if not first_line.startswith("#"):
            continue

        h = parse_header(first_line)
        if h.get("mode", "") != mode_filter:
            continue

        timestamps = []
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.split(",")
                    try:
                        timestamps.append(float(parts[0]))
                    except (ValueError, IndexError):
                        continue
        except OSError:
            pass

        t_a        = to_float(h.get("t_a"))
        full_warn  = to_float(h.get("fullwarnt"))
        early_warn = to_float(h.get("earlywarnt"))
        t_start_abs = to_float(h.get("t_start_abs"))
        max_ts     = max(timestamps) if timestamps else (t_a + 15.0 if t_a else 15.0)
        trial_dur  = (max_ts - t_a) if t_a is not None else max_ts

        trials.append({
            "file":           fpath,
            "mtime":          os.path.getmtime(fpath),
            "pid":            h.get("pid", ""),
            "mode":           mode_filter,
            "run_id":         h.get("run", ""),
            "outcome":        h.get("outcome", ""),
            "t_a":            t_a,
            "full_warn_t":    full_warn,
            "early_warn_t":   early_warn,
            "t_start_abs":    t_start_abs,
            "trial_duration": trial_dur,
        })

    if not trials:
        sys.exit(f"[ERROR] No logs for condition='{condition}' in {logs_dir}")

    run_order = load_run_order(logs_dir, mode_filter)
    if run_order:
        order_index = {rid: i for i, rid in enumerate(run_order)}
        missing = [t["run_id"] for t in trials if t["run_id"] not in order_index]
        if missing:
            print(f"[WARN] Trials not in run_order.csv, appended at end: {missing}")
        trials.sort(key=lambda t: order_index.get(t["run_id"], len(order_index)))
        print(f"[LOAD] Trial order from run_order.csv: {[t['run_id'] for t in trials]}")
    else:
        trials.sort(key=lambda t: t["mtime"])

    run_order_raw = load_run_order_raw(logs_dir, mode_filter)
    if run_order and run_order_raw:
        from collections import Counter
        dur_lookup = {t["run_id"]: t["trial_duration"] for t in trials}
        raw_counts = Counter(run_order_raw)
        occurrence_idx = Counter()
        cum_padding = 0.0
        padding_before = {}
        for rid in run_order_raw:
            occurrence_idx[rid] += 1
            if occurrence_idx[rid] == raw_counts[rid]:   # letzter (behaltener) Versuch
                padding_before[rid] = cum_padding
            else:   # verworfener Versuch — Dauer unbekannt, beste Schätzung: eigene geloggte Dauer
                cum_padding += dur_lookup.get(rid, 0.0) + DEFAULT_TRIAL_GAP_S
        for t in trials:
            t["extra_padding_before"] = padding_before.get(t["run_id"], 0.0)
        redone = [rid for rid, c in raw_counts.items() if c > 1]
        if redone:
            print(f"[LOAD] Redo detected in raw sequence: {redone} — cumulative "
                  f"estimate padded accordingly")
    else:
        for t in trials:
            t["extra_padding_before"] = 0.0

    print(f"[LOAD] {len(trials)} logs  (condition={condition}, mode={mode_filter})")
    for t in trials:
        fw = f"{t['full_warn_t']:.2f}s"  if t["full_warn_t"]  is not None else "None"
        ew = f"{t['early_warn_t']:.2f}s" if t["early_warn_t"] is not None else "None"
        print(f"  {t['run_id']:3s}  outcome={t['outcome']:14s}  "
              f"EarlyWarnT={ew}  FullWarnT={fw}"
              + ("  [abs]" if t["t_start_abs"] else ""))

    return trials


# ─────────────────────────────────────────────────────────
# TIMESTAMP MAPPING
# ─────────────────────────────────────────────────────────
def compute_video_timestamps(trials:       list[dict],
                              sync_offset:  float,
                              trial_gap_s:  float,
                              per_trial_ta: dict | None = None) -> list[dict]:
    """
    Fill video_ta and video_alarm for each trial.
    per_trial_ta (dict {idx → video_ta}) overrides cumulative estimate
    when available (auto-sync or absolute mode).
    """
    cumulative = 0.0
    for i, t in enumerate(trials):
        t_a = t["t_a"] or 0.0
        t["video_ta"] = per_trial_ta[i] if (per_trial_ta and i in per_trial_ta) \
                        else sync_offset + cumulative + t.get("extra_padding_before", 0.0)

        if t["full_warn_t"] is not None:
            t["video_alarm_vha"] = t["video_ta"] + (t["full_warn_t"] - t_a)
        else:
            t["video_alarm_vha"] = None

        if t["early_warn_t"] is not None:
            t["video_alarm_f"] = t["video_ta"] + (t["early_warn_t"] - t_a)
        else:
            t["video_alarm_f"] = None

        # Default/Fallback-Referenz (für Fälle ohne Annotation-Info):
        if t["video_alarm_vha"] is not None:
            t["video_alarm"] = t["video_alarm_vha"]
            t["warning_type"] = "vha"
        elif t["video_alarm_f"] is not None:
            t["video_alarm"] = t["video_alarm_f"]
            t["warning_type"] = "f"
        else:
            t["video_alarm"] = None
            t["warning_type"] = "none"

        cumulative += t["trial_duration"] + trial_gap_s
    return trials


def compute_timestamps_from_abs(trials: list[dict],
                                 video_start_abs: float) -> dict:
    """P12+: use T_Start_abs from log + video recording start time."""
    per_trial_ta = {}
    for i, t in enumerate(trials):
        if t["t_start_abs"] is not None and t["t_a"] is not None:
            video_ta = (t["t_start_abs"] - video_start_abs) + t["t_a"]
            per_trial_ta[i] = video_ta
            print(f"[ABS]  {t['run_id']:3s}  video_T_A={video_ta:.2f}s ({_fmt_mmss(video_ta)})")
    return per_trial_ta


# ─────────────────────────────────────────────────────────
# AUDIO LOADING
# ─────────────────────────────────────────────────────────
def load_audio(path: str, sr: int = 22050):
    import librosa as _lib

    # 1) librosa direct (handles MP4/MP3/OGG via audioread)
    try:
        y, sr_out = _lib.load(path, sr=sr, mono=True)
        print(f"[AUDIO] {Path(path).name}: {len(y)/sr_out:.1f}s at {sr_out} Hz")
        return y, sr_out
    except Exception:
        pass

    # 2) scipy.io.wavfile — handles WAV files that soundfile rejects
    try:
        import scipy.io.wavfile as _wf
        rate, data = _wf.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        else:
            data = data.astype(np.float32)
        if rate != sr:
            data = _lib.resample(data, orig_sr=rate, target_sr=sr)
        print(f"[AUDIO] {Path(path).name}: {len(data)/sr:.1f}s at {sr} Hz (scipy)")
        return data, sr
    except Exception:
        pass

    # 3) ffmpeg fallback (for video files / exotic formats)
    tmp = tempfile.mktemp(suffix=".wav")
    try:
        subprocess.run(["ffmpeg", "-i", path, "-vn", "-ar", str(sr),
                        "-ac", "1", tmp, "-y"],
                       capture_output=True, check=True)
        y, sr_out = _lib.load(tmp, sr=sr, mono=True)
        os.unlink(tmp)
        print(f"[AUDIO] {Path(path).name}: {len(y)/sr_out:.1f}s at {sr_out} Hz (ffmpeg)")
        return y, sr_out
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)

    # All methods failed — return None so caller can fall back to energy detection
    print(f"[WARN] Cannot load audio from '{Path(path).name}' — falling back to energy detection")
    return None, sr


# ─────────────────────────────────────────────────────────
# ALARM DETECTION  (librosa)
# ─────────────────────────────────────────────────────────
def detect_alarm_candidates(y, sr, template_y,
                              threshold: float = 0.07,
                              min_gap_s: float = 0.15) -> list[tuple]:
    """
    Template-match the single alarm WAV against the full video audio.
    Returns sorted list of (time_s, confidence).

    The phone plays the alarm 3× in rapid succession, so this typically
    produces 3 hits per real alarm event — cluster_alarm_events() groups them.
    """
    from scipy.signal import correlate, find_peaks

    y_n = y          / (np.max(np.abs(y))          + 1e-8)
    t_n = template_y / (np.max(np.abs(template_y)) + 1e-8)

    corr   = correlate(y_n, t_n, mode="full")
    corr   = np.maximum(corr, 0)
    lags_s = (np.arange(len(corr)) - (len(t_n) - 1)) / sr

    peaks, _ = find_peaks(corr, height=threshold,
                           distance=int(min_gap_s * sr))
    candidates = [(float(lags_s[p]), float(corr[p]))
                  for p in peaks if lags_s[p] >= 0]
    print(f"[DETECT] {len(candidates)} raw hits  (threshold={threshold})")
    return sorted(candidates, key=lambda x: x[0])


def detect_alarm_energy(y: np.ndarray, sr: int,
                         min_gap_s: float = 0.15) -> list[tuple]:
    """
    Fallback alarm detection without template.
    Computes short-time RMS energy, finds peaks above the 90th percentile.
    Works because the alarm bursts are the loudest transient events in the video.
    Returns sorted list of (time_s, normalised_energy).
    """
    from scipy.signal import find_peaks

    hop_n   = int(0.01 * sr)   # 10 ms hop
    frame_n = int(0.05 * sr)   # 50 ms frame
    n       = (len(y) - frame_n) // hop_n

    energy = np.array([
        np.sqrt(np.mean(y[i * hop_n : i * hop_n + frame_n] ** 2))
        for i in range(n)
    ])
    # Slight smoothing
    energy = np.convolve(energy, np.ones(3) / 3, mode="same")

    thresh   = np.percentile(energy, 90)
    min_dist = max(1, int(min_gap_s / 0.01))   # 0.01 = hop in seconds
    peaks, _ = find_peaks(energy, height=thresh, distance=min_dist)

    times      = np.arange(n) * 0.01            # seconds per hop
    max_e      = np.max(energy) + 1e-8
    candidates = [(float(times[p]), float(energy[p] / max_e)) for p in peaks]
    print(f"[DETECT] {len(candidates)} energy peaks  (energy threshold={thresh:.4f})")
    return sorted(candidates, key=lambda x: x[0])


def cluster_alarm_events(candidates: list[tuple],
                          window_s: float = CLUSTER_WINDOW_S) -> list[tuple]:
    """
    Collapse 3× alarm repetitions into one event.
    Within each cluster (hits within window_s of each other) keep the first
    hit — that is the true alarm onset as heard in the room.
    Returns list of (time_s, max_confidence) for each cluster.
    """
    if not candidates:
        return []
    clusters = []
    cluster_start, cluster_conf = candidates[0]
    for t, c in candidates[1:]:
        if t - cluster_start <= window_s:
            cluster_conf = max(cluster_conf, c)
        else:
            clusters.append((cluster_start, cluster_conf))
            cluster_start, cluster_conf = t, c
    clusters.append((cluster_start, cluster_conf))
    print(f"[CLUSTER] {len(candidates)} hits → {len(clusters)} alarm events")
    return clusters


def _expected_alarms(trials: list[dict]) -> list[dict]:
    """
    Build expected alarm sequence from trial metadata (same logic as run_sim.py).
    Returns list of {trial_idx, alarm_type, log_time, log_ta}.
    """
    expected = []
    for i, t in enumerate(trials):
        mode, outcome, run_id = t["mode"], t["outcome"], t["run_id"]
        log_ta = t["t_a"] or 0.0
        is_hit   = outcome in HIT_OUTCOMES
        is_catch = run_id.startswith("c")

        if mode == "a":
            # 'f' has no audio signature in this study (see chat) — only vha is
            # searched for acoustically. f's video timestamp is back-calculated
            # from video_ta in compute_video_timestamps(), which only applies
            # for P1-P11 (auto-sync); P12+ get video_ta directly from T_Start_abs.
            if is_hit and t["full_warn_t"] is not None:
                expected.append({"trial_idx": i, "alarm_type": "vha",
                                "log_time": t["full_warn_t"], "log_ta": log_ta})
        else:  # baseline
            if is_hit and t["full_warn_t"] is not None:
                expected.append({"trial_idx": i, "alarm_type": "vha",
                                "log_time": t["full_warn_t"], "log_ta": log_ta})
    return expected


def match_alarms_to_trials(events:            list[tuple],
                             trials:           list[dict],
                             min_inter_trial:  float = 8.0,
                             max_intra_trial:  float = 4.0) -> dict:
    """
    Greedy sequence match: assign alarm events → trial T_A timestamps.

    Adaptive hit trials have 2 alarms ~2.5 s apart (f + vha).
    All other alarm-bearing trials have 1 alarm.

    Returns {trial_index → video_ta}.
    """
    expected = _expected_alarms(trials)
    by_trial: dict[int, list] = {}
    for e in expected:
        by_trial.setdefault(e["trial_idx"], []).append(e)

    per_trial_ta: dict[int, float] = {}
    ci           = 0
    last_t       = -999.0

    print(f"\n[MATCH] {len(events)} alarm events → "
          f"{len(by_trial)} trials with expected alarms\n")

    for trial_idx in sorted(by_trial.keys()):
        alarms = by_trial[trial_idx]
        run_id = trials[trial_idx]["run_id"]
        search_from = last_t + min_inter_trial if per_trial_ta else 0.0

        while ci < len(events) and events[ci][0] < search_from:
            ci += 1
        if ci >= len(events):
            print(f"[MATCH] {run_id:3s}  ⚠ no events left after {search_from:.1f}s")
            break

        if len(alarms) == 1:
            matched = False
            while ci < len(events):
                vid_t, conf = events[ci]
                video_ta = vid_t - (alarms[0]["log_time"] - alarms[0]["log_ta"])
                if video_ta >= MIN_VIDEO_TA_S:
                    matched = True
                    break
                ci += 1   # zu früh, um plausibel zu sein -> verwerfen, weitersuchen
            if not matched:
                print(f"[MATCH] {run_id:3s}  ⚠ kein plausibler Treffer "
                    f"(video_ta >= {MIN_VIDEO_TA_S}s) gefunden — Cumulative-Fallback")
                continue
            per_trial_ta[trial_idx] = video_ta
            last_t = vid_t
            ci += 1
            print(f"[MATCH] {run_id:3s}  {alarms[0]['alarm_type']} "
                f"@ {vid_t:.2f}s  (conf={conf:.2f})  → video_T_A={video_ta:.2f}s")

        else:  # 2 alarms — look for pair ~2.5 s apart
            found = False
            for j in range(ci, len(events) - 1):
                if events[j][0] < search_from:
                    continue
                t1, c1 = events[j]
                t2, c2 = events[j + 1]
                if 1.5 <= (t2 - t1) <= max_intra_trial:
                    video_ta = t1 - (alarms[0]["log_time"] - alarms[0]["log_ta"])
                    per_trial_ta[trial_idx] = video_ta
                    last_t = t2
                    ci = j + 2
                    found = True
                    print(f"[MATCH] {run_id:3s}  "
                          f"f@{t1:.2f}s + vha@{t2:.2f}s  "
                          f"(gap={t2-t1:.2f}s)  → video_T_A={video_ta:.2f}s")
                    break
            if not found:
                vid_t, conf = events[ci]
                video_ta = vid_t - (alarms[-1]["log_time"] - alarms[-1]["log_ta"])
                per_trial_ta[trial_idx] = video_ta
                last_t = vid_t
                ci += 1
                print(f"[MATCH] {run_id:3s}  ⚠ pair not found — "
                      f"single @ {vid_t:.2f}s  → video_T_A={video_ta:.2f}s")

    matched = len(per_trial_ta)
    total   = len(by_trial)
    print(f"\n[MATCH] {matched}/{total} trials matched  "
          f"({total - matched} → cumulative fallback)\n")
    return per_trial_ta

def _fmt_mmss(seconds):
    if seconds is None:
        return "-"
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}:{s:05.2f}"

def match_alarms_to_trials_guided(events: list[tuple],
                                    trials:  list[dict],
                                    sync_offset: float,
                                    trial_gap_s: float,
                                    tolerance_s: float = 60.0) -> dict:
    """
    SEQUENTIAL guided matching. Each successful match becomes the new anchor
    for the NEXT trial's expected position — this resets drift to zero at
    every confirmed point, instead of accumulating error from a single
    original anchor across the whole block (see chat: that version matched
    t5 correctly but was 100-268s off for every trial after it, because it
    always measured "close to my own increasingly-wrong guess" rather than
    "close to the true, still-unknown position").
    """
    expected = _expected_alarms(trials)
    by_trial: dict[int, list] = {}
    for e in expected:
        by_trial.setdefault(e["trial_idx"], []).append(e)

    used: set[int] = set()
    per_trial_ta: dict[int, float] = {}

    last_good_ta  = sync_offset   # video_ta of the most recently CONFIRMED trial
    last_good_idx = -1            # its index (-1 = nothing confirmed yet)

    print(f"\n[MATCH-GUIDED] {len(events)} alarm events, tolerance=±{tolerance_s:.0f}s "
          f"(sequenziell, Anker wird bei jedem Treffer zurückgesetzt)\n")

    for trial_idx in sorted(by_trial.keys()):
        alarms = by_trial[trial_idx]
        run_id = trials[trial_idx]["run_id"]

        # Kumulative Dauer NUR von letztem bestätigten Punkt bis hierher —
        # nicht vom allerersten Anker aus. Fehler kann also nur über die
        # WENIGEN Trials zwischen zwei Treffern wachsen, nicht über den
        # ganzen Block.
        cumulative = sum(trials[j]["trial_duration"] + trial_gap_s
                          for j in range(last_good_idx + 1, trial_idx))
        exp_ta = last_good_ta + cumulative + trials[trial_idx].get("extra_padding_before", 0.0)

        if len(alarms) == 1:
            exp_event_time = exp_ta + (alarms[0]["log_time"] - alarms[0]["log_ta"])
            candidates = [(i, abs(t - exp_event_time)) for i, (t, c) in enumerate(events)
                          if i not in used and abs(t - exp_event_time) <= tolerance_s]
            if not candidates:
                print(f"[MATCH-GUIDED] {run_id:3s}  ⚠ kein Treffer innerhalb ±{tolerance_s:.0f}s um "
                      f"erwartete {exp_event_time:.1f}s — Anker bleibt beim letzten Treffer, "
                      f"Fehler akkumuliert weiter")
                continue
            best_i = min(candidates, key=lambda x: x[1])[0]
            used.add(best_i)
            vid_t, conf = events[best_i]
            video_ta = vid_t - (alarms[0]["log_time"] - alarms[0]["log_ta"])
            per_trial_ta[trial_idx] = video_ta
            last_good_ta, last_good_idx = video_ta, trial_idx
            print(f"[MATCH-GUIDED] {run_id:3s}  {alarms[0]['alarm_type']} @ {vid_t:.2f}s ({_fmt_mmss(vid_t)})  "
                    f"(conf={conf:.2f}, Δ={vid_t - exp_event_time:+.1f}s vs. erwartet {_fmt_mmss(exp_event_time)})  "
                    f"→ video_T_A={video_ta:.2f}s  [neuer Anker]")
        else:  # f + vha Paar
            exp_pair_time = exp_ta + (alarms[0]["log_time"] - alarms[0]["log_ta"])
            best_pair, best_dist = None, None
            for i in range(len(events) - 1):
                if i in used or (i + 1) in used:
                    continue
                t1_, _ = events[i]
                t2_, _ = events[i + 1]
                if not (1.5 <= (t2_ - t1_) <= 4.0):
                    continue
                dist = abs(t1_ - exp_pair_time)
                if dist <= tolerance_s and (best_dist is None or dist < best_dist):
                    best_dist, best_pair = dist, (i, t1_, t2_)
            if best_pair is None:
                print(f"[MATCH-GUIDED] {run_id:3s}  f@{t1_:.2f}s ({_fmt_mmss(t1_)}) + vha@{t2_:.2f}s ({_fmt_mmss(t2_)})  "
                        f"(Δ={t1_ - exp_pair_time:+.1f}s vs. erwartet {_fmt_mmss(exp_pair_time)})  "
                        f"→ video_T_A={video_ta:.2f}s  [neuer Anker]")
                continue
            i, t1_, t2_ = best_pair
            used.add(i); used.add(i + 1)
            video_ta = t1_ - (alarms[0]["log_time"] - alarms[0]["log_ta"])
            per_trial_ta[trial_idx] = video_ta
            last_good_ta, last_good_idx = video_ta, trial_idx
            print(f"[MATCH-GUIDED] {run_id:3s}  f@{t1_:.2f}s + vha@{t2_:.2f}s  "
                  f"(Δ={t1_ - exp_pair_time:+.1f}s vs. erwartet)  → video_T_A={video_ta:.2f}s  "
                  f"[neuer Anker]")

    print(f"\n[MATCH-GUIDED] {len(per_trial_ta)}/{len(by_trial)} Trials gematcht\n")
    return per_trial_ta

# ─────────────────────────────────────────────────────────
# VELOCITY-BASED REACTION DETECTION  (MediaPipe)
# ─────────────────────────────────────────────────────────
def _body_positions(cap: cv2.VideoCapture,
                    detector: mp_vision.PoseLandmarker,
                    start_s: float,
                    dur_s:   float,
                    fps:     float) -> tuple[np.ndarray, np.ndarray]:
    """
    Scan [start_s, start_s + dur_s], return (times_s, positions).

    positions columns:
      0: hip_x      — normalized [0,1] x midpoint of left/right hip
      1: hip_y      — normalized [0,1] y midpoint of left/right hip
      2: nose_y     — normalized [0,1] y of NOSE landmark
      3: shoulder_y — normalized [0,1] y midpoint of left/right shoulder

    hip_x/hip_y drive STOP/RUN/RUN_BACK detection (existing logic).
    nose_y/shoulder_y drive the new gaze/head-raise ("look up") detection:
    shoulder_y - nose_y grows when the head tilts up and away from a
    slouched, phone-down posture.

    Frames where landmarks are not detected are interpolated.
    Renamed from _hip_positions() — update all call sites.
    """
    start_f = max(0, int(start_s * fps))
    end_f   = int((start_s + dur_s) * fps)
    SEEK_BUFFER_FRAMES = 30  # ~1s Vorlauf, damit der Decoder "eingeschwungen" ist
    buffer_start = max(0, start_f - SEEK_BUFFER_FRAMES)
    cap.set(cv2.CAP_PROP_POS_FRAMES, buffer_start)
    for _ in range(start_f - buffer_start):
        cap.read()  # verwerfen, Decoder bis zum echten Zielframe vorspulen

    times, positions = [], []
    while True:
        fi = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not ret or fi > end_f:
            break

        t_s      = fi / fps
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res      = detector.detect(mp_img)  # IMAGE mode: stateless, no monotonic timestamp req.

        if res.pose_landmarks:
            lm = res.pose_landmarks[0]   # first (only) person
            lh, rh = lm[23], lm[24]      # LEFT_HIP, RIGHT_HIP
            nose    = lm[0]              # NOSE
            ls, rs  = lm[11], lm[12]     # LEFT_SHOULDER, RIGHT_SHOULDER
            hips_ok      = lh.visibility > 0.2 or rh.visibility > 0.2
            upper_ok     = nose.visibility > 0.2 and (ls.visibility > 0.2 or rs.visibility > 0.2)
            if hips_ok:
                hip_x = (lh.x + rh.x) / 2
                hip_y = (lh.y + rh.y) / 2
                nose_y     = nose.y if upper_ok else np.nan
                shoulder_y = ((ls.y + rs.y) / 2) if upper_ok else np.nan
                nose_x         = nose.x if upper_ok else np.nan
                shoulder_mid_x = ((ls.x + rs.x) / 2) if upper_ok else np.nan
                shoulder_width = abs(ls.x - rs.x) if upper_ok else np.nan
                times.append(t_s)
                positions.append([hip_x, hip_y, nose_y, shoulder_y,
                                   nose_x, shoulder_mid_x, shoulder_width])
                continue

        # No reliable pose — append NaN for later interpolation
        times.append(t_s)
        positions.append([np.nan] * 7)

    if not times:
        return np.array([]), np.array([]).reshape(0, 7)

    times = np.array(times)
    pos   = np.array(positions)

    # Linear interpolation over NaN gaps (per column)
    for col in range(pos.shape[1]):
        valid = ~np.isnan(pos[:, col])
        if valid.sum() >= 2:
            pos[:, col] = np.interp(times, times[valid], pos[valid, col])

    return times, pos


def _smooth_velocity(pos: np.ndarray, times: np.ndarray,
                     window: int = SMOOTH_FRAMES) -> np.ndarray:
    """Compute rolling-average speed (magnitude of velocity) in px/s."""
    if len(pos) < 2:
        return np.zeros(len(pos))
    dt   = np.diff(times)
    dt   = np.where(dt > 0, dt, 1e-3)
    dpos = np.diff(pos, axis=0)
    speed = np.sqrt((dpos[:, 0] / dt) ** 2 + (dpos[:, 1] / dt) ** 2)
    speed = np.concatenate([[speed[0]], speed])  # align length

    # Rolling average
    kernel = np.ones(window) / window
    return np.convolve(speed, kernel, mode="same")

def _signal_velocity(signal: np.ndarray, times: np.ndarray,
                      window: int = SMOOTH_FRAMES) -> np.ndarray:
    """Rate of change of a 1D signal over time, smoothed — generic version
    of _smooth_velocity() for scalar signals (used for gaze-raise speed
    instead of an absolute distance-from-baseline threshold, which misses
    short, jerky head movements that don't move the smoothed distance far)."""
    if len(signal) < 2:
        return np.zeros(len(signal))
    dt   = np.diff(times)
    dt   = np.where(dt > 0, dt, 1e-3)
    dsig = np.diff(signal) / dt
    dsig = np.concatenate([[dsig[0]], dsig])
    kernel = np.ones(window) / window
    return np.convolve(dsig, kernel, mode="same")

# Tuning constants for the new channels — NOT yet empirically validated,
# calibrate against a handful of known response_run / response_run_back /
# "looked up" trials using --debug_trial before trusting these on the full set.
RUN_RATIO           = 1.5    # forward speed must exceed baseline x this to count as RUN
RUN_BACK_RATIO       = 0.3   # |backward component| must exceed baseline x this
#GAZE_RISE_MARGIN     = 0.010 # absolute rise in (shoulder_y - nose_y), normalized units
GAZE_VELOCITY_THRESHOLD = 0.028
MIN_GAZE_BASELINE_FRAMES = 15
MIN_GAZE_REACTION_FRAMES = 2
MIN_VIDEO_TA_S = 3.0
#GAZE_SIDE_VELOCITY_THRESHOLD = 0.04    # ungetestet — mit --debug_trial kalibrieren
MIN_GAZE_SIDE_REACTION_FRAMES = 2
GAZE_SIDE_SD_MULTIPLIER = 4.0   # Reaktion muss >4 Baseline-Standardabweichungen
                                  # ausschlagen — ungetestet, mit --debug_trial kalibrieren


def _forward_velocity(pos: np.ndarray, times: np.ndarray,
                       heading: np.ndarray, window: int = SMOOTH_FRAMES) -> np.ndarray:
    """
    Signed velocity component of hip motion (pos[:, 0:2]) along `heading`
    (a unit vector, typically the participant's dominant pre-alarm walking
    direction). Positive = moving along heading (forward), negative =
    moving against it (backward / turned around).
    """
    if len(pos) < 2:
        return np.zeros(len(pos))
    dt   = np.diff(times)
    dt   = np.where(dt > 0, dt, 1e-3)
    dpos = np.diff(pos[:, 0:2], axis=0)
    vel  = dpos / dt[:, None]
    fwd  = vel @ heading
    fwd  = np.concatenate([[fwd[0]], fwd])
    kernel = np.ones(window) / window
    return np.convolve(fwd, kernel, mode="same")


def _sustained_crossing(times: np.ndarray, signal: np.ndarray,
                         predicate, min_frames: int) -> float | None:
    """First timestamp where `predicate(signal[i])` holds for >= min_frames
    consecutive samples. Generic helper shared by all four channels."""
    consecutive = 0
    start_t = None
    for t, v in zip(times, signal):
        if predicate(v):
            if consecutive == 0:
                start_t = t
            consecutive += 1
            if consecutive >= min_frames:
                return start_t
        else:
            consecutive = 0
            start_t = None
    return None


def find_body_reaction(cap:       cv2.VideoCapture,
                        detector:  mp_vision.PoseLandmarker,
                        alarm_s:   float,
                        search_s:  float,
                        fps:       float,
                        reaction_ratio: float = DEFAULT_REACTION_RATIO,
                        debug_run: str = None,
                        min_reaction_frames_override: int = None) -> tuple[float | None, dict]:
    """
    [Docstring unverändert lassen]
    """

    min_frames = min_reaction_frames_override or MIN_REACTION_FRAMES
    min_gaze_frames = min(MIN_GAZE_REACTION_FRAMES, min_frames)  # gaze bleibt nie strenger als der Override

    total_s   = BASELINE_WINDOW_S + search_s
    start_s   = max(0.0, alarm_s - BASELINE_WINDOW_S)
    times, pos = _body_positions(cap, detector, start_s, total_s, fps)

    debug = {"times": times, "pos": pos, "speed": None,
             "baseline": None, "threshold": None, "alarm_s": alarm_s,
             "reaction_type": None, "heading_norm": None}

    valid_pose_frames = np.isfinite(pos[:, 0]).sum() if len(pos) else 0
    debug["valid_pose_frames"] = valid_pose_frames
    if valid_pose_frames == 0:
        debug["occlusion"] = True
        return None, debug

    if len(times) < SMOOTH_FRAMES + 2:
        return None, debug

    hip_pos = pos[:, 0:2]
    speed = _smooth_velocity(hip_pos, times)
    debug["speed"] = speed

    MIN_BASELINE_FRAMES = 15
    pre_mask = times < alarm_s
    if pre_mask.sum() < MIN_BASELINE_FRAMES:
        print(f"  [WARN] only {pre_mask.sum()} pre-alarm frames — skip RT (no valid baseline)")
        return None, debug

    baseline = np.nanmean(speed[pre_mask])
    if baseline < 1e-5:
        # Participant was already stationary before alarm — STOP/RUN/RUN_BACK
        # cannot be computed from speed. GAZE_UP can still be attempted below.
        baseline = None

    post_mask  = times >= alarm_s
    post_times = times[post_mask]
    scan_mask       = post_times >= (alarm_s + MIN_PLAUSIBLE_RT)
    post_times_scan = post_times[scan_mask]
    candidates = {}  # reaction_type -> timestamp

    if baseline is not None:
        threshold = baseline * reaction_ratio
        debug["baseline"], debug["threshold"] = baseline, threshold
        post_speed = speed[post_mask]

        t_stop = _sustained_crossing(post_times_scan, post_speed[scan_mask],
                              lambda s: s <= threshold, min_frames)
        if t_stop is not None:
            candidates["stop"] = t_stop

        t_run = _sustained_crossing(post_times_scan, post_speed[scan_mask],
                             lambda s: s >= baseline * RUN_RATIO, min_frames)
        if t_run is not None:
            candidates["run"] = t_run

        # Heading: direction of net displacement during the baseline window.
        base_pts = hip_pos[pre_mask]
        if len(base_pts) >= 2:
            disp = base_pts[-1] - base_pts[0]
            norm = np.linalg.norm(disp)
            debug["heading_norm"] = norm
            if debug_run is not None:
                print(f"[RUNBACK-DEBUG] baseline displacement norm={norm:.5f}  "
                    f"(sehr klein = keine klare Gehrichtung erkennbar -> run_back unzuverlässig)")
            if norm > 1e-4:
                heading = disp / norm
                fwd = _forward_velocity(hip_pos, times, heading)
                post_fwd = fwd[post_mask]
                if debug_run is not None:
                    print(f"[RUNBACK-DEBUG] baseline_speed={baseline:.4f}  "
                        f"threshold={-baseline*RUN_BACK_RATIO:.4f}  "
                        f"post_fwd min/max={post_fwd.min():.4f}/{post_fwd.max():.4f}")
                t_back = _sustained_crossing(
                        post_times_scan, post_fwd[scan_mask],
                        lambda v: v <= -baseline * RUN_BACK_RATIO, min_frames)
                if t_back is not None:
                    candidates["run_back"] = t_back

    # --- GAZE_UP: independent of walking speed/direction ---
    gaze = pos[:, 3] - pos[:, 2]  # shoulder_y - nose_y
    if np.isfinite(gaze).sum() >= SMOOTH_FRAMES + 2:
        kernel = np.ones(SMOOTH_FRAMES) / SMOOTH_FRAMES
        gaze_smooth = np.convolve(np.nan_to_num(gaze, nan=np.nanmean(gaze)), kernel, mode="same")
        gaze_pre = gaze_smooth[pre_mask]
        if len(gaze_pre) >= MIN_GAZE_BASELINE_FRAMES:
            gaze_vel = _signal_velocity(gaze_smooth, times)
            gaze_vel_post = gaze_vel[post_mask]
            if debug_run is not None:
                print(f"[GAZE-DEBUG] velocity post min/max="
                    f"{gaze_vel_post.min():.4f}/{gaze_vel_post.max():.4f}  "
                    f"threshold={GAZE_VELOCITY_THRESHOLD}")
            t_gaze = _sustained_crossing(
                post_times_scan, gaze_vel_post[scan_mask],
                lambda v: v >= GAZE_VELOCITY_THRESHOLD, min_gaze_frames)
            if t_gaze is not None:
                candidates["gaze_up"] = t_gaze
        elif debug_run is not None:
            print(f"[GAZE-DEBUG] too little pre-alarm Baseline ({len(gaze_pre)} Frames, "
                f"need {MIN_GAZE_BASELINE_FRAMES}) — gaze_up skipped")
    elif debug_run is not None:
        print(f"[GAZE-DEBUG] too little valid nose/shoulder Frames overall "
            f"({np.isfinite(gaze).sum()}) — gaze_up skipped")

    # --- GAZE_SIDE: horizontal head turn, independent of walking speed ---
    shoulder_width = pos[:, 6]
    # Gegen Normierungs-Explosion absichern: shoulder_width muss nah an der
    # eigenen Baseline-Schulterbreite liegen, nicht nur "irgendwas über 0".
    baseline_width = np.nanmedian(shoulder_width[pre_mask]) if pre_mask.sum() > 0 else np.nan
    if np.isfinite(baseline_width) and baseline_width > 1e-3:
        valid_width = np.isfinite(shoulder_width) & (shoulder_width > 0.5 * baseline_width)
    else:
        valid_width = np.zeros(len(shoulder_width), dtype=bool)
    
    if valid_width.sum() >= SMOOTH_FRAMES + 2:
        horiz_raw = (pos[:, 4] - pos[:, 5]) / np.where(valid_width, shoulder_width, np.nan)
        kernel = np.ones(SMOOTH_FRAMES) / SMOOTH_FRAMES
        horiz_smooth = np.convolve(np.nan_to_num(horiz_raw, nan=np.nanmean(horiz_raw)),
                                    kernel, mode="same")
        horiz_pre = horiz_smooth[pre_mask]
        if len(horiz_pre) >= MIN_GAZE_BASELINE_FRAMES:
            horiz_vel = _signal_velocity(horiz_smooth, times)
            horiz_vel_baseline = horiz_vel[pre_mask]
            horiz_vel_sd = np.nanstd(horiz_vel_baseline)
            horiz_vel_post = horiz_vel[post_mask]
            if horiz_vel_sd > 1e-6:
                threshold_dynamic = GAZE_SIDE_SD_MULTIPLIER * horiz_vel_sd
                if debug_run is not None:
                    print(f"[GAZE-SIDE-DEBUG] baseline SD={horiz_vel_sd:.4f}  "
                          f"dynamic threshold={threshold_dynamic:.4f}  "
                          f"post min/max={horiz_vel_post.min():.4f}/{horiz_vel_post.max():.4f}")
                t_side = _sustained_crossing(
                    post_times_scan, np.abs(horiz_vel_post[scan_mask]),
                    lambda v: v >= threshold_dynamic, MIN_GAZE_SIDE_REACTION_FRAMES)
                if t_side is not None:
                    candidates["gaze_side"] = t_side
            elif debug_run is not None:
                print(f"[GAZE-SIDE-DEBUG] baseline SD too small/zero — gaze_side skipped")
        elif debug_run is not None:
            print(f"[GAZE-SIDE-DEBUG] too little pre-alarm baseline ({len(horiz_pre)} frames) "
                  f"— gaze_side skipped")
    elif debug_run is not None:
        print(f"[GAZE-SIDE-DEBUG] too little valid shoulder-width data — gaze_side skipped")


    if not candidates:
        if debug_run is not None and baseline is not None:
            _debug_plot(times, speed, baseline, debug["threshold"], alarm_s, None, debug_run)
        return None, debug

    # Earliest reaction across all channels wins.
    reaction_type = min(candidates, key=candidates.get)
    reaction_t    = candidates[reaction_type]
    debug["reaction_type"] = reaction_type

    if debug_run is not None and baseline is not None:
        _debug_plot(times, speed, baseline, debug["threshold"], alarm_s, reaction_t, debug_run)

    return reaction_t, debug


def _debug_plot(times, speed, baseline, threshold, alarm_s, reaction_t, label):
    """Save a velocity-over-time plot for inspection (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(times, speed, label="hip speed")
        ax.axhline(baseline,  color="green", ls="--", label=f"baseline={baseline:.4f}")
        ax.axhline(threshold, color="orange", ls="--",
                   label=f"threshold ({DEFAULT_REACTION_RATIO:.0%})={threshold:.4f}")
        ax.axvline(alarm_s, color="red", label="alarm")
        if reaction_t:
            ax.axvline(reaction_t, color="purple", label=f"reaction  RT={(reaction_t-alarm_s)*1000:.0f}ms")
        ax.set_xlabel("video time (s)")
        ax.set_ylabel("hip speed (norm/s)")
        ax.legend()
        ax.set_title(f"Hip velocity — {label}")
        fname = f"debug_velocity_{label}.png"
        fig.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"[DEBUG] Velocity plot saved → {fname}")
    except ImportError:
        print("[DEBUG] matplotlib not installed — skipping velocity plot")


def load_annotation_reactions(path=None):
    if path is None:
        # main/videos/analyze_rt_video.py -> main/trial_annotations.csv
        path = str(Path(__file__).parent.parent / "trial_annotations.csv")
    lookup = {}
    if not os.path.exists(path):
        print(f"[QC] {path} not found — cross-check disabled")
        return lookup
    import csv as _csv
    with open(path, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            pid_norm = row["PID"].strip().lstrip("Pp")   # "24" bleibt "24", "P24" wird "24"
            key = (pid_norm, row["Mode"].strip(), row["Run"].strip())
            lookup[key] = {
                "alarm_reaction": row.get("Alarm_reaction", "none"),
                "stopped_at_alarm": row.get("Stopped_at_Alarm", "no"),
            }
    return lookup

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="V2P RT Analysis — MediaPipe body-velocity reaction from block video")

    ap.add_argument("--video",       required=True)
    ap.add_argument("--logs_dir",    required=True)
    ap.add_argument("--pid",         required=True)
    ap.add_argument("--condition",   required=True, help="'Adaptive' or 'Fixed'")

    # ── Sync mode ─────────────────────────────────────────
    ap.add_argument("--sync_offset",     type=float, default=0.0)
    ap.add_argument("--trial_gap_s",     type=float, default=DEFAULT_TRIAL_GAP_S)
    ap.add_argument("--auto_sync",       action="store_true")
    ap.add_argument("--alarm_audio",     default=None,
                    help="Single alarm WAV (phone plays it 3× — provide the 1× original)")
    ap.add_argument("--alarm_threshold", type=float, default=0.07,
                    help="Template-match threshold (lower = more sensitive, default 0.07)")
    ap.add_argument("--video_start_abs", type=float, default=None,
                    help="Unix timestamp when video recording started (P12+)")
    ap.add_argument("--per_trial_ta_json", default=None,
                help='JSON-Dict {"t1": 20.77, "t5": 114.79, ...} — direct '
                     'video_ta per trial')

    # ── Reaction detection ────────────────────────────────
    ap.add_argument("--reaction_ratio",  type=float, default=DEFAULT_REACTION_RATIO,
                    help=f"Speed must drop to this fraction of baseline "
                         f"(default {DEFAULT_REACTION_RATIO})")
    ap.add_argument("--search_s",        type=float, default=8.0,
                    help="Seconds after alarm to search for reaction (default 8)")
    ap.add_argument("--debug_trial",     default=None,
                    help="Run ID to save velocity debug plot for (e.g. t3)")

    # ── Output ────────────────────────────────────────────
    ap.add_argument("--out",    default="rt_results.csv")
    ap.add_argument("--append", action="store_true")

    args = ap.parse_args()
    pid_norm = args.pid.lstrip("Pp")   # "P24" -> "24", "24" bleibt "24" — für Annotation-Lookups

    # ── Load logs ─────────────────────────────────────────
    trials = load_logs(args.logs_dir, args.pid, args.condition)
    ann_reactions = load_annotation_reactions()

    # ── Sync ──────────────────────────────────────────────
    per_trial_ta = None

    per_trial_ta_direct = None
    if args.per_trial_ta_json:
        import json as _json
        raw_map = _json.loads(args.per_trial_ta_json)
        per_trial_ta_direct = {}
        for i, t in enumerate(trials):
            if t["run_id"] in raw_map:
                per_trial_ta_direct[i] = raw_map[t["run_id"]]
        print(f"[SYNC] {len(per_trial_ta_direct)}/{len(raw_map)} direkte Anker aus "
            f"--per_trial_ta_json auf Trials gemappt")

    if per_trial_ta_direct:
        print(f"\n[SYNC] Mode: DIRECT MULTI-ANCHOR ({len(per_trial_ta_direct)} Trials)")
        per_trial_ta = per_trial_ta_direct

    if args.video_start_abs is not None:
        print(f"\n[SYNC] Mode: ABSOLUTE TIMESTAMP")
        per_trial_ta = compute_timestamps_from_abs(trials, args.video_start_abs)

    elif args.auto_sync:
        print(f"\n[SYNC] Mode: AUTO-SYNC")
        y, sr = load_audio(args.video)
        if y is None:
            sys.exit("[ERROR] Cannot load video audio — cannot auto-sync.")

        # Try template matching first; fall back to energy-based detection
        if args.alarm_audio:
            template, _ = load_audio(args.alarm_audio, sr=sr)
        else:
            template = None

        if template is not None:
            print("[SYNC] Using template matching")
            candidates = detect_alarm_candidates(y, sr, template,
                                                  threshold=args.alarm_threshold)
        else:
            print("[SYNC] Template unavailable — using energy-based detection")
            candidates = detect_alarm_energy(y, sr)

        events = cluster_alarm_events(candidates)

        if not events:
            print("[WARN] No alarm events detected — check --alarm_threshold")
            print("       Falling back to manual sync_offset=0, trial_gap_s=25s")
        else:
            per_trial_ta = match_alarms_to_trials(events, trials)

    elif args.sync_offset and args.alarm_audio:
        print(f"\n[SYNC] Mode: GUIDED (Anker={args.sync_offset}s + Audio-Erkennung)")
        y, sr = load_audio(args.video)
        if y is None:
            sys.exit("[ERROR] Cannot load video audio — cannot use guided sync.")
        template, _ = load_audio(args.alarm_audio, sr=sr)
        if template is not None:
            candidates = detect_alarm_candidates(y, sr, template,
                                                  threshold=args.alarm_threshold)
        else:
            candidates = detect_alarm_energy(y, sr)
        events = cluster_alarm_events(candidates)
        per_trial_ta = match_alarms_to_trials_guided(
            events, trials, args.sync_offset, args.trial_gap_s)

    else:
        print(f"\n[SYNC] Mode: MANUAL  (offset={args.sync_offset}s, "
              f"gap={args.trial_gap_s}s)")

    trials = compute_video_timestamps(
        trials, args.sync_offset, args.trial_gap_s, per_trial_ta)

    sync_mode = ("direct" if per_trial_ta_direct
                 else "absolute" if args.video_start_abs
                 else "auto"   if args.auto_sync
                 else "guided" if (args.sync_offset and args.alarm_audio)
                 else "manual")

    print(f"\n[SYNC] Trial timeline  (mode={sync_mode}):")
    for i, t in enumerate(trials):
        src = "matched" if (per_trial_ta and i in per_trial_ta) else "estimated"
        al  = f"{t['video_alarm']:.2f}s ({_fmt_mmss(t['video_alarm'])})" if t["video_alarm"] else "N/A "
        print(f"  {t['run_id']:3s}  video_T_A={t['video_ta']:6.2f}s ({_fmt_mmss(t['video_ta'])})  "
              f"alarm={al}  [{t['warning_type']}]  ({src})")

    # ── Open video ────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {args.video}")

    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n[VIDEO] {Path(args.video).name}  "
          f"{fps:.1f} fps  {n_frames/fps:.1f}s total")
    print(f"[RT]  reaction_ratio={args.reaction_ratio}  "
          f"(speed must drop to <{args.reaction_ratio:.0%} of pre-alarm baseline)\n")
    

    pose = _make_detector()

    # ── Per-trial analysis ────────────────────────────────
    results_rows = []

    for t in trials:
        run_id = t["run_id"]

        if run_id not in TARGET_RUNS:
            print(f"[SKIP] {run_id} — not a target trial")
            continue

        pid_norm = args.pid.lstrip("Pp")
        ann = ann_reactions.get(
            (pid_norm, CONDITION_TO_MODE[args.condition.lower()], run_id), {})
        stopped_at = ann.get("stopped_at_alarm", "no")

        if stopped_at in ("f", "both") and t.get("video_alarm_f") is not None:
            alarm_ref = t["video_alarm_f"]
            ref_type  = "f"
            if stopped_at == "both" and t.get("video_alarm_vha") is not None:
                # Bei "both" nur bis kurz vor vha suchen, sonst kann die Suche die
                # SPÄTERE (vha-getriggerte) Bewegung statt der ersten (f-getriggerten)
                # erwischen -> künstlich aufgeblähte RT (siehe P9/t5, RT=7.1s).
                gap_to_vha = t["video_alarm_vha"] - t["video_alarm_f"]
                search_s_this_trial = max(0.5, gap_to_vha - 0.3)  # kleiner Puffer vor vha
            else:
                search_s_this_trial = args.search_s
            print(f"  [REF] {run_id}: Annotation zeigt Reaktion bei f -> nutze f als Referenz"
                + (f" (Suchfenster auf {search_s_this_trial:.1f}s begrenzt wegen 'both')"
                    if stopped_at == "both" else ""))
        else:
            alarm_ref = t["video_alarm"]
            ref_type  = t["warning_type"]
            search_s_this_trial = args.search_s

        # Simulationszeit des tatsächlich genutzten Alarms — nötig, um den
        # Reaktionszeitpunkt später im Roh-Log (TTC_secondary) nachzuschlagen.
        if ref_type == "f":
            alarm_sim_s = t.get("early_warn_t")
        elif ref_type in ("vha", "vha_fallback"):
            alarm_sim_s = t.get("full_warn_t")
        else:
            alarm_sim_s = None

        if alarm_ref is None:
            print(f"[SKIP] {run_id} — no alarm in log")
            results_rows.append({
                "participant_id":  args.pid,
                "condition":       args.condition,
                "sync_mode":       sync_mode,
                "trial_id":        run_id,
                "heading_norm":    None,
                "outcome":         t["outcome"],
                "warning_type":    t["warning_type"],
                "alarm_video_s":   None,
                "alarm_video_mmss":    _fmt_mmss(alarm_ref),
                "reaction_video_s":None,
                "reaction_video_mmss": _fmt_mmss(reaction_s) if reaction_s else None,
                "rt_s":            None,
                "rt_ms":           None,
                "reaction_type":   None,
                "notes":           "no_alarm_in_log",
                "qc_mismatch":     False,
                "alarm_sim_s": round(alarm_sim_s, 4) if alarm_sim_s is not None else None,
            })
            continue

        print(f"[RT]  {run_id}  alarm @ {alarm_ref:.2f}s ({_fmt_mmss(alarm_ref)}) [{ref_type}]  → ",
              end="", flush=True)

        debug_label = run_id if run_id == args.debug_trial else None
        reaction_s, dbg = find_body_reaction(
            cap, pose,
            alarm_s=alarm_ref,
            search_s=search_s_this_trial,   # ← nicht args.search_s
            fps=fps,
            reaction_ratio=args.reaction_ratio,
            debug_run=debug_label,
        )

        reaction_type = dbg.get("reaction_type")
        heading_norm  = dbg.get("heading_norm")

        if reaction_s is None:
            print("no reaction detected")
            rt_s = rt_ms = None
            if dbg.get("occlusion"):
                notes = "occluded_or_out_of_frame"
            else:
                notes = ("baseline_zero" if (dbg.get("baseline") or 0) < 1e-5
                          else "no_reaction_in_window")
        else:
            rt_s  = reaction_s - alarm_ref
            rt_ms = rt_s * 1000.0
            print(f"reaction @ {reaction_s:.3f}s ({_fmt_mmss(reaction_s)})  →  "
                  f"RT = {rt_s:.3f}s  ({rt_ms:.0f}ms)  type={reaction_type}")
            notes = ""

        # Retry only when annotations says there is a reaction
        if reaction_s is None and stopped_at not in ("no", ""):
            print(f"  [RETRY] Annotation confirms reaction ('{stopped_at}'), "
                  f"nothing found - second attempt with relaxed threshold")
            reaction_s, dbg = find_body_reaction(
                cap, pose,
                alarm_s=alarm_ref,
                search_s=search_s_this_trial,
                fps=fps,
                reaction_ratio=args.reaction_ratio,
                debug_run=debug_label,
                min_reaction_frames_override=2,
            )
            reaction_type = dbg.get("reaction_type")
            heading_norm  = dbg.get("heading_norm")
            if reaction_s is not None:
                rt_s  = reaction_s - alarm_ref
                rt_ms = rt_s * 1000.0
                notes = "found_on_retry_relaxed_threshold"
                print(f"  [RETRY] Found: RT = {rt_s:.3f}s  type={reaction_type}")
            else:
                notes = "annotated_reaction_not_automatable"
                print(f"  [RETRY] Still not found")

            
        # Dritter Versuch: falls im f-Fenster (auch mit Retry) nichts gefunden wurde,
        # könnte die eigentliche Reaktion näher an vha liegen als an f (siehe Chat,
        # P1/t4 "jumped"-Fall). Nur relevant, wenn f als Referenz genutzt wurde.
        p1_old_protocol = args.pid.lstrip("Pp") == "1"
        allow_vha_fallback = (stopped_at == "both") if p1_old_protocol else True

        if reaction_s is None and ref_type == "f" and allow_vha_fallback \
                and t.get("video_alarm_vha") is not None:
            print(f"  [RETRY-VHA] Still nothing found in f-window — third attempt with vha as reference")
            reaction_s, dbg = find_body_reaction(
                cap, pose,
                alarm_s=t["video_alarm_vha"],
                search_s=args.search_s,
                fps=fps,
                reaction_ratio=args.reaction_ratio,
                debug_run=debug_label,
                min_reaction_frames_override=2,
            )
            reaction_type = dbg.get("reaction_type")
            heading_norm  = dbg.get("heading_norm")
            if reaction_s is not None:
                alarm_ref = t["video_alarm_vha"]
                ref_type  = "vha_fallback"
                rt_s  = reaction_s - alarm_ref
                rt_ms = rt_s * 1000.0
                notes = "found_on_retry_vha_fallback"
                print(f"  [RETRY-VHA] Found: RT = {rt_s:.3f}s  type={reaction_type}")

        annotated = ann.get("alarm_reaction", "none")
        qc_mismatch = (reaction_s is None) and (annotated not in ("none", ""))
        if qc_mismatch:
            print(f"  [QC-MISMATCH] Annotation says '{annotated}', Automation did not find anything")

        results_rows.append({
            "participant_id":  args.pid,
            "condition":       args.condition,
            "sync_mode":       sync_mode,
            "trial_id":        run_id,
            "heading_norm":    round(heading_norm, 5) if heading_norm is not None else None,
            "outcome":         t["outcome"],
            "warning_type":    ref_type,
            "alarm_video_s":   round(alarm_ref, 3),
            "reaction_video_s":round(reaction_s, 3) if reaction_s else None,
            "rt_s":            round(rt_s,  3)  if rt_s  is not None else None,
            "rt_ms":           round(rt_ms, 1)  if rt_ms is not None else None,
            "reaction_type":   reaction_type,
            "notes":           notes,
            "qc_mismatch":     qc_mismatch,
            "alarm_sim_s":     round(alarm_sim_s, 4) if alarm_sim_s is not None else None,
        })

    cap.release()
    pose.close()

    # ── Write results ─────────────────────────────────────
    fieldnames = ["participant_id", "condition", "sync_mode", "trial_id", "heading_norm",
                  "outcome", "warning_type", "alarm_video_s", "alarm_video_mmss",
                  "reaction_video_s", "reaction_video_mmss", "rt_s", "rt_ms",
                  "reaction_type", "notes", "qc_mismatch", "alarm_sim_s"]
    write_header = not (args.append and os.path.exists(args.out))
    with open(args.out, "a" if args.append else "w",
              newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(results_rows)

    print(f"\n[SAVE] {len(results_rows)} rows → {args.out}")

    # ── Summary ───────────────────────────────────────────
    valid = [r for r in results_rows if r["rt_ms"] is not None and r["rt_ms"] > 0]
    if valid:
        rts = [r["rt_ms"] for r in valid]
        print(f"\n── RT Summary  ({args.pid} | {args.condition}) ──────")
        print(f"  Valid trials : {len(valid)} / {len(results_rows)}")
        print(f"  Mean RT      : {np.mean(rts):.0f} ms")
        print(f"  SD           : {np.std(rts, ddof=1):.0f} ms")
        print(f"  Min / Max    : {min(rts):.0f} / {max(rts):.0f} ms")
    else:
        print("\n[WARN] No valid RTs.")
        if args.auto_sync:
            print("  → Alarm detection: lower --alarm_threshold (e.g. 0.04)")
        print("  → Reaction detection: lower --reaction_ratio (e.g. 0.5)")
        print("  → Check sync with --debug_trial t1  (saves velocity plot)")


if __name__ == "__main__":
    main()