#!/usr/bin/env python3
"""
analyse_rct.py
======================
Event-related visualisation of facial reactions to Wizard-of-Oz alarms
(preliminary study). Fully automated batch run over all participants
configured in PARTICIPANTS.

Expected files in --data-dir (names are matched case-insensitively):
    P1 - video.mp4
    modality_selection_P1.a.csv   (auditory task; .xlsx also accepted)
    modality_selection_P1.v.csv   (visual task;   .xlsx also accepted)
    face_landmarker.task          (or next to this script, or --model)

Per frame inside each analysis window the MediaPipe Face Landmarker
(Tasks API) yields two values - deliberately identical to the original
analysis (no additional normalization), so figures match Chapter 4:

  * eye opening (lid distance)
        |y(159) - y(145)| in normalized image coordinates
        (1.0 = full frame height).
  * head movement (nose displacement)
        Euclidean displacement of the nose tip (landmark 1) between two
        consecutive video frames, normalized image coordinates per frame
        (x relative to frame width, y relative to frame height).

Usage
-----
    python analyse_rct.py                      (script lies in the data folder)
    python analyse_rct.py --only P4 --types A VA VHA
    python analyse_rct.py --data-dir "D:/Vorstudie/Videos"

Outputs (default: <data-dir>/reaction_analysis)
-----------------------------------------------
    <id>_auditory_reaction_windows.png / .pdf   one figure per task
    <id>_visual_reaction_windows.png / .pdf
    <id>_frames.csv                             per-frame values
    <id>_alarm_summary.csv                      per-alarm summary
    all_alarm_summary.csv                       all participants combined
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =============================================================================
# 1. PARTICIPANT CONFIGURATION  (the only place to edit per participant)
# =============================================================================
# Key       = file ID used in the file names ("P1 - video.mp4", ...).
# thesis_id = ID used in outputs (thesis notation).
#             [verify] Assumed mapping P1 -> Pre1 ... P4 -> Pre4.
# anchors   = list of (task, alarm_index, video_time_s):
#             alarm_index is the 0-based row in that task's log,
#             video_time_s the second in the video at which that alarm starts.
#             One anchor is enough. Every additional anchor (ideally one per
#             task) is used to estimate the offset more robustly (median) and
#             to report the sync residual. A task with its own anchor(s) is
#             synchronised independently, which also covers recording pauses.
# exclude   = tasks without usable video data (skipped in the analysis).
PARTICIPANTS: dict[str, dict] = {
    "P1": {"thesis_id": "Pre1", "anchors": [("auditory", 1, 40.0)]},
    "P2": {"thesis_id": "Pre2", "anchors": [("visual", 13, 21.0)]},
    "P3": {"thesis_id": "Pre3", "anchors": [("auditory", 2, 44.0)],
           "exclude": ["visual"]},   # [verify] no usable video for visual task
    "P4": {"thesis_id": "Pre4", "anchors": [("visual", 0, 39.0)]},
}

# Task name -> suffix in the log file name (modality_selection_P1.a.csv)
TASK_SUFFIXES = {"auditory": "a", "visual": "v"}

# =============================================================================
# 2. ANALYSIS CONSTANTS
# =============================================================================
PRE_ONSET_S = 1.0    # window start: seconds BEFORE alarm onset (baseline)
POST_ONSET_S = 4.0   # window end: seconds AFTER alarm onset (response window)

LM_UPPER_LID = 159   # Face Mesh landmark: upper eyelid
LM_LOWER_LID = 145   # Face Mesh landmark: lower eyelid (same eye)
LM_NOSE_TIP = 1      # Face Mesh landmark: nose tip

MODEL_FILENAME = "face_landmarker.task"
VIDEO_STEMS = ("{fid} - video", "{fid}-video", "{fid}_video", "{fid} video")
VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".avi")
LOG_STEM = "modality_selection_{fid}.{suffix}"
LOG_EXTS = (".csv", ".xlsx", ".xls")
REQUIRED_LOG_COLUMNS = ("Timestamp", "Combination_Type")

ALARM_CODES = ("V", "H", "A", "VH", "VA", "HA", "VHA")   # thesis notation
SYNC_TOLERANCE_S = 0.2   # anchor residual above this triggers a warning

# =============================================================================
# 3. PLOT STYLE
# =============================================================================
EYE_NAME = "Eye opening (lid distance)"
HEAD_NAME = "Head movement (nose displacement)"
EYE_AXIS_LABEL = "Eye opening: lid distance\n[normalized to frame height]"
HEAD_AXIS_LABEL = "Head movement: nose displacement\n[normalized image units / frame]"
X_LABEL = "Time from onset (s)"
EYE_COLOR = "tab:blue"
HEAD_COLOR = "tab:orange"
ONSET_COLOR = "red"
WINDOW_SHADE = "0.88"
PANEL_WIDTH_IN = 2.3
PANEL_HEIGHT_IN = 2.4
DEFAULT_MAX_COLS = 5       # panels per row before wrapping
EXPORT_DPI = 300
PROGRESS_EVERY = 250       # progress message every N analysed frames


class AnalysisError(Exception):
    """Expected, user-facing error (missing file, bad input, sync problem)."""


# =============================================================================
# 4. FILE DISCOVERY AND LOG LOADING
# =============================================================================
def find_file(data_dir: Path, stems: list[str], exts: tuple[str, ...]) -> Path | None:
    """Return the first file matching stem+ext (case-insensitive), else None."""
    files = {f.name.lower(): f for f in data_dir.iterdir() if f.is_file()}
    for stem in stems:
        for ext in exts:
            hit = files.get(f"{stem}{ext}".lower())
            if hit is not None:
                return hit
    return None


def find_model(cli_model: Path | None, data_dir: Path) -> Path:
    """Model lookup order: --model, data dir, script dir."""
    candidates = ([cli_model] if cli_model else []) + [
        data_dir / MODEL_FILENAME, Path(__file__).resolve().parent / MODEL_FILENAME]
    for c in candidates:
        if c.is_file():
            return c
    raise AnalysisError(f"{MODEL_FILENAME} not found (looked in: "
                        f"{', '.join(str(c) for c in candidates)})")


def map_alarm_type(raw) -> str:
    """Map a logged Combination_Type to thesis notation (V, H, A, VH, VA, HA, VHA)."""
    if pd.isna(raw):
        raise AnalysisError("empty Combination_Type entry")
    text = str(raw).strip().lower()
    if text == "multimodal":
        return "VHA"
    # Fixed order V -> H -> A yields VH, VA, HA, VHA
    code = "".join(c for c, key in (("V", "visual"), ("H", "haptic"), ("A", "auditory"))
                   if key in text)
    if code not in ALARM_CODES:
        raise AnalysisError(f"unrecognised Combination_Type '{raw}'")
    return code


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV (separator sniffed: ',' or ';' from German Excel) or Excel file."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    try:
        return pd.read_excel(path)
    except ImportError as exc:
        raise AnalysisError("reading .xlsx needs openpyxl (pip install openpyxl)") from exc


def load_log(path: Path, task: str) -> pd.DataFrame:
    """Read one alarm log; alarm_index = 0-based row in the file (as in the anchors)."""
    df = read_table(path)
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_LOG_COLUMNS if c not in df.columns]
    if missing:
        raise AnalysisError(f"{path.name}: missing column(s) {missing} "
                            f"(found: {list(df.columns)})")
    if df.empty:
        raise AnalysisError(f"{path.name}: log contains no rows")
    try:
        timestamps = pd.to_datetime(df["Timestamp"])
    except (ValueError, TypeError) as exc:
        raise AnalysisError(f"{path.name}: cannot parse 'Timestamp' ({exc})") from exc
    if timestamps.isna().any():
        raise AnalysisError(f"{path.name}: empty Timestamp in row(s) "
                            f"{list(timestamps[timestamps.isna()].index)}")
    # Excel re-saving often truncates timestamps to minutes -> sync would be wrong
    if len(timestamps) > 1 and (timestamps.dt.second.eq(0)
                                & timestamps.dt.microsecond.eq(0)).all():
        raise AnalysisError(f"{path.name}: timestamps have no seconds - file was "
                            f"probably re-saved by Excel; use the original log")
    try:
        types = df["Combination_Type"].map(map_alarm_type)
    except AnalysisError as exc:
        raise AnalysisError(f"{path.name}: {exc}") from exc

    return pd.DataFrame({"task": task, "alarm_index": df.index,
                         "alarm_type": types, "timestamp": timestamps})


# =============================================================================
# 5. SYNCHRONISATION
# =============================================================================
def sync_logs(logs: dict[str, pd.DataFrame], anchors: list[tuple]) -> pd.DataFrame:
    """Map log timestamps onto the video timeline.

    offset = video_time - log_time per anchor. A task with own anchors uses
    the median of those; a task without anchors borrows the median of all
    anchors (valid only if the video is one continuous recording).
    """
    if not anchors:
        raise AnalysisError("no anchor configured")
    ref = min(df["timestamp"].min() for df in logs.values())

    rows = []
    for task, idx, video_s in anchors:
        if task not in logs:
            raise AnalysisError(f"anchor refers to task '{task}', but no log was found")
        df = logs[task]
        if not 0 <= idx < len(df):
            raise AnalysisError(f"anchor index {idx} out of range for {task} log "
                                f"(0..{len(df) - 1})")
        log_s = (df["timestamp"].iloc[idx] - ref).total_seconds()
        rows.append({"task": task, "alarm_index": idx, "video_s": float(video_s),
                     "offset_s": float(video_s) - log_s})
    anchors_df = pd.DataFrame(rows)

    parts = []
    print("  Synchronisation:")
    for task, df in logs.items():
        own = anchors_df[anchors_df["task"] == task]
        source = own if not own.empty else anchors_df
        offset = float(source["offset_s"].median())
        if own.empty:
            print(f"    {task}: no own anchor -> uses anchor(s) of "
                  f"{sorted(set(anchors_df['task']))} (assumes continuous recording)")
        else:
            print(f"    {task}: {len(own)} own anchor(s)")
        if len(own) > 1:
            for a in own.itertuples(index=False):
                residual = offset - a.offset_s   # predicted - observed
                flag = "  <-- exceeds tolerance" if abs(residual) > SYNC_TOLERANCE_S else ""
                print(f"      alarm {a.alarm_index}: residual {residual:+.3f} s{flag}")
        d = df.copy()
        d["onset_video_s"] = (d["timestamp"] - ref).dt.total_seconds() + offset
        parts.append(d)

    combined = pd.concat(parts, ignore_index=True)
    return combined.sort_values("onset_video_s").reset_index(drop=True)


# =============================================================================
# 6. VIDEO PROBING AND ALARM SELECTION
# =============================================================================
def probe_video(path: Path) -> tuple[float, float, int, int]:
    """Return (fps, duration_s, width_px, height_px)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise AnalysisError(f"OpenCV cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0 or n_frames <= 0:
        raise AnalysisError(f"cannot read fps/frame count from {path.name}")
    return fps, n_frames / fps, width, height


def select_alarms(alarms: pd.DataFrame, excluded: list[str],
                  duration_s: float) -> pd.DataFrame:
    """Drop excluded tasks and alarms whose window is not inside the video."""
    keep, skipped = [], {}
    for idx, row in alarms.iterrows():
        if row["task"] in excluded:
            reason = "task excluded"
        elif row["onset_video_s"] < 0:
            reason = "onset before video start"
        elif row["onset_video_s"] + POST_ONSET_S > duration_s:
            reason = "window exceeds video end"
        else:
            reason = None
        if reason:
            skipped[(row["task"], reason)] = skipped.get((row["task"], reason), 0) + 1
            continue
        if row["onset_video_s"] < PRE_ONSET_S:
            print(f"    note: {row['task']} alarm {row['alarm_index']}: "
                  f"baseline truncated by video start")
        keep.append(idx)

    for (task, reason), n in skipped.items():
        print(f"    skipped {n} {task} alarm(s): {reason}")
    selected = alarms.loc[keep].copy()
    if selected.empty:
        raise AnalysisError("no alarm falls inside the video - check the anchors")
    # Running number per task in video order (used in panel titles)
    selected["alarm_no"] = selected.groupby("task").cumcount() + 1
    for task, g in selected.groupby("task"):
        print(f"    {task}: {len(g)} alarm(s), video "
              f"{g['onset_video_s'].min():.1f}-{g['onset_video_s'].max():.1f} s")
    return selected.reset_index(drop=True)


# =============================================================================
# 7. FEATURE EXTRACTION (MEDIAPIPE)
# =============================================================================
def merge_intervals(intervals: list[tuple[float, float]]) -> list[list[float]]:
    """Merge overlapping time intervals so each frame is analysed once."""
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def extract_frame_features(video_path: Path, model_path: Path,
                           intervals: list[tuple[float, float]], fps: float) -> pd.DataFrame:
    """Run the Face Landmarker on all frames inside the analysis intervals."""
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise AnalysisError("mediapipe is not installed (pip install mediapipe)") from exc

    merged = merge_intervals(intervals)
    last_end = merged[-1][1]
    vision = mp.tasks.vision
    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )

    rows: list[dict] = []
    frame_idx, last_ts_ms, seg = -1, -1, 0
    cap = cv2.VideoCapture(str(video_path))
    print(f"  Face Landmarker on {len(merged)} interval(s)...")
    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            # grab() without decoding is cheap -> frames outside windows are skipped fast
            while cap.grab():
                frame_idx += 1
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                # Fallback for containers that report 0 ms: derive time from frame index
                t_s = pos_ms / 1000.0 if (pos_ms > 0 or frame_idx == 0) else frame_idx / fps
                if t_s > last_end:
                    break
                while seg < len(merged) and t_s > merged[seg][1]:
                    seg += 1
                if seg >= len(merged) or t_s < merged[seg][0]:
                    continue

                ok, frame = cap.retrieve()
                row = {"frame_idx": frame_idx, "time_s": t_s, "face_detected": False,
                       "eye_opening": np.nan, "nose_x": np.nan, "nose_y": np.nan}
                if ok:
                    # VIDEO mode requires strictly increasing timestamps
                    ts_ms = max(int(round(t_s * 1000)), last_ts_ms + 1)
                    last_ts_ms = ts_ms
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = landmarker.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
                    if result.face_landmarks:
                        lm = result.face_landmarks[0]
                        row.update(
                            face_detected=True,
                            eye_opening=abs(lm[LM_UPPER_LID].y - lm[LM_LOWER_LID].y),
                            nose_x=lm[LM_NOSE_TIP].x,
                            nose_y=lm[LM_NOSE_TIP].y,
                        )
                rows.append(row)
                if len(rows) % PROGRESS_EVERY == 0:
                    print(f"    {len(rows)} frames analysed (video t = {t_s:.1f} s)")
    finally:
        cap.release()

    frames = pd.DataFrame(rows, columns=["frame_idx", "time_s", "face_detected",
                                         "eye_opening", "nose_x", "nose_y"])
    if frames.empty:
        raise AnalysisError("no frames analysed - check video and anchors")

    # Head movement only between directly consecutive frames with a detected face.
    # Across detection gaps the value is NaN (the old script compared against the
    # last *detected* frame, which can inflate single-frame peaks).
    consecutive = frames["frame_idx"].diff().eq(1)
    displacement = np.hypot(frames["nose_x"].diff(), frames["nose_y"].diff())
    frames["head_movement"] = displacement.where(consecutive)

    print(f"    done: {len(frames)} frames, face detected in "
          f"{frames['face_detected'].mean() * 100:.1f} %")
    return frames


# =============================================================================
# 8. WINDOW TABLE AND SUMMARY
# =============================================================================
FRAME_COLUMNS = ["pid", "task", "alarm_index", "alarm_type", "time_rel_s",
                 "eye_opening", "head_movement", "alarm_no", "frame_idx", "face_detected"]


def build_window_table(frames: pd.DataFrame, alarms: pd.DataFrame, pid: str) -> pd.DataFrame:
    """Cut one window [-PRE, +POST] per alarm; frames may belong to overlapping windows."""
    parts = []
    for a in alarms.itertuples(index=False):
        rel = frames["time_s"] - a.onset_video_s
        win = frames.loc[(rel >= -PRE_ONSET_S) & (rel <= POST_ONSET_S)]
        win = win.sort_values("frame_idx").copy()
        if win.empty:
            print(f"    warn: {a.task} alarm {a.alarm_index}: no frames in window")
            continue
        win["time_rel_s"] = win["time_s"] - a.onset_video_s
        # First frame of every window has no predecessor inside the window
        win.loc[win.index[0], "head_movement"] = np.nan
        win = win.assign(pid=pid, task=a.task, alarm_index=a.alarm_index,
                         alarm_type=a.alarm_type, alarm_no=a.alarm_no)
        parts.append(win[FRAME_COLUMNS])
    if not parts:
        raise AnalysisError("no analysis window contains frames")
    return pd.concat(parts, ignore_index=True)


def summarize_alarms(win: pd.DataFrame, alarms: pd.DataFrame, fps: float,
                     frame_w: int, frame_h: int) -> pd.DataFrame:
    """Per alarm: head-movement maximum in the response window (0..+POST)."""
    records = []
    for (task, alarm_no), g in win.groupby(["task", "alarm_no"], sort=False):
        post = g[g["time_rel_s"] >= 0]
        pre = g[g["time_rel_s"] < 0]
        hm = post["head_movement"]
        if hm.notna().any():
            i_max = hm.idxmax()
            max_hm, t_max = hm.loc[i_max], post.loc[i_max, "time_rel_s"]
        else:
            max_hm, t_max = np.nan, np.nan
        onset = alarms.loc[(alarms["task"] == task) & (alarms["alarm_no"] == alarm_no),
                           "onset_video_s"].iloc[0]
        records.append({
            "pid": g["pid"].iloc[0], "task": task, "alarm_no": alarm_no,
            "alarm_index": g["alarm_index"].iloc[0], "alarm_type": g["alarm_type"].iloc[0],
            "onset_video_s": round(float(onset), 3),
            "max_head_movement_post": max_hm,
            "time_of_max_s": t_max,
            # Same metric in the 1-s baseline, to judge the peak against tracking jitter
            "max_head_movement_pre": pre["head_movement"].max(),
            "n_frames_post": len(post),
            "n_face_detected_post": int(post["face_detected"].sum()),
            # Recording properties needed to interpret the normalized units
            "fps": round(fps, 2), "frame_w_px": frame_w, "frame_h_px": frame_h,
        })
    return pd.DataFrame(records)


# =============================================================================
# 9. PLOTTING
# =============================================================================
def compute_ylims(win: pd.DataFrame, eye_override, head_override):
    """Common y-limits for all panels (and both tasks) so amplitudes are comparable."""
    eye = win["eye_opening"].dropna()
    head = win["head_movement"].dropna()
    if eye.empty:
        raise AnalysisError("no face detected in any analysis window")
    if eye_override:
        eye_lim = tuple(eye_override)
    else:
        # Lid distance has a true zero (closed eye): always start the axis at 0
        eye_lim = (0.0, float(eye.max()) * 1.05)
    if head_override:
        head_lim = tuple(head_override)
    else:
        head_lim = (0.0, (float(head.max()) if not head.empty else 1e-3) * 1.08)
    return eye_lim, head_lim


def plot_task(win_task: pd.DataFrame, pid: str, task: str, eye_lim, head_lim,
              out_dir: Path, ncols: int | None, fps: float, show: bool) -> Path:
    """One panel per alarm; one figure per distraction task. Returns the PDF path."""
    alarms = (win_task[["alarm_no", "alarm_type"]].drop_duplicates()
              .sort_values("alarm_no").reset_index(drop=True))
    n = len(alarms)
    ncols = min(ncols or DEFAULT_MAX_COLS, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, sharex=True, sharey=True, squeeze=False, layout="constrained",
        figsize=(PANEL_WIDTH_IN * ncols + 1.6, PANEL_HEIGHT_IN * nrows + 1.0))
    flat = axes.ravel()

    for i, a in enumerate(alarms.itertuples(index=False)):
        ax = flat[i]
        d = win_task[win_task["alarm_no"] == a.alarm_no]
        first_col = i % ncols == 0
        # Right-axis scale only in the rightmost grid column. A label on the last
        # panel of a partial row would widen its whole column (gap in row 1).
        last_col = i % ncols == ncols - 1
        # sharex hides x tick labels unless the panel is the bottom one in its column
        bottom_in_col = i + ncols >= n

        ax.axvspan(0, POST_ONSET_S, color=WINDOW_SHADE, lw=0, zorder=0)
        ax.axvline(0, color=ONSET_COLOR, ls="--", lw=1.0, zorder=1)
        ax.plot(d["time_rel_s"], d["eye_opening"], color=EYE_COLOR, lw=1.1, zorder=2)
        ax.set_xlim(-PRE_ONSET_S, POST_ONSET_S)
        ax.set_ylim(*eye_lim)
        ax.tick_params(axis="y", colors=EYE_COLOR)
        ax.grid(alpha=0.3)
        ax.set_title(f"{a.alarm_type} (alarm {a.alarm_no})", fontsize=10)
        if bottom_in_col:
            ax.tick_params(axis="x", labelbottom=True)
        if first_col:
            ax.set_ylabel(EYE_AXIS_LABEL, color=EYE_COLOR, fontsize=9)

        ax2 = ax.twinx()
        ax2.plot(d["time_rel_s"], d["head_movement"], color=HEAD_COLOR, lw=1.1)
        ax2.set_ylim(*head_lim)
        if last_col:
            ax2.tick_params(axis="y", colors=HEAD_COLOR)
            ax2.set_ylabel(HEAD_AXIS_LABEL, color=HEAD_COLOR, fontsize=9)
        else:
            # Inner panels: no right-hand tick marks, numbers or label
            ax2.tick_params(axis="y", right=False, labelright=False)

    for ax in flat[n:]:
        ax.axis("off")

    fig.supxlabel(X_LABEL)
    handles = [
        Line2D([], [], color=EYE_COLOR, label=EYE_NAME),
        Line2D([], [], color=HEAD_COLOR, label=HEAD_NAME),
        Line2D([], [], color=ONSET_COLOR, ls="--", label="Alarm onset"),
        Patch(color=WINDOW_SHADE, label=f"Response window (0-{POST_ONSET_S:g} s)"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False,
               fontsize=9)
    # Unit note so a value like 0.004 is interpretable without the caption
    fig.text(0.0, -0.01,
             f"{pid}, {task} distraction. Normalized image coordinates: 1.0 = full frame "
             f"height (eye) / frame width or height (head). Head movement per frame "
             f"at {fps:.0f} fps.",
             ha="left", va="top", fontsize=7.5, color="0.35")

    stem = out_dir / f"{pid}_{task}_reaction_windows"
    fig.savefig(stem.with_suffix(".png"), dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return stem.with_suffix(".pdf")


# =============================================================================
# 10. PER-PARTICIPANT PIPELINE
# =============================================================================
def process_participant(fid: str, cfg: dict, args, model: Path) -> pd.DataFrame:
    """Run the full pipeline for one participant; returns its alarm summary."""
    pid = cfg.get("thesis_id", fid)
    excluded = [] if args.include_excluded else list(cfg.get("exclude", []))
    print(f"\n=== {fid} -> {pid} ===")

    video = find_file(args.data_dir, [s.format(fid=fid) for s in VIDEO_STEMS], VIDEO_EXTS)
    if video is None:
        raise AnalysisError(f"no video found for {fid} (expected '{fid} - video.mp4')")

    logs: dict[str, pd.DataFrame] = {}
    anchor_tasks = {a[0] for a in cfg.get("anchors", [])}
    for task, suffix in TASK_SUFFIXES.items():
        path = find_file(args.data_dir, [LOG_STEM.format(fid=fid, suffix=suffix)], LOG_EXTS)
        if path is None:
            if task in anchor_tasks:
                raise AnalysisError(f"anchor log for task '{task}' not found")
            print(f"  note: no {task} log found - task skipped")
            continue
        logs[task] = load_log(path, task)
        print(f"  {task:>8}: {path.name} ({len(logs[task])} alarms)")

    alarms = sync_logs(logs, cfg.get("anchors", []))
    fps, duration_s, frame_w, frame_h = probe_video(video)
    print(f"  Video: {video.name}, {fps:.2f} fps, {frame_w}x{frame_h} px, {duration_s:.1f} s")
    alarms = select_alarms(alarms, excluded, duration_s)

    intervals = [(t - PRE_ONSET_S, t + POST_ONSET_S) for t in alarms["onset_video_s"]]
    frames = extract_frame_features(video, model, intervals, fps)

    win = build_window_table(frames, alarms, pid)
    summary = summarize_alarms(win, alarms, fps, frame_w, frame_h)
    win.to_csv(args.out_dir / f"{pid}_frames.csv", index=False)
    summary.to_csv(args.out_dir / f"{pid}_alarm_summary.csv", index=False)

    eye_lim, head_lim = compute_ylims(win, args.ylim_eye, args.ylim_head)
    for task in sorted(win["task"].unique()):
        win_task = win[win["task"] == task]
        if args.types:
            win_task = win_task[win_task["alarm_type"].isin(args.types)]
        if win_task.empty:
            print(f"  no selected alarms to plot for '{task}'")
            continue
        pdf = plot_task(win_task, pid, task, eye_lim, head_lim, args.out_dir,
                        args.ncols, fps, args.show)
        print(f"  figure: {pdf.name} (+ .png)")
    return summary


# =============================================================================
# 11. COMMAND LINE
# =============================================================================
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch MediaPipe reaction analysis "
                                            "(preliminary study).")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Folder with videos and logs (default: folder of this script)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output folder (default: <data-dir>/reaction_analysis)")
    p.add_argument("--only", nargs="+", metavar="ID",
                   help="Process only these file IDs, e.g. --only P1 P4")
    p.add_argument("--model", type=Path, default=None,
                   help=f"Path to {MODEL_FILENAME} (default: data dir, then script dir)")
    p.add_argument("--types", nargs="+", choices=ALARM_CODES,
                   help="Plot only these alarm types (CSV output always contains all)")
    p.add_argument("--ncols", type=int, default=None,
                   help=f"Panels per row (default: {DEFAULT_MAX_COLS})")
    p.add_argument("--ylim-eye", nargs=2, type=float, metavar=("MIN", "MAX"),
                   help="Fixed y-limits for eye opening (e.g. identical for all participants)")
    p.add_argument("--ylim-head", nargs=2, type=float, metavar=("MIN", "MAX"),
                   help="Fixed y-limits for head movement")
    p.add_argument("--include-excluded", action="store_true",
                   help="Also analyse tasks listed under 'exclude'")
    p.add_argument("--show", action="store_true", help="Show figures interactively")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    # Default: the folder containing this script (independent of the working directory)
    args.data_dir = args.data_dir or Path(__file__).resolve().parent
    if not args.data_dir.is_dir():
        print(f"ERROR: data folder not found: {args.data_dir}", file=sys.stderr)
        return 1
    if args.ncols is not None and args.ncols < 1:
        print("ERROR: --ncols must be >= 1", file=sys.stderr)
        return 1
    args.out_dir = args.out_dir or args.data_dir / "reaction_analysis"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ids = args.only or list(PARTICIPANTS)
    unknown = [i for i in ids if i not in PARTICIPANTS]
    if unknown:
        print(f"ERROR: not configured in PARTICIPANTS: {unknown}", file=sys.stderr)
        return 1
    try:
        model = find_model(args.model, args.data_dir)
    except AnalysisError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summaries, failed = [], {}
    for fid in ids:
        try:
            summaries.append(process_participant(fid, PARTICIPANTS[fid], args, model))
        except AnalysisError as exc:   # one participant failing does not stop the batch
            print(f"  ERROR ({fid}): {exc}", file=sys.stderr)
            failed[fid] = str(exc)

    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
        combined.to_csv(args.out_dir / "all_alarm_summary.csv", index=False)
        print("\nHead movement maximum per alarm (response window 0-"
              f"{POST_ONSET_S:g} s):")
        cols = ["pid", "task", "alarm_no", "alarm_type", "max_head_movement_post",
                "time_of_max_s", "max_head_movement_pre", "n_face_detected_post"]
        print(combined[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nOutputs in: {args.out_dir.resolve()}")
    if failed:
        print(f"Failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())