#!/usr/bin/env python3
"""
run_analysis.py — V2P Study | Batch RT Analysis Wrapper
========================================================

Sucht automatisch alle P??_A.mp4 / P??_B.mp4 in videos/ und analysiert sie.
A = Adaptive, B = Baseline — keine weitere Konfiguration nötig.

Ordnerstruktur (Wrapper liegt in main/):
    main/
      run_analysis.py
      analyze_rt_video.py
      videos/
        P01_A.mp4
        P01_B.mp4
        P04_B.mp4        (fehlende Blöcke werden automatisch übersprungen)
        alert_tone.wav
      logs/
        P01/
          run_order.csv
          ...

Aufruf:
    python run_analysis.py                    # alle Teilnehmer
    python run_analysis.py --pid P01          # nur P01
    python run_analysis.py --pid P01 P04      # P01 und P04
    python run_analysis.py --dry_run          # nur Kommandos anzeigen
    python run_analysis.py --debug_trial t3   # Velocity-Plot für t3
"""

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd

# ─── KONFIGURATION ────────────────────────────────────────────────────────────
ALARM_THRESHOLD = 0.07   # Template-Match Sensitivität (niedriger = sensitiver)
REACTION_RATIO  = 0.35   # Reaktionsschwelle (Anteil des Baseline-Speeds)
OUTPUT_CSV      = "rt_results.csv"

# ─── PFADE (relativ zu main/) ─────────────────────────────────────────────────
HERE        = Path(__file__).parent
VIDEOS_DIR  = HERE / "videos"
LOGS_DIR    = HERE / "logs"
ALARM_WAV   = VIDEOS_DIR / "alert_tone.wav"
ANALYZE_PY  = VIDEOS_DIR / "analyze_rt_video.py"

BLOCK_TO_CONDITION = {"A": "Adaptive", "B": "Baseline"}


def run_block(pid: str, block: str, debug_trial: str | None, dry_run: bool) -> bool:
    condition = BLOCK_TO_CONDITION[block]
    video     = VIDEOS_DIR / f"{pid}_{block}.mp4"

    # Logs-Ordner: Videos heißen P01, Logs-Ordner heißen P1 (ohne führende Null)
    pid_short = "P" + str(int(pid[1:]))   # "P01" → "P1"
    logs = LOGS_DIR / pid_short
    if not logs.exists():
        logs = LOGS_DIR / pid             # Fallback: mit führender Null

    if not video.exists():
        print(f"[SKIP]  {pid}_{block}.mp4 — nicht gefunden")
        return False
    if not logs.exists():
        print(f"[SKIP]  {pid} logs/ — nicht gefunden ({LOGS_DIR / pid_short})")
        return False

    cmd = [
        sys.executable, str(ANALYZE_PY),
        "--video",           str(video),
        "--logs_dir",        str(logs),
        "--pid",             pid_short,
        "--condition",       condition,
        "--auto_sync",
        "--alarm_audio",     str(ALARM_WAV),
        "--alarm_threshold", str(ALARM_THRESHOLD),
        "--reaction_ratio",  str(REACTION_RATIO),
        "--out",             OUTPUT_CSV,
        "--append",
    ]
    if debug_trial:
        cmd += ["--debug_trial", debug_trial]

    print(f"\n{'─'*60}")
    print(f"[RUN]   {pid} | Block {block} ({condition})")
    print(f"{'─'*60}")

    if dry_run:
        print("        " + " ".join(cmd))
        return True

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid",         nargs="*", default=None,
                    help="Nur diese Teilnehmer (z.B. --pid P01 P04)")
    ap.add_argument("--debug_trial", default=None,
                    help="Run-ID für Velocity-Debug-Plot (z.B. t3)")
    ap.add_argument("--dry_run",     action="store_true")
    args = ap.parse_args()

    # Voraussetzungen prüfen
    if not ALARM_WAV.exists():
        sys.exit(f"[ERROR] alert_tone.wav nicht gefunden: {ALARM_WAV}")
    if not ANALYZE_PY.exists():
        sys.exit(f"[ERROR] analyze_rt_video.py nicht gefunden: {ANALYZE_PY}")

    # Alle P??_A/B.mp4 Videos finden
    all_videos = sorted(VIDEOS_DIR.glob("P??_[AB].mp4"))
    if not all_videos:
        sys.exit(f"[ERROR] Keine P??_A/B.mp4 Videos in {VIDEOS_DIR}")

    # Auf gewünschte PIDs filtern
    if args.pid:
        pids = set(args.pid)
        all_videos = [v for v in all_videos
                      if v.stem.rsplit("_", 1)[0] in pids]
        if not all_videos:
            sys.exit(f"[ERROR] Keine Videos für: {args.pid}")

    # Alten Output löschen wenn alle Teilnehmer verarbeitet werden
    out = Path(OUTPUT_CSV)
    if out.exists() and not args.pid:
        out.unlink()
        print(f"[INFO]  Alte {OUTPUT_CSV} gelöscht\n")

    ok = skip = 0
    for video in all_videos:
        pid, block = video.stem.rsplit("_", 1)
        if run_block(pid, block, args.debug_trial, args.dry_run):
            ok += 1
        else:
            skip += 1

    print(f"\n{'═'*60}")
    print(f"Fertig: {ok} erfolgreich, {skip} übersprungen/fehlgeschlagen")
    if ok:
        print(f"Ergebnisse: {OUTPUT_CSV}")

    # ── Heading-Reliability-Check ─────────────────────────
    if ok:
        try:
            rt = pd.read_csv(OUTPUT_CSV)
            if "heading_norm" in rt.columns:
                flag = rt["heading_norm"].notna() & (rt["heading_norm"] < 0.05)
                if flag.any():
                    print(f"\n{'─'*60}")
                    print(f"[QC] {flag.sum()} Trial(s) mit unsicherer Baseline-Richtung "
                          f"(heading_norm < 0.05) — manuell gegenprüfen:")
                    print(rt[flag][["participant_id", "condition", "trial_id",
                                     "reaction_type", "rt_s"]].to_string(index=False))
            else:
                print("\n[QC] Spalte 'heading_norm' fehlt in rt_results.csv — "
                      "analyze_rt_video.py Patch noch nicht eingebaut?")
        except Exception as e:
            print(f"\n[QC] Konnte {OUTPUT_CSV} nicht für den Heading-Check lesen: {e}")


if __name__ == "__main__":
    main()