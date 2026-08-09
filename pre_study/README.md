# Pre-Study Scripts

Wizard-of-Oz and calibration scripts used during the pilot/pre-study phase, prior to the main
study (N=30). These scripts are **finished and were not modified** for the main study; the main
study uses its own scenario/bridge scripts (see [`../main_study`](../main_study)).

## Contents

| File | Purpose |
|---|---|
| `sim_TTC.py` | CARLA playback tool for calibrating/testing specific time-to-collision (TTC) values: spawns a vehicle at a distance computed from a manually entered target TTC and constant speed, then plays back the approach from the pedestrian's point of view |
| `sim_urgency.py` | Pilot study script capturing participants' self-assessed urgency: drives a vehicle toward a fixed pedestrian viewpoint at constant speed; the experimenter logs distance/TTC at keypress moments, saved per participant to CSV |
| `sim_alarm.py` | Wizard-of-Oz control script — experimenter manually triggers one of 7 modality combinations (visual/haptic/auditory, single or combined) via keypress, sent to the phone over UDP; logs each triggered modality with a timestamp for later analysis |

## Requirements

- CARLA 0.9.15 (Python API)
- Python 3.10.11
- Python packages: `pygame` (`sim_TTC.py`, `sim_urgency.py`), `pynput` (`sim_alarm.py`)
- CARLA map: `Town10HD`

## Usage

**`sim_TTC.py`** — run with the CARLA server active:
```bash
python sim_TTC.py
```
Prompts for a target TTC (seconds) in the terminal, spawns the scene accordingly.
Controls: `S` = start playback, `R` = reset / enter a new TTC.

**`sim_urgency.py`** — run with the CARLA server active:
```bash
python sim_urgency.py
```
Prompts for a participant ID, then logs to `logs_urgency/ttc_log_<participant_id>.csv`.
Controls: `S` = start vehicle, `Space` = log current distance/TTC, `R` = save and reset.

**`sim_alarm.py`** — run standalone (communicates with the phone app over UDP):
```bash
python sim_alarm.py
```
Prompts for a participant ID, then logs to `logs_alarm/modality_selection_<participant_id>.csv`.
Controls: keys `1`–`7` trigger a modality combination (see table below), `q` = quit.

| Key | Modality |
|---|---|
| 1 | Visual only |
| 2 | Haptic only |
| 3 | Auditory only |
| 4 | Visual + Haptic |
| 5 | Visual + Auditory |
| 6 | Haptic + Auditory |
| 7 | Multimodal (all three) |

## Network configuration (`sim_alarm.py`)

Uses the same port scheme later reused by `main_bridge.py` in the main study:

| Port | Direction | Purpose |
|---|---|---|
| 5006 | Phone → Script | Phone connection detection |
| 5007 | Script → Phone | Alarm command (Flutter-side receiver) |
| 5008 | Script → Phone | Alarm command (native Kotlin overlay service) |

## Related

Sends alarm commands to the Android app in [`../smombie_bridge`](../smombie_bridge) over UDP.
