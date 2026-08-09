# Main Study Scripts

Simulation and communication scripts used to run the main study (N=30, within-subject,
Adaptive System vs. Fixed Baseline).

## Contents

| File | Purpose |
|---|---|
| `run_sim.py` | Runs the CARLA driving scenario logic: scenario sequencing, trial triggering (Target/Catch/Safe), and alarm-timing state machine |
| `main_bridge.py` | Handles the UDP bridge between the host laptop (CARLA) and the smartphone running `smombie_bridge` |

## Requirements

- CARLA [version]
- Python [version]
- `keyboard` library (QWERTZ global keypress handling, Y/X/C control scheme)
- [Insert: any other dependencies, e.g. requirements.txt]

## Study design summary

- N=30 participants, within-subject design
- Two counterbalanced blocks: Adaptive System vs. Fixed Baseline
- 9 trials per block (5 Target, 2 Catch, 2 Safe)
- Counterbalancing: odd participant IDs = Adaptive first, even PIDs = Baseline first
- Two-stage warning escalation: early "f" alarm at 5.0 s TTC → full multimodal (VHA) alert at
  2.5 s TTC, with a `MIN_TOTAL_TTC` floor introducing a systematic, logged `TimingError`

## Usage

[Insert: exact command(s) to launch a session, e.g. `python run_sim.py --participant P01 --condition adaptive`]

## Related

Communicates with the Android app in [`../smombie_bridge`](../smombie_bridge) over UDP unicast
(not broadcast), via the phone's mobile hotspot.
