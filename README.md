# Master Thesis V2P — Context-Aware Vehicle-to-Pedestrian Warning System

Code repository for the Master's thesis *"Balancing Safety and Acceptance: A Context-Aware
Approach to V2P Interventions for Distracted Pedestrians"* (UX Design, Technische Hochschule
Ingolstadt, 2026).

The project investigates a smartphone-based Vehicle-to-Pedestrian (V2P) warning system,
comparing an **adaptive, context-aware** alert strategy against a **fixed multimodal baseline**
in a within-subject user study (N=30).

## Repository structure

| Folder | Description |
|---|---|
| [`pre_study/`](./pre_study) | Wizard-of-Oz scripts used for the pilot/pre-study phase |
| [`main_study/`](./main_study) | Simulation and bridge scripts used to run the main study (N=30) |
| [`smombie_bridge/`](./smombie_bridge) | Flutter Android app running on the participant's smartphone, receiving V2P alerts over UDP (used in both the pre-study and main study) |

Each folder contains its own README with setup and usage instructions.

## System overview

The study uses a CARLA-based driving simulation on a host laptop, communicating over a UDP
link (via the phone's mobile hotspot) with a Flutter app running on a Samsung Galaxy S25.
The app renders a two-stage escalating alert:

1. **Early warning** ("f" alarm) — visual red bar + silent haptic, triggered at 5.0 s TTC
2. **Full multimodal alert** (VHA) — visual + haptic + audio, triggered at 2.5 s TTC (adaptive
   condition additionally shows a directional arrow)

## Author

Vanessa Scherer
