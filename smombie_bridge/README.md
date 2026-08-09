# smombie_bridge — V2P Warning App (Android)

Flutter/Kotlin Android app that runs on the participant's smartphone (Samsung Galaxy S25) and
renders the Vehicle-to-Pedestrian (V2P) warning alerts sent from the CARLA simulation host over
UDP.

Used across **both** the pre-study and the main study; the overlay/alert logic was extended for
the main study to support the two-stage escalation and the adaptive vs. baseline conditions.

## Key files

| File | Purpose |
|---|---|
| `lib/main.dart` | App entry point, UI, and state management |
| `android/app/src/main/kotlin/.../MainActivity.kt` | Main Android activity |
| `android/app/src/main/kotlin/.../OverlayHelper.kt` | Renders the overlay UI (visual alert: red bar, directional arrow) on top of other apps |
| `android/app/src/main/kotlin/.../OverlayReceiver.kt` | Receives overlay trigger broadcasts |
| `android/app/src/main/kotlin/.../UdpOverlayService.kt` | UDP listener — receives alarm commands from the host laptop (`sim_alarm.py` in the pre-study, `main_bridge.py` in the main study) |
| `android/app/src/main/kotlin/.../SmombieBackgroundService.kt` | Keeps the UDP listener and overlay service alive in the background |

## Alert logic

Two-stage escalation, driven by commands received over UDP from the host laptop:

1. **Early warning ("f" alarm)** — visual red bar + silent haptic vibration only (no audio)
2. **Full alert (VHA)** — visual + haptic + audio; adaptive condition adds a directional arrow

Alert components map to single-character UDP message flags handled in `UdpOverlayService.kt`:
`v` = visual overlay, `a` = audio (3× beep pulse), `h` = haptic (3× vibration pulse).

## Build & run

```bash
flutter clean
flutter pub get
flutter build apk --release
```

Output: `build/app/outputs/flutter-apk/app-release.apk`

A pre-built APK for the exact version used in the main study is attached to the corresponding
[GitHub Release] [Insert link once created].

## Network configuration

- Communication: UDP **unicast** (not broadcast) between the CARLA host laptop and the phone
- Connection: phone's mobile hotspot (host laptop connects to the phone's hotspot)
- **Port: 5008** (`DatagramSocket(5008)` in `UdpOverlayService.kt`)

## Used by

- `pre_study/sim_alarm.py` — Wizard-of-Oz alarm triggering (pre-study)
- `main_study/main_bridge.py` — automated UDP bridge (main study)

## Requirements / build environment

- Flutter 3.41.6 (stable channel) • Dart 3.11.4
- Android Gradle Plugin 8.11.1
- Kotlin 2.2.20
- `minSdk` / `targetSdk` / `compileSdk`: inherited from Flutter tooling defaults (not hardcoded
  in `build.gradle.kts`)
- Tested on: Samsung Galaxy S25
