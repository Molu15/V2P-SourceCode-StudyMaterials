"""
calibrate_camera.py
─────────────────────────────────────────────────────────
Interactive tool to determine CAM_POS / CAM_ROT for the fixed
projector setup (and any other reference points, e.g. PED_REFERENCE
or vehicle spawn/path waypoints in SCENARIOS).

USAGE
  1. Start CARLA with rendering enabled (NOT -RenderOffScreen).
  2. Run this script:  python calibrate_camera.py
  3. In the CARLA window, free-fly the spectator with the mouse +
     WASD (and Q/E for up/down) to the desired viewpoint.
  4. Switch to THIS terminal and press:
       [c]  -> capture as CAM_POS / CAM_ROT
       [p]  -> capture as PED_REFERENCE (carla.Transform)
       [v]  -> capture as a custom-named waypoint (you'll be
               prompted for a label, e.g. "t1_left_spawn")
       [l]  -> quit
  5. Copy the printed snippet(s) into run_sim.py.

All captures are also appended to camera_calibration.json so
nothing gets lost between sessions.

NOTE: The spectator can only be free-flown manually while CARLA's
own window has focus and no other client is overriding the
spectator transform (e.g. stop any running run_sim.py first).
"""

import json
import os
import time

import carla
from pynput import keyboard

# ─────────────────────────────────────────────────────────
HOST, PORT  = 'localhost', 2000
OUTPUT_FILE = "camera_calibration.json"
POLL_HZ     = 5   # live readout refresh rate
# ─────────────────────────────────────────────────────────

client = carla.Client(HOST, PORT)
client.set_timeout(10.0)
world      = client.get_world()
spectator  = world.get_spectator()

# ── Load previous captures (if any) ─────────────────────────
captured = {}
if os.path.exists(OUTPUT_FILE):
    try:
        with open(OUTPUT_FILE, "r") as f:
            captured = json.load(f)
    except (json.JSONDecodeError, OSError):
        captured = {}

running = True


def _save():
    with open(OUTPUT_FILE, "w") as f:
        json.dump(captured, f, indent=2)


def print_transform(tf: carla.Transform, label: str):
    loc, rot = tf.location, tf.rotation
    print("\n" + "=" * 60)
    print(f"[{label}] captured transform")
    print("-" * 60)
    print(f"carla.Location(x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f})")
    print(f"carla.Rotation(pitch={rot.pitch:.2f}, yaw={rot.yaw:.2f}, "
          f"roll={rot.roll:.2f})")
    print("--- as carla.Transform (e.g. for PED_REFERENCE) ---")
    print(f"carla.Transform(\n"
          f"    carla.Location(x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f}),\n"
          f"    carla.Rotation(pitch={rot.pitch:.2f}, yaw={rot.yaw:.2f}, "
          f"roll={rot.roll:.2f}))")
    print("--- as (x, y) waypoint tuple (e.g. for v_path) ---")
    print(f"({loc.x:.2f}, {loc.y:.2f})")
    print("=" * 60 + "\n")


def capture(key: str, label: str):
    tf = spectator.get_transform()
    print_transform(tf, label)
    loc, rot = tf.location, tf.rotation
    captured[key] = {
        "label": label,
        "x": round(loc.x, 3), "y": round(loc.y, 3), "z": round(loc.z, 3),
        "pitch": round(rot.pitch, 2), "yaw": round(rot.yaw, 2),
        "roll": round(rot.roll, 2),
    }
    _save()
    print(f"[INFO] Saved under key '{key}' in {OUTPUT_FILE}\n")


def on_press(key):
    global running
    try:
        ch = key.char
    except AttributeError:
        return

    if ch == 'c':
        capture("camera", "CAM")
    elif ch == 'p':
        capture("ped_reference", "PED_REFERENCE")
    elif ch == 'v':
        label = input("\nLabel for this waypoint "
                       "(e.g. t1_left_spawn): ").strip()
        if label:
            capture(label, label.upper())
    elif ch == 'l':
        running = False


def main():
    print("Free-fly the spectator in the CARLA window "
          "(mouse + WASD, Q/E for height).")
    print("Press  [c]=camera  [p]=pedestrian reference  "
          "[v]=named waypoint  [q]=quit\n")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    try:
        while running:
            tf = spectator.get_transform()
            loc, rot = tf.location, tf.rotation
            print(f"\rLive: x={loc.x:7.2f}  y={loc.y:7.2f}  z={loc.z:6.2f} | "
                  f"pitch={rot.pitch:6.1f}  yaw={rot.yaw:6.1f}  "
                  f"roll={rot.roll:5.1f}    ", end="", flush=True)
            time.sleep(1.0 / POLL_HZ)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        print(f"\n\nAll captured points saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
