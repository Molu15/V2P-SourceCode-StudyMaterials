"""
calibrate_setup.py
────────────────────────────────────────────────────────────
Combined calibration tool:
  1. Opens pygame window with study camera POV (FOV=110, identical to study)
  2. Shows marker lines at validated CARLA positions
  3. You place physical tape marks where lines appear on the projected floor
  4. You measure the distances with a tape measure
  5. Press M — enter your measured values (saved to pedestrian_config.json)
  6. Press ENTER to save and print run_sim.py constants

MARKER LEGEND:
  GREEN  — Marker A: Keypress Y (experimenter presses when P steps here)
  YELLOW — Marker B: Keypress X (experimenter presses when P steps here)
  RED    — Vehicle lane (danger zone, where trigger car passes)

Keys: M=measure  ENTER=save  Q=quit
"""

import json
import os
import threading
import time
import math

import carla
import pygame

# ─── FILES ────────────────────────────────────────────────
CALIB_FILE  = os.path.join(os.path.dirname(__file__), "camera_calibration.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "pedestrian_config.json")

CARLA_HOST = 'localhost'
CARLA_PORT = 2000
CAMERA_FOV = '110'

# ─── LINE GEOMETRY (validated in calibrate_markers_visual.py) ─────────────
LINE_Y_START = 20.0
LINE_Y_END   = 55.0
LINE_Z       = 0.06
LINE_THICK   = 0.06

# ─── LOAD CALIBRATION ─────────────────────────────────────
with open(CALIB_FILE) as f:
    calib = json.load(f)

cam     = calib["camera"]
ped     = calib["ped_reference"]
CAM_POS = carla.Location(x=cam["x"], y=cam["y"], z=cam["z"])
CAM_ROT = carla.Rotation(pitch=cam["pitch"], yaw=cam["yaw"], roll=cam["roll"])
PED_REF_X = ped["x"]
PED_REF_Y = ped["y"]

# Add these two lines:
PED_FWD_X = math.cos(math.radians(ped["yaw"]))
PED_FWD_Y = math.sin(math.radians(ped["yaw"]))

XS = {
    "A": round(PED_REF_X + 0.0  * PED_FWD_X, 3),
    "B": round(PED_REF_X + 1.5  * PED_FWD_X, 3),
    "D": round(PED_REF_X + 17.7 * PED_FWD_X, 3),
}
# ─── COLOURS ──────────────────────────────────────────────
def _cc(r, g, b): return carla.Color(r, g, b)
C_GREEN  = _cc(0,   130, 0)
C_YELLOW = _cc(180, 160, 0)
C_RED    = _cc(160, 0,   0)
PG_GREEN  = (0,   200, 0)
PG_YELLOW = (220, 200, 0)
PG_RED    = (220, 60,  60)
PG_GRAY   = (160, 160, 160)


# ─── CAMERA ───────────────────────────────────────────────
class CalibCamera:
    W, H = 1280, 720

    def __init__(self, world, display):
        self.display = display
        self.surface = None
        self._lock   = threading.Lock()

        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self.W))
        bp.set_attribute('image_size_y', str(self.H))
        bp.set_attribute('fov', CAMERA_FOV)
        for attr in ('motion_blur_intensity',
                     'motion_blur_max_distortion',
                     'motion_blur_min_object_screen_size',
                     'lens_circle_falloff', 'lens_circle_multiplier',
                     'lens_k', 'lens_kcube',
                     'lens_x_size', 'lens_y_size'):
            bp.set_attribute(attr, '0.0')

        self.sensor = world.spawn_actor(
            bp, carla.Transform(CAM_POS, CAM_ROT))
        self.sensor.listen(self._on_image)

    def _on_image(self, image):
        import numpy as np
        arr = np.frombuffer(image.raw_data, dtype='uint8')
        arr = arr.reshape((image.height, image.width, 4))
        arr = arr[:, :, :3][:, :, ::-1]
        surf = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        with self._lock:
            self.surface = surf

    def render(self):
        with self._lock:
            if self.surface is not None:
                scaled = pygame.transform.scale(
                    self.surface,
                    (self.display.get_width(), self.display.get_height()))
                self.display.blit(scaled, (0, 0))

    def destroy(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()


# ─── DEBUG LINES ──────────────────────────────────────────
def draw_marker(debug, x, colour, label):
    p1 = carla.Location(x=x, y=LINE_Y_START, z=LINE_Z)
    p2 = carla.Location(x=x, y=LINE_Y_END,   z=LINE_Z)
    debug.draw_line(p1, p2, thickness=LINE_THICK,
                    color=colour, life_time=0.12)
    lbl = carla.Location(x=x, y=LINE_Y_END - 2.0, z=LINE_Z + 0.3)
    debug.draw_string(lbl, label, draw_shadow=True,
                      color=colour, life_time=0.12)


def draw_all(debug):
    draw_marker(debug, XS["A"], C_GREEN,  "A (Y-key)")
    draw_marker(debug, XS["B"], C_YELLOW, "B (X-key)")
    draw_marker(debug, XS["D"], C_RED,    "Vehicle lane")


# ─── HUD ──────────────────────────────────────────────────
def draw_hud(display, cms, state):
    font = pygame.font.SysFont("monospace", 16)

    panel = pygame.Surface((500, 140), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 160))
    display.blit(panel, (16, 16))

    y = 22
    for text, col in [
        (f"GREEN  Marker A (Y-key)  : {cms['A']:>6.1f} cm from start", PG_GREEN),
        (f"YELLOW Marker B (X-key)  : {cms['B']:>6.1f} cm from start", PG_YELLOW),
        (f"RED    Vehicle lane      : {cms['D']:>6.1f} cm from start", PG_RED),
    ]:
        display.blit(font.render(text, True, col), (24, y))
        y += 24

    y += 4
    if state == "show":
        hint = "Mark floor positions, measure, then press M to enter values"
    elif state == "measured":
        hint = "Values saved — press ENTER to confirm or M to re-measure"
    else:
        hint = ""

    display.blit(font.render(hint, True, PG_GRAY), (24, y))
    y += 20
    display.blit(font.render("M=measure  ENTER=save  Q=quit", True, PG_GRAY),
                 (24, y))


# ─── MEASUREMENT INPUT ────────────────────────────────────
def ask_float(prompt, default):
    while True:
        raw = input(f"  {prompt} [{default:.1f} cm]: ").strip()
        if raw == "":
            return float(default)
        try:
            val = float(raw)
            if val < 0:
                print("  Please enter a positive value.")
                continue
            return val
        except ValueError:
            print("  Please enter a number.")


def run_measurement(cms):
    print("\n  ── Enter tape-measured distances (cm from START mark) ──")
    print("  Press ENTER to keep current value.\n")
    cms["total"] = ask_float("Total crossing width (kerb to kerb)", cms["total"])
    cms["A"]     = ask_float("Marker A — GREEN line position",      cms["A"])
    cms["B"]     = ask_float("Marker B — YELLOW line position",     cms["B"])
    cms["D"]     = ask_float("Vehicle lane (RED line) near edge",   cms["D"])
    print("\n  Values recorded. Press ENTER to save.\n")


# ─── SAVE ─────────────────────────────────────────────────
def save_config(cms):
    dist_a_to_b      = round((cms["B"] - cms["A"]) / 100.0, 4)
    dist_b_to_danger = round((cms["D"] - cms["B"]) / 100.0, 4)

    config = {
            "_note":             "Physical distances in cm from START mark.",
            "total_cm":          cms["total"],
            "marker_A_cm":       cms["A"],
            "marker_B_cm":       cms["B"],
            "danger_near_cm":    cms["D"],
            "DIST_A_TO_B":       dist_a_to_b,
            "DIST_B_TO_DANGER":  dist_b_to_danger,
            # CARLA x-coords derived from PED_REFERENCE + physical distances
            "marker_A_x":  round(PED_REF_X + (cms["A"] / 100.0) * PED_FWD_X, 3),
            "marker_B_x":  round(PED_REF_X + (cms["B"] / 100.0) * PED_FWD_X, 3),
            "danger_x":    round(PED_REF_X + (cms["D"] / 100.0) * PED_FWD_X, 3),
            "ref_y":       PED_REF_Y,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 55)
    print(f"  Saved → {OUTPUT_FILE}")
    print("=" * 55)
    print(f"  Marker A  (GREEN)  : {cms['A']:.1f} cm")
    print(f"  Marker B  (YELLOW) : {cms['B']:.1f} cm")
    print(f"  Vehicle lane (RED) : {cms['D']:.1f} cm")
    print()
    print("  Paste into run_sim.py:")
    print(f"    DIST_A_TO_B      = {dist_a_to_b:.4f}")
    print(f"    DIST_B_TO_DANGER = {dist_b_to_danger:.4f}")
    print("=" * 55 + "\n")


# ─── MAIN ─────────────────────────────────────────────────
def main():
    print("\n╔══════════════════════════════════════════╗")
    print("║         SETUP CALIBRATION TOOL          ║")
    print("╚══════════════════════════════════════════╝")
    print()
    print("  Step 1: Lines appear in projection window")
    print("  Step 2: Mark floor where each line lands")
    print("  Step 3: Measure distances with tape measure")
    print("  Step 4: Press M to enter values")
    print("  Step 5: Press ENTER to save\n")

    print("  Connecting to CARLA...", end="", flush=True)
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)
    world  = client.get_world()
    debug  = world.debug

    settings = world.get_settings()
    settings.synchronous_mode    = False
    settings.fixed_delta_seconds = 0.0
    world.apply_settings(settings)
    print(" OK\n")

    pygame.init()
    info    = pygame.display.Info()
    display = pygame.display.set_mode(
        (info.current_w, info.current_h), pygame.NOFRAME)
    pygame.display.set_caption("Setup Calibration")

    camera = CalibCamera(world, display)
    time.sleep(1.5)

    # Physical cm values — filled in by experimenter after measuring
    cms = {"total": 315.0, "A": 0.0, "B": 0.0, "D": 0.0}

    state     = "show"
    clock     = pygame.time.Clock()
    last_draw = 0.0
    running   = True

    while running:
        world.tick()
        camera.render()
        draw_hud(display, cms, state)
        pygame.display.flip()

        now = time.perf_counter()
        if now - last_draw >= 1.0 / 15:
            draw_all(debug)
            last_draw = now

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_m:
                    run_measurement(cms)
                    state = "measured"
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    if state == "measured":
                        save_config(cms)
                    else:
                        print("  Press M first to enter measurements.")
                    running = False

        clock.tick(60)

    camera.destroy()
    pygame.quit()
    print("  Done.")


if __name__ == "__main__":
    main()