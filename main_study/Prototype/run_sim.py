import carla
import json
import socket
import threading
import pygame
import math
import time
import os
import csv
import random
import subprocess
import sys
import keyboard
import numpy as np

# ─── BRIDGE CONFIG ────────────────────────────────────────
ALARM_OUT_PORT = 5011   # UDP port: alarms sent to Bridge → Phone

# ─── SIMULATION PHYSICS ───────────────────────────────────
TARGET_SPEED_MS  = 40.0 / 3.6   # Target vehicle speed  (~11.1 m/s = 40 km/h)
AMBIENT_SPEED_MS = 40.0 / 3.6   # Ambient vehicle speed (same as target)
DT               = 1.0 / 60.0   # CARLA fixed timestep (60 Hz)

# ─── WARNING THRESHOLDS ───────────────────────────────────
TTC_EARLY_WARN = 5.0   # TTC at which 'f'   alarm fires (adaptive mode only)
TTC_FULL_WARN  = 2.5   # TTC at which 'vha' alarm fires (both modes)

# ─── PEDESTRIAN TIMING ────────────────────────────────────
# Measured distances between the two floor markers in the real setup.
# Loaded from pedestrian_config.json; these are the fallback defaults.
DIST_A_TO_B      = 1.6   # m  kerb marker (Y) → midpoint marker (X)
DIST_B_TO_DANGER = 0.9   # m  midpoint marker → vehicle lane (danger zone)

_PED_CONFIG_FILE = os.path.join(os.path.dirname(__file__),
                                "pedestrian_config.json")
try:
    with open(_PED_CONFIG_FILE) as _f:
        _pc = json.load(_f)
    DIST_A_TO_B      = _pc["DIST_A_TO_B"]
    DIST_B_TO_DANGER = _pc["DIST_B_TO_DANGER"]
    print(f"[SIM] Loaded pedestrian config: "
          f"A→B={DIST_A_TO_B:.3f}m, B→D={DIST_B_TO_DANGER:.3f}m")
except Exception:
    print("[SIM] pedestrian_config.json not found — using defaults.")


# ─── GEOMETRY ─────────────────────────────────────────────
# Coordinate naming convention:
#   SPAWN / END : path start / end point
#   V / H       : Vertical road (±X axis) / Horizontal road (±Y axis)
#   L / R       : Left / Right from camera POV (vertical road only)
#   F / B       : Frontal / Behind from camera POV (horizontal road only)
#   O / I       : Outer / Inner lane
#   Y_V_*       : y-coordinate of a vertical lane
#   X_H_*       : x-coordinate of a horizontal lane

COLLISION_X  = -52.6   # Intersection centre — where target vehicle turns
LANE_Z       =   0.5   # Vertical offset for all vehicle spawns
TURN_OFFSET  =   8.0   # y-offset added after the turn for a smooth curve
STOP_Y_CROSS =  42.5   # y-position where catch-trial vehicles stop in the cross street

# Vertical lane y-coordinates
Y_V_L_O =  28.4    # Outer left  (+X vehicles: t1/t2/t5/c1/s2)
Y_V_L_I =  13.21   # Inner left  (ambient for t2)
Y_V_R_O =  16.63   # Outer right (-X vehicles: t3/t4)
Y_V_R_I =  13.21   # Inner right (ambient for t3/t4)

# Horizontal lane x-coordinates
X_H_F_O = -52.0    # Outer frontal lane (ambient t1/t4/s1; cross-street path)
X_H_F_I = -48.5    # Inner frontal lane (ambient t2)
X_H_B   = -45.0    # Behind lane       (ambient c1)

# Vertical road spawn / end x-coordinates
SPAWN_V_L_O = -60.0   # Left outer spawn
SPAWN_V_R_O =  10.0   # Right outer spawn  (-X vehicles)
SPAWN_V_R_I =  10.0   # Right inner spawn  (-X vehicles, different lane)
END_V_L_O   = -90.0   # Left  end  (-X vehicles despawn here)
END_V_L_I   = -75.0   # Left  end  (inner lane)
END_V_R_O   = -15.0   # Right end  (+X vehicles despawn here)

# Horizontal road spawn / end y-coordinates
SPAWN_H_F   =  20.0    # Frontal spawn (top of cross street)
SPAWN_H_B_Y = 45.0    # Behind spawn y-coordinate (c1 ambient)
END_H_F     = 70.0    # Frontal end (+Y vehicles despawn here)
END_H_B     = -40.0   # Behind end  (-Y vehicles despawn here, c1 ambient)

# ─── DISPLAY POSITION ─────────────────────────────────────
# Top-left pixel position of the pygame window.
# Example — projector as a second monitor to the right at 1920×1080:
#   WINDOW_X, WINDOW_Y = 1920, 0
WINDOW_X = 2048
WINDOW_Y = 0

# ─── CAMERA CALIBRATION ───────────────────────────────────
# Camera position, rotation, and pedestrian reference point
# are loaded from camera_calibration.json so they can be
# adjusted without editing this file.
CALIBRATION_FILE = os.path.join(os.path.dirname(__file__),
                                "camera_calibration.json")

def _load_calibration(path=CALIBRATION_FILE):
    with open(path, "r") as f:
        return json.load(f)

_calib = _load_calibration()
_cam   = _calib["camera"]
_ped   = _calib["ped_reference"]

CAM_POS = carla.Location(x=_cam["x"], y=_cam["y"], z=_cam["z"])
CAM_ROT = carla.Rotation(pitch=_cam["pitch"], yaw=_cam["yaw"], roll=_cam["roll"])

# Pedestrian reference — used to derive the collision point.
# Do not edit manually; adjust camera_calibration.json instead.
PED_REF_X = _ped["x"]
PED_REF_Y = _ped["y"]

# ─── COLLISION POINT ──────────────────────────────────────
# The exact map coordinate the target vehicle passes through
# when it enters the pedestrian's path.
COLLISION_POINT_X = -51.7        # measured: x ≈ -51.69
COLLISION_POINT_Y =  PED_REF_Y   # same y-axis as pedestrian walking path


# ─────────────────────────────────────────────────────────
# SCENARIO DEFINITIONS
#
# outcome:
#   "hit"  → target vehicle drives through the collision point (target trials)
#   "stop" → target vehicle stops just before the collision point (catch trials)
#   "safe" → target vehicle never approaches the collision point  (filler trials)
#
# ambient: background vehicle that moves independently of keypresses.
#   Starts driving as soon as setup_world() completes.
#   Set to None to run a scenario without an ambient vehicle.
#
# v_rot: 0 = vehicle travels in +X direction (left→right on screen)
#        180 = vehicle travels in -X direction (right→left on screen)
# ─────────────────────────────────────────────────────────
SCENARIOS = {
    "t1": {
        "outcome": "hit", "direction": "right",
        "model": "vehicle.tesla.model3",       "color": "200,0,0",
        "v_spawn": (SPAWN_V_L_O, Y_V_L_O, LANE_Z), "v_rot": 0,
        "v_path":  [(SPAWN_V_L_O, Y_V_L_O),
                    (COLLISION_X, Y_V_L_O),
                    (COLLISION_X, Y_V_L_O + TURN_OFFSET),
                    (X_H_F_O,    END_H_F)],
        # Ambient: enters from top of cross street, turns left, exits left
        "ambient": {
            "model": "vehicle.toyota.prius", "color": "50,180,50",
            "spawn": (X_H_F_O, SPAWN_H_F, LANE_Z), "rot": 90,
            "path":  [(X_H_F_O, SPAWN_H_F),
                      (X_H_F_O, Y_V_L_I),
                      (END_V_L_O, Y_V_L_I)],
        },
    },
    "t2": {
        "outcome": "hit", "direction": "right",
        "model": "vehicle.audi.tt",            "color": "0,80,200",
        "v_spawn": (SPAWN_V_L_O, Y_V_L_O, LANE_Z), "v_rot": 0,
        "v_path":  [(SPAWN_V_L_O, Y_V_L_O),
                    (COLLISION_X, Y_V_L_O),
                    (COLLISION_X, Y_V_L_O + TURN_OFFSET),
                    (X_H_F_O,    END_H_F)],
        # Ambient: enters from top of cross street (inner lane), drives straight through
        "ambient": {
            "model": "vehicle.toyota.prius", "color": "50,180,50",
            "spawn": (X_H_F_I, SPAWN_H_F, LANE_Z), "rot": 90,
            "path":  [(X_H_F_I, SPAWN_H_F),
                      (X_H_F_I, END_H_F)],
        },
    },
    "t3": {
        "outcome": "hit", "direction": "right",
        "model": "vehicle.bmw.grandtourer",    "color": "30,160,30",
        "v_spawn": (SPAWN_V_R_O, Y_V_R_O, LANE_Z), "v_rot": 180,
        "v_path":  [(SPAWN_V_R_O, Y_V_R_O),
                    (COLLISION_X, Y_V_R_O),
                    (COLLISION_X, Y_V_R_O + TURN_OFFSET),
                    (X_H_F_O,    END_H_F)],
        # Ambient: inner right lane, drives straight through intersection
        "ambient": {
            "model": "vehicle.toyota.prius", "color": "50,180,50",
            "spawn": (SPAWN_V_R_O, Y_V_R_I, LANE_Z), "rot": 180,
            "path":  [(SPAWN_V_R_O, Y_V_R_I),
                      (END_V_L_I,   Y_V_R_I)],
        },
    },
    "t4": {
        "outcome": "hit", "direction": "right",
        "model": "vehicle.dodge.charger_2020", "color": "220,220,0",
        "v_spawn": (SPAWN_V_R_O, Y_V_R_O, LANE_Z), "v_rot": 180,
        "v_path":  [(SPAWN_V_R_O, Y_V_R_O),
                    (COLLISION_X, Y_V_R_O),
                    (COLLISION_X, Y_V_R_O + TURN_OFFSET),
                    (X_H_F_O,    END_H_F)],
        # Ambient: enters from top of cross street (outer lane), drives straight through
        "ambient": {
            "model": "vehicle.toyota.prius", "color": "50,180,50",
            "spawn": (X_H_F_O, SPAWN_H_F, LANE_Z), "rot": 90,
            "path":  [(X_H_F_O, SPAWN_H_F),
                      (X_H_F_O, END_H_F)],
        },
    },
    "t5": {
        "outcome": "hit", "direction": "right",
        "model": "vehicle.mercedes.coupe_2020","color": "80,80,80",
        "v_spawn": (SPAWN_V_L_O, Y_V_L_O, LANE_Z), "v_rot": 0,
        "v_path":  [(SPAWN_V_L_O, Y_V_L_O),
                    (COLLISION_X, Y_V_L_O),
                    (COLLISION_X, Y_V_L_O + TURN_OFFSET),
                    (X_H_F_O,    END_H_F)],
        # Ambient: inner right lane, drives straight through intersection
        "ambient": {
            "model": "vehicle.toyota.prius", "color": "50,180,50",
            "spawn": (SPAWN_V_R_I, Y_V_R_O, LANE_Z), "rot": 180,
            "path":  [(SPAWN_V_R_I, Y_V_R_O),
                      (END_V_L_I,   Y_V_R_O)],
        },
    },
    "c1": {
        "outcome": "stop", "direction": "right",
        "model": "vehicle.seat.leon",          "color": "180,180,180",
        "v_spawn": (SPAWN_V_L_O, Y_V_L_O, LANE_Z), "v_rot": 0,
        "v_path":  [(SPAWN_V_L_O, Y_V_L_O),
                    (COLLISION_X, Y_V_L_O),
                    (COLLISION_X, Y_V_L_O + TURN_OFFSET),
                    (X_H_F_O,    STOP_Y_CROSS)],   # stops before collision point
        # Ambient: drives from behind toward camera and exits below
        "ambient": {
            "model": "vehicle.toyota.prius", "color": "50,180,50",
            "spawn": (X_H_B, SPAWN_H_B_Y, LANE_Z), "rot": 270,
            "path":  [(X_H_B, SPAWN_H_B_Y),
                      (X_H_B, END_H_B)],
        },
    },
    "c2": {
        "outcome": "stop", "direction": "right",
        "model": "vehicle.audi.etron",         "color": "100,0,150",
        "v_spawn": (SPAWN_V_R_O, Y_V_R_O, LANE_Z), "v_rot": 180,
        "v_path":  [(SPAWN_V_R_O, Y_V_R_O),
                    (COLLISION_X, Y_V_R_O),
                    (COLLISION_X, Y_V_R_O + TURN_OFFSET),
                    (X_H_F_O,    STOP_Y_CROSS)],   # stops before collision point
        # Ambient: outer left lane, drives straight through intersection
        "ambient": {
            "model": "vehicle.toyota.prius", "color": "50,180,50",
            "spawn": (SPAWN_V_L_O, Y_V_L_O, LANE_Z), "rot": 0,
            "path":  [(SPAWN_V_L_O, Y_V_L_O),
                      (END_V_R_O,   Y_V_L_O)],
        },
    },
    "s1": {
        "outcome": "safe", "direction": "right",
        "model": "vehicle.nissan.micra",       "color": "255,128,0",
        "v_spawn": (SPAWN_V_R_O, Y_V_R_O, LANE_Z), "v_rot": 180,
        "v_path":  [(SPAWN_V_R_O, Y_V_R_O),
                    (END_V_L_O,   Y_V_R_O)],        # drives straight, no turn
        # Ambient: enters from top of cross street (outer lane), drives straight through
        "ambient": {
            "model": "vehicle.toyota.prius", "color": "50,180,50",
            "spawn": (X_H_F_O, SPAWN_H_F, LANE_Z), "rot": 90,
            "path":  [(X_H_F_O, SPAWN_H_F),
                      (X_H_F_O, END_H_F)],
        },
    },
    "s2": {
        "outcome": "safe", "direction": "right",
        "model": "vehicle.lincoln.mkz_2020",   "color": "30,30,200",
        "v_spawn": (SPAWN_V_L_O, Y_V_L_O, LANE_Z), "v_rot": 0,
        "v_path":  [(SPAWN_V_L_O, Y_V_L_O),
                    (END_V_R_O,   Y_V_L_O)],        # drives straight, no turn
        # Ambient: outer right lane, drives straight through intersection
        "ambient": {
            "model": "vehicle.toyota.prius", "color": "50,180,50",
            "spawn": (SPAWN_V_R_O, Y_V_R_O, LANE_Z), "rot": 180,
            "path":  [(SPAWN_V_R_O, Y_V_R_O),
                      (END_V_L_O,   Y_V_R_O)],
        },
    },
}

# Quick lookup: scenario key → trial type string
TRIAL_TYPE = {k: v["outcome"] for k, v in SCENARIOS.items()}


# ─────────────────────────────────────────────────────────
# CONSTRAINED RANDOMIZATION
#
# Generates a shuffled trial order satisfying:
#   1. First trial must be a target (hit)
#   2. Last trial must be a target (hit)
#   3. At most 1 catch trial in the final 2 positions
#   4. No two catch trials may be adjacent
# ─────────────────────────────────────────────────────────
def generate_run_order(max_attempts: int = 1000) -> list[str]:
    keys    = list(SCENARIOS.keys())
    targets = [k for k in keys if TRIAL_TYPE[k] == "hit"]
    catches = [k for k in keys if TRIAL_TYPE[k] == "stop"]
    safes   = [k for k in keys if TRIAL_TYPE[k] == "safe"]

    for _ in range(max_attempts):
        order = targets + catches + safes
        random.shuffle(order)

        if TRIAL_TYPE[order[0]] != "hit":
            continue
        if TRIAL_TYPE[order[-1]] != "hit":
            continue
        if sum(1 for k in order[-2:] if TRIAL_TYPE[k] == "stop") > 1:
            continue
        if any(TRIAL_TYPE[order[i]] == "stop" and TRIAL_TYPE[order[i+1]] == "stop"
               for i in range(len(order) - 1)):
            continue

        return order

    # Fallback: manually construct a valid order
    random.shuffle(catches)
    random.shuffle(safes)
    random.shuffle(targets)
    return [targets[0], targets[1], targets[2], catches[0],
            targets[3], safes[0], catches[1], safes[1], targets[4]]


# ─── KEYBOARD INPUT ───────────────────────────────────────
# Edge-detection wrapper around the `keyboard` library.
# Returns True only on the first frame a key transitions from up → down,
# regardless of OS focus or how long the key is held.
_prev_keys: set = set()

def key_just_pressed(key: str) -> bool:
    global _prev_keys
    currently = keyboard.is_pressed(key)
    was_pressed = key in _prev_keys
    if currently and not was_pressed:
        _prev_keys.add(key)
        return True
    if not currently:
        _prev_keys.discard(key)
    return False


# ─────────────────────────────────────────────────────────
# DATA LOGGER
# ─────────────────────────────────────────────────────────
class DataLogger:
    """
    Collects per-frame vehicle telemetry and writes a CSV on scenario end.

    Header line (metadata):
        PID, Mode, Run, Outcome, T_A, T_B, T_C,
        TTC_at_response, EarlyWarnT, FullWarnT

    Per-frame columns:
        timestamp, v_x, v_y, v_yaw, TTC_secondary, is_active, warning_code
    """

    def __init__(self, p_id, mode, run_id):
        self.p_id, self.mode, self.run_id = p_id, mode, run_id
        self.data = []

        # Experimenter keypress timestamps (set externally by SmombieScenario)
        self.timestamp_A     = None   # Y key: participant at kerb
        self.timestamp_B     = None   # X key: participant at midpoint
        self.timestamp_C     = None   # C key: alarms aborted
        self.early_warn_time = None   # Wall-clock time 'f'   alarm was sent
        self.full_warn_time  = None   # Wall-clock time 'vha' alarm was sent
        self.timing_error    = None

        self.t_start_abs = None

        log_dir = os.path.join('logs', f'P{p_id}')
        os.makedirs(log_dir, exist_ok=True)
        self.filepath = os.path.join(log_dir, f'{mode}_{run_id}.csv')

    def log_frame(self, timestamp, v_loc, v_yaw, ttc, is_active, warning_code):
        self.data.append({
            "timestamp":     round(timestamp, 4),
            "v_x":           round(v_loc.x, 3),
            "v_y":           round(v_loc.y, 3),
            "v_yaw":         round(v_yaw, 2),
            # TTC_secondary: kinematic estimate from vehicle position.
            # For sanity-checking only — primary DV is TTC_at_response in header.
            "TTC_secondary": round(ttc, 3) if ttc < 999 else "N/A",
            "is_active":     is_active,
            "warning_code":  warning_code,
        })

    def save_and_close(self, outcome):
        if not self.data:
            return

        # TTC_at_response: time remaining until the pedestrian reaches the
        # danger zone, estimated from their walking speed between the two
        # floor markers (A→B) and the remaining distance (B→danger zone).
        ttc_at_response = "N/A"
        if self.timestamp_A is not None and self.timestamp_B is not None:
            elapsed_AB = self.timestamp_B - self.timestamp_A
            if elapsed_AB > 0:
                v_ped           = DIST_A_TO_B / elapsed_AB
                ttc_at_response = round(DIST_B_TO_DANGER / v_ped, 4)

        with open(self.filepath, 'w', newline='') as f:
            f.write(
                f"# PID:{self.p_id} Mode:{self.mode} Run:{self.run_id} "
                f"Outcome:{outcome} "
                f"T_Start_abs:{self.t_start_abs} "
                f"T_A:{self.timestamp_A} "
                f"T_B:{self.timestamp_B} "
                f"T_C:{self.timestamp_C} "
                f"TTC_at_response:{ttc_at_response} "
                f"TimingError:{self.timing_error} "
                f"EarlyWarnT:{self.early_warn_time} "
                f"FullWarnT:{self.full_warn_time}\n"
            )
            writer = csv.DictWriter(f, fieldnames=self.data[0].keys())
            writer.writeheader()
            writer.writerows(self.data)

        print(f"[LOG] Saved → {self.filepath}")
        if ttc_at_response != "N/A":
            print(f"[LOG] TTC-at-response: {ttc_at_response:.3f}s")


# ─────────────────────────────────────────────────────────
# KINEMATIC VEHICLE DRIVER
# ─────────────────────────────────────────────────────────
class VehicleDriver:
    """
    Moves a CARLA vehicle along a list of (x, y) waypoints at a fixed speed.
    Physics are disabled; position is set directly each tick (kinematic control).
    """

    def __init__(self, vehicle, path, speed_ms, spawn_z):
        self.vehicle  = vehicle
        self.path     = [carla.Location(x=p[0], y=p[1], z=spawn_z) for p in path]
        self.speed_ms = speed_ms
        self.spawn_z  = spawn_z
        self.idx      = 0
        self.done     = False

    def tick(self):
        if self.done or self.idx >= len(self.path):
            self.done = True
            return

        v_tf   = self.vehicle.get_transform()
        target = self.path[self.idx]
        diff   = target - v_tf.location

        # Advance to next waypoint when within 1.5 m
        if math.sqrt(diff.x**2 + diff.y**2) < 1.5:
            self.idx += 1
            if self.idx >= len(self.path):
                self.done = True
            return

        # Move vehicle by one DT step toward current target
        unit   = carla.Vector3D(diff.x, diff.y, 0)
        length = math.sqrt(unit.x**2 + unit.y**2)
        unit.x /= length
        unit.y /= length

        new_loc   = v_tf.location + carla.Location(
            x=unit.x * self.speed_ms * DT,
            y=unit.y * self.speed_ms * DT,
            z=0)
        new_loc.z = self.spawn_z
        new_yaw   = math.degrees(math.atan2(unit.y, unit.x))

        self.vehicle.set_transform(
            carla.Transform(new_loc, carla.Rotation(yaw=new_yaw)))


# ─────────────────────────────────────────────────────────
# WIDE-ANGLE CAMERA
# ─────────────────────────────────────────────────────────
class WideCamera:
    """
    Spawns a CARLA rgb_camera sensor and renders each frame to a pygame surface.
    Runs in a background thread to avoid blocking the simulation loop.
    """

    DISPLAY_W = 1280
    DISPLAY_H = 720

    def __init__(self, world, display):
        self.display = display
        self.surface = None
        self._lock   = threading.Lock()

        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self.DISPLAY_W))
        bp.set_attribute('image_size_y', str(self.DISPLAY_H))
        bp.set_attribute('fov', '110')

        # Disable motion blur (prevents smearing on fast-moving vehicles)
        bp.set_attribute('motion_blur_intensity', '0.0')
        bp.set_attribute('motion_blur_max_distortion', '0.0')
        bp.set_attribute('motion_blur_min_object_screen_size', '0.0')

        # Disable lens distortion
        bp.set_attribute('lens_circle_falloff', '0.0')
        bp.set_attribute('lens_circle_multiplier', '0.0')
        bp.set_attribute('lens_k', '0.0')
        bp.set_attribute('lens_kcube', '0.0')
        bp.set_attribute('lens_x_size', '0.0')
        bp.set_attribute('lens_y_size', '0.0')

        self.sensor = world.spawn_actor(
            bp, carla.Transform(CAM_POS, CAM_ROT))
        self.sensor.listen(self._on_image)

    def _on_image(self, image):
        """Convert raw CARLA image (BGRA) to a pygame RGB surface."""
        array = np.frombuffer(image.raw_data, dtype='uint8')
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3][:, :, ::-1]   # BGRA → RGB
        surf  = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        with self._lock:
            self.surface = surf

    def render(self):
        """Scale and blit the latest camera frame to the pygame display."""
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


# ─────────────────────────────────────────────────────────
# MAIN SCENARIO CLASS
# ─────────────────────────────────────────────────────────
class SmombieScenario:
    """
    Manages a single trial: world setup, vehicle driving, alarm logic,
    collision detection, and data logging.

    Keypress sequence (experimenter-operated):
        Y → participant reaches kerb marker     (logs T_A, starts timing)
        X → participant reaches midpoint marker  (spawns vehicle, starts alarms)
        C → abort pending alarms (participant reacted)
        Q → force-quit current scenario
    """

    def __init__(self, mode, run_id, p_id, display):
        self.mode, self.run_id, self.p_id = mode, run_id, p_id
        self.config  = SCENARIOS[run_id]
        self.display = display

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world  = self.client.get_world()

        self.logger     = DataLogger(p_id, mode, run_id)
        self.alarm_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self._session_start = None

        self.actor_list     = []   # all spawned actors; destroyed in finally
        self.ambient_actor  = None
        self.ambient_driver = None
        self.target_vehicle = None
        self.vehicle_driver = None
        self.camera         = None

        # ── Trial state ───────────────────────────────────
        self.active     = False       # True after X is pressed (vehicle moving)
        self.outcome    = "no_event"
        self.target_hit = False       # True once vehicle crosses collision point

        # ── Keypress state machine ────────────────────────
        self._press_count      = 0     # 0=idle, 1=Y done, 2=X done
        self._t_A              = None  # Wall-clock timestamp of Y press
        self._t_B              = None  # Wall-clock timestamp of X press
        self._warn_early_fired = False
        self._warn_full_fired  = False
        self._warn_aborted     = False # True after C press

        # ── Tick-based alarm scheduling ───────────────────
        # Alarms are counted in simulation ticks (not wall-clock seconds)
        # to stay accurate when CARLA runs slower than real-time.
        self._tick_count       = 0
        self._tick_B           = None  # Tick number when X was pressed
        self._tick_early_alarm = None  # Tick to fire 'f'   alarm
        self._tick_full_alarm  = None  # Tick to fire 'vha' alarm
        self._vehicle_start_tick = 0

        # NEU — hier einfügen:
        self._ped_ttc_at_f   = None
        self._ped_ttc_at_vha = None

        # ── Sound ─────────────────────────────────────────
        self._sound_channel = None

    # ── Alarm → Bridge ──────────────────────────────────
    def _send_alarm(self, code: str, note: str = ""):
        try:
            self.alarm_sock.sendto(code.encode(), ('127.0.0.1', ALARM_OUT_PORT))
            print(f"\n[SIM] Alarm sent: '{code}'" + (f"  {note}" if note else ""))
        except OSError as e:
            print(f"[SIM] Alarm send error: {e}")

    # ── Spawn helper ─────────────────────────────────────
    def _spawn_and_settle(self, bp, transform, ticks=20):
        """
        Spawn a vehicle, let physics run briefly so it drops to road level,
        record the settled z-height, then disable physics and hide the actor
        below the map until it is needed.
        Returns (actor, spawn_z).
        """
        actor = self.world.spawn_actor(bp, transform)
        actor.set_simulate_physics(True)
        for _ in range(ticks):
            self.world.tick()
        spawn_z = actor.get_location().z
        actor.set_simulate_physics(False)
        actor.set_transform(carla.Transform(
            carla.Location(x=transform.location.x,
                           y=transform.location.y,
                           z=-10.0),
            transform.rotation))
        return actor, spawn_z

    # ── World setup ──────────────────────────────────────
    def setup_world(self):
        """
        Clear the map, enable synchronous mode, spawn the camera sensor,
        target vehicle, and ambient vehicle. Ambient starts moving immediately.
        """
        # Remove any leftover actors from previous scenarios
        for a in self.world.get_actors().filter('vehicle.*'): a.destroy()
        for a in self.world.get_actors().filter('walker.*'):  a.destroy()
        for a in self.world.get_actors().filter('sensor.*'):  a.destroy()

        bp_lib = self.world.get_blueprint_library()

        # Enable synchronous mode with fixed timestep
        settings = self.world.get_settings()
        settings.synchronous_mode    = True
        settings.fixed_delta_seconds = DT
        self.world.apply_settings(settings)

        # Spawn camera sensor
        self.camera = WideCamera(self.world, self.display)
        self.actor_list.append(self.camera.sensor)

        # Spawn target vehicle (hidden below map until X is pressed)
        v_cfg = self.config
        v_bp  = bp_lib.find(v_cfg["model"])
        v_bp.set_attribute('color', v_cfg["color"])
        v_tf  = carla.Transform(
            carla.Location(*v_cfg["v_spawn"]),
            carla.Rotation(yaw=v_cfg["v_rot"]))
        try:
            self.target_vehicle, spawn_z = self._spawn_and_settle(v_bp, v_tf)
        except RuntimeError as e:
            raise RuntimeError(
                f"[SIM] Target vehicle spawn failed at {v_tf.location}: {e}") from e
        self.actor_list.append(self.target_vehicle)

        self.vehicle_driver = VehicleDriver(
            self.target_vehicle, v_cfg["v_path"], TARGET_SPEED_MS, spawn_z)

        # Spawn ambient vehicle (if defined for this scenario)
        amb = v_cfg.get("ambient")
        if amb:
            a_bp = bp_lib.find(amb["model"])
            a_bp.set_attribute('color', amb["color"])
            a_tf = carla.Transform(
                carla.Location(*amb["spawn"]),
                carla.Rotation(yaw=amb["rot"]))
            try:
                self.ambient_actor, a_z = self._spawn_and_settle(a_bp, a_tf)
                self.ambient_driver = VehicleDriver(
                    self.ambient_actor, amb["path"], AMBIENT_SPEED_MS, a_z)
                self.actor_list.append(self.ambient_actor)
            except RuntimeError:
                print("[SIM] Ambient spawn failed — skipping ambient vehicle.")

        # Release ambient vehicle to its start position immediately
        if self.ambient_actor:
            path0 = self.ambient_driver.path[0]
            self.ambient_actor.set_transform(carla.Transform(
                carla.Location(x=path0.x, y=path0.y,
                               z=self.ambient_driver.spawn_z),
                carla.Rotation(yaw=self.config["ambient"]["rot"])))
            self.world.tick()

        print(f"\n[SIM] '{self.run_id}' ready  |  "
              f"outcome={v_cfg['outcome'].upper()}  |  mode={self.mode.upper()}")
        print("      Y=Start (kerb)  X=Midpoint  C=Abort  Q=Quit")

        self._session_start = time.perf_counter()   # ← Zeitreferenz = Ambient Sound Start
        self.logger.t_start_abs = time.time()   # ← Unix-Zeit beim Ambient Sound Start
        if city_sound:
            self._sound_channel = city_sound.play(loops=-1)

    # ── TTC calculation (per-frame, secondary measure) ───
    def _calc_ttc(self) -> float:
        """
        Two-phase kinematic TTC estimate based on vehicle position.
        Phase 1 (main road): remaining distance to COLLISION_X divided by speed,
                             plus the fixed cross-street travel time (seg2_time).
        Phase 2 (cross street): remaining y-distance to COLLISION_POINT_Y.

        Note: this is a sanity-check column in the CSV.
              The primary DV (TTC_at_response) is computed from pedestrian timing.
        """
        v_loc     = self.target_vehicle.get_location()
        v_rot     = self.config["v_rot"]
        spawn_y   = self.config["v_spawn"][1]
        seg2_time = abs(COLLISION_POINT_Y - spawn_y) / TARGET_SPEED_MS

        # Vehicle is in the cross street once its y has moved >2 m from the main road
        in_cross = v_loc.y > spawn_y + 2.0

        if not in_cross:
            dist_main = (COLLISION_X - v_loc.x) if v_rot == 0 else (v_loc.x - COLLISION_X)
            return max(0.0, dist_main / TARGET_SPEED_MS) + seg2_time

        return max(0.0, (COLLISION_POINT_Y - v_loc.y) / TARGET_SPEED_MS)

    # ── Main simulation loop ──────────────────────────────
    def run(self):
        global _prev_keys
        _prev_keys.clear()   # discard any stale key state from previous scenario

        session_start = self._session_start if self._session_start is not None \
                else time.perf_counter()

        try:
            while True:
                self.world.tick()
                self._tick_count += 1
                now = time.perf_counter() - session_start

                # Render camera and process pygame events
                self.camera.render()
                pygame.display.flip()
                for event in pygame.event.get():
                    pass   # window-close events are ignored; use Q to quit

                # ── Keypresses (focus-independent) ─────────
                if key_just_pressed('y') and self._press_count == 0:
                    self._handle_keydown(pygame.K_y, now)
                if key_just_pressed('x') and self._press_count == 1:
                    self._handle_keydown(pygame.K_x, now)
                if key_just_pressed('c') and self._press_count >= 1:
                    self._handle_keydown(pygame.K_c, now)
                if key_just_pressed('q'):
                    self._handle_keydown(pygame.K_q, now)

                # ── Drive ambient vehicle ───────────────────
                if self.ambient_driver:
                    self.ambient_driver.tick()

                # ── Drive target vehicle (after X press) ────
                if self.active and self._tick_count >= self._vehicle_start_tick:
                    self.vehicle_driver.tick()

                # ── Alarm logic ─────────────────────────────
                # Alarms are checked against simulation tick count so timing
                # is correct even when CARLA runs slower than real-time.
                if (self.active
                        #and not self._warn_aborted -> to abort the alarms when stopped
                        and self._tick_early_alarm is not None):

                    outcome = self.config["outcome"]

                    # 'f' alarm: fires for hit AND catch trials, adaptive mode only
                    if (not self._warn_early_fired
                            and self._tick_count >= self._tick_early_alarm
                            and outcome in ("hit", "stop")):
                        self._warn_early_fired = True
                        if self.mode == 'a':
                            self._send_alarm(f"f:{self.config['direction']}",
                                                f"(ped TTC={self._ped_ttc_at_f:.2f}s)")
                            self.logger.early_warn_time = round(now, 4)

                    # 'vha' alarm: fires for hit trials only (both modes)
                    if (not self._warn_full_fired
                            and self._tick_count >= self._tick_full_alarm
                            and outcome == "hit"):
                        self._warn_full_fired = True
                        alarm = ("vha:none" if self.mode == 'b'
                                 else f"vha:{self.config['direction']}")
                        self._send_alarm(alarm,
                                            f"(ped TTC={self._ped_ttc_at_vha:.2f}s)")
                        self.logger.full_warn_time = round(now, 4)

                # ── Collision detection ─────────────────────
                # Virtual collision: vehicle crosses COLLISION_POINT_Y in the cross street.
                if (self.active
                        and not self.target_hit
                        and self.config["outcome"] == "hit"):

                    v_loc = self.target_vehicle.get_location()
                    v_rot = self.config["v_rot"]

                    if v_rot == 0:
                        # +X vehicle: must be in cross street AND past pedestrian y
                        in_cross = (v_loc.x >= COLLISION_X
                                    and v_loc.y > Y_V_L_O + 2.0)
                        hit = in_cross and v_loc.y >= COLLISION_POINT_Y
                    else:
                        # -X vehicle: must have turned AND reached pedestrian y
                        hit = (v_loc.x <= COLLISION_POINT_X
                               and v_loc.y >= COLLISION_POINT_Y)

                    if hit:
                        self.target_hit = True
                        self.outcome    = "collision"
                        print(f"\n*** COLLISION: {self.run_id.upper()} ***")

                # ── Per-frame logging (active frames only) ──
                ttc = self._calc_ttc() if self.active else 999.0
                if self.logger.full_warn_time is not None:
                    warning_code = "vha"
                elif self.logger.early_warn_time is not None:
                    warning_code = "f"
                else:
                    warning_code = None

                v_tf = self.target_vehicle.get_transform()
                if self.active:
                    self.logger.log_frame(
                        now, v_tf.location, v_tf.rotation.yaw,
                        ttc, self.active, warning_code)

                # ── Status line ─────────────────────────────
                elapsed_str = f"{now - self._t_A:.2f}s" if self.active else "waiting"
                print(f"  elapsed={elapsed_str} | "
                      f"v_x={v_tf.location.x:.1f} | "
                      f"TTC≈{ttc:.2f}s", end="\r")

                # ── Exit conditions ──────────────────────────
                if self.target_hit:
                    # C was pressed before vehicle arrived → participant reacted
                    if self._warn_aborted:
                        self.outcome = "response_stop"
                    print("\n[SIM] Hit — ending run.")
                    break

                if self.active and self.vehicle_driver.done:
                    if self.config["outcome"] == "hit" and not self.target_hit:
                        # Vehicle completed its path without a detected collision
                        self.target_hit = True
                        self.outcome = ("response_stop" if self._warn_aborted
                                        else "collision")
                        print(f"\n[SIM] Vehicle path complete — {self.outcome}.")
                    else:
                        outcome_map = {"stop": "safe_stop", "safe": "safe_turn"}
                        self.outcome = outcome_map.get(self.config["outcome"], "no_event")
                        print("\n[SIM] Vehicle path complete.")
                    break

                # Safety timeout: 30 s after X press
                if self.active and self._t_B is not None and now - self._t_B > 30.0:
                    print("\n[SIM] Timeout — ending run.")
                    break

        finally:
            if self._sound_channel:
                self._sound_channel.stop()
                self._sound_channel = None
            if self.camera:
                self.camera.destroy()
            self.logger.save_and_close(self.outcome)
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            for actor in self.actor_list:
                try:
                    actor.destroy()
                except Exception:
                    pass
            print(f"[SIM] '{self.run_id}' complete. Outcome: {self.outcome}")

    # ── Keypress handler ─────────────────────────────────
    def _handle_keydown(self, key, now: float):
        """
        Y (press 1) — participant reached the kerb marker.
                       Records T_A; vehicle does NOT spawn yet.
        X (press 2) — participant reached the midpoint marker.
                       Measures walking speed, computes t_to_danger,
                       spawns vehicle at the correct distance, schedules alarms.
        C (press 3) — abort pending alarms (participant reacted).
        Q           — force-quit the current scenario.
        """
        if key == pygame.K_y and self._press_count == 0:
            self._press_count       = 1
            self._t_A               = now
            self.logger.timestamp_A = round(now, 4)
            print(f"\n[SIM] *** Y pressed at t={now:.3f}s — participant at kerb ***")

        elif key == pygame.K_x and self._press_count == 1:
            self._press_count       = 2
            self._t_B               = now
            self.logger.timestamp_B = round(now, 4)

            elapsed_AB = now - self._t_A
            if elapsed_AB <= 0:
                print("[SIM] Warning: elapsed_AB=0 — skipping vehicle launch.")
                return

            # Estimate pedestrian walking speed and time to reach the danger zone
            v_ped       = DIST_A_TO_B / elapsed_AB
            t_to_danger = DIST_B_TO_DANGER / v_ped

            print(f"\n[SIM] *** X pressed at t={now:.3f}s ***")
            print(f"  v_ped       = {v_ped:.3f} m/s  ({v_ped*100:.1f} cm/s)")
            print(f"  t_to_danger = {t_to_danger:.3f} s")

            # ── Compute spawn position ────────────────────
            # The vehicle must travel from spawn to COLLISION_POINT in t_to_danger
            # simulation seconds. The path has two segments:
            #   seg1: main road  → COLLISION_X  (variable length, set by spawn_x)
            #   seg2: cross street → COLLISION_POINT_Y  (fixed length)
            #
            # A minimum seg1 time (MIN_SEG1_TIME) ensures the vehicle is visible
            # for long enough and does not spawn too close to the intersection.

            v_rot = self.config["v_rot"]
            seg2      = abs(COLLISION_POINT_Y - self.config["v_spawn"][1])
            seg2_time = seg2 / TARGET_SPEED_MS

            MIN_TOTAL_TTC = 5.2
            effective_t = max(t_to_danger, MIN_TOTAL_TTC)
            print(f"  [INFO] effective_t={effective_t:.2f}s  "
                f"(t_to_danger={t_to_danger:.2f}s, cap={MIN_TOTAL_TTC}s)")

            total_dist = TARGET_SPEED_MS * effective_t
            seg1       = total_dist - seg2

            if v_rot == 0:
                # +X vehicle: spawns to the left of COLLISION_X
                spawn_x = max(COLLISION_X - seg1, -117.0)
                actual_travel = (abs(spawn_x - COLLISION_X) + seg2) / TARGET_SPEED_MS
            else:
                # -X vehicle: spawns to the right of COLLISION_POINT_X
                spawn_x = min(COLLISION_POINT_X + seg1, 100.0)
                actual_travel = (abs(spawn_x - COLLISION_POINT_X) + seg2) / TARGET_SPEED_MS

            # Place vehicle at computed spawn position and rebuild its path
            spawn_y  = self.config["v_spawn"][1]
            spawn_tf = carla.Transform(
                carla.Location(x=spawn_x, y=spawn_y,
                               z=self.vehicle_driver.spawn_z),
                carla.Rotation(yaw=v_rot))

            self.target_vehicle.set_transform(spawn_tf)
            new_path = [(spawn_x, spawn_y)] + self.config["v_path"][1:]
            self.vehicle_driver = VehicleDriver(
                self.target_vehicle, new_path,
                TARGET_SPEED_MS, self.vehicle_driver.spawn_z)

            self.world.tick()
            self.active = True

            # ── Schedule alarms ───────────────────────────
            self._tick_B = self._tick_count

            wait_time = max(0.0, t_to_danger - actual_travel)
            self._vehicle_start_tick = self._tick_count + int(wait_time / DT)
            if wait_time > 0.01:
                print(f"  [INFO] spawn capped — vehicle waits {wait_time:.2f}s before moving")

            t_arrival = actual_travel + wait_time

            self._tick_early_alarm = self._tick_B + int((t_arrival - TTC_EARLY_WARN) / DT)
            self._tick_full_alarm  = self._tick_B + int((t_arrival - TTC_FULL_WARN)  / DT)

            ticks_to_early = self._tick_early_alarm - self._tick_count
            ticks_to_full  = self._tick_full_alarm  - self._tick_count

            timing_err = t_arrival - t_to_danger
            self.logger.timing_error = round(timing_err, 4)
            self._ped_ttc_at_f   = TTC_EARLY_WARN - timing_err
            self._ped_ttc_at_vha = TTC_FULL_WARN  - timing_err

            print(f"  spawn_x      = {spawn_x:.2f}")
            print(f"  vehicle ETA  = {actual_travel:.3f} s")
            if timing_err > TTC_FULL_WARN:
                print(f"  timing error = {timing_err:.3f}s  ⚠ vha feuert nach Danger-Passage — ggf. wiederholen!")
            else:
                print(f"  timing error = {timing_err:.3f}s  OK")
            print(f"  f   in {max(0, ticks_to_early * DT):.2f}s → ped TTC={self._ped_ttc_at_f:.2f}s "
                    f"({'BASELINE — not sent' if self.mode == 'b' else 'adaptive only'})")
            print(f"  vha in {max(0, ticks_to_full  * DT):.2f}s → ped TTC={self._ped_ttc_at_vha:.2f}s (both modes)")

        elif key == pygame.K_c and self._press_count >= 1:
            self._warn_aborted      = True
            self.logger.timestamp_C = round(now, 4)
            #print(f"\n[SIM] *** C pressed at t={now:.3f}s — alarms aborted ***")
            print(f"\n[SIM] *** C pressed at t={now:.3f}s — participant reacted (alarms still fire) ***")

        elif key == pygame.K_q:
            print("\n[SIM] Manual quit.")
            self.outcome    = "quit"
            self.target_hit = True


# ─────────────────────────────────────────────────────────
# SCENARIO RUNNER
# ─────────────────────────────────────────────────────────
def _run_scenario(run_id: str, mode: str, p_id: str, display):
    """Instantiate, run, and clean up a single scenario."""
    print(f"\n--- Starting '{run_id}' ({mode.upper()}) ---")
    mgr = SmombieScenario(mode, run_id, p_id, display)
    try:
        mgr.setup_world()
        mgr.run()
    except RuntimeError as e:
        print(f"[ERROR] {e}")

    print("\n  Participant walks back to start mark.")
    print("  Press ENTER when ready for next trial...")
    while True:
        for event in pygame.event.get():
            pass
        pygame.display.flip()
        if keyboard.is_pressed('enter'):
            while keyboard.is_pressed('enter'):
                time.sleep(0.01)
            break
        time.sleep(0.016)


# ─────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Launch Bridge in a separate console window
    bridge_proc = subprocess.Popen(
        [sys.executable, "main_bridge.py"],
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    time.sleep(2.0)
    print("[Bridge] Started.")

    # Initialise pygame
    pygame.init()
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

    # Load city ambience sound (optional — scenario still runs without it)
    city_sound  = None
    _sound_path = os.path.join(os.path.dirname(__file__), "city_ambience.wav")
    if os.path.exists(_sound_path):
        city_sound = pygame.mixer.Sound(_sound_path)
        city_sound.set_volume(0.5)
        print("[SIM] City ambience loaded.")
    else:
        print("[SIM] city_ambience.wav not found — running without sound.")

    # Open full-screen borderless window at the configured position
    os.environ['SDL_VIDEO_WINDOW_POS'] = f'{WINDOW_X},{WINDOW_Y}'
    info    = pygame.display.Info()
    display = pygame.display.set_mode(
        (info.current_w, info.current_h), pygame.NOFRAME)
    WideCamera.DISPLAY_W = display.get_width()
    WideCamera.DISPLAY_H = display.get_height()
    pygame.display.set_caption("Smombie Sim")

    # ── Participant ID ───────────────────────────────────
    while True:
        p_id = input("\nParticipant number (integer, e.g. 1): ").strip()
        if p_id.isdigit():
            break
        print("  Enter a plain integer.")
    p_num = int(p_id)

    # Counterbalancing: even → baseline first; odd → adaptive first
    block_order  = ['b', 'a'] if p_num % 2 == 0 else ['a', 'b']
    block_labels = {'b': 'BASELINE', 'a': 'ADAPTIVE'}
    print(f"\n  Participant {p_num}: "
          f"{block_labels[block_order[0]]} first, "
          f"then {block_labels[block_order[1]]}")

    # ── Session loop (two blocks) ────────────────────────
    for block_idx, mode in enumerate(block_order):
        print(f"\n{'='*55}")
        print(f"  BLOCK {block_idx + 1}: {block_labels[mode]}")
        print(f"{'='*55}")

        run_order = generate_run_order()
        actual_order = []
        print(f"  Run order: {run_order}")
        print("  Press ENTER to follow order, or type a run ID. 'q' to quit block.\n")

        order_iter = iter(run_order)
        running    = True

        while running:
            next_suggested = next(order_iter, None)

            if next_suggested:
                print(f"\n  Next: [{next_suggested}] ({TRIAL_TYPE[next_suggested].upper()})"
                      f" — ENTER to start, or type ID / 'q': ", end='', flush=True)
            else:
                print("\n  All trials done. ENTER to continue or 'q': ",
                      end='', flush=True)

            r = ''
            print("  > ", end='', flush=True)
            while True:
                for event in pygame.event.get():
                    pass
                pygame.display.flip()

                if keyboard.is_pressed('enter'):
                    while keyboard.is_pressed('enter'):
                        time.sleep(0.01)
                    print()
                    break
                elif keyboard.is_pressed('backspace'):
                    if r:
                        r = r[:-1]
                        print(f'\r  > {r}  ', end='', flush=True)
                    while keyboard.is_pressed('backspace'):
                        time.sleep(0.01)
                else:
                    for char in 'abcdefghijklmnopqrstuvwxyz1234567890':
                        if keyboard.is_pressed(char):
                            r += char
                            print(f'\r  > {r}', end='', flush=True)
                            while keyboard.is_pressed(char):
                                time.sleep(0.01)
                            break

                time.sleep(0.016)

            if r == 'q':
                running = False
            elif r == '' and next_suggested:
                actual_order.append(next_suggested)
                _run_scenario(next_suggested, mode, p_id, display)
            elif r in SCENARIOS:
                actual_order.append(r)
                _run_scenario(r, mode, p_id, display)
            elif r:
                print(f"  Unknown ID. Valid: {list(SCENARIOS.keys())}")

        if actual_order:
            # Keep last occurrence of each run, in relative order
            seen = set()
            deduped = []
            for run_id in reversed(actual_order):
                if run_id not in seen:
                    seen.add(run_id)
                    deduped.insert(0, run_id)

            log_dir = os.path.join('logs', f'P{p_id}')
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, 'run_order.csv'), 'a', encoding='utf-8') as f:
                f.write(f"{mode.upper()}_raw {','.join(actual_order)}\n")
                f.write(f"{mode.upper()} {','.join(deduped)}\n")

        print(f"\n  Block {block_idx + 1} complete.")
        if block_idx == 0:
            input("  Press Enter when ready to start the next block...")

    pygame.quit()
    print("\nSession ended.")
    bridge_proc.terminate()
    keyboard.unhook_all()