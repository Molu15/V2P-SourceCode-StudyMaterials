import carla
import pygame
import math
import sys
import os

# --- Configuration ---
MAP_NAME = 'Town10HD'
TARGET_SPEED_KMH = 50.0
TARGET_SPEED_MS = TARGET_SPEED_KMH / 3.6
P_LOC = carla.Location(x=-55.0, y=-22.0, z=1.0) # Ground Truth Observation Point

class PlaybackManager:
    def __init__(self):
        # 1. CARLA Setup
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(60.0)
        print(f"Loading {MAP_NAME}...")
        self.world = self.client.load_world(MAP_NAME)
        
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05 
        self.world.apply_settings(settings)
        self.clock = pygame.time.Clock()

        # 2. State Variables
        self.vehicle = None
        self.pedestrian = None
        self.is_running = False
        self.is_frozen = False
        self.current_ttc = 5.0 # Default starting TTC

        # 3. Pygame Setup
        pygame.init()
        self.screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("TTC Playback Tool")
        self.font = pygame.font.SysFont("Arial", 18)

    def calculate_spawn_y(self, ttc):
        """Calculates Y coordinate based on desired TTC and constant speed."""
        # Distance = Speed * Time
        required_distance = TARGET_SPEED_MS * ttc
        # Car spawns North (+Y) of the pedestrian
        return P_LOC.y + required_distance

    def setup_trial(self):
        """Resets the scene and asks for a new TTC value via terminal."""
        self.cleanup()
        self.is_running = False
        self.is_frozen = False

        # Get User Input for TTC from Terminal
        try:
            print("\n" + "="*30)
            val = input(f"Enter desired TTC in seconds (Current: {self.current_ttc}s): ")
            if val.strip():
                self.current_ttc = float(val)
        except ValueError:
            print(">> Invalid input. Keeping previous TTC.")

        # Spawn Pedestrian
        #underground_loc = carla.Location(x=P_LOC.x, y=P_LOC.y, z=P_LOC.z - 0.2)
        p_bp = self.world.get_blueprint_library().filter("walker.pedestrian.0001")[0]
        #p_bp.set_attribute('role_name', 'invisible_point')
        self.pedestrian = self.world.try_spawn_actor(p_bp, carla.Transform(P_LOC))
        #self.pedestrian = self.world.try_spawn_actor(p_bp, carla.Transform(underground_loc))


        # 2. Spawn Red Vehicle at calculated TTC distance
        v_bp = self.world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
        if v_bp.has_attribute('color'):
            v_bp.set_attribute('color', '255,0,0') # Red
        
        spawn_y = self.calculate_spawn_y(self.current_ttc)
        v_loc = carla.Location(x=-52.0, y=spawn_y, z=1.0)
        
        # Yaw -90 faces South (toward the pedestrian)
        self.vehicle = self.world.try_spawn_actor(v_bp, carla.Transform(v_loc, carla.Rotation(yaw=-90)))

        # 3. Set Spectator POV (The "Eye" of the Participant)
        # Set Pedestrian POV for the Researcher/Participant
        spectator = self.world.get_spectator()
        spec_transform = carla.Transform(
            P_LOC + carla.Location(z=0.95), # Eyes at 1.7m
            carla.Rotation(pitch=-5, yaw=90) # Looking north toward approaching car
        )
        spectator.set_transform(spec_transform)

        # Let physics settle
        for _ in range(10): 
            self.world.tick()
            
        print(f">> READY: TTC {self.current_ttc}s | Start Dist: {TARGET_SPEED_MS * self.current_ttc:.2f}m")
        print(">> Press 'S' in Pygame window to Start | 'R' to change TTC")

    def start_driving(self):
        """Activates constant velocity using your proven direction logic."""
        self.is_running = True
        self.is_frozen = False
        
        # Get car forward vector
        fwd = self.vehicle.get_transform().get_forward_vector()
        
        # Using your -forward_vec.y logic that keeps the car driving straight
        target_velocity = carla.Vector3D(
            x = -fwd.y * TARGET_SPEED_MS, 
            y = 0, 
            z = 0
        )
        
        self.vehicle.enable_constant_velocity(target_velocity)
        print(f"Playing back: {TARGET_SPEED_KMH} km/h")

    def run(self):
        try:
            self.setup_trial()
            while True:
                # Sync loop to 20 FPS (matches fixed_delta_seconds 0.05)
                self.clock.tick(20) 
                
                if not self.is_frozen:
                    self.world.tick()

                # UI Update
                self.screen.fill((20, 20, 20))
                status_color = (0, 255, 0) if self.is_running and not self.is_frozen else (200, 200, 200)
                status_txt = "FROZEN" if self.is_frozen else ("PLAYING" if self.is_running else "READY")
                
                self.screen.blit(self.font.render(f"Mode: TTC Playback", True, (255,255,255)), (20, 40))
                self.screen.blit(self.font.render(f"Target TTC: {self.current_ttc}s", True, (255,255,255)), (20, 70))
                self.screen.blit(self.font.render(f"Status: {status_txt}", True, status_color), (20, 110))
                self.screen.blit(self.font.render("S: Start | R: Reset/Change TTC", True, (120,120,120)), (20, 220))
                pygame.display.flip()

                # Input Handling
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: 
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_s and not self.is_running:
                            self.start_driving()
                        if event.key == pygame.K_r:
                            self.setup_trial()

                # Auto-freeze logic: 10m past the observation point
                if self.is_running and not self.is_frozen:
                    v_loc = self.vehicle.get_location()
                    # If vehicle has passed P_LOC.y by 10m
                    if v_loc.y < (P_LOC.y - 10.0):
                        self.is_frozen = True
                        self.vehicle.enable_constant_velocity(carla.Vector3D(0,0,0))
                        print(">> Playback finished. Car is 10m past. Press 'R' to reset.")

        finally:
            # Restore original CARLA settings on exit
            self.world.apply_settings(self.original_settings)
            self.cleanup()
            pygame.quit()

    def cleanup(self):
        """Safely destroy actors."""
        if self.vehicle: self.vehicle.destroy()
        if self.pedestrian: self.pedestrian.destroy()
        self.vehicle = None
        self.pedestrian = None

if __name__ == "__main__":
    try:
        sim = PlaybackManager()
        sim.run()
    except KeyboardInterrupt:
        print("\nSimulation stopped.")