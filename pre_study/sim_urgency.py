import carla
import pygame
import csv
import time
import math
import sys
import os


# --- Configuration ---
MAP_NAME = 'Town10HD'
TARGET_SPEED_KMH = 50.0
TARGET_SPEED_MS = TARGET_SPEED_KMH / 3.6
START_DISTANCE = 100.0


# Coordinates for a clear straightaway in Town10HD
#P_LOC = carla.Location(x=-52.5, y=-40.0, z=0.2)
P_LOC = carla.Location(x=-55.0, y=-22.0, z=1.0)



class SimulationManager:

    def __init__(self):
        # 1. Initialize CARLA Client
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(60.0)

       
        print(f"Loading {MAP_NAME} (this may take a moment)...")
        self.world = self.client.load_world(MAP_NAME)
        self.blueprint_library = self.world.get_blueprint_library()

       
        # 2. Synchronous Mode Settings (Crucial for TTC accuracy)
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05 # 20 FPS
        self.world.apply_settings(settings)
        self.clock = pygame.time.Clock()


        # 3. Participant Setup
        self.participant_id = input("Enter Participant ID: ")

        self.log_folder = "logs_urgency"
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
            
        # Combine folder and filename
        self.filename = os.path.join(self.log_folder, f"ttc_log_{self.participant_id}.csv")
        #self.filename = f"ttc_log_{self.participant_id}.csv"

       
        # 4. State Variables
        self.vehicle = None
        self.pedestrian = None
        self.is_running = False
        self.is_frozen = False
        self.log_data = []

       
        # 5. Pygame Setup
        pygame.init()
        self.screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption(f"Urgency Study")
        self.font = pygame.font.SysFont("Arial", 18)


    def setup_actors(self):
        """Spawns actors and resets the trial state."""
        self.cleanup()
        self.is_running = False
        self.is_frozen = False
        self.log_data = []


        # Spawn Pedestrian
        #underground_loc = carla.Location(x=P_LOC.x, y=P_LOC.y, z=P_LOC.z - 0.2)
        p_bp = self.blueprint_library.filter("walker.pedestrian.0001")[0]

        p_bp.set_attribute('role_name', 'invisible_point')

        self.pedestrian = self.world.try_spawn_actor(p_bp, carla.Transform(P_LOC))
        #self.pedestrian = self.world.try_spawn_actor(p_bp, carla.Transform(underground_loc))


        # Spawn Vehicle 100m away (y+100)
        # Note: z=1.0 prevents the car from sticking to the ground on spawn
        v_bp = self.blueprint_library.filter("vehicle.tesla.model3")[0]

        # --- NEW: SET VEHICLE COLOR TO RED ---
        if v_bp.has_attribute('color'):
            v_bp.set_attribute('color', '255,0,0')

        #v_loc = carla.Location(x=P_LOC.x, y=P_LOC.y + START_DISTANCE, z=1.0)
        v_loc = carla.Location(x=-52.0, y=P_LOC.y + START_DISTANCE, z=1.0)
        self.vehicle = self.world.try_spawn_actor(v_bp, carla.Transform(v_loc, carla.Rotation(yaw=-90)))

        if not self.vehicle or not self.pedestrian:
            print("ERROR: Failed to spawn actors. Check for collisions at spawn points.")
            return


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


        print(f"\n--- TRIAL READY (ID: {self.participant_id}) ---")
        print("S: Start | Space: Log TTC | R: Save & Reset")


    def log_ttc(self):
        """Calculates and stores distance and TTC."""
        v_loc = self.vehicle.get_location()
        p_loc = self.pedestrian.get_location()

       
        # Calculate Euclidean Distance
        dist = v_loc.distance(p_loc)

       
        # Formula: TTC = Distance / Constant Velocity
        ttc = dist / TARGET_SPEED_MS

       
        self.log_data.append([time.time(), round(dist, 2), round(ttc, 3)])
        print(f"REGISTERED: Distance = {dist:.2f}m, TTC = {ttc:.2f}s")


    def start_driving(self):
        self.is_running = True
        self.is_frozen = False

       
        # 1. Clear any previous physics state
        forward_vec = self.vehicle.get_transform().get_forward_vector()
        
        target_velocity = carla.Vector3D(
            x = -forward_vec.y * TARGET_SPEED_MS, 
            y = 0, 
            z = 0
        )
       
        self.vehicle.enable_constant_velocity(target_velocity)
        print(f"Vehicle moving forward at {TARGET_SPEED_KMH} km/h")


    def save_to_csv(self):
        """Saves recorded data to the participant's file."""
        if not self.log_data:
            return

        # 'a' mode allows multiple runs (R key) to be appended to the same file
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["Unix_Timestamp", "Distance_m", "TTC_s"])
            writer.writerows(self.log_data)
        print(f"Data saved to: {os.path.abspath(self.filename)}")


    def run(self):
        try:
            self.setup_actors()

           
            while True:
                self.clock.tick(20)
                if not self.is_frozen:
                    self.world.tick()

               
                # Simple Pygame UI
                self.screen.fill((20, 20, 20))
                color = (0, 255, 0) if self.is_running and not self.is_frozen else (200, 200, 200)
                status_txt = "FROZEN" if self.is_frozen else ("MOVING" if self.is_running else "READY")

               
                #self.screen.blit(self.font.render(f"Participant ID: {self.participant_id}", True, (255,255,255)), (20, 50))
                self.screen.blit(self.font.render(f"Status: {status_txt}", True, color), (20, 100))
                self.screen.blit(self.font.render("S: Start | Space: Log | R: Reset", True, (150,150,150)), (20, 200))
                pygame.display.flip()


                # Event Handling
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_s and not self.is_running:
                            self.start_driving()
                        if event.key == pygame.K_SPACE and self.is_running and not self.is_frozen:
                            self.log_ttc()
                        if event.key == pygame.K_r:
                            self.save_to_csv()
                            self.setup_actors()

                if self.is_running and not self.is_frozen:
                    v_loc = self.vehicle.get_location()
                    p_loc = self.pedestrian.get_location()
                    
                    # Logic to freeze 10m past the pedestrian
                    # In Town10 moving down this road, "past" means vehicle.y < pedestrian.y
                    if v_loc.y < (p_loc.y - 10.0):
                        self.is_frozen = True
                        self.vehicle.enable_constant_velocity(carla.Vector3D(0,0,0))
                        print("Trial Ended: 10m Past Pedestrian. Simulation Frozen. Press 'R' to reset.")


 #               # Auto-stop if vehicle passes pedestrian
 #               if self.is_running:
 #                   current_dist = self.vehicle.get_location().distance(self.pedestrian.get_location())
 #                   if current_dist < 1.5: # Stop just before impact
 #                       self.is_running = False
 #                       self.vehicle.enable_constant_velocity(carla.Vector3D(0,0,0))
 #                       print("Trial reached end point. Press 'R' to save and reset.")


        finally:
            # Clean up on exit
            self.save_to_csv()
            self.world.apply_settings(self.original_settings)
            self.cleanup()
            pygame.quit()


    def cleanup(self):
        if self.vehicle: self.vehicle.destroy()
        if self.pedestrian: self.pedestrian.destroy()
        self.vehicle = None
        self.pedestrian = None


if __name__ == "__main__":
    try:
        sim = SimulationManager()
        sim.run()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")