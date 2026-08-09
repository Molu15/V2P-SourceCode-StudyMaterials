import socket
import csv
import os
import threading
from datetime import datetime
from pynput import keyboard

# --- Konfiguration ---
LAPTOP_IP = "0.0.0.0" 
#PHONE_IP = "10.198.69.212"  # Deine S25 IP
IN_PORT = 5006              # Empfang vom Handy
OUT_PORT = 5007             # Senden an Flutter
OUT_PORT_KOTLIN = 5008      # Senden an Kotlin Overlay

LOG_DIR = "logs_alarm"

class WoZBridge:
    def __init__(self):
        # UDP Sockets
        self.sock_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_in.bind((LAPTOP_IP, IN_PORT))
        self.sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        self.is_running = True
        self.phone_ip = None  # ← neu

        # Modality Mapping
        self.MODALITY_MAP = {
            '1': {"name": "Visual only", "cmd": "v"},
            '2': {"name": "Haptic only", "cmd": "h"},
            '3': {"name": "Auditory only", "cmd": "a"},
            '4': {"name": "Visual + Haptic", "cmd": "vh"},
            '5': {"name": "Visual + Auditory", "cmd": "va"},
            '6': {"name": "Haptic + Auditory", "cmd": "ha"},
            '7': {"name": "Multimodal", "cmd": "vha"},
        }

        # Logging Setup
        self.participant_id = input("Enter Participant ID: ")
        self.log_filename = os.path.join(LOG_DIR, f"modality_selection_{self.participant_id}.csv")
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)

    def log_event(self, combination_type):
        """Speichert den Tastendruck für die spätere Analyse von RQ1a/b."""
        file_exists = os.path.isfile(self.log_filename)
        with open(self.log_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Combination_Type"])
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), combination_type])

    def listen_to_phone(self):
        print(f"[*] Höre auf Port {IN_PORT} für Handy-Daten...")
        while self.is_running:
            try:
                data, addr = self.sock_in.recvfrom(1024)
                if self.phone_ip != addr[0]:
                    self.phone_ip = addr[0]
                    print(f"\n[WoZ] Handy-IP erkannt: {self.phone_ip}")
            except:
                break

    def send_alarm(self, mode):
        if self.phone_ip is None:
            print("[!] Handy noch nicht verbunden — warte auf erstes Paket")
            return
        print(f"[!] Sende Alarm '{mode}' → {self.phone_ip}")
        try:
            self.sock_out.sendto(mode.encode(), (self.phone_ip, OUT_PORT))
            self.sock_out.sendto(mode.encode(), (self.phone_ip, OUT_PORT_KOTLIN))
        except Exception as e:
            print(f"Sende-Fehler: {e}")

    def on_press(self, key):
        try:
            if hasattr(key, 'char'):
                if key.char in self.MODALITY_MAP:
                    selection = self.MODALITY_MAP[key.char]
                    self.send_alarm(selection['cmd'])
                    self.log_event(selection['name'])
                
                if key.char == 'q':
                    print("\nBeende System...")
                    self.is_running = False
                    return False
        except Exception as e:
            print(f"Fehler: {e}")

if __name__ == "__main__":
    bridge = WoZBridge()
    
    # Empfangs-Thread starten
    threading.Thread(target=bridge.listen_to_phone, daemon=True).start()

    print(f"\n--- Wizard-of-Oz Study Control ---")
    print(f"Proband: {bridge.participant_id}")
    print("Tasten 1-7: Alarme senden | q: Beenden")
    print("-" * 35)

    with keyboard.Listener(on_press=bridge.on_press) as listener:
        listener.join()