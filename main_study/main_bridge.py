import socket
import threading
import time

# ─── CONFIG ───────────────────────────────────────────────
LAPTOP_IP       = "0.0.0.0"
PHONE_IN_PORT   = 5006   # Any packet from phone → bridge learns its IP
PHONE_OUT_PORT  = 5007   # Alarms to phone (Flutter)
PHONE_OUT_PORT2 = 5008   # Alarms to phone (Kotlin overlay)
CARLA_IN_PORT   = 5011   # Alarms from CARLA


class Bridge:
    def __init__(self):
        self.phone_ip   = None
        self.is_running = True

        # Receives any packet from phone to learn its IP address
        self.sock_from_phone = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_from_phone.bind((LAPTOP_IP, PHONE_IN_PORT))
        self.sock_from_phone.settimeout(0.1)

        # Sends alarms to phone
        self.sock_to_phone = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Receives alarm strings from CARLA
        self.sock_from_carla = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_from_carla.bind((LAPTOP_IP, CARLA_IN_PORT))
        self.sock_from_carla.settimeout(0.1)

    def _detect_phone_ip(self):
        """Wait for any incoming packet from the phone to record its IP."""
        while self.is_running:
            try:
                _, addr = self.sock_from_phone.recvfrom(1024)
                if self.phone_ip != addr[0]:
                    self.phone_ip = addr[0]
                    print(f"\n[Bridge] Phone connected: {self.phone_ip}\n")
            except socket.timeout:
                pass
            except Exception as e:
                print(f"\n[Bridge] Phone listener error: {e}")

    def _relay_alarms(self):
        """Forward CARLA alarm strings to both phone ports."""
        while self.is_running:
            try:
                data, _ = self.sock_from_carla.recvfrom(1024)
                alarm   = data.decode().strip()
                print(f"\n[Bridge] Alarm: '{alarm}' → Phone")
                if self.phone_ip:
                    self.sock_to_phone.sendto(alarm.encode(),
                                              (self.phone_ip, PHONE_OUT_PORT))
                    self.sock_to_phone.sendto(alarm.encode(),
                                              (self.phone_ip, PHONE_OUT_PORT2))
                else:
                    print("[Bridge] Phone not connected — alarm dropped.")
            except socket.timeout:
                pass
            except Exception as e:
                print(f"\n[Bridge] CARLA listener error: {e}")

    def run(self):
        # Determine laptop IP so the experimenter can enter it in the phone app
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            laptop_ip = s.getsockname()[0]
            s.close()
        except Exception:
            laptop_ip = "unknown — check ipconfig"

        print("\n╔══════════════════════════════════════════╗")
        print("║              SMOMBIE BRIDGE              ║")
        print("╚══════════════════════════════════════════╝")
        print(f"\n  Laptop IP : {laptop_ip}")
        print(f"  Port      : {PHONE_IN_PORT}")
        print(f"\n  Enter this IP in the phone app, then start the session.")
        print(f"  Waiting for phone connection...\n")

        threading.Thread(target=self._detect_phone_ip, daemon=True).start()
        threading.Thread(target=self._relay_alarms,    daemon=True).start()

        while not self.phone_ip:
            time.sleep(0.2)

        print("[Bridge] Running. Press q + Enter to quit.\n")

        while True:
            try:
                cmd = input().strip().lower()
            except EOFError:
                break
            if cmd == 'q':
                self.is_running = False
                break


if __name__ == "__main__":
    Bridge().run()