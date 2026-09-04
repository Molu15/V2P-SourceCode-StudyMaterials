import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
VIDEO_PATH = "P1 - video.mp4"
MODEL_PATH = "face_landmarker.task"
LOG_PATHS = ["modality_selection_P1.a.csv", "modality_selection_P1.v.csv"]

# NEU: Die passenden Beschriftungen für die beiden Dateien oben (gleiche Reihenfolge!)
DISTRACTION_LABELS = ["Auditory Distraction", "Visual Distraction"]

ANKER_LOG_INDEX = 0
ANKER_ALARM_INDEX = 1 
ANKER_VIDEO_ZEIT = 40.0

WINDOW_SIZE = 4.0 

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def map_to_short_modality(full_type):
    if pd.isna(full_type): return ""
    t = str(full_type).strip().lower()
    if t == "visual only": return "V"
    if t == "haptic only": return "H"
    if t == "auditory only": return "A"
    if t == "multimodal": return "V+H+A"
    parts = []
    if "visual" in t: parts.append("V")
    if "haptic" in t: parts.append("H")
    if "auditory" in t: parts.append("A")
    return "+".join(parts) if parts else full_type

def sync_multiple_logs(log_paths, anker_log_idx, anker_alarm_idx, anker_video_time):
    ref_df = pd.read_csv(log_paths[anker_log_idx], parse_dates=['Timestamp'])
    anker_timestamp = ref_df['Timestamp'].iloc[anker_alarm_idx]
    all_alarms = []
    
    for path in log_paths:
        try:
            df = pd.read_csv(path, parse_dates=['Timestamp'])
            df['Video_Second'] = (df['Timestamp'] - anker_timestamp).dt.total_seconds() + anker_video_time
            df['Short_Type'] = df['Combination_Type'].apply(map_to_short_modality)
            all_alarms.append(df)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            
    combined_df = pd.concat(all_alarms, ignore_index=True)
    combined_df = combined_df.sort_values(by='Video_Second').reset_index(drop=True)
    
    # UNIVERSELLER DYNAMISCHER FIX: 
    # Behalte nur Alarme, die >= 0 Sekunden sind (also tatsächlich im Video existieren).
    # Alle Vortests, die vor dem Videoschnitt stattfanden, fliegen automatisch raus!
    combined_df = combined_df[combined_df['Video_Second'] >= 0].reset_index(drop=True)
    
    return combined_df

# ==========================================
# 3. ANALYSIS WITH HEAD MOVEMENT (JERK)
# ==========================================
def analyze_video_tasks(video_path, alarm_times, model_path):
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1
    )

    results_data = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    
    # Variablen für Bewegungstracking
    last_nose_x, last_nose_y = None, None

    print("Starting MediaPipe analysis...")
    with FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            if frame_count % 500 == 0 or frame_count == total_frames:
                print(f"Progress: Frame {frame_count}/{total_frames} ({int((frame_count / total_frames) * 100)}%)")
            
            frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            current_second = frame_timestamp_ms / 1000.0
            
            # Prüfen, in WELCHEM Alarmfenster wir uns befinden (für saubere Trennung im Plot)
            current_alarm_id = None
            for idx, t in enumerate(alarm_times):
                if t <= current_second <= t + WINDOW_SIZE:
                    current_alarm_id = idx
                    break
            
            if current_alarm_id is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                
                if detection_result.face_landmarks:
                    mesh = detection_result.face_landmarks[0]
                    
                    # EAR (Eye Opening)
                    ear = abs(mesh[159].y - mesh[145].y)
                    
                    # Kopfbewegung (Nase Index 1) berechnen
                    nose_x, nose_y = mesh[1].x, mesh[1].y
                    
                    if last_nose_x is not None:
                        # Euklidische Distanzänderung der Nase zum vorherigen Frame (Geschwindigkeit)
                        movement = np.sqrt((nose_x - last_nose_x)**2 + (nose_y - last_nose_y)**2)
                    else:
                        movement = 0.0
                    
                    results_data.append({
                        'time': current_second,
                        'ear': ear,
                        'movement': movement,
                        'alarm_id': current_alarm_id # Verhindert das Zusammenzeichnen der Intervalle
                    })
                    
                    last_nose_x, last_nose_y = nose_x, nose_y
            else:
                # Außerhalb der Fenster setzen wir das Tracking zurück
                last_nose_x, last_nose_y = None, None
                    
    cap.release()
    return pd.DataFrame(results_data)

# ==========================================
# 4. MAIN WORKFLOW & PLOTTING
# ==========================================
try:
    df_logs = sync_multiple_logs(LOG_PATHS, DISTRACTION_LABELS, ANKER_LOG_INDEX, ANKER_ALARM_INDEX, ANKER_VIDEO_ZEIT)
    analysis_df = analyze_video_tasks(VIDEO_PATH, df_logs['Video_Second'].tolist(), MODEL_PATH)

    if not analysis_df.empty:
        # Erstelle ein Diagramm mit 2 separaten Achsen (Y1 = Augen, Y2 = Bewegung)
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax2 = ax1.twinx() # Gemeinsame X-Achse

        # Zeichne Daten blockweise, um lange Verbindungslinien zu verhindern
        for alarm_id in analysis_df['alarm_id'].unique():
            block = analysis_df[analysis_df['alarm_id'] == alarm_id]
            ax1.plot(block['time'], block['ear'], color='blue', alpha=0.6, label="Eye Opening (EAR)" if alarm_id == 0 else "")
            ax2.plot(block['time'], block['movement'], color='darkorange', alpha=0.7, label="Head Movement (Jerk)" if alarm_id == 0 else "")

        # Alarme einzeichnen
        for idx, row in df_logs.iterrows():
            ax1.axvline(x=row['Video_Second'], color='red', linestyle='--', alpha=0.6)
            ax1.text(row['Video_Second'], ax1.get_ylim()[1] * 0.85, f" {row['Short_Type']}", rotation=90, color='red', fontweight='bold')

        # Achsen-Styling
        ax1.set_xlabel("Video Timeline (Seconds)", fontsize=12)
        ax1.set_ylabel("Relative Eye Opening (EAR)", color='blue', fontsize=12)
        ax2.set_ylabel("Head Movement Magnitude (Jerk)", color='darkorange', fontsize=12)
        
        # Kombinierte Legende
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.title("Participant Reaction Analysis: Eyes vs. Head Movement (P1)", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"Reaction_Analysis_{VIDEO_PATH.split('.')[0]}.png", dpi=300)
        plt.show()
    else:
        print("No data detected.")
except Exception as e:
    print(f"An error occurred: {e}")