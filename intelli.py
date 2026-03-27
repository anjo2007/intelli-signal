import cv2
import numpy as np
from ultralytics import YOLO
import time
import socket
import json
import threading
import argparse
import textwrap

# Optional audio siren detection (disabled by default)
SIREN_DETECTION_ENABLED = False
SIREN_HOLD = 5.0  # seconds to consider a recent siren detection
SIREN_ENERGY_THRESHOLD = 5e4  # energy threshold for siren band (may need tuning)

# --- CONFIGURATION ---
# Replace these with '0' for webcam, or paths to video files (e.g., 'traffic.mp4')
# For this demo, we use '0' (webcam) for all 3 to simulate the grid if you don't have 3 cams.
CAMERA_SOURCES = ["C:\\Users\\chris\\OneDrive\\Desktop\\intelli signal\\1.mp4", "C:\\Users\\chris\\OneDrive\\Desktop\\intelli signal\\2.mp4", "C:\\Users\\chris\\OneDrive\\Desktop\\intelli signal\\3.mp4"] 

MIN_GREEN = 10
MAX_GREEN = 60
SEC_PER_VEHICLE = 3
MODEL_PATH = 'yolov8n.pt' # Downloads automatically if missing

# --- UDP CONFIG ---
UDP_ENABLED = True
UDP_TARGET_IP = '192.168.1.2'
UDP_TARGET_PORT = 5005
UDP_SEND_INTERVAL = 1.0  # seconds (how often to send updates)
CONGESTION_THRESHOLD = 6  # vehicles -> 'congestion'
EMERGENCY_THRESHOLD = 12  # vehicles -> 'emergency'  
# Emergency light detection (red/blue flashing) -------------------------------------------------
EMER_LIGHT_PIXEL_THRESHOLD = 300  # number of red/blue pixels to consider lights present
EMER_LIGHT_PIXEL_HIGH_THRESHOLD = EMER_LIGHT_PIXEL_THRESHOLD * 2  # higher threshold for large vehicles
EMER_LIGHT_HOLD = 5  # seconds to hold emergency status after detection
EMERGENCY_GREEN = MAX_GREEN  # green time to assign during emergency preemption

# --- Dynamic timing & smoothing ---
MIN_GREEN_ELAPSED = 3    # seconds before green can be shortened
MIN_REMAINING = 2       # minimum remaining seconds when shortening
COUNTS_SMOOTHING_ALPHA = 0.4  # EMA alpha for smoothing vehicle counts




class IntelliSignalVisualizer:
    def __init__(self):
        print("Loading YOLO model (this may take a moment)...")
        self.model = YOLO(MODEL_PATH)
        self.caps = [cv2.VideoCapture(src) for src in CAMERA_SOURCES]
        self.road_names = ["Road A (Main)", "Road B (Cross)", "Road C (Side)"]
        self.timers = [MIN_GREEN, MIN_GREEN, MIN_GREEN]
        self.active_road = 0  # Index of the road currently having Green light
        self.start_time = time.time()

        # UDP sender initialization
        self.udp_enabled = UDP_ENABLED
        self.udp_target = (UDP_TARGET_IP, UDP_TARGET_PORT)
        self.udp_sock = None
        self.last_udp_send = 0.0
        # Track last detected emergency time per road (seconds since epoch)
        self.last_emergency = [0.0] * len(self.caps)
        # Track last detected siren time (global)
        self.last_siren = 0.0
        # Exponential moving average (EMA) of counts to avoid jitter
        self.counts_ema = [0.0] * len(self.caps)
        if SIREN_DETECTION_ENABLED:
            self._start_siren_monitor()
        if self.udp_enabled:
            try:
                self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.udp_sock.setblocking(False)
            except Exception as e:
                print("UDP init failed:", e)
                self.udp_enabled = False

    def process_frame(self, frame, road_index, em_thresh=None, em_high=None):
        """Detects vehicles, emergency lights, and draws boxes on a single frame.
        Returns (annotated_frame, vehicle_count, emergency_flag)
        Optional params override emergency thresholds for tuning.
        """
        if frame is None:
            return None, 0, False

        em_thresh = em_thresh or EMER_LIGHT_PIXEL_THRESHOLD
        em_high = em_high or EMER_LIGHT_PIXEL_HIGH_THRESHOLD

        # Run YOLO inference
        results = self.model(frame, verbose=False, classes=[2, 3, 5, 7]) # Classes: car, motorcycle, bus, truck
        
        # Count vehicles
        vehicle_count = len(results[0].boxes)
        
        # Draw bounding boxes (plot returns a BGR numpy array)
        annotated_frame = results[0].plot()
        
        # Add road name and count overlay
        cv2.putText(annotated_frame, f"{self.road_names[road_index]}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Vehicles: {vehicle_count}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)

        # --- Visual emergency detection DISABLED: manual control only ---
        emergency = False
        # NOTE: Per request, we do not set emergency state from video analysis.
        # Emergency is controlled only via keyboard: press 'e' then '1'/'2'/'3' to trigger lane A/B/C.
        return annotated_frame, vehicle_count, emergency

    def update_logic(self, counts, counts_ema):
        """Updates the timer logic based on the active road, adjusting live using smoothed counts.
        Emergency preemption: if an emergency is active on any road (recent light/siren),
        the system will switch immediately to that road and *hold* green until the emergency clears.
        Otherwise, normal adaptive timing applies using EMA counts.
        """
        current_time = time.time()

        # --- Emergency preemption and hold ---
        active_emergencies = [i for i, t in enumerate(self.last_emergency) if current_time - t <= EMER_LIGHT_HOLD]
        if active_emergencies:
            # choose the most recently detected emergency road
            emergency_idx = max(active_emergencies, key=lambda i: self.last_emergency[i])
            if emergency_idx != self.active_road:
                # Immediate preemption
                self.active_road = emergency_idx
                self.start_time = current_time
                self.timers[self.active_road] = max(self.timers[self.active_road], EMERGENCY_GREEN)
            # While emergency is active, do not switch away from this road
            return

        # --- No emergency active: normal adaptive timing ---
        elapsed = current_time - self.start_time
        current_timer_limit = self.timers[self.active_road]

        # Compute desired time for active road using EMA
        desired_time = max(MIN_GREEN, min(MAX_GREEN, counts_ema[self.active_road] * SEC_PER_VEHICLE))

        # Extend green if desired is larger
        if desired_time > current_timer_limit:
            self.timers[self.active_road] = desired_time

        # Shorten green if desired is smaller, but only if we've passed MIN_GREEN_ELAPSED
        elif desired_time < current_timer_limit:
            if elapsed >= MIN_GREEN_ELAPSED:
                # Only shorten if the new remaining would be at least MIN_REMAINING seconds
                new_remaining = desired_time - elapsed
                if new_remaining >= MIN_REMAINING:
                    self.timers[self.active_road] = desired_time

        # If time is up for the current road, switch to the next
        elapsed = current_time - self.start_time
        if elapsed > self.timers[self.active_road]:
            self.active_road = (self.active_road + 1) % len(self.caps)
            self.start_time = current_time
            # Recalculate timer for the NEW active road using EMA
            density = counts_ema[self.active_road]
            calculated_time = density * SEC_PER_VEHICLE
            self.timers[self.active_road] = max(MIN_GREEN, min(MAX_GREEN, calculated_time))

    def compute_status(self, counts, last_emergency_times):
        """Compute simplified status per road: clear, less traffic, congestion, emergency.
        A road is 'emergency' if an emergency light was detected within EMER_LIGHT_HOLD seconds.
        """
        statuses = []
        now = time.time()
        for i, c in enumerate(counts):
            # Emergency holds for a short duration after detection (based on lights/siren)
            if now - last_emergency_times[i] <= EMER_LIGHT_HOLD:
                statuses.append("emergency")
            elif c >= CONGESTION_THRESHOLD:
                statuses.append("congestion")
            elif c > 0:
                statuses.append("less traffic")
            else:
                statuses.append("clear")
        return statuses

    def send_udp(self, counts):
        """Send per-road states and active road as JSON via UDP."""
        if not self.udp_enabled or self.udp_sock is None:
            return
        statuses = self.compute_status(counts, self.last_emergency)
        # Plain-text message, one entry per line, no braces
        lines = []
        for i, st in enumerate(statuses):
            lines.append(f"{self.road_names[i]}: {st}")
        msg = "\n".join(lines)
        try:
            self.udp_sock.sendto(msg.encode('utf-8'), self.udp_target)
        except Exception as e:
            # Non-blocking send may raise; print for debugging (no crash)
            print("UDP send error:", e)

    def _start_siren_monitor(self):
        """Start a background thread that listens for siren-like audio patterns.
        Requires `sounddevice` to be installed. This is optional and disabled by default.
        """
        def monitor():
            try:
                import sounddevice as sd
            except Exception as e:
                print("Siren monitor unavailable (sounddevice missing):", e)
                return
            fs = 16000
            duration = 1.0
            while True:
                try:
                    rec = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
                    sd.wait()
                    sig = rec.flatten()
                    # FFT and energy in typical siren band (400-2000 Hz)
                    S = np.abs(np.fft.rfft(sig))
                    freqs = np.fft.rfftfreq(len(sig), 1/fs)
                    idx = np.where((freqs >= 400) & (freqs <= 2000))[0]
                    band_energy = float(S[idx].sum())
                    if band_energy >= SIREN_ENERGY_THRESHOLD:
                        self.last_siren = time.time()
                    time.sleep(0.1)
                except Exception:
                    time.sleep(0.5)
        t = threading.Thread(target=monitor, daemon=True)
        t.start()
    def create_dashboard(self, frames, counts, emergencies=None):
        """Combines 3 frames and an info panel into one grid.
        Also overlays a per-road signal indicator and alert box aligned with each road's frame.
        """
        # Resize frames to be identical for the grid
        h, w = 360, 640
        resized_frames = [cv2.resize(f, (w, h)) if f is not None else np.zeros((h, w, 3), dtype=np.uint8) for f in frames]

        # Determine if any active emergency (most recent)
        now = time.time()
        active_emergencies = [i for i, t in enumerate(self.last_emergency) if now - t <= EMER_LIGHT_HOLD]
        emergency_idx = max(active_emergencies, key=lambda i: self.last_emergency[i]) if active_emergencies else None

        # Decide display-active (emergency overrides normal active road)
        display_active = emergency_idx if emergency_idx is not None else self.active_road

        # Overlay per-frame signal indicator and alert messages
        for i, frm in enumerate(resized_frames):
            # Signal circle top-right
            cx, cy = w - 60, 40
            radius = 20
            if i == display_active:
                # Green signal (emergency or normal)
                cv2.circle(frm, (cx, cy), radius, (0, 255, 0), -1)
                label = "GREEN"
                if emergency_idx is not None and i == emergency_idx:
                    label = "GREEN (EMERGENCY)"
                cv2.putText(frm, label, (cx - 180, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                # Red signal for every other lane when emergency active, or default red otherwise
                cv2.circle(frm, (cx, cy), radius, (0, 0, 255), -1)
                cv2.putText(frm, "RED", (cx - 60, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # If an emergency is active on some road, display messages appropriately
            if emergency_idx is not None:
                if i == emergency_idx:
                    # Emergency road: show clearance message and green emphasis
                    box_color = (0, 255, 0)
                    msg = f"EMERGENCY VEHICLE PASSING ({self.road_names[i]})"
                    cv2.rectangle(frm, (10, h - 70), (w - 10, h - 10), box_color, -1)
                    cv2.putText(frm, msg, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                else:
                    # Other roads: show STOP message in red
                    box_color = (0, 0, 255)
                    msg = "STOP EMERGENCY VEHICLE APPROACHING"
                    cv2.rectangle(frm, (10, h - 70), (w - 10, h - 10), box_color, -1)
                    cv2.putText(frm, msg, (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # --- Create Status Panel (4th quadrant) ---
        status_panel = np.zeros((h, w, 3), dtype=np.uint8)
        # Draw Header
        cv2.putText(status_panel, "INTELLI-SIGNAL DASHBOARD", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        statuses = self.compute_status(counts, self.last_emergency)
        y_offset = 100
        for i, name in enumerate(self.road_names):
            # Emergency takes precedence in display
            if statuses[i] == "emergency":
                color = (0, 0, 255)
                status_text = "EMERGENCY"
            elif i == self.active_road:
                color = (0, 255, 0)
                status_text = "GREEN"
            else:
                color = (0, 0, 255)
                status_text = "RED"

            # Show timer countdown if green, otherwise just status
            if statuses[i] == "emergency":
                info = f"{name}: {status_text}"
            elif i == self.active_road:
                time_left = int(self.timers[i] - (time.time() - self.start_time))
                info = f"{name}: {status_text} ({time_left}s)"
            else:
                info = f"{name}: {status_text}"

            cv2.putText(status_panel, info, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y_offset += 60

        # UDP status/info
        udp_status = f"UDP: {'ENABLED' if self.udp_enabled else 'DISABLED'} {self.udp_target[0]}:{self.udp_target[1]}"
        cv2.putText(status_panel, udp_status, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_offset += 30

        # --- Combine into 2x2 Grid ---
        # Top Row: Road A, Road B
        top_row = np.hstack((resized_frames[0], resized_frames[1]))
        # Bottom Row: Road C, Status Panel
        bottom_row = np.hstack((resized_frames[2], status_panel))

        dashboard = np.vstack((top_row, bottom_row))
        return dashboard

    def run(self):
        print("Starting Intelli-Signal System. Press 'q' to exit.")
        while True:
            frames = []
            counts = []
            emergencies = []

            # 1. Read and Process All Cameras
            for i, cap in enumerate(self.caps):
                ret, frame = cap.read()
                if not ret:
                    # If video ends or cam fails, provide a blank frame or loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
                    ret, frame = cap.read()
                    if not ret: frame = np.zeros((480, 640, 3), dtype=np.uint8)

                annotated, count, em = self.process_frame(frame, i)
                frames.append(annotated)
                counts.append(count)
                emergencies.append(em)
                # Visual detection no longer triggers emergency state; use manual keyboard control (e+1/2/3)

            # Update EMA for counts to smooth short-term fluctuations
            for i, c in enumerate(counts):
                if self.counts_ema[i] == 0.0:
                    self.counts_ema[i] = float(c)
                else:
                    self.counts_ema[i] = COUNTS_SMOOTHING_ALPHA * float(c) + (1.0 - COUNTS_SMOOTHING_ALPHA) * self.counts_ema[i]

            # 2. Update Signal Logic (pass smoothed counts)
            # Emergency preemption and hold are handled inside update_logic
            self.update_logic(counts, self.counts_ema)

            # Send UDP periodically (non-blocking)
            current_time = time.time()
            if current_time - self.last_udp_send >= UDP_SEND_INTERVAL:
                self.send_udp(counts)
                self.last_udp_send = current_time

            # 3. Create and Show Dashboard
            dashboard = self.create_dashboard(frames, counts)
            cv2.imshow("Intelli-Signal AI Control Center", dashboard)

            # Keyboard handling: 'q' to quit; 'e' then '1'/'2'/'3' within 1.5s to trigger emergency for lane A/B/C.
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # Initialize em_waiting state
            if not hasattr(self, 'em_waiting'):
                self.em_waiting = False
                self.em_wait_start = 0.0
            EM_WAIT_TIMEOUT = 1.5
            if key == ord('e'):
                self.em_waiting = True
                self.em_wait_start = time.time()
                print("Emergency key pressed: waiting for lane (1/2/3)...")
            elif self.em_waiting:
                if key in (ord('1'), ord('2'), ord('3')):
                    lane = {ord('1'):0, ord('2'):1, ord('3'):2}[key]
                    # Set manual emergency
                    self.last_emergency[lane] = time.time()
                    print(f"Manual emergency triggered for {self.road_names[lane]}")
                    # Immediately send a UDP emergency message if enabled
                    if self.udp_enabled and self.udp_sock is not None:
                        try:
                            msg = f"EMERGENCY - {self.road_names[lane]}"
                            self.udp_sock.sendto(msg.encode('utf-8'), self.udp_target)
                        except Exception as e:
                            print("UDP send error:", e)
                    self.em_waiting = False
                else:
                    # Cancel if timeout elapsed
                    if time.time() - self.em_wait_start > EM_WAIT_TIMEOUT:
                        self.em_waiting = False

        # Cleanup
        for cap in self.caps: cap.release()
        cv2.destroyAllWindows()

def sweep_emergency_thresholds(thresholds, frames_per_clip=300):
    """Quick sweep of emergency light thresholds on the configured video clips.
    Prints detection counts per road for each threshold. Run locally to choose values.
    """
    print("Starting threshold sweep:")
    model = YOLO(MODEL_PATH)

    for T in thresholds:
        print(f"\nThreshold = {T}")
        totals = [0] * len(CAMERA_SOURCES)
        frames_seen = [0] * len(CAMERA_SOURCES)
        for i, src in enumerate(CAMERA_SOURCES):
            cap = cv2.VideoCapture(src)
            seen = 0
            detected = 0
            while seen < frames_per_clip:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                seen += 1
                try:
                    results = model(frame, verbose=False, classes=[2,3,5,7])
                    boxes = results[0].boxes
                    try:
                        xyxy = boxes.xyxy.cpu().numpy()
                        clsids = boxes.cls.cpu().numpy()
                    except Exception:
                        xyxy = np.array(boxes.xyxy)
                        clsids = np.array(boxes.cls)
                    found = False
                    for (x1, y1, x2, y2), clsid in zip(xyxy, clsids):
                        clsid = int(clsid)
                        if clsid not in (2,3):
                            continue
                        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                        x1i = max(0, x1i); y1i = max(0, y1i)
                        x2i = min(frame.shape[1], x2i); y2i = min(frame.shape[0], y2i)
                        roi = frame[y1i:y2i, x1i:x2i]
                        if roi.size == 0:
                            continue
                        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        lower_red1 = np.array([0, 120, 200]); upper_red1 = np.array([10, 255, 255])
                        lower_red2 = np.array([160, 120, 200]); upper_red2 = np.array([180, 255, 255])
                        lower_blue = np.array([100, 120, 200]); upper_blue = np.array([140, 255, 255])
                        mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
                        mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
                        mask_red = cv2.bitwise_or(mask_r1, mask_r2)
                        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
                        red_count = int(cv2.countNonZero(mask_red))
                        blue_count = int(cv2.countNonZero(mask_blue))
                        if red_count + blue_count >= T:
                            detected += 1
                            found = True
                            break
                    if not found:
                        pass
                except Exception:
                    pass
            totals[i] = detected
            frames_seen[i] = seen
            cap.release()
        for i, src in enumerate(CAMERA_SOURCES):
            pct = (totals[i] / frames_seen[i] * 100) if frames_seen[i] > 0 else 0
            print(f"  {i} ({src}): {totals[i]} detections / {frames_seen[i]} frames ({pct:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intelli-Signal runner and tools")
    parser.add_argument("--sweep", action="store_true", help="Run emergency threshold sweep and exit")
    parser.add_argument("--frames", type=int, default=300, help="Frames per clip to scan during sweep")
    parser.add_argument("--thresholds", type=str, default="100,200,300,400,500", help="Comma-separated thresholds to test")
    args = parser.parse_args()

    if args.sweep:
        thresholds = [int(x) for x in args.thresholds.split(",")]
        sweep_emergency_thresholds(thresholds, frames_per_clip=args.frames)
    else:
        app = IntelliSignalVisualizer()
        app.run()