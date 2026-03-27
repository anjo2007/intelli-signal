# Intelli-Signal 🚦

An AI-powered intelligent traffic light control system that uses real-time computer vision to detect vehicles and emergency vehicles, then dynamically adjusts signal timing to optimize traffic flow and prioritize emergency response.

## Features

- **Real-time vehicle detection** using YOLOv8 Nano — detects cars, motorcycles, buses, and trucks
- **Adaptive signal timing** — green light duration adjusts automatically based on traffic density
- **Emergency vehicle preemption** — detects flashing red/blue lights and immediately grants green to the emergency vehicle's lane
- **Multi-road support** — monitors up to 3 intersections simultaneously
- **Live visual dashboard** — 2×2 grid showing all video feeds with overlays and a status panel
- **UDP status broadcast** — sends real-time traffic status to external systems every second
- **Optional audio siren detection** — background thread listens for siren frequencies via microphone

## Demo

The dashboard shows three road feeds (Road A, Road B, Road C) alongside a live status panel:

```
┌──────────────────┬──────────────────┐
│  Road A (Main)   │  Road B (Cross)  │
│  🟢 GREEN 18s    │  🔴 RED          │
├──────────────────┼──────────────────┤
│  Road C (Side)   │  STATUS PANEL    │
│  🔴 RED          │  Road A: GREEN   │
│                  │  Road B: RED     │
│                  │  Road C: RED     │
│                  │  UDP: connected  │
└──────────────────┴──────────────────┘
```

## Requirements

- Python 3.8+
- A camera or video file for each monitored road

## Installation

```bash
# Clone the repository
git clone https://github.com/anjo2007/intelli-signal.git
cd intelli-signal

# Install dependencies
pip install opencv-python ultralytics numpy

# Optional: enable audio siren detection
pip install sounddevice
```

The YOLOv8 Nano model (`yolov8n.pt`) is included in the repository.

## Configuration

Open `intelli.py` and edit the constants near the top of the file:

### Camera sources

```python
CAMERA_SOURCES = [
    "path/to/road_a.mp4",   # Road A
    "path/to/road_b.mp4",   # Road B
    "path/to/road_c.mp4",   # Road C
]
# Use 0, 1, 2 … for live webcams
```

### Traffic light timing

| Constant | Default | Description |
|---|---|---|
| `MIN_GREEN` | `10` | Minimum green duration (seconds) |
| `MAX_GREEN` | `60` | Maximum green duration (seconds) |
| `SEC_PER_VEHICLE` | `3` | Extra green seconds added per detected vehicle |

### Emergency detection

| Constant | Default | Description |
|---|---|---|
| `EMER_LIGHT_PIXEL_THRESHOLD` | `300` | Red/blue pixels required to trigger emergency |
| `EMER_LIGHT_PIXEL_HIGH_THRESHOLD` | `600` | Threshold for larger vehicles |
| `EMER_LIGHT_HOLD` | `5` | Seconds to keep emergency status active |
| `EMERGENCY_GREEN` | `MAX_GREEN` | Green time granted during emergency |

### UDP output

```python
UDP_ENABLED        = True
UDP_TARGET_IP      = '192.168.1.2'
UDP_TARGET_PORT    = 5005
UDP_SEND_INTERVAL  = 1.0   # seconds between updates
```

## Usage

### Start the traffic control system

```bash
python intelli.py
```

#### Keyboard controls

| Key | Action |
|---|---|
| `Q` | Quit |
| `E` then `1` | Trigger emergency on Road A |
| `E` then `2` | Trigger emergency on Road B |
| `E` then `3` | Trigger emergency on Road C |

### Calibrate emergency-light thresholds

Run a sweep over different pixel-count thresholds to find the best value for your video clips:

```bash
# Default sweep (thresholds 100 – 900, 300 frames per clip)
python intelli.py --sweep

# Custom thresholds
python intelli.py --sweep --thresholds 100,200,300,400,500

# Custom frame limit per clip
python intelli.py --sweep --frames 500
```

### Test UDP connectivity

```bash
python udp_sender.py
```

An interactive prompt lets you send arbitrary messages to the configured UDP target, useful for verifying network connectivity before deployment.

## How It Works

```
Video feeds (3 cameras)
        │
        ▼
YOLOv8 vehicle detection
        │
        ▼
Vehicle counting + emergency light detection (HSV colour analysis)
        │
        ▼
Adaptive timing logic (EMA-smoothed counts)
        │
   ┌────┴────────────────────────────┐
   ▼                                 ▼
Signal control decision       UDP status broadcast
        │
        ▼
Live dashboard (2×2 grid + status panel)
```

1. **Capture** a frame from each of the three cameras.
2. **Detect** vehicles with YOLOv8 (classes: car, motorcycle, bus, truck).
3. **Smooth** per-road counts with an exponential moving average (α = 0.4).
4. **Check** for emergency lights by scanning red/blue color density in each bounding box.
5. **Decide** which road gets green next (round-robin, interrupted by emergency preemption).
6. **Calculate** green duration: `clamp(count × SEC_PER_VEHICLE, MIN_GREEN, MAX_GREEN)`.
7. **Render** the dashboard and send a UDP status update.

## UDP Message Format

Each update is a plain-text, newline-delimited list of road statuses:

```
Road A (Main): less traffic
Road B (Cross): congestion
Road C (Side): clear
```

Traffic status levels: `clear` · `less traffic` · `congestion` · `emergency`

## Project Structure

```
intelli-signal/
├── intelli.py      # Main application — detection, timing logic, dashboard
├── udp_sender.py   # UDP test client
└── yolov8n.pt      # YOLOv8 Nano pre-trained weights
```

## License

This project is provided as-is for research and educational purposes.
