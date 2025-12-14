# GazeRead: Hands-Free Content Viewer

A hands-free document viewer that enables navigation through facial gestures using only a standard webcam.

## Features

- **Gaze-Based Page Navigation**: Look left or right to turn pages
- **Nostril Flare Zoom**: Flare nostrils to toggle zoom mode
- **Double-Blink Scroll**: Double-blink to scroll when zoomed in
- **Personalized Calibration**: 7-state calibration adapts to individual users
- **Real-Time HUD**: Visual feedback overlay showing gesture metrics
- **False Activation Prevention**: Multiple strategies including dwell time, cooldowns, and gesture isolation

## Requirements

- Python 3.8+
- Webcam (720p recommended)
- dlib shape predictor model (`shape_predictor_68_face_landmarks.dat`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/daggergad7/HCI-Project.git
cd project2
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python main.py
```

2. Complete the calibration process:
   - Press `c` to capture each calibration state
   - Follow on-screen instructions for eyes open/closed, double-blink, nose flare, and gaze directions

3. Once calibrated, use gestures to navigate:
   - **Look left/right** (hold for 0.5s) → Previous/Next page
   - **Flare nostrils** → Toggle zoom
   - **Double-blink** (when zoomed) → Scroll down

4. Controls:
   - `r` - Reset calibration
   - `q` or `ESC` - Quit

## Project Structure

```
project2/
├── main.py              # Main application with gesture recognition
├── requirements.txt     # Python dependencies
├── page_*.png          # Document pages to display
├── config.json         # Saved calibration data (generated)
└── README.md           # This file
```

## Technical Details

- **Face Detection**: dlib frontal face detector
- **Landmark Detection**: 68-point shape predictor
- **Blink Detection**: Eye Aspect Ratio (EAR) algorithm
- **Gaze Detection**: Pupil position ratio within eye region
- **Flare Detection**: Nostril width to inter-ocular distance ratio

## False Activation Mitigation

- **Temporal Filtering**: 3 consecutive frames required for blinks/flares
- **Dwell Time**: 15 frames (~0.5s) for gaze-based page turns
- **Cooldown Periods**: 2.0s cooldown after zoom toggle
- **Gesture Isolation**: Prevents gesture interference
- **Personalized Calibration**: Thresholds adapted to each user
