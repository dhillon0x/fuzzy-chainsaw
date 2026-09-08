# 🦾 AI-Enabled 6-Axis Robotic Arm Controller & Vision Pipeline

An end-to-end, open-source software stack integrating a dual-model real-time computer vision pipeline (YOLOv8 + FastSAM) with a 6-DOF Inverse Kinematics solver and high-speed serial motion control. 

This repository houses the complete algorithmic and driver codebase tailored for the 3D-printable 6-axis robot arm hardware designed by Event Horizon Research.

---

## 📌 Attribution & Hardware Credits

The mechanical CAD chassis, gear systems, and physical dimensions referenced by this control stack originate from:

* **Design:** **6-Axis Robotic Arm**[cite: 1]
* **Author / Creator:** **Event Horizon Research** (`@EventHorizonR_151042`)[cite: 1]
* **Platform:** [Printables Model #742414](https://www.printables.com/model/742414-6-axis-robotic-arm)[cite: 1]
* **License:** [Creative Commons (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)[cite: 1]
* **Original Reddit Demo:** [Reddit /r/3Dprinting 6-Axis Arm running](https://www.reddit.com/r/3Dprinting/comments/1azw67y/3d_printed_6axis_robot/)[cite: 1]

> *Note: All mechanical component STLs (`Base`, `Bicep_Tricep`, `Elbow`, `Forearm`, `Gears`) belong to Event Horizon Research[cite: 1]. This repository supplies an independent, modern Python-based AI perception and kinematic trajectory system.*

---

## 🧠 System Overview & Architecture

```text
       [ Live USB / IP Camera ]
                  │
                  ▼
   ┌─────────────────────────────┐
   │    vision/detector.py       │  <-- Model 1: YOLOv8n (Ultra-fast object bounding box)
   └──────────────┬──────────────┘
                  │ [BBox Coordinates]
                  ▼
   ┌─────────────────────────────┐
   │    vision/segmenter.py      │  <-- Model 2: FastSAM-s (CNN Prompted Instance Mask)
   └──────────────┬──────────────┘
                  │ [Exact Geometric Centroid (cx, cy)]
                  ▼
   ┌─────────────────────────────┐
   │     vision_pipeline.py      │  <-- Perspective Matrix: Camera UV -> 3D World (X, Y, Z)
   └──────────────┬──────────────┘
                  │ [Cartesian Targets (mm)]
                  ▼
   ┌─────────────────────────────┐
   │      src/kinematics.py      │  <-- Denavit-Hartenberg (DH) & Bounded Levenberg-Marquardt IK
   └──────────────┬──────────────┘
                  │ [Joint Angles: J1...J6]
                  ▼
   ┌─────────────────────────────┐
   │      src/controller.py      │  <-- UART / Serial Packet Pipeline (G-code style commands)
   └─────────────────────────────┘


6-axis-robotic-arm/
├── docs/                                 # Assembly manuals, schematics, pinout guides
│   ├── 742414-6-axis-robotic-arm.pdf
│   └── robotic-arm-instruction-manual.pdf
├── hardware/                             # STL files cataloged by joint subassembly
│   ├── Base/
│   ├── Bicep_Tricep/
│   ├── Elbow/
│   ├── Forearm/
│   └── Gears/
├── src/                                  # Motion control & kinematic mathematics
│   ├── __init__.py
│   ├── controller.py                     # Serial port stream with simulated fallback
│   └── kinematics.py                     # 6-DOF forward and inverse kinematic solvers
├── vision/                               # AI dual-model perception package
│   ├── __init__.py
│   ├── detector.py                       # YOLOv8 object identification
│   └── segmenter.py                      # FastSAM mask segmenter & moment centroid
├── vision_pipeline.py                    # Top-level executable running camera-to-joint loop
├── requirements.txt                      # Project dependencies
└── README.md                             # System manual and documentation.

##  Installation & Setup Instructions

```bash
# 1. Clone the repository
git clone [https://github.com/](https://github.com/)<your-username>/6-axis-robotic-arm.git
cd 6-axis-robotic-arm

# 2. Create and activate a virtual environment
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate
# On Windows (PowerShell / Command Prompt):
python -m venv venv
venv\Scripts\activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

*Required weights (`yolov8n.pt` and `FastSAM-s.pt`) download automatically from the Ultralytics CDN during the initial run.*

---

## 💻 How to Run the Software

Launch the main computer vision and tracking loop:

```bash
python vision_pipeline.py






[HW WARN] Running in SIMULATION mode: could not open port COM3
[SIM OUT] G0 J1:24.12 J2:45.00 J3:-12.40 J4:0.00 J5:32.18 J6:0.00

🔌 Microcontroller Firmware Protocol

The serial driver outputs ASCII G-code packets at 115200 baud. If building custom Arduino or ESP32 firmware, format your serial parser to expect

( G0 J1:<angle> J2:<angle> J3:<angle> J4:<angle> J5:<angle> J6:<angle>\n )
Example: G0 J1:15.20 J2:30.00 J3:-45.10 J4:0.00 J5:60.00 J6:0.00\n
Joint Units: Degrees ($\pm 0.01^\circ$ resolution).
Firmware Handshake: Configure your microcontroller to echo ok\n after enqueuing step pulses.

📐 Workspace Calibration (Pixel to 3D Coordinates)
The script maps camera pixels to robot base coordinates using a flat planar projection in pixel_to_world_coords():
( [Camera Frame: cx, cy] ---> Scale Factor (mm/px) + Axis Offset ---> [Robot Base: X, Y, Z (mm)] )

⚠️ Troubleshooting & FAQ
Camera index not detected: If OpenCV fails to start, change cv2.VideoCapture(0) to index 1 or 2 in vision_pipeline.py to target an external webcam.

Weights download stall: If automatic downloads fail due to network restrictions, download yolov8n.pt and FastSAM-s.pt directly from Ultralytics releases and place them in the root directory.

Target unreachable (None output): If target coordinates exceed the arm's mechanical envelope or violate the joint limits set in src/kinematics.py, the solver returns None and discards that position frame.

Linux serial permission errors: Run sudo usermod -a -G dialout $USER and log back into your session to enable USB device access.

📺 Recommended Visual Guides
If you need a walkthrough of the mathematical principles or vision integration, refer to these guides:

Inverse Kinematics Concepts & Math: Robotics: Forward and Inverse Kinematics Explained by Angela Sodemann

YOLO + OpenCV Coordinate Extraction: YOLO Object Detection & Centroid Tracking with OpenCV by Murtaza's Workshop

