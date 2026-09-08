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
