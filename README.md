
# Eye-Gesture Controlled Human–Computer Interface

Hands-free accessibility system using real-time gaze and blink detection.

---

## Overview

This repository contains the complete implementation of a hands-free human–computer interaction system. The system detects facial landmarks, estimates gaze direction using a deep learning model, stabilizes the signals with real-time smoothing, interprets gaze into directional commands through calibration, recognizes blink gestures, and communicates the final commands to a virtual keyboard or remote device. The final user interface is a complete on-screen keyboard controlled entirely through gaze and blinking.

The system is intended for assistive technologies, accessibility interfaces, medical communication, and hands-free computer control.

---

## Repository Structure

```
Final Integration/
│
├── models/                         Pre-trained ONNX inference models
│
├── Blinking_PI_v2.py               Blink detection engine
├── Calibration.py                  Gaze direction calibration tool
├── Calibration_Pi.py               Alternative calibration script
├── Gaze.py                         Low-level gaze estimation from camera
├── Receiver.py                     Remote data receiver
├── Sender.py                       Remote data sender
├── User_Guide.py                   Usage instructions and documentation
├── demo_utils.py                   YOLOX face detection utilities
├── gaze_detector_integration.py    Unified gaze estimation pipeline
├── gaze_direction_calibration.json Saved calibration thresholds
├── gaze_utils.py                   Gaze interpretation utilities
├── keyko.py                        Virtual keyboard and UI interaction
└── smoother.py                     Temporal smoothing filters
```

---

## System Architecture

```
Camera Input
    ↓
Face Detection (YOLOX + demo_utils)
    ↓
Landmark Extraction (MediaPipe)
    ↓
Head Pose and Eye Region Normalization
    ↓
Gaze Estimation Model (ONNX)
    ↓
Continuous Gaze Output (pitch, yaw)
    ↓
Temporal Smoothing (One-Euro Filter – smoother.py)
    ↓
Calibration Mapping (Calibration.py + gaze_utils.py)
    ↓
Discrete Directions (left, right, up, down, center)
    ↓
Blink Detection (Blinking_PI_v2.py)
    ↓
Remote Communication (Sender.py / Receiver.py)
    ↓
Interaction Layer (Virtual Keyboard – keyko.py)
```

---

# Module Descriptions

## 1. Gaze Estimation

### `Gaze.py`

Implements real-time gaze estimation from a video frame. It handles:

* face detection
* landmark extraction
* head pose normalization
* ONNX gaze inference
* returning continuous gaze angle `(pitch, yaw)`

This script is responsible for converting a raw camera frame into a stable continuous gaze vector.

---

### `demo_utils.py`

Utility functions supporting YOLOX-based face detection:

* bounding box extraction
* non-maximum suppression
* result normalization

Used internally by gaze detector modules.

---

### `gaze_detector_integration.py`

Full integration layer that combines:

* ONNX face detector
* MediaPipe landmark extraction
* ONNX gaze estimation model
* temporal smoothing
* calibration thresholds

Provides a single callable function:

```
get_gaze_from_frame(frame) → (pitch, yaw)
```

This file is the **core inference engine** used by all higher-level modules.

---

### `smoother.py`

Implements temporal stabilization using **One-Euro Filtering** to reduce jitter from:

* camera noise
* sudden head motion
* ambient lighting variations

Provides smoothing for:

* `(pitch, yaw)` values
* optional landmark stabilization

---

## 2. Calibration and Interpretation

### `Calibration.py`

Interactive calibration tool. The user looks at predefined regions (center, left, right, up, down), and the system samples gaze values to compute personalized thresholds.

Results are saved in:

```
gaze_direction_calibration.json
```

---

### `gaze_direction_calibration.json`

Stores the numerical boundaries that map continuous gaze to discrete directions.

---

### `gaze_utils.py`

Uses calibration thresholds to convert `(pitch, yaw)` into discrete commands:

```
"left", "right", "up", "down", "center"
```

Also supports confidence scoring to stabilize direction switching.

---

## 3. Blink and Gesture Recognition

### `Blinking_PI_v2.py`

Detects blink gestures using Eye Aspect Ratio (EAR) and timing thresholds. Supports:

* single blink → selection
* long blink or double blink → alternative actions

Used to trigger clicking without hands.

---

## 4. Communication Layer

### `Sender.py`

Transmits the interpreted commands (direction + blink) to another machine or UI layer via:

* network sockets
* serial communication
* or any configured transport method

Allows computation and UI to run on separate devices.

---

### `Receiver.py`

Receives streamed commands and passes them to the local interaction layer, enabling remote execution.

This design decouples:

* computer vision processing
* user interface logic

Allowing thin-client deployment.

---

## 5. User Interaction Layer

### `keyko.py`

Implements a fully usable gaze-controlled virtual keyboard:

* navigation between keys using gaze direction
* blink activation for typing
* keyboard switching and UI control
* underlying OS keyboard input via pynput or similar libraries

The final user-facing interface enables full text entry without physical input devices.

---

### `User_Guide.py`

Contains instructions for:

* setup
* calibration
* usage
* troubleshooting

---

# Notes

* Models inside `/models` are replaceable if ONNX format and compatible input shapes are maintained
* Calibration is mandatory for accuracy and personalization
* Smoothing strongly improves interaction stability

---

# Use Cases

* Assistive and accessibility devices
* Medical communication tools
* Hands-free computer navigation

---

# Model Source Attribution

The gaze-estimation models used in this project are adapted from the open-source implementation provided by **MIT Han Lab** under the `proxyless_gaze` framework.  
We acknowledge and thank the authors for their contribution to the research community.

🔗 Repository: https://github.com/mit-han-lab/proxylessnas/tree/master/proxyless_gaze

> Note: All pretrained models and related components included from this repository remain under their original open-source license terms. The authors of this thesis do not claim ownership over the model architecture or pretrained weights.

---

# Legacy Development Folders (Trials & Experiments)

This repository also contains several auxiliary folders such as **Gaze CH**, **Gaze V1**, and **TinyTrackerS**.
These folders represent **early experimental stages, prototypes, or isolated trials** that were explored during development. They are intentionally kept for:

* traceability and documentation
* reproducibility of previous model evaluations
* academic transparency for the thesis
* future reference or benchmarking

**Important Note:**

> These folders are **not part of the final system architecture**.
> The complete, production-ready implementation used in our thesis and demo is located exclusively inside the **`Final Integration/`** directory.

All source code required to run the full system, including calibration, gaze estimation, blink detection, communication, smoothing, and UI interaction, exists solely within the **Final Integration** folder.
