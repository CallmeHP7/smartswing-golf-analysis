# SmartSwing – Golf Swing Analysis (Assignment)

SmartSwing is a prototype design for a mobile-first, computer-vision–based golf swing
analysis system. It uses pose estimation and simple biomechanical features to compute
a 0–100 **Swing Accuracy Score** and detect common issues like:

- Over-the-top downswing
- Early extension
- Improper wrist hinge

This repository contains **reference code** for the core analysis pipeline:
pose extraction, swing detection, feature extraction, and scoring.

> NOTE: This is assignment-level prototype code. It focuses on clean structure and
> interpretability, not on fully production-ready performance.

## Pipeline Overview

1. **Pose Estimation (`pose_estimation.py`)**
   - Uses a pose model (e.g., MediaPipe Pose) to extract 2D keypoints for each frame.

2. **Swing Detection (`swing_detector.py`)**
   - Uses wrist motion to automatically detect the main swing interval.

3. **Feature Extraction (`features.py`)**
   - Computes swing-plane angles, downswing alignment, spine-angle change,
     head movement, and simple wrist hinge approximation.

4. **Scoring (`scoring.py`)**
   - Converts features into three sub-scores:
     - Plane Consistency Score (PCS)
     - Alignment Score (AS)
     - Rotation Stability Score (RSS)
   - Aggregates them into a 0–100 Swing Accuracy Score and flags common faults.

5. **Command-line Tool (`analyze_swing.py`)**
   - Example script to run the entire pipeline on a video file and print results.

## Installation

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

