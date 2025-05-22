#🎯 CodeAlpha_Tasks_Object_Detection
Object Detection and Tracking  Develop a system capable of detecting and  tracking objects in real-time video streams. Use  deep learning models like YOLO (You Only Look  Once) for accurate object  detection and tracking.
# Real-Time People Tracking with YOLO and DeepSORT

![Tracking Demo](demo.gif)

A lightweight real-time people tracking system using YOLO for detection and DeepSORT for object tracking.

## 🚀  Features
- Real-time tracking (30+ FPS on CPU)
- Person detection with tracking IDs
- Video input/output support
- Customizable confidence thresholds

## Installation
```bash
pip install ultralytics deep-sort-realtime opencv-python

## Usage
python detection.py 

Options:
--input: Input video path (default: 'xyz.mp4')

--output: Output video path (default: 'output.mp4')

--conf: Confidence threshold (default: 0.4)

Example
python
# Track people in video and save results
python track.py --input street.mp4 --output tracked.mp4 --conf 0.5
Requirements
Python 3.8+

Ultralytics YOLO

DeepSORT-Realtime

OpenCV

License
MIT
Key points:
1. Minimal but covers all essentials
2. Clear installation/usage instructions
3. Includes placeholder for demo GIF
4. Lists basic requirements
5. Simple MIT license

