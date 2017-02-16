# Advanced Lane Finding

The goal of this project was to prepare a processing pipeline to identify the lane boundaries in a video. 

## Project structure

| File                         | Description                                                                        |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| `lanetracker/camera.py`      | Implements camera calibration based on the set of calibration images. |
| `lanetracker/tracker.py`     | Implements lane tracking by applying a processing pipeline to consecutive frames in a video. |
| `lanetracker/gradients.py`   | Set of edge-detecting routines based on gradients and color. |
| `lanetracker/perspective.py` | Set of perspective transformation routines. |
| `lanetracker/line.py` 	   | `Line` class representing a single lane boundary line. |
| `lanetracker/window.py`      | `Window` class representing a scanning window used to detect points likely to represent lines. |
