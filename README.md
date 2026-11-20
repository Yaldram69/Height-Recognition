# Height Recognition

Webcam tool that segments objects with the pretrained YOLOv8 segmentation model (`yolov8n-seg.pt`), skips the COCO `person` class, and draws each remaining object's contour and bounding box with its height in pixels. The code also contains a pixel-to-centimetre conversion driven by a reference object, but that reference is not set, so the output shows "Calibrating..." instead of a real height.

## Features

- YOLOv8 nano instance segmentation on a live webcam feed
- Ignores the COCO `person` class (class ID 0)
- Height in pixels from the bounding box of each object mask
- Optional pixel-to-cm conversion through `REFERENCE_HEIGHT_REAL` and `REFERENCE_HEIGHT_PIXELS`

## Project structure

- `test.py`: the webcam script, with the person filter.
- `Height_recognition.ipynb`: notebook with two versions of the same loop. The first has no person filter; the second adds it under the markdown cell "Ignoring Person class from pre train YOLO model".
- `yolov8n-seg.pt`: pretrained YOLOv8 nano segmentation weights.

## Requirements

Imports used: `cv2`, `numpy`, `ultralytics`.

```
pip install ultralytics opencv-python numpy
```

## Installation

```
git clone https://github.com/Yaldram69/Height-Recognition.git
cd Height-Recognition
pip install ultralytics opencv-python numpy
```

## Usage

Run from the repository folder so `yolov8n-seg.pt` is found:

```
python test.py
```

The script reads from the default webcam (index 0) and takes no arguments. Press `q` in the window to quit.

To run the notebook, open `Height_recognition.ipynb` in Jupyter and run the last code cell, which contains the version with the person filter.

## Notes

- `REFERENCE_HEIGHT_PIXELS` is `None`, so the displayed value stays "Calibrating...". To calibrate, measure the pixel height of an object of known height `REFERENCE_HEIGHT_REAL` (cm) placed at a known distance, set `REFERENCE_HEIGHT_PIXELS` to that value, and rerun.
- The scale factor is a single number. It is only valid for objects at the same distance from the camera as the reference object.
- The measured value is the image height of the mask's bounding box. It is not a true 3D height and is affected by perspective and camera angle.
- The contour filter `len(contour) > 5` only removes very small contours.
- Only the segmentation weights are included; no training code is present.

## Pretrained YOLO weights (not included)

The following off-the-shelf Ultralytics checkpoints were removed from this repository because they are publicly downloadable. The code loads them by name, and Ultralytics downloads them automatically on first use:

- `yolov8n-seg.pt`

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # downloaded automatically if not present
```

Or download them manually from https://github.com/ultralytics/assets/releases
