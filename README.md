# Road Defect Detector

A PyQt5-based desktop application for detecting road pavement defects using YOLOv8-seg neural network.

## Features

- Process single images or entire folders.
- Segmentation masks with per-class colors.
- Interactive "Before/After" view (hold button to see original image).
- Export results to:
  - **PNG** (overlay with contours and labels)
  - **JSON** (LabelMe-compatible polygon annotations)
  - **CSV** (summary with bounding boxes and areas)
- Batch export for all processed images.
- Customizable class names and colors (persisted between sessions).
- Load any YOLOv8-seg model (`.pt` file).

## Requirements

- Python 3.8+
- PyTorch
- Ultralytics YOLO
- PyQt5
- OpenCV
- Pillow
- NumPy

Install dependencies:

```bash
pip install -r requirements.txt
```
## Getting the Model

This repository does **not** include the pre‑trained model file `best.pt` because of its size (208 MB). I do not recommend using this file, as it is currently of extremely questionable quality. However, you can download it using the link below as an example.

Download the model from Google Drive:  
[Download best.pt](https://drive.google.com/file/d/16-1jIjH9n-3GtGhWBkgF5IF4NK956dAK/view?usp=sharing)

Place the file in the project root directory (next to `main_window.py`).

## Usage

Run the application:

```bash
python main_window.py
```
### Main Workflow

1. **Open image(s)** – use `File → Open Image` or `File → Open Folder`.
2. **Process** – select `Process → Process Current` for the selected image, or `Process → Process All` to process all images in the list.
3. **View** – click on an image in the list to view its result. Hold the **"Original Image"** button to see the unprocessed version.
4. **Export** – export the current image or all processed images via the `Export` menu.

### Customization

- **Class names & colors** – go to `Classes → Class Settings`. In the dialog, you can rename any class and change its display color. Changes are saved automatically.
- **Change model** – use `Model → Load Model...` to select another `.pt` YOLO‑seg model.

### File Structure
- main_window.py – main GUI application.

- inference.py – YOLO detection wrapper with tiling and merging.

- export_utils.py – utilities for saving annotated images, JSON, CSV.

- settings.py – persistent storage for user preferences (class names, colors).

- best.pt – (not included) pre-trained model.

### License
TThis project is distributed under the Apache 2.0 License. ee the [LICENSE](https://github.com/AoDhcA/RoadDefectDetector?tab=Apache-2.0-1-ov-file) file for details.

### Acknowledgments

- Built with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
- GUI framework: [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) (GPL v3).
- Computer vision: [OpenCV](https://opencv.org/) (Apache 2.0).
- Image processing: [Pillow](https://python-pillow.org/) (MIT-CMU).
- Numerical operations: [NumPy](https://numpy.org/) (BSD-3-Clause).

