# YOLO Models Testing Benchmark

The models included in the repository ready to be tested:

- Yolov5 Testing ✔️
- Yolov6 Testing ❌
- Yolov7 Testing ❌
- Yolov8 Testing ❌

## YOLOv5 Testing Benchmark

This repository contains a script to validate a trained YOLOv5 detection model on a detection dataset. The purpose is to create a benchmark for different YOLO models.

### Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/S-Y-3-D/Benchmark.git
    cd yolov5
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

### Usage

#### Running the Validation Script

The script `val.py` is used to validate the YOLOv5 model. You can customize various parameters such as data path, weights, batch size, image size, and thresholds.

#### Example Command

To run the validation script with default parameters:
```sh
python val.py
```
This will run the `run` function defined in the `val.py`

#### Parameters

- `--data`: Path to the dataset configuration file.
- `--weights`: Path to the trained model weights file.
- `--batch-size`: Batch size for validation.
- `--imgsz`: Image size for inference.
- `--conf-thres`: Confidence threshold for predictions.
- `--iou-thres`: IoU threshold for non-max suppression.
- `--device`: Device to run the validation on (e.g., `cpu`, `cuda:0`).
- `--project`: Directory to save validation results.
- `--name`: Name of the validation run.
- `--workers`: Number of data loader workers.
- `--half`: Use half-precision (FP16) inference.


### Results

The results will be saved in the `runs/val` directory by default. This includes:

- Validation images with labels and predictions.
- Metrics including precision, recall, mAP@0.5, and mAP@0.5:0.95.
- Confusion matrix plots.
