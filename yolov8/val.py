from ultralytics import YOLO
import os
# Load a model
model = YOLO("yolov8n.pt")

# Customize validation settings
if __name__ == "__main__":

    metrics = model.val(data="coco8.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, save_json=True)
    results = metrics.results_dict
    # Creating the formatted text string
    output = f"P\t\t\t\t\t\tR\t\t\t\tmAP50\t\t\tmAP50-95\n"
    output += f"{results['metrics/precision(B)']:.3f}\t\t\t{results['metrics/recall(B)']:.3f}\t\t\t{results['metrics/mAP50(B)']:.4f}\t\t{results['metrics/mAP50-95(B)']:.4f}"

    dir_path = f"{metrics.save_dir}/metrics"
    os.makedirs(dir_path, exist_ok=True)
    # Writing to a .txt file
    file_path = f"{dir_path}/results.txt"
    with open(file_path, "w") as file:
        file.write(output)