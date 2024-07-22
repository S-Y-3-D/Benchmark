# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Validate a trained YOLOv5 detection model on a detection dataset.

"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utilities.dataloaders import create_dataloader
from utilities.general import (
    check_dataset,
    check_img_size,
    colorstr,
    increment_path,
    non_max_suppression,
    scale_boxes,
    xywh2xyxy,
    check_version
)
from utilities.plots import ConfusionMatrix, ap_per_class, box_iou, output_to_target, plot_images



def smart_inference_mode(torch_1_9=check_version(torch.__version__, "1.9.0")):
    """Applies torch.inference_mode() if torch>=1.9.0, else torch.no_grad() as a decorator for functions."""

    def decorate(fn):
        return (torch.inference_mode if torch_1_9 else torch.no_grad)()(fn)

    return decorate

def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix.

    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


def smart_inference_mode(torch_1_9= check_version(torch.__version__, "1.9.0")):
    """Applies torch.inference_mode() if torch>=1.9.0, else torch.no_grad() as a decorator for functions."""

    def decorate(fn):
        return (torch.inference_mode if torch_1_9 else torch.no_grad)()(fn)

    return decorate

@smart_inference_mode()
def run(
    data,
    weights=None,  # model.pt path(s)
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    max_det=300,  # maximum detections per image
    task="val",  # train, val, test, speed or study
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    single_cls=False,  # treat as single-class dataset
    augment=False,  # augmented inference
    project=ROOT / "runs/val",  # save to project/name
    name="exp",  # save to project/name
    half=True,  # use FP16 half-precision inference
    model=None,
    compute_loss=None,
):
    # Initialize/load model and set device
    training = model is not None
    
    pt=True
    
    device = "cpu"
    # Directories
    save_dir = increment_path(Path(project) / name)  # increment run
    (save_dir / "labels").mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / "metrics").mkdir(parents=True, exist_ok=True)  # make dir
    # Load model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)
    stride = 32
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    half = False  # FP16 supported on limited backends with CUDA
    # Data
    data = check_dataset(data)  # check

    # Configure
    model.eval()
    nc = 1 if single_cls else int(data["nc"])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        
        pad, rect = (0.0, False) if task == "speed" else (0.5, pt)  # square inference for benchmarks
        task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images
        dataloader = create_dataloader(
            data[task],
            imgsz,
            batch_size,
            stride,
            single_cls,
            pad=pad,
            workers=workers,
        )[0]

    seen = 0
    
    confusion_matrix = ConfusionMatrix(nc=nc)
    
    names = model.names if hasattr(model, "names") else model.module.names  # get class names
    
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    
    s = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
    
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    loss = torch.zeros(3, device=device)
    
    jdict, stats, ap, ap_class = [], [], [], []
    
    for batch_i, (im, targets, paths, shapes) in enumerate(dataloader):
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        preds, train_out = model(im) if compute_loss else (model(im, augment=augment), None)
        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
    
        preds = non_max_suppression(
            preds, conf_thres, iou_thres, labels=[], multi_label=True, agnostic=single_cls, max_det=max_det
        )

        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1
            plots=True
            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

        # Plot images
        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir / f"val_batch{batch_i}_labels.jpg", names)  # labels
            plot_images(im, output_to_target(preds), paths, save_dir / f"val_batch{batch_i}_pred.jpg", names)  # pred

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    txt = open(save_dir / "metrics" / "metrics.txt", "w")
    
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # print format
    print(s,"\n", pf % ("all", seen, nt.sum(), mp, mr, map50, map), file = txt)
    if nt.sum() == 0:
        print(f"WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels")

    # Print results per
    verbose = False
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]), file = txt)
    
    txt.close()

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))


if __name__ == "__main__":

    run(data="coco128.yaml", weights="yolov5s.pt", batch_size=32,  imgsz=640,  conf_thres=0.001, iou_thres=0.6)
