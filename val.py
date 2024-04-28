# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Validate a trained YOLOv5 detection model on a detection dataset

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s_openvino_model     # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlmodel            # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import subprocess
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

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size, check_requirements,
                           check_yaml, coco80_to_coco91_class, colorstr, increment_path, non_max_suppression,
                           print_args, scale_boxes, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torchvision

import torch

from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
object_tracker = DeepSort(max_age=10,
    n_init=3,
    nms_max_overlap=1.0,
    max_cosine_distance=0.2,
    nn_budget=None,
    override_track_class=None,
    embedder="clip_ViT-B/32",
    half=False,
    bgr=False,
    embedder_gpu=True,
    embedder_model_name=None,
    embedder_wts=None,
    polygon=False,
    today=None)

def calculate_iou(box1, box2):
    # box1 and box2 should be in the format (x1, y1, x2, y2)

    # Calculate the intersection rectangle
    intersection_x1 = torch.max(box1[0], box2[0])
    intersection_y1 = torch.max(box1[1], box2[1])
    intersection_x2 = torch.min(box1[2], box2[2])
    intersection_y2 = torch.min(box1[3], box2[3])

    # Calculate the area of intersection rectangle
    intersection_area = torch.clamp(intersection_x2 - intersection_x1, min=0) * torch.clamp(intersection_y2 - intersection_y1, min=0)

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate IoU
    iou = intersection_area / (box1_area + box2_area - intersection_area + 1e-5)  # Adding a small epsilon to avoid division by zero

    return iou

# Example usage
box1 = torch.tensor([50, 50, 150, 150])
box2 = torch.tensor([100, 100, 200, 200])
iou = calculate_iou(box1, box2)
print("IoU:", iou.item())


def nms_all_classes(boxes, scores, iou_threshold=0.9, ind=False):
    # Convert boxes and scores to PyTorch tensors
    boxes_tensor = torch.tensor(boxes)
    scores_tensor = torch.tensor(scores)
    # if ind:

    #     iou = calculate_iou(boxes_tensor[5], boxes_tensor[8])

    # Combine boxes and scores into a single tensor
    detections = torch.cat((boxes_tensor, scores_tensor.view(-1, 1)), dim=1)

    # Apply NMS using torchvision.ops.nms
    keep_indices = torchvision.ops.nms(detections[:, :4], detections[:, 4], 0.80)

    # Initialize mask to keep all boxes
    keep_mask = torch.zeros_like(scores_tensor, dtype=torch.bool)

    # Update the mask based on the indices returned by NMS
    keep_mask[keep_indices] = True

    return keep_indices

def plot_image_with_bboxes(image_path, yolo_predictions_path, output_path):
    # Load the image
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = image.shape  # Get image dimensions

    # Read YOLO predictions from the txt file
    with open(yolo_predictions_path, 'r') as file:
        yolo_predictions = [line.strip().split() for line in file]

    # Define classes and colors
    classes = {0: 'car', 1: 'van', 2: 'misc', 3: 'truck', 4: 'tram'}
    colors = {0: 'r', 1: 'g', 2: 'b', 3: 'y', 4: 'c'}

    # Plot the image
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)

    # Parse YOLO predictions and draw bounding boxes
    for pred in yolo_predictions:
        class_idx = int(pred[0])
        x, y, w, h = map(float, pred[1:5])
        confidence = float(pred[5])
        # Convert normalized coordinates to pixel coordinates
        x_center = int(x * image_width)
        y_center = int(y * image_height)
        w = int(w * image_width)
        h = int(h * image_height)
        x = x_center - w / 2
        y = y_center - h / 2

        #label = f'{classes[class_idx]}: {confidence:.2f}'
        label = f'{classes[class_idx]}: {confidence}'
        #label = str(class_idx)
        color = colors[class_idx]

        # Draw bounding box
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none', label=label)
        plt.gca().add_patch(rect)
        #plt.text(x, y, label, color=color, backgroundcolor='none', fontsize=8, verticalalignment='top')
        plt.text(x, y + 2, label, color=color, backgroundcolor='white', fontsize=8, verticalalignment='bottom')

    # Add legend
    # plt.legend(loc='upper right')
    # plt.show

    # Save the image with bounding boxes
    plt.savefig(output_path)
    plt.show()
    plt.close()

def yolo_to_tlbr(yolo_bbox, image_width, image_height):
    """
    Convert YOLO format bounding box to (top, left, bottom, right) format.

    Parameters:
    - yolo_bbox (list or tuple): YOLO format bounding box [center_x, center_y, width, height].
    - image_width (int): Width of the image.
    - image_height (int): Height of the image.

    Returns:
    - list: (top, left, bottom, right) format bounding box.
    """
    center_x, center_y, width, height = yolo_bbox

    # Calculate the top-left and bottom-right coordinates
    x_min = int((center_x - width / 2) * image_width)
    y_min = int((center_y - height / 2) * image_height)
    x_max = int((center_x + width / 2) * image_width)
    y_max = int((center_y + height / 2) * image_height)

    return [y_min, x_min, y_max, x_max]

def calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    - bbox1 (list or tuple): Bounding box in the format (top, left, bottom, right).
    - bbox2 (list or tuple): Bounding box in the format (top, left, bottom, right).

    Returns:
    - float: Intersection over Union (IoU) score.
    """
    # Calculate the intersection coordinates
    x_min = max(bbox1[1], bbox2[1])
    y_min = max(bbox1[0], bbox2[0])
    x_max = min(bbox1[3], bbox2[3])
    y_max = min(bbox1[2], bbox2[2])

    # Calculate the area of intersection
    intersection_area = max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1)

    # Calculate the area of both bounding boxes
    area_bbox1 = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    area_bbox2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    # Calculate the union area
    union_area = area_bbox1 + area_bbox2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

def calculate_iou_matrix(bboxes1, bboxes2):
    """
    Calculate IoU matrix between two sets of bounding boxes.

    Parameters:
    - bboxes1 (numpy array): Array of bounding boxes in tlbr format for the first set.
    - bboxes2 (numpy array): Array of bounding boxes in tlbr format for the second set.

    Returns:
    - numpy array: IoU matrix between each pair of bounding boxes.
    """
    iou_matrix = np.zeros((bboxes1.shape[0], bboxes2.shape[0]))

    for i in range(bboxes1.shape[0]):
        for j in range(bboxes2.shape[0]):
            iou_matrix[i, j] = calculate_iou(bboxes1[i], bboxes2[j])

    return iou_matrix


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
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


@smart_inference_mode()
def run(
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        max_det=300,  # maximum detections per image
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                              f'classes). Pass correct combination of --weights and --data that are trained together.'
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad, rect = (0.0, False) if task == 'speed' else (0.5, pt)  # square inference for benchmarks
        rect = False
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, 'names') else model.module.names  # get class names
    ##################### CV #################################################################
    # classes = [2, 7]
    # labels = {}
    # for i in classes:
    #     labels[i] = names[i]
    # names= labels
    #########################################################################################

    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile()  # profiling times
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar

    predn_with_track = []

    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
            preds, train_out = model(im) if compute_loss else (model(im, augment=augment), None)

        # Loss
        if compute_loss:
            loss += compute_loss(train_out, targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling

        

        with dt[2]:
            preds = non_max_suppression(preds,
                                        conf_thres,
                                        iou_thres,
                                        labels=lb,
                                        multi_label=True,
                                        agnostic=single_cls,
                                        max_det=max_det)
            

            
            for b, pred in enumerate(preds):
                # gg = False
                # path = Path(paths[b])
                # frame_id = int(path.stem)
                # if frame_id ==73:
                #     print("HHHHHHHHHHHHHHHHHHHHHHHHHH")
                #     gg = True
                mask_nms = nms_all_classes(pred[:,0:4], pred[:,4:5], iou_threshold=0.8)
                preds[b] = pred[mask_nms]

            
            ############################### Tracking ########################################


            
            for si, pred in enumerate(preds):
                detections = []
                
                path, shape = Path(paths[si]), shapes[si][0]
                frame_id = int(path.stem)
                # if frame_id ==73:
                #     print("HHHHHHHHHHHHHHHHHHHHHHHHHH")
                predn = pred.clone()
                scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred
                predn = predn.cpu()
                x1, y1, x2, y2, score, cls = predn[:, 0], predn[:, 1], predn[:, 2], predn[:, 3], predn[:, 4], predn[:, 5]
                cls = [model.names[int(idx)] for idx in cls]
                for i in range(predn.shape[0]):
                    detections.append(([x1[i], y1[i], int(x2[i]-x1[i]), int(y2[i]-y1[i])], score[i], cls[i]))
                image = cv2.imread(path.as_posix())
                # Convert the image from BGR to RGB (if needed)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                tracks = object_tracker.update_tracks(detections, frame=image) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )    
                track_id = []
                bbox_track = []
                track_score = []
                for track in tracks:
                    if not track.is_confirmed() or (track.det_conf is None):
                        continue
                    track_id.append(track.track_id)
                    track_score.append(track.det_conf)
                    # if track.det_conf is None:
                    #     print("HHHHHHHHHHHHHHHHHHHHHHHHHH")
                    #     print(track.to_ltrb(orig = True, orig_strict=False))
                    bbox_track.append(track.to_ltrb(orig = True, orig_strict=False))

                track_score = np.array(track_score)
                track_id = np.array(track_id)
                bbox_track = np.array(bbox_track)
                yolo_bbox = np.stack([x1, y1, x2, y2], axis=1)
                track_ids = np.ones((yolo_bbox.shape[0],1)) * -1
                # iou = calculate_iou_matrix(yolo_bbox, bbox_track)
                # if iou.shape[0]!=0 and iou.shape[1]!=0:
                #     yolo_with_track = np.argmax(iou, axis=1)
                #     tracks_with_yolo = np.argmax(iou, axis=0)
                #     # Create a mask to ensure uniqueness
                #     mask = np.zeros_like(yolo_with_track, dtype=bool)
                #     for i, track_idx in enumerate(tracks_with_yolo):
                #         if yolo_with_track[track_idx] == i:
                #             mask[track_idx] = True

                #     iou_potential_track = np.max(iou, axis=1) >=0.9
                #     track_id = track_id[yolo_with_track].reshape(-1,1)
                #     track_ids[iou_potential_track* mask] = track_id[iou_potential_track * mask] 
                if bbox_track.shape[0] != 0:
                    for i, score in enumerate(predn[:, 4]):
                        # Find indices where track_score is equal to the current score
                        indices = np.where(track_score == score.numpy())[0]

                        # Check if there is exactly one matching element
                        if len(indices) == 1:
                            track_ids[i] = track_id[indices[0]]
                

                # Check if there are at least two non -1 elements that are the same
                # A =(track_ids[track_ids != -1]).shape[0]
                # B = (np.unique(track_ids[track_ids != -1])).shape[0]
                # if (track_ids[track_ids != -1]).shape[0] != (np.unique(track_ids[track_ids != -1])).shape[0]:
                #     print("hi")


                # count_zeros = np.count_nonzero(track_ids == 0)
                # if count_zeros >1:
                #     print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
                # predn_with_track.append(np.stack([predn.numpy(), track_ids], axis=1))
                # if batch_i ==13:
                #     print("HHHHHHHHHHHHHHHHh")
                predn_with_track.append(np.hstack((predn.numpy(), track_ids)))

                neighbor_frame = {}
                matched_track_ind = {}
                neighbor_cls = {}
                max_neighbors = 10
                if si>=max_neighbors or batch_i>0:
                    
                    index = batch_i*batch_size + si
                    current_frame = predn_with_track[index]
                    for i in range(1,max_neighbors+1):
                        frame_ind = f'current_frame_minus_{i}'
                        matched_track_name = f'current_ind_minus_{i}'
                        matched_cls_name = f'current_cls_minus_{i}'
                        matched_track_ind[matched_track_name] = []
                        neighbor_cls[matched_cls_name] = []
                        neighbor_frame[frame_ind] = predn_with_track[index-i]
                    # current_frame_minus_1 = predn_with_track[index-1]
                    # current_frame_minus_2 = predn_with_track[index-2]
                    # current_frame_minus_3 = predn_with_track[index-3]
                    # current_frame_minus_4 = predn_with_track[index-4]
                    # current_frame_minus_5 = predn_with_track[index-5]
                    # current_frame_minus_6 = predn_with_track[index-6]
                    # current_frame_minus_7 = predn_with_track[index-7]
                    # current_frame_minus_8 = predn_with_track[index-8]
                    # current_frame_minus_9 = predn_with_track[index-9]
                    # current_frame_minus_10 = predn_with_track[index-10]

                    

                    for k in range(current_frame.shape[0]):
                        if current_frame[k,-1]!=-1:
                            id = current_frame[k,-1]
                            current_score  = current_frame[k,-3]
                            cls_current = current_frame[k,-2]
                            for i in range(1,max_neighbors+1):
                                frame_ind_name = f'current_frame_minus_{i}'
                                matched_track_name = f'current_ind_minus_{i}'
                                matched_cls_name = f'current_cls_minus_{i}'
                                matched_track_ind[matched_track_name] = neighbor_frame[frame_ind_name][:, -1] == current_frame[k,-1]
                                neighbor_cls[matched_cls_name] = neighbor_frame[frame_ind_name][matched_track_ind[matched_track_name], -2]


                            # current_ind_minus_1 = current_frame_minus_1[:,-1]== current_frame[k,-1]
                            # cls_current_minus_1 = current_frame_minus_1[current_ind_minus_1,-2]
                            # current_ind_minus_2 = current_frame_minus_2[:,-1]== current_frame[k,-1]
                            # cls_current_minus_2 = current_frame_minus_2[current_ind_minus_2,-2]
                            # current_ind_minus_3 = current_frame_minus_3[:,-1]== current_frame[k,-1]
                            # cls_current_minus_3 = current_frame_minus_3[current_ind_minus_3,-2]
                            # current_ind_minus_4 = current_frame_minus_4[:,-1]== current_frame[k,-1]
                            # cls_current_minus_4 = current_frame_minus_4[current_ind_minus_4,-2]
                            # current_ind_minus_5 = current_frame_minus_5[:,-1]== current_frame[k,-1]
                            # cls_current_minus_5 = current_frame_minus_5[current_ind_minus_5,-2]
                            # current_ind_minus_6 = current_frame_minus_6[:,-1]== current_frame[k,-1]
                            # cls_current_minus_6 = current_frame_minus_6[current_ind_minus_6,-2]
                            # current_ind_minus_7 = current_frame_minus_7[:,-1]== current_frame[k,-1]
                            # cls_current_minus_7 = current_frame_minus_7[current_ind_minus_7,-2]
                            # current_ind_minus_8 = current_frame_minus_8[:,-1]== current_frame[k,-1]
                            # cls_current_minus_8 = current_frame_minus_8[current_ind_minus_8,-2]
                            # current_ind_minus_9 = current_frame_minus_9[:,-1]== current_frame[k,-1]
                            # cls_current_minus_9 = current_frame_minus_9[current_ind_minus_9,-2]
                            # current_ind_minus_10 = current_frame_minus_10[:,-1]== current_frame[k,-1]
                            # cls_current_minus_10 = current_frame_minus_10[current_ind_minus_10,-2]
                            # past_ind = past_frame[:,-1] == current_frame[k,-1]
                            # cls_past = past_frame[past_ind,-2]
                            # score_previous = previous_frame[previous_frame[:,-1] == current_frame[k,-1],-3]
                            # score_past = past_frame[past_frame[:,-1]== current_frame[k,-1],-3]
                            # classes= np.hstack((cls_current_minus_1, cls_current_minus_2, cls_current_minus_3, cls_current_minus_4, cls_current_minus_5,
                            #                     cls_current_minus_6, cls_current_minus_7, cls_current_minus_8, cls_current_minus_9, cls_current_minus_10))
                            classes = []
                            
                            for i in range(1,max_neighbors+1):
                                matched_cls_name = f'current_cls_minus_{i}'
                                classes.append(neighbor_cls[matched_cls_name])
                            
                            classes = np.array([arr[0] for arr in classes if arr.shape[0] > 0])

                            # classes = np.hstack([arr for arr in classes if arr.shape[0] > 0])
                        

                            # classes= np.hstack((cls_current_minus_1, cls_current_minus_2, cls_current_minus_3, cls_current_minus_4, cls_current_minus_5))
                            # classes = np.array([cls_current_minus_one[0], cls_current_minus_two[0], cls_current_minus_three[0], cls_current_minus_four[0], cls_current_minus_five[0]])
                            neighbors = 6
                            score_thresh = 0.8
                            unique_elements, counts = np.unique(classes[-1-neighbors:-1], return_counts=True)
                            # Find the element with the maximum count
                            if counts.shape[0] != 0 :
                                estimated_classs = unique_elements[np.argmax(counts)]
                            else:
                                estimated_classs = -1
                            # if cls_previous.size != 0 and cls_past.size !=0:
                            #     classes = np.array([cls_current, cls_previous[0], cls_past[0]])
                            # neighbors = 5
                            # score_thresh = 0.8
                            if estimated_classs != -1:
                                if estimated_classs != cls_current and classes.shape[0]  >= neighbors and  max(counts) >=neighbors and  current_score <= score_thresh:
                                    # # classes = np.array([cls_current, cls_previous[0], cls_past[0]])
                                    # unique_elements, counts = np.unique(classes, return_counts=True)
                                    # # Find the element with the maximum count
                                    # estimated_classs = unique_elements[np.argmax(counts)]
                                    # i = 0
                                    for i in range(neighbors+1):
                                        if i == 0:
                                            predn_with_track[index][k, -2] = estimated_classs
                                        else:
                                            # continue
                                            current_ind = f'current_ind_minus_{i}'
                                            predn_with_track[index - i][np.where(matched_track_ind[current_ind])[0], -2] = estimated_classs

                                        


                                    # predn_with_track[index][k, -2] = estimated_classs
                                    # predn_with_track[index-1][np.where(current_ind_minus_1)[0], -2] = estimated_classs
                                    # predn_with_track[index-2][np.where(current_ind_minus_2)[0], -2] = estimated_classs
                                    # predn_with_track[index-3][np.where(current_ind_minus_3)[0], -2] = estimated_classs
                                    # predn_with_track[index-4][np.where(current_ind_minus_4)[0], -2] = estimated_classs
                                    # predn_with_track[index-5][np.where(current_ind_minus_5)[0], -2] = estimated_classs
                                    

                                    
                                # if cls_previous == cls_past and cls_previous != current_frame[k,-1]:
                                #     predn_with_track[si][k, -2] = cls_previous
                                # elif cls_previous != cls_current or cls_past != cls_current:
                                #     if score_previous >= score_past:
                                #         predn_with_track[si][k, -2] = cls_previous
                                #     else:
                                #         predn_with_track[si][k, -2] = cls_past
                    
                    # cv2.rectangle(img,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(0,0,255),2)
                    # cv2.putText(img, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Example usage
                # for j, path in enumerate(paths):
                #     image_path = path
                #     im_number = os.path.splitext(os.path.basename(image_path))[0]
                #     yolo_predictions_path = '/media/hamed/Data/CV_PRJ/yolov5/runs/val/exp/labels/' + im_number + '.txt'
                #     output_path = '/media/hamed/Data/CV_PRJ/yolov5/runs/val/exp/output/' + im_number + '.jpg' 

                #     plot_image_with_bboxes(image_path, yolo_predictions_path, output_path)
            ############################### class change ############################################################
                                    estimated_classs = torch.tensor(estimated_classs, dtype=torch.float32).to(device='cuda').reshape(-1,1)
                                    # # A = np.where(previous_ind)[0]
                                    # # ind = torch.tensor(np.where(previous_ind)[0])
                                    # i = 0
                                    for i in range(neighbors+1):
                                        if i == 0:
                                            
                                            preds[si][k, -1] = estimated_classs
                                        else:
                                            matched_track_name = f'current_ind_minus_{i}'
                                            preds[si-i][np.where(matched_track_ind[matched_track_name])[0], -1] = estimated_classs

                                    # preds[si][k, -1] = estimated_classs
                                    # preds[si-1][np.where(current_ind_minus_1)[0], -1] = estimated_classs
                                    # preds[si-2][np.where(current_ind_minus_2)[0], -1] = estimated_classs
                                    # preds[si-3][np.where(current_ind_minus_3)[0], -1] = estimated_classs
                                    # preds[si-4][np.where(current_ind_minus_4)[0], -1] = estimated_classs
                                    # preds[si-5][np.where(current_ind_minus_5)[0], -1] = estimated_classs
                    # if si >=5:
                    #     preds[si][:, -2] = torch.from_numpy(predn_with_track[index][:, -1])
                    #     preds[si-1][:, -2] = torch.from_numpy(predn_with_track[index-1][:, -1])
                    #     preds[si-2][:, -2] = torch.from_numpy(predn_with_track[index-2][:, -1])
                    #     preds[si-3][:, -2] = torch.from_numpy(predn_with_track[index-3][:, -1])
                    #     preds[si-4][:, -2] = torch.from_numpy(predn_with_track[index-4][:, -1])
                    #     preds[si-5][:, -2] = torch.from_numpy(predn_with_track[index-5][:, -1])
                    # elif preds[si].shape[0] != 0:
                    #     # if(torch.from_numpy(predn_with_track[index][:, -1]).shape != preds[si][:, -2].shape):
                    #     #     print(torch.from_numpy(predn_with_track[index][:, -1]).shape)
                    #     #     print(preds[si][:, -2].shape)
                    #     #     print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
                    #     preds[si][:, -2] = torch.from_numpy(predn_with_track[index][:, -1])


            #################################################################################

        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

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

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

            # Tracking output
            ######################################################################################
            # for j, path in enumerate(paths):
            image_path = path
            im_number = os.path.splitext(os.path.basename(image_path))[0]
            yolo_predictions_path = '/media/hamed/Data/CV_PRJ/yolov5/runs/val/exp/labels/' + im_number + '.txt'
            output = '/media/hamed/Data/CV_PRJ/yolov5/runs/val/exp/output/'
            output_path = output + im_number + '.jpg' 
            if not os.path.exists(output):
                os.makedirs(output)
            plot_image_with_bboxes(image_path, yolo_predictions_path, output_path)
            ######################################################################################

        # Plot images
        if plots and batch_i < 3:
            # plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
            plot_images(im, output_to_target(preds), paths, save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred

        callbacks.run('on_val_batch_end', batch_i, im, targets, paths, shapes, preds)

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end', nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path('../datasets/coco/annotations/instances_val2017.json'))  # annotations
        if not os.path.exists(anno_json):
            anno_json = os.path.join(data['path'], 'annotations', 'instances_val2017.json')
        pred_json = str(save_dir / f'{w}_predictions.json')  # predictions
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements('pycocotools>=2.0.6')
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/CV.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', default=True, action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results')
        if opt.save_hybrid:
            LOGGER.info('WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != 'cpu'  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            subprocess.run(['zip', '-r', 'study.zip', 'study_*.txt'])
            plot_val_study(x=x)  # plot
        else:
            raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test", "speed", "study")')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
