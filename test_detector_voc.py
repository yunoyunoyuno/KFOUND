#!/usr/bin/env python

import os
import json
import glob
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from tqdm import tqdm
import argparse;

import torch
torch.cuda.empty_cache()
from torch.amp import autocast;

# -------------------------------
# Detectron2 Imports
# -------------------------------
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import Boxes

# -------------------------------
# pycocotools for Evaluation
# -------------------------------
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from detectron2.layers import get_norm
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads

import random

def seed_everything(seed=42):
    random.seed(seed)
    print("set seed to",seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (at some potential performance cost):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(99);

@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsExtraNorm(Res5ROIHeads):
    def _build_res5_block(self, cfg):
        seq, out_channels = super()._build_res5_block(cfg)
        norm_type = cfg.MODEL.RESNETS.NORM
        norm_layer = get_norm(norm_type, out_channels)
        seq.add_module("norm", norm_layer)
        return seq, out_channels

# ---------------------------------------------------------
# 1) PATH CONFIGURATION
# ---------------------------------------------------------
parser = argparse.ArgumentParser("Faster RCNN on VOC");
parser.add_argument("--weight",
                    type = str,
                    default="./output_faster_rcnn_u/small_size/KanG7D5/pseudo_bboxes_0.004_mk/model_final.pth",
                    required=True);
parser.add_argument("--testset",
                    type=str,
                    default="07",
                    required=True);

args = parser.parse_args();

if args.testset == "07":
    VOC_ANNOTATION_DIR = "./data/VOC/2007/train_val/Annotations"
    VOC_IMAGE_DIR      = "./data/VOC/2007/train_val/JPEGImages"
    CONFIG_FILE        = "./configs/RN50_DINO_FRCNN_VOC07_CAD.yaml" 
elif args.testset == "12":
    VOC_IMAGE_DIR      = "./data/VOC/2012/train_val/JPEGImages"
    VOC_ANNOTATION_DIR = "./data/VOC/2012/train_val/Annotations"
    CONFIG_FILE        = "./configs/RN50_DINO_FRCNN_VOC12_CAD.yaml";
    
MODEL_WEIGHTS = args.weight;
TEMP_COCO_JSON     = "temp_voc_annotations.json"
PRED_JSON          = "predictions.json"

# ---------------------------------------------------------
# 2) CONVERT VOC TO COCO (CLASS-AGNOSTIC)
# ---------------------------------------------------------
coco_dataset = {
    "info": {
        "description": "VOC to COCO format for class-agnostic detection",
        "version": "1.0",
        "year": 2025,
        "contributor": "Auto-generated",
        "date_created": "2025"
    },
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "object"}],  # single class
}

image_id_map = {}
image_id_counter = 1
ann_id_counter = 1

xml_files = [f for f in os.listdir(VOC_ANNOTATION_DIR) if f.endswith(".xml")]
for xml_file in tqdm(xml_files, desc="Converting VOC to COCO"):
    xml_path = os.path.join(VOC_ANNOTATION_DIR, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    base_name = os.path.splitext(filename)[0]

    # Create an image entry if new
    if base_name not in image_id_map:
        image_id_map[base_name] = image_id_counter
        size = root.find("size")
        width  = int(size.find("width").text)
        height = int(size.find("height").text)
        coco_dataset["images"].append({
            "id": image_id_counter,
            "file_name": filename,
            "width": width,
            "height": height
        })
        image_id_counter += 1

    # Add bounding boxes as single-class
    image_id = image_id_map[base_name]
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        w = xmax - xmin
        h = ymax - ymin

        coco_dataset["annotations"].append({
            "id": ann_id_counter,
            "image_id": image_id,
            "category_id": 1,  # single class
            "bbox": [xmin, ymin, w, h],
            "area": w * h,
            "iscrowd": 0,
        })
        ann_id_counter += 1

# Save the temporary COCO file
with open(TEMP_COCO_JSON, "w") as f:
    json.dump(coco_dataset, f)

# ---------------------------------------------------------
# 3) BUILD DETECTRON2 MODEL
# ---------------------------------------------------------
cfg = get_cfg()
cfg.merge_from_file(CONFIG_FILE)
cfg.MODEL.WEIGHTS = MODEL_WEIGHTS
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # class-agnostic
predictor = DefaultPredictor(cfg)

# ---------------------------------------------------------
# 4) RUN INFERENCE AND COLLECT PREDICTIONS (COCO FORMAT)
# ---------------------------------------------------------
predictions = []

all_images = [f for f in os.listdir(VOC_IMAGE_DIR)
              if f.lower().endswith((".jpg", ".png", ".jpeg"))]

# Wrap the loop in `torch.no_grad()` so we don't store grads
with torch.no_grad():
    for img_file in tqdm(all_images, desc="Running Inference"):
        base_name = os.path.splitext(img_file)[0]
        if base_name not in image_id_map:
            # The image might not exist in annotations, skip
            continue

        image_id = image_id_map[base_name]
        img_path = os.path.join(VOC_IMAGE_DIR, img_file)

        # Load image
        img = cv2.imread(img_path)

        # Here is where we enable AMP for FP16 inference
        with autocast(device_type="cuda", dtype=torch.float16):
            outputs = predictor(img)

        # Extract boxes (Detectron2 is XYXY format)
        instances = outputs["instances"].to("cpu")
        boxes_xyxy = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
        scores = instances.scores.numpy() if instances.has("scores") else []

        # Convert XYXY -> XYWH for COCO
        for box, score in zip(boxes_xyxy, scores):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            predictions.append({
                "image_id": image_id,
                "category_id": 1,  # single class
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(score)
            })

# Save predictions
with open(PRED_JSON, "w") as f:
    json.dump(predictions, f)

# ---------------------------------------------------------
# 5) EVALUATE WITH PYCOCOTOOLS (CLASS-AGNOSTIC)
# ---------------------------------------------------------
coco_gt = COCO(TEMP_COCO_JSON)
coco_dt = coco_gt.loadRes(PRED_JSON)
coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

# Class-agnostic: don't separate by category
coco_eval.params.useCats = 0

# Evaluate
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

stats = coco_eval.stats
metric_descriptions = [
    "AP @[IoU=0.50:0.95]",
    "AP @[IoU=0.50]",
    "AP @[IoU=0.75]",
    "AP (Small)",
    "AP (Medium)",
    "AP (Large)",
    "AR @[maxDets=1]",
    "AR @[maxDets=10]",
    "AR @[maxDets=100]",
    "AR (Small)",
    "AR (Medium)",
    "AR (Large)",
]

print("\nClass-Agnostic Detection Metrics:")
for desc, val in zip(metric_descriptions, stats):
    print(f"{desc}: {val:.6f}")

# Cleanup (optional)
os.remove(TEMP_COCO_JSON)
os.remove(PRED_JSON)
print("\nEvaluation complete.")