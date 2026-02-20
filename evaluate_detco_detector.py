#!/usr/bin/env python
"""
Evaluate DetCo-based Class-Agnostic Object Detector

This script evaluates a trained DetCo detector on COCO and VOC datasets.
Similar to evaluate_frcnn_multiple_seeds.py but for a single model.

Usage:
    python3 evaluate_detco_detector.py --model_weights ./output_detco/model_final.pth
    
    
python3 evaluate_detco_detector.py \
  --model_weights ./output_detco/full_size/g9/pseudo_bboxes_0.004_mk/model_final.pth \
  --output_dir ./output_detco_evaluation
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import random
from tqdm import tqdm
from collections import OrderedDict

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets import load_coco_json
from detectron2.layers import get_norm
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads

# VOC imports
import xml.etree.ElementTree as ET
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.amp import autocast


def seed_everything(seed=99):
    """Set seed for reproducibility"""
    print(f"Setting seed to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsExtraNorm(Res5ROIHeads):
    """
    Custom ROI head with extra normalization layer
    Same as used in DetCo and MoCo training
    """
    def _build_res5_block(self, cfg):
        seq, out_channels = super()._build_res5_block(cfg)
        norm_type = cfg.MODEL.RESNETS.NORM
        norm_layer = get_norm(norm_type, out_channels)
        seq.add_module("norm", norm_layer)
        return seq, out_channels


class CustomCOCOEvaluator(COCOEvaluator):
    """Extends COCOEvaluator to include Average Recall (AR) metrics"""
    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        results = super()._derive_coco_results(coco_eval, iou_type, class_names)
        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return results
        
        # Add AR metrics
        results["AR1"] = coco_eval.stats[6]
        results["AR10"] = coco_eval.stats[7]
        results["AR100"] = coco_eval.stats[8]
        results["ARs"] = coco_eval.stats[9]
        results["ARm"] = coco_eval.stats[10]
        results["ARl"] = coco_eval.stats[11]
        return results


def register_class_agnostic_dataset(name, json_file, image_root):
    """Register COCO dataset with class-agnostic conversion"""
    def wrapper():
        temp_name = "temp_" + name
        dataset = load_coco_json(json_file, image_root, temp_name)
        # Convert all categories to class 0 (class-agnostic)
        for item in dataset:
            for anno in item["annotations"]:
                anno["category_id"] = 0
        return dataset

    # Remove existing registrations
    for ds_name in [name, "temp_" + name]:
        if ds_name in DatasetCatalog.list():
            DatasetCatalog.remove(ds_name)
        if ds_name in MetadataCatalog.list():
            MetadataCatalog.remove(ds_name)

    DatasetCatalog.register(name, wrapper)
    metadata = MetadataCatalog.get(name)
    metadata.set(image_root=image_root, thing_classes=["object"], evaluator_type="coco")
    return metadata


def setup_cfg(model_weights, num_classes=1, freeze_at=2):
    """Setup Detectron2 config for DetCo model"""
    cfg = get_cfg()
    
    # ============ MODEL ============
    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
    
    # Backbone: ResNet-50 with SyncBN
    cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    cfg.MODEL.BACKBONE.FREEZE_AT = freeze_at
    
    cfg.MODEL.RESNETS.DEPTH = 50
    cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
    cfg.MODEL.RESNETS.NORM = "SyncBN"
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res4"]
    
    # RPN
    cfg.MODEL.RPN.IN_FEATURES = ["res4"]
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    
    # ROI Heads
    cfg.MODEL.ROI_HEADS.NAME = "Res5ROIHeadsExtraNorm"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["res4"]
    
    # ROI Box Head
    cfg.MODEL.ROI_BOX_HEAD.NORM = "SyncBN"
    
    # Weights
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.MASK_ON = False
    
    # Pixel normalization (ImageNet-style)
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]
    
    # Device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ============ INPUT ============
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.FORMAT = "RGB"
    
    # ============ DATALOADER ============
    cfg.DATALOADER.NUM_WORKERS = 4
    
    # ============ TEST ============
    cfg.TEST.PRECISE_BN.ENABLED = False
    
    return cfg


def evaluate_coco_dataset(model_weights, dataset_name, test_json, test_images, output_dir):
    """Evaluate on COCO dataset"""
    print(f"\n{'='*80}")
    print(f"Evaluating COCO Dataset: {dataset_name}")
    print(f"{'='*80}")
    
    dataset_key = f"class_agnostic_{dataset_name}"
    register_class_agnostic_dataset(dataset_key, test_json, test_images)
    
    # Setup config
    cfg = setup_cfg(model_weights)
    cfg.DATASETS.TEST = (dataset_key,)
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Build predictor
    predictor = DefaultPredictor(cfg)
    
    # Build evaluator
    evaluator = CustomCOCOEvaluator(dataset_key, cfg, False, output_dir=cfg.OUTPUT_DIR)
    
    # Build test loader
    test_loader = build_detection_test_loader(cfg, dataset_key)
    
    # Run inference with FP16
    print("\nRunning inference...")
    predictor.model.eval()
    evaluator.reset()
    
    with torch.no_grad():
        for inputs in tqdm(test_loader, desc=f"Evaluating {dataset_name}"):
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                outputs = predictor.model(inputs)
            evaluator.process(inputs, outputs)
    
    # Evaluate
    print("\nComputing metrics...")
    results = evaluator.evaluate()
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Extract metrics
    metrics = {
        'AP': results['bbox']['AP'],
        'AP50': results['bbox']['AP50'],
        'AP75': results['bbox']['AP75'],
        'APs': results['bbox']['APs'],
        'APm': results['bbox']['APm'],
        'APl': results['bbox']['APl'],
        'AR1': results['bbox']['AR1'],
        'AR10': results['bbox']['AR10'],
        'AR100': results['bbox']['AR100'],
        'ARs': results['bbox']['ARs'],
        'ARm': results['bbox']['ARm'],
        'ARl': results['bbox']['ARl']
    }
    
    return metrics


def evaluate_voc_dataset(model_weights, voc_year, voc_annotation_dir, voc_image_dir, output_dir):
    """Evaluate on VOC dataset"""
    print(f"\n{'='*80}")
    print(f"Evaluating VOC{voc_year} Dataset")
    print(f"{'='*80}")
    
    # Convert VOC to COCO format
    print("\nConverting VOC to COCO format...")
    coco_dataset = {
        "info": {
            "description": f"VOC{voc_year} in COCO format",
            "version": "1.0",
            "year": int(f"20{voc_year}") if len(voc_year) == 2 else int(voc_year),
            "date_created": "2026"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}],
    }
    
    image_id_map = {}
    image_id_counter = 1
    ann_id_counter = 1
    
    xml_files = [f for f in os.listdir(voc_annotation_dir) if f.endswith(".xml")]
    for xml_file in tqdm(xml_files, desc=f"Converting VOC{voc_year} annotations"):
        xml_path = os.path.join(voc_annotation_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        filename = root.find("filename").text
        base_name = os.path.splitext(filename)[0]
        
        if base_name not in image_id_map:
            image_id_map[base_name] = image_id_counter
            size = root.find("size")
            width = int(size.find("width").text)
            height = int(size.find("height").text)
            coco_dataset["images"].append({
                "id": image_id_counter,
                "file_name": filename,
                "width": width,
                "height": height
            })
            image_id_counter += 1
        
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
                "category_id": 1,
                "bbox": [xmin, ymin, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id_counter += 1
    
    # Save temporary COCO file
    temp_coco_json = os.path.join(output_dir, f"temp_voc{voc_year}_annotations.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(temp_coco_json, "w") as f:
        json.dump(coco_dataset, f)
    
    # Build model
    cfg = setup_cfg(model_weights)
    predictor = DefaultPredictor(cfg)
    
    # Run inference
    print("\nRunning inference...")
    predictions = []
    all_images = [f for f in os.listdir(voc_image_dir)
                  if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    
    with torch.no_grad():
        for img_file in tqdm(all_images, desc=f"Inference on VOC{voc_year}"):
            base_name = os.path.splitext(img_file)[0]
            if base_name not in image_id_map:
                continue
            
            image_id = image_id_map[base_name]
            img_path = os.path.join(voc_image_dir, img_file)
            img = cv2.imread(img_path)
            
            with autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                outputs = predictor(img)
            
            instances = outputs["instances"].to("cpu")
            boxes_xyxy = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
            scores = instances.scores.numpy() if instances.has("scores") else []
            
            for box, score in zip(boxes_xyxy, scores):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                predictions.append({
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score)
                })
    
    # Save predictions
    pred_json = os.path.join(output_dir, f"predictions_voc{voc_year}.json")
    with open(pred_json, "w") as f:
        json.dump(predictions, f)
    
    # Evaluate
    print("\nComputing metrics...")
    coco_gt = COCO(temp_coco_json)
    coco_dt = coco_gt.loadRes(pred_json)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.useCats = 0  # Class-agnostic
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    stats = coco_eval.stats
    
    # Cleanup
    os.remove(temp_coco_json)
    os.remove(pred_json)
    torch.cuda.empty_cache()
    
    # Extract metrics
    metrics = {
        'AP': stats[0],
        'AP50': stats[1],
        'AP75': stats[2],
        'APs': stats[3],
        'APm': stats[4],
        'APl': stats[5],
        'AR1': stats[6],
        'AR10': stats[7],
        'AR100': stats[8],
        'ARs': stats[9],
        'ARm': stats[10],
        'ARl': stats[11]
    }
    
    return metrics


def print_results(all_results, model_weights):
    """Print results in a nice format"""
    print("\n" + "="*100)
    print("EVALUATION RESULTS")
    print("="*100)
    print(f"\nModel: {model_weights}")
    
    for dataset_name, metrics in all_results.items():
        print("\n" + "-"*100)
        print(f"Dataset: {dataset_name}")
        print("-"*100)
        
        # Print main metrics
        print(f"\nAverage Precision (AP):")
        print(f"  AP      (IoU=0.50:0.95): {metrics['AP']:.4f}")
        print(f"  AP50    (IoU=0.50)     : {metrics['AP50']:.4f}")
        print(f"  AP75    (IoU=0.75)     : {metrics['AP75']:.4f}")
        print(f"  APs     (small)        : {metrics['APs']:.4f}")
        print(f"  APm     (medium)       : {metrics['APm']:.4f}")
        print(f"  APl     (large)        : {metrics['APl']:.4f}")
        
        print(f"\nAverage Recall (AR):")
        print(f"  AR@1    (maxDets=1)    : {metrics['AR1']:.4f}")
        print(f"  AR@10   (maxDets=10)   : {metrics['AR10']:.4f}")
        print(f"  AR@100  (maxDets=100)  : {metrics['AR100']:.4f}")
        print(f"  ARs     (small)        : {metrics['ARs']:.4f}")
        print(f"  ARm     (medium)       : {metrics['ARm']:.4f}")
        print(f"  ARl     (large)        : {metrics['ARl']:.4f}")
    
    print("\n" + "="*100)


def save_results_to_file(all_results, model_weights, output_file):
    """Save results to a text file"""
    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("DetCo Detector Evaluation Results\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Model: {model_weights}\n")
        f.write(f"Date: 2026-01-30\n\n")
        
        for dataset_name, metrics in all_results.items():
            f.write("\n" + "-"*100 + "\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write("-"*100 + "\n\n")
            
            f.write("Average Precision (AP):\n")
            f.write(f"  AP      (IoU=0.50:0.95): {metrics['AP']:.6f}\n")
            f.write(f"  AP50    (IoU=0.50)     : {metrics['AP50']:.6f}\n")
            f.write(f"  AP75    (IoU=0.75)     : {metrics['AP75']:.6f}\n")
            f.write(f"  APs     (small)        : {metrics['APs']:.6f}\n")
            f.write(f"  APm     (medium)       : {metrics['APm']:.6f}\n")
            f.write(f"  APl     (large)        : {metrics['APl']:.6f}\n\n")
            
            f.write("Average Recall (AR):\n")
            f.write(f"  AR@1    (maxDets=1)    : {metrics['AR1']:.6f}\n")
            f.write(f"  AR@10   (maxDets=10)   : {metrics['AR10']:.6f}\n")
            f.write(f"  AR@100  (maxDets=100)  : {metrics['AR100']:.6f}\n")
            f.write(f"  ARs     (small)        : {metrics['ARs']:.6f}\n")
            f.write(f"  ARm     (medium)       : {metrics['ARm']:.6f}\n")
            f.write(f"  ARl     (large)        : {metrics['ARl']:.6f}\n")
        
        f.write("\n" + "="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate DetCo-based Class-Agnostic Object Detector',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model_weights', type=str, required=True,
                        help='Path to model weights (.pth file)')
    parser.add_argument('--output_dir', type=str, default='./output_detco_evaluation',
                        help='Directory for saving evaluation results')
    parser.add_argument('--seed', type=int, default=99,
                        help='Random seed for reproducibility')
    
    # Dataset options
    parser.add_argument('--eval_coco', action='store_true', default=True,
                        help='Evaluate on COCO datasets')
    parser.add_argument('--eval_voc', action='store_true', default=True,
                        help='Evaluate on VOC datasets')
    
    args = parser.parse_args()
    
    # Set seed
    seed_everything(args.seed)
    
    # Check if model exists
    if not os.path.exists(args.model_weights):
        print(f"Error: Model weights not found at {args.model_weights}")
        sys.exit(1)
    
    print(f"\n{'='*100}")
    print(f"DetCo Detector Evaluation")
    print(f"{'='*100}")
    print(f"Model: {args.model_weights}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*100}\n")
    
    all_results = OrderedDict()
    
    # Evaluate on COCO datasets
    if args.eval_coco:
        # COCO20k training set
        if os.path.exists('./data/coco20k/annotations/coco20k.json'):
            try:
                results = evaluate_coco_dataset(
                    model_weights=args.model_weights,
                    dataset_name='COCO20k_train',
                    test_json='./data/coco20k/annotations/coco20k.json',
                    test_images='./data/coco20k/images/train/',
                    output_dir=os.path.join(args.output_dir, 'coco20k_train')
                )
                all_results['COCO20k_train'] = results
            except Exception as e:
                print(f"Error evaluating COCO20k_train: {e}")
        
        # COCO val2017
        if os.path.exists('./data/coco20k/annotations/instances_val2017.json'):
            try:
                results = evaluate_coco_dataset(
                    model_weights=args.model_weights,
                    dataset_name='COCO_val2017',
                    test_json='./data/coco20k/annotations/instances_val2017.json',
                    test_images='./data/coco20k/images/val/',
                    output_dir=os.path.join(args.output_dir, 'coco_val2017')
                )
                all_results['COCO_val2017'] = results
            except Exception as e:
                print(f"Error evaluating COCO_val2017: {e}")
    
    # Evaluate on VOC datasets
    if args.eval_voc:
        # VOC 2007
        if os.path.exists('./data/VOC/2007/train_val/Annotations'):
            try:
                results = evaluate_voc_dataset(
                    model_weights=args.model_weights,
                    voc_year='07',
                    voc_annotation_dir='./data/VOC/2007/train_val/Annotations',
                    voc_image_dir='./data/VOC/2007/train_val/JPEGImages',
                    output_dir=os.path.join(args.output_dir, 'voc2007')
                )
                all_results['VOC2007'] = results
            except Exception as e:
                print(f"Error evaluating VOC2007: {e}")
        
        # VOC 2012
        if os.path.exists('./data/VOC/2012/train_val/Annotations'):
            try:
                results = evaluate_voc_dataset(
                    model_weights=args.model_weights,
                    voc_year='12',
                    voc_annotation_dir='./data/VOC/2012/train_val/Annotations',
                    voc_image_dir='./data/VOC/2012/train_val/JPEGImages',
                    output_dir=os.path.join(args.output_dir, 'voc2012')
                )
                all_results['VOC2012'] = results
            except Exception as e:
                print(f"Error evaluating VOC2012: {e}")
    
    # Print and save results
    if len(all_results) > 0:
        print_results(all_results, args.model_weights)
        
        # Save results to file
        output_file = os.path.join(args.output_dir, "evaluation_results.txt")
        os.makedirs(args.output_dir, exist_ok=True)
        save_results_to_file(all_results, args.model_weights, output_file)
        print(f"\nResults saved to: {output_file}")
        
        # Also save as JSON
        json_file = os.path.join(args.output_dir, "evaluation_results.json")
        with open(json_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to: {json_file}")
    else:
        print("\nNo successful evaluations!")


if __name__ == "__main__":
    main()
