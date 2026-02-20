#!/usr/bin/env python
"""
Evaluate Faster R-CNN models with multiple seeds and calculate mean ± std
This script runs evaluation on multiple model weights from different seeds
and calculates the average performance with standard deviation.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import random
from tqdm import tqdm

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator
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
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsExtraNorm(Res5ROIHeads):
    """Custom ROI head with extra normalization layer"""
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
        for item in dataset:
            for anno in item["annotations"]:
                anno["category_id"] = 0
        return dataset

    for ds_name in [name, "temp_" + name]:
        if ds_name in DatasetCatalog.list():
            DatasetCatalog.remove(ds_name)
        if ds_name in MetadataCatalog.list():
            MetadataCatalog.remove(ds_name)

    DatasetCatalog.register(name, wrapper)
    metadata = MetadataCatalog.get(name)
    metadata.set(image_root=image_root, thing_classes=["object"], evaluator_type="coco")
    return metadata


def evaluate_coco_dataset(model_weights, dataset_name, test_json, test_images, config_file, output_dir):
    """Evaluate on COCO dataset"""
    dataset_key = f"class_agnostic_{dataset_name}"
    register_class_agnostic_dataset(dataset_key, test_json, test_images)
    
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.DATASETS.TEST = (dataset_key,)
    cfg.MODEL.ROI_HEADS.NAME = "Res5ROIHeadsExtraNorm"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = model_weights
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    predictor = DefaultPredictor(cfg)
    evaluator = CustomCOCOEvaluator(dataset_key, cfg, False, output_dir=cfg.OUTPUT_DIR)
    test_loader = build_detection_test_loader(cfg, dataset_key)
    
    # Inference with FP16
    predictor.model.eval()
    evaluator.reset()
    with torch.no_grad():
        for inputs in tqdm(test_loader, desc=f"Evaluating {dataset_name}"):
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = predictor.model(inputs)
            evaluator.process(inputs, outputs)
    
    results = evaluator.evaluate()
    torch.cuda.empty_cache()
    
    return {
        'AP': results['bbox']['AP'],
        'AP50': results['bbox']['AP50'],
        'AP75': results['bbox']['AP75'],
        'AR1': results['bbox']['AR1'],
        'AR10': results['bbox']['AR10'],
        'AR100': results['bbox']['AR100']
    }


def evaluate_voc_dataset(model_weights, voc_year, voc_annotation_dir, voc_image_dir, config_file):
    """Evaluate on VOC dataset"""
    # Convert VOC to COCO format
    coco_dataset = {
        "info": {
            "description": f"VOC{voc_year} in COCO format",
            "version": "1.0",
            "year": int(f"20{voc_year}") if len(voc_year) == 2 else int(voc_year),
            "date_created": "2025"
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
    for xml_file in tqdm(xml_files, desc=f"Converting VOC{voc_year} to COCO"):
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
    temp_coco_json = f"temp_voc{voc_year}_annotations.json"
    with open(temp_coco_json, "w") as f:
        json.dump(coco_dataset, f)
    
    # Build model
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    predictor = DefaultPredictor(cfg)
    
    # Run inference
    predictions = []
    all_images = [f for f in os.listdir(voc_image_dir)
                  if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    
    with torch.no_grad():
        for img_file in tqdm(all_images, desc=f"Running inference VOC{voc_year}"):
            base_name = os.path.splitext(img_file)[0]
            if base_name not in image_id_map:
                continue
            
            image_id = image_id_map[base_name]
            img_path = os.path.join(voc_image_dir, img_file)
            img = cv2.imread(img_path)
            
            with autocast(device_type="cuda", dtype=torch.float16):
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
    pred_json = f"predictions_voc{voc_year}.json"
    with open(pred_json, "w") as f:
        json.dump(predictions, f)
    
    # Evaluate
    coco_gt = COCO(temp_coco_json)
    coco_dt = coco_gt.loadRes(pred_json)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.useCats = 0
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    stats = coco_eval.stats
    
    # Cleanup
    os.remove(temp_coco_json)
    os.remove(pred_json)
    torch.cuda.empty_cache()
    
    return {
        'AP': stats[0],
        'AP50': stats[1],
        'AP75': stats[2],
        'AR1': stats[6],
        'AR10': stats[7],
        'AR100': stats[8]
    }


def evaluate_single_model(model_weights, datasets):
    """Evaluate a single model on all datasets"""
    print(f"\n{'='*80}")
    print(f"Evaluating model: {model_weights}")
    print(f"{'='*80}")
    
    all_results = {}
    
    for dataset_info in datasets:
        dataset_name = dataset_info['name']
        print(f"\n--- Evaluating on {dataset_name} ---")
        
        if 'coco' in dataset_name.lower():
            results = evaluate_coco_dataset(
                model_weights=model_weights,
                dataset_name=dataset_name,
                test_json=dataset_info['json'],
                test_images=dataset_info['images'],
                config_file=dataset_info['config'],
                output_dir=dataset_info['output_dir']
            )
        else:  # VOC
            results = evaluate_voc_dataset(
                model_weights=model_weights,
                voc_year=dataset_info['year'],
                voc_annotation_dir=dataset_info['annotations'],
                voc_image_dir=dataset_info['images'],
                config_file=dataset_info['config']
            )
        
        all_results[dataset_name] = results
        print(f"Results for {dataset_name}:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.6f}")
    
    return all_results


def calculate_statistics(all_results):
    """Calculate mean and std for each dataset and metric"""
    stats = {}
    
    # Get dataset names from first result
    if len(all_results) > 0:
        dataset_names = all_results[0].keys()
        
        for dataset_name in dataset_names:
            stats[dataset_name] = {}
            
            # Get metric names
            if dataset_name in all_results[0]:
                metric_names = all_results[0][dataset_name].keys()
                
                for metric_name in metric_names:
                    values = [result[dataset_name][metric_name] 
                             for result in all_results 
                             if dataset_name in result and metric_name in result[dataset_name]]
                    
                    if len(values) > 0:
                        stats[dataset_name][metric_name] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'values': values
                        }
    
    return stats


def print_statistics(stats, model_configs):
    """Print statistics in a nice format"""
    print("\n" + "="*100)
    print("FINAL RESULTS - Mean ± Standard Deviation")
    print("="*100)
    
    print("\nModel Configurations:")
    for i, config in enumerate(model_configs, 1):
        print(f"  {i}. {config['weights']}")
    
    print(f"\nNumber of experiments: {len(model_configs)}")
    
    for dataset_name, dataset_stats in stats.items():
        print("\n" + "="*100)
        print(f"Dataset: {dataset_name}")
        print("-"*100)
        
        for metric_name, metric_data in dataset_stats.items():
            mean = metric_data['mean']
            std = metric_data['std']
            values = metric_data['values']
            
            print(f"\n{metric_name}:")
            print(f"  Mean ± Std: {mean:.6f} ± {std:.6f}")
            print(f"  Individual values: {[f'{v:.6f}' for v in values]}")
            print(f"  Min: {min(values):.6f}, Max: {max(values):.6f}")
    
    print("\n" + "="*100)


def save_results_to_file(stats, model_configs, output_file):
    """Save results to a text file"""
    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("Faster R-CNN Evaluation Results - Multiple Seeds\n")
        f.write("="*100 + "\n\n")
        
        f.write("Model Configurations:\n")
        for i, config in enumerate(model_configs, 1):
            f.write(f"  {i}. {config['weights']}\n")
        
        f.write(f"\nNumber of experiments: {len(model_configs)}\n")
        
        for dataset_name, dataset_stats in stats.items():
            f.write("\n" + "="*100 + "\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write("-"*100 + "\n\n")
            
            for metric_name, metric_data in dataset_stats.items():
                mean = metric_data['mean']
                std = metric_data['std']
                values = metric_data['values']
                
                f.write(f"{metric_name}:\n")
                f.write(f"  Mean ± Std: {mean:.6f} ± {std:.6f}\n")
                f.write(f"  Individual values: {[f'{v:.6f}' for v in values]}\n")
                f.write(f"  Min: {min(values):.6f}, Max: {max(values):.6f}\n\n")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Faster R-CNN with multiple seeds',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    args = parser.parse_args()
    
    # Set seed
    seed_everything(99)
    
    # Define model configurations (3 seeds) - IoU-based Grid 9 Degree 4
    model_configs = [
        {
            'name': 'KanG9D4_seed0_iou',
            'weights': './output_faster_rcnn_u/iou_based/KanG9D4_seed0/model_final.pth'
        },
        {
            'name': 'KanG9D4_seed100_iou',
            'weights': './output_faster_rcnn_u/iou_based/KanG9D4_seed100/model_final.pth'
        },
        {
            'name': 'KanG9D4_seed200_iou',
            'weights': './output_faster_rcnn_u/iou_based/KanG9D4_seed200/model_final.pth'
        }
    ]
    
    # Define datasets
    datasets = [
        {
            'name': 'COCO20k_train',
            'json': './data/coco20k/annotations/coco20k.json',
            'images': './data/coco20k/images/train/',
            'config': './configs/RN50_DINO_FRCNN_COCO20k_CAD.yaml',
            'output_dir': './output_evaluation/coco20k'
        },
        {
            'name': 'COCO20k_val',
            'json': './data/coco20k/annotations/instances_val2017.json',
            'images': './data/coco20k/images/val/',
            'config': './configs/RN50_DINO_FRCNN_COCO20k_CAD.yaml',
            'output_dir': './output_evaluation/cocoval2017'
        },
        {
            'name': 'VOC07',
            'year': '07',
            'annotations': './data/VOC/2007/train_val/Annotations',
            'images': './data/VOC/2007/train_val/JPEGImages',
            'config': './configs/RN50_DINO_FRCNN_VOC07_CAD.yaml'
        },
        {
            'name': 'VOC12',
            'year': '12',
            'annotations': './data/VOC/2012/train_val/Annotations',
            'images': './data/VOC/2012/train_val/JPEGImages',
            'config': './configs/RN50_DINO_FRCNN_VOC12_CAD.yaml'
        }
    ]
    
    print(f"\n{'='*100}")
    print(f"Starting Faster R-CNN evaluation with {len(model_configs)} different seeds")
    print(f"Datasets: {[d['name'] for d in datasets]}")
    print(f"{'='*100}\n")
    
    # Evaluate all models
    all_results = []
    for i, model_config in enumerate(model_configs, 1):
        print(f"\n[{i}/{len(model_configs)}] Evaluating {model_config['name']}...")
        try:
            results = evaluate_single_model(
                model_weights=model_config['weights'],
                datasets=datasets
            )
            all_results.append(results)
        except Exception as e:
            print(f"Error evaluating {model_config['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Calculate and print statistics
    if len(all_results) > 0:
        stats = calculate_statistics(all_results)
        print_statistics(stats, model_configs)
        
        # Save results
        output_file = "frcnn_evaluation_results_iou_based_g9d4.txt"
        save_results_to_file(stats, model_configs, output_file)
        print(f"\nResults saved to: {output_file}")
    else:
        print("No successful evaluations!")


if __name__ == "__main__":
    main()
