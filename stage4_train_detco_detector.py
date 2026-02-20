#!/usr/bin/env python
"""
DetCo-based Class-Agnostic Object Detector Training Script

This script trains a Faster R-CNN detector using DetCo pretrained weights
on the same pseudo-label data used in stage4_train_class_agnostic_detector.py.

Key features:
- Uses DetCo-200ep-AA pretrained weights (contrastive learning backbone)
- Converts DetCo weights to Detectron2 format inline
- Single GPU training with batch size 3, scaled LR (0.00375)
- FREEZE_AT=2 for backbone freezing (stem + res2)
- Class-agnostic detection (NUM_CLASSES=1)

python3 stage4_train_detco_detector.py

Usage:
    python stage4_train_detco_detector.py \
        --annotation_train ./pred_annotations/full_size/g9/train/pseudo_bboxes_0.004_mk.json \
        --annotation_val ./pred_annotations/full_size/g9/val/pseudo_bboxes_0.004_mk.json \
        --output_dir ./output_detco/
"""

import os
import torch
import random
import numpy as np
import argparse
import warnings
import pickle as pkl
import tempfile
from collections import OrderedDict

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Detectron2 Imports
# ---------------------------------------------------------
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.layers import get_norm
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads
import json


# ---------------------------------------------------------
# 1. Parse Command-Line Arguments
# ---------------------------------------------------------
def get_args():
    thresh = 0.004
    mk = "mk"
    
    parser = argparse.ArgumentParser(description="DetCo-based Class-Agnostic Detector Training")
    
    # Annotation paths (same as stage4)
    parser.add_argument('--annotation_train', type=str, 
                        default=f"./pred_annotations/full_size/g9/train/pseudo_bboxes_{thresh}_{mk}.json",
                        help="Path to training annotations (COCO json format)")
    parser.add_argument('--annotation_val', type=str, 
                        default=f"./pred_annotations/full_size/g9/val/pseudo_bboxes_{thresh}_{mk}.json",
                        help="Path to validation annotations (COCO json format)")
    
    # Image directories (same as stage4)
    parser.add_argument('--train_image_dir', type=str, default="./data/coco20k/images/train",
                        help="Directory with training images")
    parser.add_argument('--val_image_dir', type=str, default="./data/coco20k/images/val",
                        help="Directory with validation images")
    
    # Output directory
    parser.add_argument('--output_dir', type=str, 
                        default=f"./output_detco/full_size/g9/pseudo_bboxes_{thresh}_{mk}/",
                        help="Directory for saving outputs")
    
    # DetCo pretrained weights
    parser.add_argument('--detco_weights', type=str, 
                        default="./DetCo/detco_200ep_AA.pth",
                        help="Path to DetCo pretrained weights (.pth)")
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=3,
                        help="Images per batch (default: 3 for single GPU)")
    parser.add_argument('--base_lr', type=float, default=0.00375,
                        help="Base learning rate (scaled: 0.02 * 3/16 = 0.00375)")
    parser.add_argument('--max_iter', type=int, default=24000,
                        help="Maximum training iterations")
    parser.add_argument('--eval_period', type=int, default=5000,
                        help="Evaluation period (iterations)")
    
    # Other options
    parser.add_argument('--fp16', type=int, default=1,
                        help="Enable FP16 (mixed precision) training")
    parser.add_argument('--seed', type=int, default=99, 
                        help="Random seed for reproducibility")
    parser.add_argument('--resume', action='store_true',
                        help="Resume training from last checkpoint")
    parser.add_argument('--freeze_at', type=int, default=2,
                        help="Freeze backbone stages (0=none, 2=stem+res2)")
    
    args = parser.parse_args()
    return args


# ---------------------------------------------------------
# 2. DetCo Weight Conversion
# ---------------------------------------------------------
def convert_detco_weights(input_path, output_path=None):
    """
    Convert DetCo pretrained weights from OpenSelfSup format to Detectron2 format.
    
    This is adapted from DetCo/benchmarks/detection/convert-pretrain-to-detectron2.py
    
    Args:
        input_path: Path to DetCo .pth file
        output_path: Optional path to save .pkl file (if None, uses temp file)
    
    Returns:
        Path to the converted .pkl file
    """
    print(f"Converting DetCo weights from: {input_path}")
    
    # Load the DetCo checkpoint
    obj = torch.load(input_path, map_location="cpu")
    
    # Extract state dict (DetCo saves under "state_dict" key)
    if "state_dict" in obj:
        state_dict = obj["state_dict"]
    elif "model" in obj:
        state_dict = obj["model"]
    else:
        state_dict = obj
    
    newmodel = {}
    for k, v in state_dict.items():
        old_k = k
        
        # Skip momentum encoder keys (MoCo uses encoder_k)
        if "encoder_k" in k or "queue" in k:
            continue
        
        # Remove 'module.' prefix if present
        if k.startswith("module."):
            k = k[7:]
        
        # Remove 'encoder_q.' prefix if present (MoCo encoder)
        if k.startswith("encoder_q."):
            k = k[10:]
        
        # Remove 'backbone.' prefix if present
        if k.startswith("backbone."):
            k = k[9:]
        
        # Convert layer naming: add 'stem.' prefix for non-layer keys
        if "layer" not in k:
            k = "stem." + k
        
        # Convert layer numbering: layer1 -> res2, layer2 -> res3, etc.
        for t in [1, 2, 3, 4]:
            k = k.replace(f"layer{t}", f"res{t + 1}")
        
        # Convert batch norm naming: bn1/2/3 -> conv1/2/3.norm
        for t in [1, 2, 3]:
            k = k.replace(f"bn{t}", f"conv{t}.norm")
        
        # Convert downsample naming
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        
        # Convert tensor to numpy
        if isinstance(v, torch.Tensor):
            v = v.numpy()
        
        newmodel[k] = v
        # print(f"  {old_k} -> {k}")
    
    # Create Detectron2 compatible checkpoint
    res = {
        "model": newmodel,
        "__author__": "DetCo",
        "matching_heuristics": True
    }
    
    # Determine output path
    if output_path is None:
        output_path = input_path.replace('.pth', '_d2_format.pkl')
    
    # Save the converted weights
    with open(output_path, "wb") as f:
        pkl.dump(res, f)
    
    print(f"Converted weights saved to: {output_path}")
    print(f"  Total keys converted: {len(newmodel)}")
    
    return output_path


# ---------------------------------------------------------
# 3. Dataset Registration
# ---------------------------------------------------------
def fix_coco_annotation_file(annotation_path):
    """
    Ensure the COCO annotation file has required fields like 'info', 'licenses', and 'categories'.
    """
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    
    # Add 'info' field if missing
    if 'info' not in data:
        data['info'] = {
            "description": "Custom Dataset for DetCo Training",
            "url": "",
            "version": "1.0",
            "year": 2026,
            "contributor": "",
            "date_created": "2026/01/30"
        }
    
    # Add 'licenses' field if missing
    if 'licenses' not in data:
        data['licenses'] = []
    
    # Ensure 'categories' exists (class-agnostic: single "object" class)
    if 'categories' not in data:
        data['categories'] = [{"id": 1, "name": "object", "supercategory": "none"}]
    
    # Save the fixed file
    with open(annotation_path, 'w') as f:
        json.dump(data, f)
    
    print(f"Fixed annotation file: {annotation_path}")


def register_datasets(args):
    """Register custom datasets with Detectron2."""
    # Fix annotation files to ensure they have required COCO fields
    print("Fixing annotation files...")
    fix_coco_annotation_file(args.annotation_train)
    fix_coco_annotation_file(args.annotation_val)
    
    # Clear any previous registrations to avoid stale metadata
    DatasetCatalog.clear()
    
    # Register the datasets using the provided paths
    register_coco_instances("detco_dataset_train", {}, args.annotation_train, args.train_image_dir)
    register_coco_instances("detco_dataset_val", {}, args.annotation_val, args.val_image_dir)

    # Force the correct image_root into MetadataCatalog
    MetadataCatalog.get("detco_dataset_train").set(image_root=args.train_image_dir)
    MetadataCatalog.get("detco_dataset_val").set(image_root=args.val_image_dir)

    # Get dataset dicts and convert to absolute paths
    dataset_dicts_train = DatasetCatalog.get("detco_dataset_train")
    dataset_dicts_val = DatasetCatalog.get("detco_dataset_val")
    
    # Print sample info
    print("\nTrain sample file names:")
    for d in dataset_dicts_train[:3]:
        print(f"  {d['file_name']}")
    
    # Convert file names to absolute paths
    for record in dataset_dicts_train:
        record["file_name"] = os.path.abspath(record["file_name"])
    for record in dataset_dicts_val:
        record["file_name"] = os.path.abspath(record["file_name"])

    print(f"\nRegistered datasets:")
    print(f"  Train: {len(dataset_dicts_train)} images")
    print(f"  Val: {len(dataset_dicts_val)} images")
    
    return dataset_dicts_train, dataset_dicts_val


# ---------------------------------------------------------
# 4. Custom ROI Heads (same as DetCo/MoCo)
# ---------------------------------------------------------
@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsExtraNorm(Res5ROIHeads):
    """
    As described in the MoCo/DetCo paper, there is an extra BN layer
    following the res5 stage for better transfer learning.
    """
    def _build_res5_block(self, cfg):
        seq, out_channels = super()._build_res5_block(cfg)
        norm = cfg.MODEL.RESNETS.NORM
        norm = get_norm(norm, out_channels)
        seq.add_module("norm", norm)
        return seq, out_channels


# ---------------------------------------------------------
# 5. Custom Trainer with COCO Evaluator
# ---------------------------------------------------------
class DetCoTrainer(DefaultTrainer):
    """Custom trainer that uses COCO evaluator for class-agnostic detection."""
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, distributed=False, output_dir=output_folder)


# ---------------------------------------------------------
# 6. Configuration Setup
# ---------------------------------------------------------
def setup_cfg(args, converted_weights_path):
    """
    Create and configure the Detectron2 config for DetCo-based training.
    
    Key settings:
    - ResNet-50 C4 backbone with SyncBN
    - Faster R-CNN with custom ROI head (extra BN after res5)
    - Class-agnostic detection (NUM_CLASSES=1)
    - FREEZE_AT=2 (freeze stem + res2)
    - Scaled learning rate for batch size 3
    """
    cfg = get_cfg()
    
    # ============ MODEL ============
    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
    
    # Backbone: ResNet-50 with SyncBN
    cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    cfg.MODEL.BACKBONE.FREEZE_AT = args.freeze_at  # Freeze stem + res2
    
    cfg.MODEL.RESNETS.DEPTH = 50
    cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False  # Use stride in 3x3 conv (like torchvision)
    cfg.MODEL.RESNETS.NORM = "SyncBN"
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res4"]  # For C4 architecture
    
    # RPN settings
    cfg.MODEL.RPN.IN_FEATURES = ["res4"]
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    
    # ROI Heads: Custom head with extra BN
    cfg.MODEL.ROI_HEADS.NAME = "Res5ROIHeadsExtraNorm"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Class-agnostic detection
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["res4"]
    
    # ROI Box Head with SyncBN
    cfg.MODEL.ROI_BOX_HEAD.NORM = "SyncBN"
    
    # Pretrained weights (converted DetCo)
    cfg.MODEL.WEIGHTS = converted_weights_path
    
    # Disable mask head
    cfg.MODEL.MASK_ON = False
    
    # Pixel normalization (ImageNet-style, RGB format)
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]
    
    # Device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ============ INPUT ============
    cfg.INPUT.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.FORMAT = "RGB"
    
    # ============ DATASETS ============
    cfg.DATASETS.TRAIN = ("detco_dataset_train",)
    cfg.DATASETS.TEST = ("detco_dataset_val",)
    
    # ============ DATALOADER ============
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    
    # ============ SOLVER ============
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = (int(0.6 * args.max_iter), int(0.8 * args.max_iter))  # LR decay steps
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = 100
    cfg.SOLVER.WARMUP_FACTOR = 0.001
    cfg.SOLVER.CHECKPOINT_PERIOD = args.eval_period
    
    # Optimizer
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.MOMENTUM = 0.9
    
    # Mixed precision training
    cfg.SOLVER.AMP.ENABLED = bool(args.fp16)
    
    # ============ TEST ============
    cfg.TEST.EVAL_PERIOD = args.eval_period
    cfg.TEST.PRECISE_BN.ENABLED = True
    
    # ============ OUTPUT ============
    cfg.OUTPUT_DIR = args.output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg


# ---------------------------------------------------------
# 7. Seed Everything
# ---------------------------------------------------------
def seed_everything(seed=42):
    """Set random seed for reproducibility."""
    print(f"Setting seed to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------
# 8. Main Training Loop
# ---------------------------------------------------------
def main():
    # Parse arguments
    args = get_args()
    
    print("=" * 60)
    print("DetCo-based Class-Agnostic Object Detector Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  DetCo weights: {args.detco_weights}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Base LR: {args.base_lr}")
    print(f"  Max iterations: {args.max_iter}")
    print(f"  Freeze at: {args.freeze_at}")
    print(f"  FP16: {bool(args.fp16)}")
    print(f"  Output dir: {args.output_dir}")
    print()
    
    # Set seed for reproducibility
    seed_everything(args.seed)
    
    # Convert DetCo weights to Detectron2 format
    converted_weights_path = args.detco_weights.replace('.pth', '_d2_format.pkl')
    if not os.path.exists(converted_weights_path):
        converted_weights_path = convert_detco_weights(args.detco_weights, converted_weights_path)
    else:
        print(f"Using existing converted weights: {converted_weights_path}")
    
    # Register datasets
    register_datasets(args)
    
    # Setup configuration
    cfg = setup_cfg(args, converted_weights_path)
    
    print("\n" + "=" * 60)
    print("Model Configuration Summary")
    print("=" * 60)
    print(f"  Architecture: GeneralizedRCNN (Faster R-CNN C4)")
    print(f"  Backbone: ResNet-50 with SyncBN")
    print(f"  ROI Heads: Res5ROIHeadsExtraNorm (with extra BN)")
    print(f"  Num Classes: {cfg.MODEL.ROI_HEADS.NUM_CLASSES} (class-agnostic)")
    print(f"  Freeze At: {cfg.MODEL.BACKBONE.FREEZE_AT}")
    print(f"  Input sizes: {cfg.INPUT.MIN_SIZE_TRAIN}")
    print()
    
    # Create trainer
    trainer = DetCoTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60 + "\n")
    
    # Train
    trainer.train()
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Running Final Evaluation on Validation Set...")
    print("=" * 60 + "\n")
    
    evaluator = COCOEvaluator("detco_dataset_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "detco_dataset_val")
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    
    print("\n" + "=" * 60)
    print("Final Evaluation Results")
    print("=" * 60)
    print(results)
    
    # Save results to file
    results_path = os.path.join(cfg.OUTPUT_DIR, "final_evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    main()
