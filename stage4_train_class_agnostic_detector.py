#!/usr/bin/env python
import os
import torch
import random
import numpy as np
import argparse
import warnings
warnings.filterwarnings("ignore")
# ---------------------------------------------------------
# Detectron2 and Other Imports
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
import copy

# ---------------------------------------------------------
# 1. Parse Command-Line Arguments
# ---------------------------------------------------------

def get_args():
    thresh=0.004
    mk="mk"
    
    parser = argparse.ArgumentParser(description="Detectron2 Training with FP16 and Custom Paths")
    # parser.add_argument('--annotation_train', type=str, default=f"./pred_annotations/found/train/pseudo_bboxes_{thresh}_{mk}.json",
    #                     help="Path to training annotations (COCO json format)")
    # parser.add_argument('--annotation_val', type=str, default=f"./pred_annotations/found/val/pseudo_bboxes_{thresh}_{mk}.json",
    #                     help="Path to validation annotations (COCO json format)")
    
    # parser.add_argument('--annotation_train', type=str, default=f"./pred_annotations/train/pseudo_bboxes_{thresh}_cut.json",
    #                     help="Path to training annotations (COCO json format)")
    # parser.add_argument('--annotation_val', type=str, default=f"./pred_annotations/val/pseudo_bboxes_{thresh}_cut.json",
    #                     help="Path to validation annotations (COCO json format)")
    
    parser.add_argument('--annotation_train', type=str, default=f"./pred_annotations/full_size/g9/train/pseudo_bboxes_{thresh}_{mk}.json",
                        help="Path to training annotations (COCO json format)")
    parser.add_argument('--annotation_val', type=str, default=f"./pred_annotations/full_size/g9/val/pseudo_bboxes_{thresh}_{mk}.json",
                        help="Path to validation annotations (COCO json format)")
    
    
    parser.add_argument('--train_image_dir', type=str, default="./data/coco20k/images/train",
                        help="Directory with training images")
    parser.add_argument('--val_image_dir', type=str, default="./data/coco20k/images/val",
                        help="Directory with validation images")
    
    # parser.add_argument('--train_image_dir', type=str, default="./data/coco20k/imagesv2/train",
    #                     help="Directory with training images")
    # parser.add_argument('--val_image_dir', type=str, default="./data/coco20k/imagesv2/val",
    #                     help="Directory with validation images")
    
    # parser.add_argument('--output_dir', type=str, default=f"./output_faster_rcnn_u/pseudo_bboxes_{thresh}_{mk}/",
    #                     help="Directory for saving outputs")
    # parser.add_argument('--output_dir', type=str, default=f"./output_faster_rcnn_u/pseudo_bboxes_{thresh}_cut/",
    #                     help="Directory for saving outputs")
    # parser.add_argument('--output_dir', type=str, default=f"./output_faster_rcnn_u/found/pseudo_bboxes_{thresh}_{mk}/",
    #                     help="Directory for saving outputs")
    parser.add_argument('--output_dir', type=str, default=f"./output_faster_rcnn_u/full_size/g9/pseudo_bboxes_{thresh}_{mk}/",
                        help="Directory for saving outputs")
    
    parser.add_argument('--fp16', type=int,default=1,
                        help="Enable FP16 (mixed precision) training")
    parser.add_argument('--seed', type=int, default=99, help="Random seed for reproducibility")
    args = parser.parse_args()
    return args

# ---------------------------------------------------------
# 2. Register the Dataset
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
            "description": "Custom Dataset",
            "url": "",
            "version": "1.0",
            "year": 2025,
            "contributor": "",
            "date_created": "2025/01/01"
        }
    
    # Add 'licenses' field if missing
    if 'licenses' not in data:
        data['licenses'] = []
    
    # Ensure 'categories' exists
    if 'categories' not in data:
        data['categories'] = [{"id": 1, "name": "object", "supercategory": "none"}]
    
    # Save the fixed file
    with open(annotation_path, 'w') as f:
        json.dump(data, f)
    
    print(f"Fixed annotation file: {annotation_path}")

def register_datasets(args):
    # Fix annotation files to ensure they have required COCO fields
    print("Fixing annotation files...")
    fix_coco_annotation_file(args.annotation_train)
    fix_coco_annotation_file(args.annotation_val)
    
    # Clear any previous registrations to avoid stale metadata.
    DatasetCatalog.clear()
    
    # Register the datasets using the provided paths.
    register_coco_instances("custom_dataset_train", {}, args.annotation_train, args.train_image_dir)
    register_coco_instances("custom_dataset_val",   {}, args.annotation_val,   args.val_image_dir)

    # Force the correct image_root into MetadataCatalog.
    MetadataCatalog.get("custom_dataset_train").set(image_root=args.train_image_dir)
    MetadataCatalog.get("custom_dataset_val").set(image_root=args.val_image_dir)

    # (Optional) Print sample file names
    dataset_dicts_train = DatasetCatalog.get("custom_dataset_train")
    dataset_dicts_val   = DatasetCatalog.get("custom_dataset_val")
    print("Train sample file names (before abs path):")
    for d in dataset_dicts_train[:5]:
        print(d["file_name"])
    print("\nVal sample file names (before abs path):")
    for d in dataset_dicts_val[:5]:
        print(d["file_name"])

    # Convert file names to absolute paths.
    for record in dataset_dicts_train:
        record["file_name"] = os.path.abspath(record["file_name"])
    for record in dataset_dicts_val:
        record["file_name"] = os.path.abspath(record["file_name"])

    print("\nTrain sample file names (absolute):")
    for d in dataset_dicts_train[:5]:
        print(d["file_name"])
    print("\nVal sample file names (absolute):")
    for d in dataset_dicts_val[:5]:
        print(d["file_name"])

    print("\nAvailable Datasets:", DatasetCatalog.list())
    print("Train length:", len(dataset_dicts_train))
    print("Val length:", len(dataset_dicts_val))
    return dataset_dicts_train, dataset_dicts_val

# ---------------------------------------------------------
# 3. Create a Custom ROI Heads for Unsupervised Weights
# ---------------------------------------------------------
@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsExtraNorm(Res5ROIHeads):
    """
    Custom ROI head with an extra normalization layer after the res5 stage.
    This modification is inspired by the DINO (unsupervised) pretraining approach.
    """
    def _build_res5_block(self, cfg):
        seq, out_channels = super()._build_res5_block(cfg)
        norm_type = cfg.MODEL.RESNETS.NORM
        norm_layer = get_norm(norm_type, out_channels)
        seq.add_module("norm", norm_layer)
        return seq, out_channels

# ---------------------------------------------------------
# 4. Create a Custom Trainer with a COCO Evaluator
# ---------------------------------------------------------
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, distributed=False, output_dir=output_folder)

# ---------------------------------------------------------
# 5. Set up Config for Unsupervised (DINO) Training
# ---------------------------------------------------------
def setup_cfg(args):
    cfg = get_cfg()
    # Merge configuration from your unsupervised training config file.
    cfg.merge_from_file("./configs/RN50_DINO_FRCNN_COCO20k_CAD.yaml")
    
    # Set the training and testing datasets.
    cfg.DATASETS.TRAIN = ("custom_dataset_train",)
    cfg.DATASETS.TEST  = ("custom_dataset_val",)
    
    # Update the number of classes to match your dataset.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    
    # Load unsupervised (DINO) weights.
    cfg.MODEL.WEIGHTS = "./weights/dino_RN50_pretrain_d2_format.pkl"
    
    # Use the custom ROI head that adds an extra normalization layer.
    cfg.MODEL.ROI_HEADS.NAME = "Res5ROIHeadsExtraNorm"
    
    # Use GPU if available.
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Solver (Hyperparameters)
    cfg.SOLVER.IMS_PER_BATCH = 3
    cfg.SOLVER.MAX_ITER = 24000
    cfg.SOLVER.STEPS = [int(0.6 * cfg.SOLVER.MAX_ITER), int(0.8 * cfg.SOLVER.MAX_ITER)]
    cfg.TEST.EVAL_PERIOD = 5000
    
    # Set output directory from args.
    cfg.OUTPUT_DIR = args.output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    cfg.SOLVER.CHECKPOINT_PERIOD = cfg.TEST.EVAL_PERIOD;
    
    # Enable FP16 training if requested.
    # (AMP stands for Automatic Mixed Precision)
    cfg.SOLVER.AMP.ENABLED = args.fp16
    
    return cfg

def seed_everything(seed=42):
    print("Setting seed to", seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------
# 6. Main: Build Model, Train, and Evaluate
# ---------------------------------------------------------
if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    register_datasets(args)
    cfg = setup_cfg(args)
    print(cfg);
    # Option A: Let the trainer build the model internally.
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=True)
    print(args.__dict__)
    trainer.train()

    # Optionally, run final evaluation on the validation dataset.
    print("Running final evaluation on validation set...")
    evaluator = COCOEvaluator("custom_dataset_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    
    from detectron2.data import build_detection_test_loader
   
    val_loader = build_detection_test_loader(cfg, "custom_dataset_val")
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    print("Final evaluation results:", results)
