#!/usr/bin/env python
import os
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import load_coco_json
from tqdm import tqdm
import torch
import random, numpy as np
import argparse;
# ---------------------------------------------------------
# Import for the custom ROI head (used during training)
# ---------------------------------------------------------
from detectron2.layers import get_norm
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads


def custom_inference_with_fp16_and_tqdm(model, data_loader, evaluator):
    model.eval()
    evaluator.reset()
    with torch.no_grad():
        for inputs in tqdm(data_loader, desc="Evaluating"):
            # Use AMP autocast to enable mixed-precision
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
            evaluator.process(inputs, outputs)
    return evaluator.evaluate()

def seed_everything(seed=42):
    print("set seed to",seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (at some potential performance cost):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsExtraNorm(Res5ROIHeads):
    """
    Custom ROI head with an extra normalization layer after the res5 stage.
    This modification is used during unsupervised (DINO) training.
    """
    def _build_res5_block(self, cfg):
        seq, out_channels = super()._build_res5_block(cfg)
        norm_type = cfg.MODEL.RESNETS.NORM
        norm_layer = get_norm(norm_type, out_channels)
        seq.add_module("norm", norm_layer)
        return seq, out_channels

# ---------------------------------------------------------
# Custom Evaluator (extends COCOEvaluator to report AR metrics)
# ---------------------------------------------------------
class CustomCOCOEvaluator(COCOEvaluator):
    """
    Extends COCOEvaluator to include Average Recall (AR) metrics.
    """
    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        # Get standard AP metrics (keys: AP, AP50, AP75, APs, APm, APl)
        results = super()._derive_coco_results(coco_eval, iou_type, class_names)
        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return results

        results["AR1"]   = coco_eval.stats[6]
        results["AR10"]  = coco_eval.stats[7]
        results["AR100"] = coco_eval.stats[8]
        results["ARs"]   = coco_eval.stats[9]
        results["ARm"]   = coco_eval.stats[10]
        results["ARl"]   = coco_eval.stats[11]
        return results

# ---------------------------------------------------------
# Class-Agnostic Dataset Registration Function
# ---------------------------------------------------------
def register_class_agnostic_dataset(name, json_file, image_root):
    """
    Register a COCO-format dataset for evaluation while converting all
    category IDs to 0 (for class-agnostic detection).
    """
    def wrapper():
        temp_name = "temp_" + name
        print('json_file',json_file);
        print('image_root',image_root);
        dataset = load_coco_json(json_file, image_root, temp_name)
        # Convert every annotation's category_id to 0.
        for item in dataset:
            for anno in item["annotations"]:
                anno["category_id"] = 0
        return dataset

    # Remove previous registrations if they exist.
    for ds_name in [name, "temp_" + name]:
        if ds_name in DatasetCatalog.list():
            DatasetCatalog.remove(ds_name)
        if ds_name in MetadataCatalog.list():
            MetadataCatalog.remove(ds_name)

    DatasetCatalog.register(name, wrapper)
    metadata = MetadataCatalog.get(name)
    metadata.set(image_root=image_root, thing_classes=["object"], evaluator_type="coco")
    return metadata

def main():
    # --------------------- CONFIGURATION ---------------------
    parser = argparse.ArgumentParser("Faster-RCNN on COCO");
    
    parser.add_argument("--weight",
                      type=str,
                      required=True,
                      default="./output_faster_rcnn_u/small_size/KanG7D5/pseudo_bboxes_0.004_mk/model_final.pth");
    parser.add_argument("--testset",
                      type=str,
                      required=True,
                      default="val");
    
    args = parser.parse_args();
    
    DATASET_NAME = "class_agnostic_test";
    
    MODEL_WEIGHTS = args.weight;
    # b2 44999, b3 64999
    if args.testset == "val":
        TEST_IMAGES  = "./data/coco20k/images/val/"
        TEST_JSON    = "./data/coco20k/annotations/instances_val2017.json"
        output       = "output_evaluation/cocoval2017"
    elif args.testset == "train":
        TEST_IMAGES  = "./data/coco20k/images/train/"   
        TEST_JSON    = "./data/coco20k/annotations/coco20k.json" 
        output= "output_evaluation/coco20k"

    register_class_agnostic_dataset(DATASET_NAME, TEST_JSON, TEST_IMAGES)

    cfg = get_cfg()
    cfg.merge_from_file("./configs/RN50_DINO_FRCNN_COCO20k_CAD.yaml")
    cfg.DATASETS.TEST = (DATASET_NAME,)
    cfg.MODEL.ROI_HEADS.NAME = "Res5ROIHeadsExtraNorm"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = MODEL_WEIGHTS
    cfg.OUTPUT_DIR = output
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    predictor = DefaultPredictor(cfg)
    evaluator = CustomCOCOEvaluator(DATASET_NAME, cfg, False, output_dir=cfg.OUTPUT_DIR)
    test_loader = build_detection_test_loader(cfg, DATASET_NAME)

    print("\nStarting class-agnostic evaluation with the new model...")

    # --------------------- INFERENCE (with progress + FP16) ---------------------
    results = custom_inference_with_fp16_and_tqdm(predictor.model, test_loader, evaluator)

    # --------------------- PRINT RESULTS ---------------------
    print("\n================== Evaluation Metrics ==================")
    print(f"{'AP (IoU=0.50:0.95):':<25} {results['bbox']['AP']:.7f}")
    print(f"{'AP50:':<25} {results['bbox']['AP50']:.7f}")
    print(f"{'AP75:':<25} {results['bbox']['AP75']:.7f}")
    print(f"{'AP Small:':<25} {results['bbox']['APs']:.7f}")
    print(f"{'AP Medium:':<25} {results['bbox']['APm']:.7f}")
    print(f"{'AP Large:':<25} {results['bbox']['APl']:.7f}")

    print("\n================== Average Recall (AR) ==================")
    print(f"{'AR@1:':<25} {results['bbox']['AR1']*100:.7f}")
    print(f"{'AR@10:':<25} {results['bbox']['AR10']*100:.7f}")
    print(f"{'AR@100:':<25} {results['bbox']['AR100']*100:.7f}")
    print(f"{'AR Small:':<25} {results['bbox']['ARs']*100:.7f}")
    print(f"{'AR Medium:':<25} {results['bbox']['ARm']*100:.7f}")
    print(f"{'AR Large:':<25} {results['bbox']['ARl']*100:.7f}")

if __name__ == "__main__":
    seed_everything(99)
    main()