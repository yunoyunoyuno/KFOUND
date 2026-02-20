#!/usr/bin/env python
import os
import argparse
import random
import numpy as np
import cv2
import torch
from tqdm import tqdm
from PIL import Image
from scipy import ndimage

# Import your required modules
from maskcut import maskcut, densecrf  # Assuming maskcut.py contains required functions
import dino_ as dino               # Assuming DINO implementation
from utils_function.metric import IoU;

# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 0.003 --upper-threshold 1.01 --mode train --use_maskcut 1
# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 0.003 --upper-threshold 1.01 --mode val --use_maskcut 1

# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 0.005 --upper-threshold 1.01 --mode train --use_maskcut 1
# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 0.005 --upper-threshold 1.01 --mode val --use_maskcut 1

#   
# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 0.004 --upper-threshold 1.01 --mode val --use_maskcut 1
# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 0.004 --upper-threshold 1.01 --mode train --use_maskcut 0
# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 0.004 --upper-threshold 1.01 --mode val --use_maskcut 0 

# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 0.004 --mode train --use_maskcut 0
# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 0.004 --mode val --use_maskcut 0

# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 0.003 --mode train --use_maskcut 1
# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 0.003 --mode val --use_maskcut 1

# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 0.003 --mode train --use_maskcut 0
# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 0.003 --mode val --use_maskcut 0

# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 1 --mode train --use_maskcut 1
# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 1 --mode val --use_maskcut 1

# Remove the problematics label
# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 0.004 --upper-threshold 1.01 --mode train --use_maskcut 1 --labels_output_dir data/coco20k_kfound/train/pseudo_bboxes_0.004_cut --verbose
# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 0.004 --upper-threshold 1.01 --mode val --use_maskcut 1 --labels_output_dir data/coco20k_kfound/val/pseudo_bboxes_0.004_cut --verbose


# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 0.004 --upper-threshold 1.01 --mode train --use_maskcut 1 --labels_output_dir ./data/coco20k_found/ --verbose --mask_folder ./data/coco20k_found/
# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 0.004 --upper-threshold 1.01 --mode val --use_maskcut 1 --labels_output_dir ./data/coco20k_found/ --verbose --mask_folder ./data/coco20k_found/

# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 0.004 --upper-threshold 1.01 --mode train --use_maskcut 1 --labels_output_dir  ./data/coco20k_kfound_crf/Conv/ --verbose --mask_folder ./data/coco20k_kfound_crf/Conv/
# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 0.004 --upper-threshold 1.01 --mode val --use_maskcut 1 --labels_output_dir ./data/coco20k_kfound_crf/Conv/ --verbose --mask_folder ./data/coco20k_kfound_crf/Conv/

# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 0.004 --upper-threshold 1.01 --mode train --use_maskcut 1 --labels_output_dir  ./data/coco20k_kfound_crf/g5/ --verbose --mask_folder ./data/coco20k_kfound_crf/g5/
# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 0.004 --upper-threshold 1.01 --mode val --use_maskcut 1 --labels_output_dir  ./data/coco20k_kfound_crf/g5/ --verbose --mask_folder ./data/coco20k_kfound_crf/g5/

# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 0.004 --upper-threshold 1.01 --mode train --use_maskcut 1 --labels_output_dir  ./data/coco20k_kfound_crf/g7/ --verbose --mask_folder ./data/coco20k_kfound_crf/g7/
# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 0.004 --upper-threshold 1.01 --mode val --use_maskcut 1 --labels_output_dir  ./data/coco20k_kfound_crf/g7/ --verbose --mask_folder ./data/coco20k_kfound_crf/g7/

# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 0.004 --upper-threshold 1.01 --mode train --use_maskcut 1 --labels_output_dir  ./data/coco20k_kfound_crf/g9/ --verbose --mask_folder ./data/coco20k_kfound_crf/g9/
# python3 ./stage2_3_create_pseudo_bboxes.py --lower-threshold 0.004 --upper-threshold 1.01 --mode val --use_maskcut 1 --labels_output_dir  ./data/coco20k_kfound_crf/g9/ --verbose --mask_folder ./data/coco20k_kfound_crf/g9/

def process_image(image, 
                  mask_path, 
                  image_file,
                  backbone,
                  class_id,
                  normalized_threshold, 
                  large_box_threshold, 
                  maskcut_N,
                  n_components,
                  use_maskcut_flag,
                  bad_labels, 
                  idx=-1):
    height, width = image.shape[:2]

    # Load and process the mask image
    mask = cv2.imread(mask_path, 0)
    if mask is None:
        raise ValueError("GG no mask",mask_path);

    original_bboxes = []
    computed_need_maskcut = False;
    
    if len(np.unique(mask)) == 1:
        computed_need_maskcut = True;
        bad_labels.append(idx);
    else:
        _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, n_components)

        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            if (w * h) / (height * width) >= normalized_threshold:
                original_bboxes.append((class_id,
                    (x + w / 2) / width,
                    (y + h / 2) / height,
                    w / width,
                    h / height
                ))
                
        if len(original_bboxes) == 0:
            normalized_threshold = 0.001
            for i in range(1, num_labels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                if (w * h) / (height * width) > normalized_threshold:
                    original_bboxes.append((class_id,
                        (x + w / 2) / width,
                        (y + h / 2) / height,
                        w / width,
                        h / height
                    ))
            if len(original_bboxes) == 0:
                normalized_threshold = 0.0001
                for i in range(1, num_labels):
                    x = stats[i, cv2.CC_STAT_LEFT]
                    y = stats[i, cv2.CC_STAT_TOP]
                    w = stats[i, cv2.CC_STAT_WIDTH]
                    h = stats[i, cv2.CC_STAT_HEIGHT]
                    if (w * h) / (height * width) > normalized_threshold:
                        original_bboxes.append((class_id,
                            (x + w / 2) / width,
                            (y + h / 2) / height,
                            w / width,
                            h / height
                        ))
                if len(original_bboxes) == 0:
                    computed_need_maskcut = True
                    bad_labels.append(idx)

    final_bboxes = original_bboxes.copy()

    # Apply MaskCut if needed and enabled via command-line flag
    if computed_need_maskcut and use_maskcut_flag:
        #return computed_need_maskcut,[(class_id, 0.5, 0.5, 1.0, 1.0)]
        # The "cpu" parameter is set based on whether CUDA is available.
        bipartitions, _, I_new = maskcut(
            image_file,
            backbone,
            patch_size=8,
            tau=0.15,
            N=maskcut_N,
            fixed_size=480,
            cpu=(not torch.cuda.is_available())
        )
        maskcut_bboxes = []
        for bipartition in bipartitions:
            # Refine the mask with dense CRF and fill holes
            pseudo_mask = densecrf(np.array(I_new), bipartition)
            pseudo_mask = ndimage.binary_fill_holes(pseudo_mask >= 0.5)
            
            # mask1 = torch.from_numpy(bipartition).cuda()
            # mask2 = torch.from_numpy(pseudo_mask).cuda()
            
            # if IoU(mask1, mask2) < 0.5:
            #     pseudo_mask = pseudo_mask * -1

            pseudo_mask[pseudo_mask < 0] = 0
            pseudo_mask = Image.fromarray(np.uint8(pseudo_mask*255))
            pseudo_mask = np.asarray(pseudo_mask.resize((width, height)))

            pseudo_mask = pseudo_mask.astype(np.uint8)
            upper = np.max(pseudo_mask)
            lower = np.min(pseudo_mask)
            thresh = upper / 2.0
            pseudo_mask[pseudo_mask > thresh] = upper
            pseudo_mask[pseudo_mask <= thresh] = lower
            
            # # Resize the mask to original image dimensions
            # final_mask = Image.fromarray((pseudo_mask.a stype(np.uint8) * 255))
            # final_mask = np.array(final_mask.resize((width, height))) > 128

            # Compute bounding box from the refined mask
            rows = np.any(pseudo_mask > 0, axis=1)
            cols = np.any(pseudo_mask > 0, axis=0)
            try:
                y1, y2 = np.where(rows)[0][[0, -1]]
                x1, x2 = np.where(cols)[0][[0, -1]]
            except IndexError:
                continue  # Skip if mask is invalid

            box_w = (x2 - x1) / width
            box_h = (y2 - y1) / height

            maskcut_bboxes.append((
                class_id,
                (x1 + x2) / (2 * width),
                (y1 + y2) / (2 * height),
                box_w,
                box_h
            ))

        # Replace original boxes with maskcut boxes or use fallback if none valid
        final_bboxes = maskcut_bboxes if maskcut_bboxes else [(class_id, 0.5, 0.5, 1.0, 1.0)]
    elif computed_need_maskcut and not use_maskcut_flag:
        # If maskcut is indicated but disabled, use original boxes or fallback default.
        if not final_bboxes:
            final_bboxes = [(class_id, 0.5, 0.5, 1.0, 1.0)]

    return computed_need_maskcut, final_bboxes


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


def main():
    parser = argparse.ArgumentParser(description="Process images with optional MaskCut refinement.")
    parser.add_argument("--mode", type=str, default="train", help="Training or test mode",choices=["train","val"])
    parser.add_argument("--mask_folder", type=str,
                        default="./data/coco20k_kfound/",
                        help="Folder containing mask images")
    parser.add_argument("--image_path", type=str,
                        default="./data/coco20k/images/",
                        help="Path to the original images")
    parser.add_argument("--labels_output_dir", type=str,
                        default="./data/coco20k_kfound/",
                        help="Directory to save output label files")
    parser.add_argument("--use_maskcut", 
                        type=int,
                        help="Enable MaskCut processing if set",
                        default=1)
    parser.add_argument("--seed", type=int, default=99, help="Random seed for reproducibility");
    parser.add_argument("--lower-threshold",type=float,default=0.004,help="Normalized threshold for small bboxes");
    parser.add_argument("--upper-threshold",type=float,default=1.0,help="Upper threshold for a large bbox");
    parser.add_argument("--N",type=int,default=3,help="# Maskcut component");
    parser.add_argument("--n_components",type=int,default=8,choices=[4,8],help="# connected component");
    parser.add_argument("--verbose",action="store_true",default=True);
    
    
    args = parser.parse_args()
    if args.verbose:
        print(args.__dict__)
    assert args.mode in ["train","val"], "Only 'train' or 'val' mode"
    image_path = os.path.join(args.image_path ,args.mode);

    mask_folder = os.path.join(args.mask_folder,args.mode,"predicted_masks_crf")
    if "bs" in args.mask_folder:
        mask_folder = os.path.join(args.mask_folder,args.mode,"predicted_masks_bs")
    
    seed_everything(99)

    # Create output directory if it doesn't exist
    os.makedirs(args.labels_output_dir, exist_ok=True)

    # Initialize the DINO backbone
    backbone = dino.ViTFeat(
        "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
        768, 'base', 'k', 8
    )
    backbone.eval()
    if torch.cuda.is_available():
        backbone.cuda()

    # Set additional parameters
    class_id = 0
    normalized_threshold = args.lower_threshold;
    large_box_threshold = args.upper_threshold;
    maskcut_N = args.N;
    bad_labels = []

    mask_files = sorted(os.listdir(mask_folder))
    for idx, file_name in tqdm(enumerate(mask_files),total=len(mask_files)):
        
        mask_path = os.path.join(mask_folder , file_name)
        base_name = os.path.splitext(file_name)[0]
        image_file = os.path.join(image_path, f"{base_name}.jpg")

        image = cv2.imread(image_file)
        assert image is not None, f"{image_file} is not exist"

        computed_need_maskcut, bboxes = process_image(
            image, 
            mask_path,
            image_file, 
            backbone,
            class_id,
            normalized_threshold, 
            large_box_threshold,
            maskcut_N,          # now 8th parameter
            args.n_components,  # now 9th parameter
            args.use_maskcut,
            bad_labels, 
            idx=idx
        )

        # Save bounding boxes to the output label file
        #print((args.labels_output_dir if args.labels_output_dir[-1] != "/" else args.labels_output_dir[:-1]));
        #print(f"/{args.mode}",f"/pseudo_bboxes_{args.lower_threshold}",('mk' if args.use_maskcut else ''))
        
        output_based_path = str((args.labels_output_dir if args.labels_output_dir[-1] != "/" else args.labels_output_dir[:-1])) + \
            f"/{args.mode}/pseudo_bboxes_{args.lower_threshold}_{('mk' if args.use_maskcut else '')}";
        
        #output_based_path = str(args.labels_output_dir)
        os.makedirs(output_based_path,exist_ok=True);
        output_path = os.path.join(output_based_path, 
                                   f"{base_name}.txt");
        
        #if not computed_need_maskcut:
        with open(output_path, "w") as f:
            for box in bboxes:
                line = f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n"
                f.write(line)

    if args.verbose:
        print(f"Process completed. Bad labels count: {len(bad_labels)}");
        print("Which are\n",bad_labels)
        print(args.__dict__)
    

if __name__ == '__main__':
    main()
