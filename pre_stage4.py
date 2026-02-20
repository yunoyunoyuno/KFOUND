#!/usr/bin/env python
import os
import json
import cv2
from tqdm import tqdm;

import argparse;



def main():
    # Set mode ("train" or "val")
    parser = argparse.ArgumentParser("Prestage 4");
    parser.add_argument("--mode",type=str,default="train",choices=["train","val"]);
    parser.add_argument("--type",type=str,default="mk",choices=["","mk","cut"]);
    parser.add_argument("--t",type=float,default=0.004);
    parser.add_argument("--label_basename",type=str,required=True);
    parser.add_argument("--output_path",type=str,required=True);
    
    args = parser.parse_args();
    
    mode = args.mode;
    thresh= args.t;
    t=args.type;
    
    
    # Set annotation file based on mode
    annotation_file = f"./data/coco20k/annotations/instances_{mode}{(2014 if mode == 'train' else 2017)}.json"
    
    # Set directories for images and YOLO annotations
    image_dir = f"./data/coco20k/images/{mode}";
    # yolo_annotation_path = f"./data/coco20k_kfound/{mode}/pseudo_bboxes_0.003_/"
    # yolo_annotation_path = f"./data/coco20k_kfound/{mode}/pseudo_bboxes_{thresh}_{mk}/"
    # yolo_annotation_path = f"./data/coco20k_found/{mode}/pseudo_bboxes_{thresh}_{mk}/"
    #yolo_annotation_path = f"./data/coco20k_kfound_crf/Conv/{mode}/pseudo_bboxes_{thresh}_{mk}/"
    #yolo_annotation_path = f"./data/coco20k_kfound/{mode}/pseudo_bboxes_{thresh}_cut/"
    
    yolo_annotation_path = os.path.join(args.label_basename,mode,f"pseudo_bboxes_{thresh}_{t}");
    
    # Set output annotations path and file name
    #output_annotations_path = f"./pred_annotations/{mode}"
    # output_annotations_path = f"./pred_annotations/found/{mode}"
    #output_annotations_path = f"./pred_annotations/full_size/conv/{mode}"
    output_annotations_path = os.path.join(args.output_path,mode);
    #output_annotations_path = f"./pred_annotations/full_size/g9/{mode}"
    # Extract a name for the new annotation file from the YOLO annotation path
    new_ann_files = yolo_annotation_path.strip().split('/')[-2]
    out_annotation_file = os.path.join(output_annotations_path, new_ann_files + '.json')

    
    # Ensure output directory exists
    os.makedirs(output_annotations_path, exist_ok=True)
    
    # Load original COCO dataset annotations
    with open(annotation_file, "r") as f:
        coco_data = json.load(f)
    
    # Create a mapping from file name to image ID from COCO data
    image_id_map = {img["file_name"]: img["id"] for img in coco_data["images"]}
    
    # Initialize COCO-style annotations dictionary
    coco_annotations = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}]  # Single class dataset
    }
    
    # Track annotation IDs (unique per annotation)
    annotation_id = 0
    
    # Get all image filenames in the dataset directory
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
    if not image_files:
        raise FileNotFoundError(f"No readable image formats (jpg, png) found in {image_dir}")
    
    # Process each image in the directory
    for image_filename in tqdm(image_files, desc="Processing Images",total=len(image_files)):
        # Construct corresponding annotation file name (assumes .txt extension)
        annotation_filename = image_filename[:-4] + ".txt"
        annotation_path = os.path.join(yolo_annotation_path, annotation_filename)
        image_path = os.path.join(image_dir, image_filename)
    
        # Skip if YOLO annotation does not exist
        if not os.path.exists(annotation_path):
            print(f"Skipping {image_filename}: No annotation found");
            continue;
            #raise ValueError(f"{image_filename}: No annotation found");
    
        # Ensure the image exists in the original COCO dataset
        if image_filename not in image_id_map:
            raise ValueError(f"{image_filename}: Not found in original COCO dataset")
    
        # Retrieve the image ID from the COCO mapping
        image_id = image_id_map[image_filename]
    
        # Load image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            ValueError(f"{image_filename}: Unable to read image")
        height, width, _ = image.shape
    
        # Append image metadata
        coco_annotations["images"].append({
            "id": image_id,
            "file_name": image_filename,
            "width": width,
            "height": height
        })
    
        # Read YOLO annotation file
        with open(annotation_path, "r") as f:
            lines = f.readlines()
    
        # Process each bounding box in the YOLO annotation file
        for line in lines:
            values = line.strip().split()
            # Assuming YOLO format: <class> <x_center> <y_center> <box_width> <box_height>
            # For single-class, we fix class_id to 1
            x_center, y_center, box_width, box_height = map(float, values[1:])
    
            # Convert YOLO (normalized) format to COCO (absolute) format
            x_min = (x_center - box_width / 2) * width
            y_min = (y_center - box_height / 2) * height
            box_width_abs = box_width * width
            box_height_abs = box_height * height
    
            # Ensure bounding box values are within valid limits
            x_min, y_min = max(0, x_min), max(0, y_min)
            box_width_abs, box_height_abs = max(1, box_width_abs), max(1, box_height_abs)
    
            # Append converted annotation
            annotation_id += 1
            coco_annotations["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # Single class
                "bbox": [x_min, y_min, box_width_abs, box_height_abs],
                "area": box_width_abs * box_height_abs,
                "iscrowd": 0
            })
    
    # Save the COCO-format annotations to the output file
    with open(out_annotation_file, "w") as f:
        json.dump(coco_annotations, f, indent=4)
    
    print(f"Annotation file saved as {out_annotation_file}")

if __name__ == "__main__":
    main()
