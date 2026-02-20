import os
import yaml
import random
import argparse
import torch
import numpy as np
import torch.nn as nn;
from torch.amp import autocast  # Updated import for mixed precision
from KFOUND.misc import batch_apply_bilateral_solver;
from KFOUND.KANLinear import KAN_Convolutional_Layer;
import torch.nn.functional as F

# Argument parser for modifying input data, output path, and mode
# python3 ./stage1_background_extraction.py --mode train
# python3 ./stage1_background_extraction.py --mode val

parser = argparse.ArgumentParser(description="Run KFOUND Model on COCO dataset.")
parser.add_argument("--mode", type=str, default="train", choices=["train", "val"],
                    help="Mode for dataset (train/val).")
parser.add_argument("--input_dir", type=str, default="./data/COCO/coco20k/",
                    help="Path to input dataset.")
# parser.add_argument("--output_dir", type=str, default="./data/coco20k_kfound/",
#                     help="Path to save output results.")
parser.add_argument("--output_dir", type=str, default="./data/coco20k_kfound_crf/Conv/",
                    help="Path to save output results.")
parser.add_argument("--weight", type=str, default=None,help="Path to KFOUND weight",required=True);

parser.add_argument("--decoder",type=str,default="Conv",choices=["KAN","Conv","KanConv"]);
parser.add_argument("--g",type=int,default=5);
parser.add_argument("--s",type=str,default="dcrf",choices=["bs","dcrf"]);
parser.add_argument("--d",type=int,default=3);
parser.add_argument('--seed', type=int, default=99, help="Random seed for reproducibility")


args = parser.parse_args();

# Set variables based on arguments
mode = args.mode
root_dir = os.path.join(args.input_dir, mode)
output_dir = os.path.join(args.output_dir, mode)

# Set a fixed seed for reproducibility
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# For CUDA determinism (may impact performance)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"Set seed to {seed}")

from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage
from utils_function.crf import densecrf
import sys

# Ensure KFOUND/ path is correctly added
found_path = os.path.abspath("./KFOUND/")
if os.path.exists(found_path):
    sys.path.append(found_path)
else:
    print(f"Error: Path does not exist -> {found_path}")

print(f"Current Working Directory: {os.getcwd()}")
print(f"KFOUND Path: {found_path}")

from datasets.datasets import build_dataset
from model import KFoundModel

# Set paths and configuration
dataset_name = "COCO20k"
dataset_set = mode
for_eval = True  # True for evaluation, False for training

evaluation_type = "saliency"
config_path = "./KFOUND/configs/kfound_DUTS-TR.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

print("root_dir",root_dir)
print("dataset name",dataset_name)
print("dataset_set",dataset_set)
# Build the dataset
coco_dataset = build_dataset(
    root_dir=root_dir,
    dataset_name=dataset_name,
    dataset_set=dataset_set,
    for_eval=for_eval,
    config=config,
    evaluation_type=evaluation_type,
)

print(f"Loaded {len(coco_dataset)} images from COCO {dataset_set}.")


def IoU(mask1, mask2):
    mask1, mask2 = (mask1 > 0.5).to(torch.bool), (mask2 > 0.5).to(torch.bool)
    intersection = torch.sum(mask1 * (mask1 == mask2), dim=[-1, -2]).squeeze()
    union = torch.sum(mask1 + mask2, dim=[-1, -2]).squeeze()
    return (intersection.to(torch.float) / union).mean().item()

# Ensure CUDA memory is cleared
torch.cuda.empty_cache()

# Initialize the FOUND Model
model = KFoundModel(
    vit_model=config["model"]["pre_training"],
    vit_arch=config["model"]["arch"],
    vit_patch_size=config["model"]["patch_size"],
    enc_type_feats=config["kfound"]["feats"],
    bkg_type_feats=config["kfound"]["feats"],
    bkg_th=config["kfound"]["bkg_th"],
)

if args.decoder == "KAN":
    model.decoder = KAN_Convolutional_Layer(
                            in_channels=384,  # Number of input channels
                            out_channels=1,  # Number of output channels to match 1x1 convolution
                            grid_size=args.g,
                            spline_order=args.d,
                            scale_noise=0.1,
                            scale_base=1.0,
                            scale_spline=1.0,
                            base_activation=nn.SiLU,
                            grid_eps=0.02,
                            grid_range=[-1, 1]
                        )

elif args.decoder == "Conv" : 
    model.decoder = nn.Conv2d(384, 1, (1, 1))

# Convert model to float16 for efficiency
# model.half()


# Load Decoder Weights

decoder_weight = args.weight

model.decoder_load_weights(decoder_weight)
model.eval()
model.cuda()

print(f"Loaded decoder weights: {decoder_weight}")

# Create output directories
os.makedirs(output_dir, exist_ok=True);
if args.decoder == "Conv":
    mask_crf_dir = os.path.join(output_dir, "predicted_masks_bs")
elif args.decoder == "KAN":
    mask_crf_dir = os.path.join(output_dir, "predicted_masks_crf")
#mask_bs_dir = os.path.join(output_dir, "predicted_masks_bs");

#os.makedirs(mask_dir, exist_ok=True)
os.makedirs(mask_crf_dir, exist_ok=True)
#os.makedirs(mask_bs_dir, exist_ok=True)

# DataLoader for the dataset
batch_size = 1
dataloader = torch.utils.data.DataLoader(
    dataset=coco_dataset,
    batch_size=batch_size,
    shuffle=False,
)

# Prediction Loop
sigmoid = torch.nn.Sigmoid()
show = 0

for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
    inputs, img_init, gt_labels, img_paths = data

    #inputs = inputs.cuda()
    inputs = inputs.cuda().half()
    with torch.no_grad():
        with autocast('cuda'):
            preds, _, _, _ = model.forward_step(inputs, for_eval=True)
    
    # Convert predictions back to float32 for safe processing
    preds_up = F.interpolate(
        preds.float(),
        size=img_init.shape[2:],
        mode="bicubic",
        align_corners=False,
    )

    #BS
    if args.s == "bs":
        preds_mask = (sigmoid(preds_up.detach()) > 0.5).float()
        preds_mask_bs, _ = batch_apply_bilateral_solver(
                                        data,
                                        preds_mask.detach()
                                    );
        
        # Save the pseudo mask
        img_name = os.path.basename(img_paths[0])
        pred_mask_crf_path = os.path.join(mask_crf_dir, img_name.replace(".jpg", ".png"))
        plt.imsave(pred_mask_crf_path, preds_mask_bs.cpu().numpy().astype(np.uint8), cmap="gray")

    
    
    # denseCRF
    elif args.s == "dcrf":
        preds_mask = (sigmoid(preds_up)).float().cpu().squeeze().numpy()
        img_init_uint8 = (img_init.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pseudo_mask_crf = densecrf(img_init_uint8, preds_mask)
        pseudo_mask = ndimage.binary_fill_holes(pseudo_mask_crf >= 0.5)

        # Compare the original mask and CRF mask
        # mask1 = torch.from_numpy(preds_mask).half()
        # mask2 = torch.from_numpy(pseudo_mask.astype(float)).half()
        
        # mask1 = torch.from_numpy(preds_mask)
        # mask2 = torch.from_numpy(pseudo_mask.astype(float))

        # if torch.cuda.is_available():
        #     mask1 = mask1.cuda()
        #     mask2 = mask2.cuda()

        # if IoU(mask1.float(), mask2.float()) < 0.5:
        #     pseudo_mask = ~pseudo_mask  

        # pseudo_mask = pseudo_mask.astype(bool)
        # pseudo_mask_uint8 = (pseudo_mask * 255).astype(np.uint8)
        # pseudo_mask = Image.fromarray(pseudo_mask_uint8)

        # # Resize and binarize
        # pseudo_mask = np.asarray(pseudo_mask.resize((img_init.shape[-1], img_init.shape[-2]), Image.BICUBIC))
        # pseudo_mask = (pseudo_mask > 127).astype(np.uint8) * 255
    
        pseudo_mask[pseudo_mask < 0] = 0
        pseudo_mask = Image.fromarray(np.uint8(pseudo_mask*255))
        pseudo_mask = np.asarray(pseudo_mask.resize((img_init.shape[-1], img_init.shape[-2])))

        pseudo_mask = pseudo_mask.astype(np.uint8)
        upper = np.max(pseudo_mask)
        lower = np.min(pseudo_mask)
        thresh = upper / 2.0
        pseudo_mask[pseudo_mask > thresh] = upper
        pseudo_mask[pseudo_mask <= thresh] = lower

        # Save the pseudo mask
        img_name = os.path.basename(img_paths[0])
        #pred_mask_path = os.path.join(mask_dir, img_name.replace(".jpg", ".png"))
        pred_mask_crf_path = os.path.join(mask_crf_dir, img_name.replace(".jpg", ".png"))
        #pred_mask_bs_path = os.path.join(mask_bs_dir, img_name.replace(".jpg", ".png"))
        
        #plt.imsave(pred_mask_path, preds_mask, cmap="gray")
        plt.imsave(pred_mask_crf_path, pseudo_mask_crf, cmap="gray")
        #plt.imsave(pred_mask_bs_path, preds_mask_bs.cpu().detach(), cmap="gray")


print(f"Processed all images. Results saved in {output_dir}.")
