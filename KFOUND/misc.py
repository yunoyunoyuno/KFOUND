import re
import os
import cv2
import yaml
import math
import random
import scipy.ndimage
import numpy as np

import torch
import torch.nn.functional as F

from typing import List
from torchvision import transforms as T

from bilateral_solver import bilateral_solver_output


loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

class Struct:
    def __init__(self, **entries): 
        self.__dict__.update(entries)

def load_config(config_file):
    with open(config_file, errors='ignore') as f:
        # conf = yaml.safe_load(f)  # load config
        conf = yaml.load(f, Loader=loader)
    print('hyperparameters: ' + ', '.join(f'{k}={v}' for k, v in conf.items()))
        
    #TODO yaml_save(save_dir / 'config.yaml', conf)
    return Struct(**conf)

def set_seed(seed: int) -> None:
    """
    Set all seeds to make results reproducible
    """
    # env
    os.environ["PYTHONHASHSEED"] = str(seed)

    # python
    random.seed(seed)

    # numpy
    np.random.seed(seed)

    # torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    print(f"setted all seeds to {seed}")

def IoU(mask1, mask2):
    """
    Code adapted from TokenCut: https://github.com/YangtaoWANG95/TokenCut
    """
    mask1, mask2 = (mask1 > 0.5).to(torch.bool), (mask2 > 0.5).to(torch.bool)
    intersection = torch.sum(mask1 * (mask1 == mask2), dim=[-1, -2]).squeeze()
    union = torch.sum(mask1 + mask2, dim=[-1, -2]).squeeze()
    return (intersection.to(torch.float) / union).mean().item()

from typing import Tuple
import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from scipy import ndimage
from PIL import Image;

MAX_ITER = 10
POS_W = 7 
POS_XY_STD = 3
Bi_W = 10
Bi_XY_STD = 50 
Bi_RGB_STD = 5

# CRFs are a type of probabilistic graphical model used for structured prediction. In image processing, they are often employed 
# for tasks like image segmentation, where the goal is to assign a label to each pixel in an image in a way that takes into account 
# both the individual pixel's information and the contextual information from neighboring pixels.
# DenseCRF is a specific type of CRF that utilizes fully connected pairwise potentials. 
# This means that every pixel in the image can potentially interact with every other pixel,
# allowing for more detailed and refined segmentation results.

#DenseCRF considers both spatial (geometric) and appearance (color) information to improve the segmentation accuracy.

def densecrf(image : np.array, mask : np.array):
    # image is a 2D numpy array representing the input image.
    # mask is a binary mask indicating the foreground and background.
    h, w = mask.shape
    mask = mask.reshape(1, h, w); # (1, h, w)
    fg = mask.astype(float); # # (1, h, w)
    bg = 1 - fg; # (1, h, w)
    output_logits = torch.from_numpy(np.concatenate((bg,fg), axis=0)) # (1,h,w) cat (1,h,w) -> (2,h,w)

    H, W = image.shape[:2]
    image = np.ascontiguousarray(image); # (H, W, 3)
    
    output_logits: torch.Tensor = F.interpolate(output_logits.unsqueeze(0), size=(H, W), mode="bilinear").squeeze(); #(2, H, W)
    output_probs = F.softmax(output_logits, dim=0).cpu().numpy(); # (2, H, W)
    
    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs); # numpy : (c, H * W)
    U = np.ascontiguousarray(U); # (c, H * W)

    d = dcrf.DenseCRF2D(w, h, c);
    """  Unary potentials: represent the cost of assigning each pixel to a particular label (foreground or background).
                           These are derived from the softmax probabilities.
    """
    # Convert probabilities to unary potentials (energy values for each pixel label).
    d.setUnaryEnergy(U);
    """ Gaussian Potentials (pairwise potentials): 
        Consider the spatial closeness of pixels. Pixels that are closer together are more likely to have the same label.
    """
    # Consider the spatial closeness of pixels. Pixels that are closer together are more likely to have the same label.
    # sxy (standard deviation in XY plane): Controls the spatial proximity influence.
    # compat (compatibility): Defines the penalty for different labels.
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W);
    """ Bilateral Potentials (pairwise potentials): Consider both spatial closeness and color similarity. 
                                                    This helps to preserve object boundaries where there are color differences.
    """
    # sxy (spatial proximity standard deviation).
    # srgb (color similarity standard deviation).
    # compat (compatibility).
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)

    # Inference is run for a specified number of iterations (MAX_ITER), resulting in the most probable label for each pixel.
    Q: list = d.inference(MAX_ITER) # len = H * W * c
    Q: np.array = np.array(Q).reshape((c, h, w)) # (c, H, W)
    # The final segmentation map (MAP) is generated by taking the label with the highest probability for each pixel after inference.
    MAP: np.array = np.argmax(Q, axis=0).reshape((h,w)).astype(np.float32);# (H, W)
    return MAP #[H,W]

def batch_apply_dense_crf(
    data: Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], torch.Tensor, str],
    mask_preds: torch.Tensor,
    get_all_cc: bool = True
):
    inputs, init_imgs, gt_labels, img_path = data
    masks_crf = []

    for i in range(inputs.shape[0]):
        mask_pred = mask_preds[i]
        init_img = init_imgs[i]

        # Convert logits to probability map
        preds_mask = F.sigmoid(mask_pred).float().cpu().squeeze().numpy()
        img_init_uint8 = (init_img.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Apply DenseCRF
        mask_crf = densecrf(img_init_uint8, preds_mask)
        pseudo_mask = ndimage.binary_fill_holes(mask_crf >= 0.5)
        pseudo_mask = Image.fromarray((pseudo_mask * 255).astype(np.uint8))
        pseudo_mask = np.asarray(pseudo_mask.resize((mask_pred.shape[-1], mask_pred.shape[-2])))
        #pseudo_mask = np.asarray(pseudo_mask.resize((init_img.shape[-1], init_img.shape[-2])))
        pseudo_mask = pseudo_mask.astype(np.uint8)
        pseudo_mask = (pseudo_mask > pseudo_mask.max() / 2.0).astype(np.uint8)  # Threshold

        # CC Filtering Based on `get_all_cc`
        labeled, nr_objects = ndimage.label(pseudo_mask)
        if nr_objects > 0:
            nb_pixel = [np.sum(labeled == k) for k in range(nr_objects + 1)]
            pixel_order = np.argsort(nb_pixel)[::-1]  # Descending order

            if not get_all_cc:
                # Single mode: Keep only the largest CC
                largest_cc = labeled == pixel_order[1]  # Skip background (pixel_order[0])
                pseudo_mask = largest_cc.astype(np.uint8)
            # else:
            #     # Multi mode: Remove background CC (e.g., CC touching image border)
            #     border_mask = np.zeros_like(pseudo_mask)
            #     border_mask[[0, -1], :] = 1
            #     border_mask[:, [0, -1]] = 1
            #     touching_border = np.unique(labeled * border_mask)
            #     background_cc = touching_border[touching_border != 0]

            #     # Retain all non-background CCs
            #     foreground_ccs = [k for k in range(1, nr_objects + 1) if k not in background_cc]
            #     pseudo_mask = np.isin(labeled, foreground_ccs).astype(np.uint8)

        # Finalize mask
        pseudo_mask = torch.from_numpy(pseudo_mask).cuda().float().squeeze()
        masks_crf.append(pseudo_mask[None, :, :])

    return torch.cat(masks_crf, dim=0).squeeze()
        

def batch_apply_bilateral_solver(data,
                                 masks,
                                 get_all_cc=True,
                                 shape=None):

    cnt_bs = 0
    masks_bs = []
    inputs, init_imgs, gt_labels, img_path = data

    for id in range(inputs.shape[0]):
        _, bs_mask, use_bs = apply_bilateral_solver(
            mask=masks[id].squeeze().cpu().numpy(),
            img=init_imgs[id],
            img_path=img_path[id],
            im_fullsize=False,
            # Careful shape should be opposed
            shape=(gt_labels.shape[-1], gt_labels.shape[-2]),
            get_all_cc=get_all_cc,
        )
        cnt_bs += use_bs

        # use the bilateral solver output if IoU > 0.5
        if use_bs:
            if shape is None:
                shape = masks.shape[-2:]
            # Interpolate to downsample the mask back
            bs_ds = F.interpolate(
                torch.Tensor(bs_mask).unsqueeze(0).unsqueeze(0),
                shape,  # TODO check here
                mode="bilinear",
                align_corners=False,
            )
            masks_bs.append(bs_ds.bool().cuda().squeeze()[None, :, :])
        else:
            # Use initial mask
            masks_bs.append(masks[id].cuda().squeeze()[None, :, :])
    
    return torch.cat(masks_bs).squeeze(), cnt_bs


def apply_bilateral_solver(
    mask,
    img,
    img_path,
    shape,
    im_fullsize=False,
    get_all_cc=False,
    bs_iou_threshold: float = 0.5,
    reshape: bool = True,
):
    # Get initial image in the case of using full image
    img_init = None
    if not im_fullsize:
        # Use the image given by dataloader
        shape = (img.shape[-1], img.shape[-2])
        t = T.ToPILImage()
        img_init = t(img)

    if reshape:
        # Resize predictions to image size
        resized_mask = cv2.resize(mask, shape)
        sel_obj_mask = resized_mask
    else:
        resized_mask = mask
        sel_obj_mask = mask

    # Apply bilinear solver
    _, binary_solver = bilateral_solver_output(
        img_path,
        resized_mask,
        img=img_init,
        sigma_spatial=16,
        sigma_luma=16,
        sigma_chroma=8,
        get_all_cc=get_all_cc,
    )

    mask1 = torch.from_numpy(resized_mask).cuda()
    mask2 = torch.from_numpy(binary_solver).cuda().float()

    use_bs = 0
    # If enough overlap, use BS output
    if IoU(mask1, mask2) > bs_iou_threshold:
        sel_obj_mask = binary_solver.astype(float)
        use_bs = 1

    return resized_mask, sel_obj_mask, use_bs

def get_bbox_from_segmentation_labels(
    segmenter_predictions: torch.Tensor,
    initial_image_size: torch.Size,
    scales: List[int],
) -> np.array:
    """
    Find the largest connected component in foreground, extract its bounding box
    """
    objects, num_objects = scipy.ndimage.label(segmenter_predictions)

    # find biggest connected component
    all_foreground_labels = objects.flatten()[objects.flatten() != 0]
    most_frequent_label = np.bincount(all_foreground_labels).argmax()
    mask = np.where(objects == most_frequent_label)
    # Add +1 because excluded max
    ymin, ymax = min(mask[0]), max(mask[0]) + 1
    xmin, xmax = min(mask[1]), max(mask[1]) + 1

    if initial_image_size == segmenter_predictions.shape:
        # Masks are already upsampled
        pred = [xmin, ymin, xmax, ymax]
    else:
        # Rescale to image size
        r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
        r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
        pred = [r_xmin, r_ymin, r_xmax, r_ymax]

    # Check not out of image size (used when padding)
    if initial_image_size:
        pred[2] = min(pred[2], initial_image_size[1])
        pred[3] = min(pred[3], initial_image_size[0])

    return np.asarray(pred)


def bbox_iou(
    box1: np.array,
    box2: np.array,
    x1y1x2y2: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7,
):
    # https://github.com/ultralytics/yolov5/blob/develop/utils/general.py
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(
            b1_x1, b2_x1
        )  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * torch.pow(
                    torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
                )
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU
