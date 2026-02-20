# Code from TokenCut

import PIL.Image as Image 
import torch;

def resize_pil(I, patch_size=16) : 
    w, h = I.size

    new_w, new_h = int(round(w / patch_size)) * patch_size, int(round(h / patch_size)) * patch_size
    feat_w, feat_h = new_w // patch_size, new_h // patch_size

    return I.resize((new_w, new_h), resample=Image.LANCZOS), w, h, feat_w, feat_h


# Code from Maskcut

import numpy as np;
from scipy import ndimage

# def detect_box(bipartition, seed,  dims, initial_im_size=None, scales=None, principle_object=True):
#     """
#     Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
#     """

#     w_featmap, h_featmap = dims
#     objects, num_objects = ndimage.label(bipartition) 
#     cc = objects[np.unravel_index(seed, dims)]
    

#     if principle_object:
#         mask = np.where(objects == cc)
#        # Add +1 because excluded max
#         ymin, ymax = min(mask[0]), max(mask[0]) + 1
#         xmin, xmax = min(mask[1]), max(mask[1]) + 1
#         # Rescale to image size
#         r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
#         r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
#         pred = [r_xmin, r_ymin, r_xmax, r_ymax]
         
#         # Check not out of image size (used when padding)
#         if initial_im_size:
#             pred[2] = min(pred[2], initial_im_size[1])
#             pred[3] = min(pred[3], initial_im_size[0])
        
#         # Coordinate predictions for the feature space
#         # Axis different then in image space
#         pred_feats = [ymin, xmin, ymax, xmax]

#         return pred, pred_feats
#     else:
#         raise NotImplementedError
    
    
def detect_box(bipartition, seed, dims, initial_im_size=None, scales=None, principle_object=True):
    objects, num_objects = ndimage.label(bipartition)
    cc = objects[np.unravel_index(seed, dims)]
    if principle_object:
        mask = np.where(objects == cc)
        ymin, ymax = min(mask[0]), max(mask[0]) + 1
        xmin, xmax = min(mask[1]), max(mask[1]) + 1
        r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
        r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
        pred = [r_xmin, r_ymin, r_xmax, r_ymax]
        if initial_im_size:
            pred[2] = min(pred[2], initial_im_size[1])
            pred[3] = min(pred[3], initial_im_size[0])
        pred_feats = [ymin, xmin, ymax, xmax]
        return pred, pred_feats, objects, mask
    else:
        raise NotImplementedError


    
def IoU(mask1, mask2):
    mask1, mask2 = (mask1>0.5).to(torch.bool), (mask2>0.5).to(torch.bool)
    intersection = torch.sum(mask1 * (mask1 == mask2), dim=[-1, -2]).squeeze()
    union = torch.sum(mask1 + mask2, dim=[-1, -2]).squeeze()
    return (intersection.to(torch.float) / union).mean().item()
