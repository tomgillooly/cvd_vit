"""
This file define the contrast metric as well as the local and global contrast metrics
used in prior works
"""

import torch

from itertools import product
from kornia.color import rgb_to_lab

def local_contrast(img, window_size=5, gaussian=True):
    """
    Get local contrast within defined window by taking eucliden pixel distance from
    each centre point to its neighbours. Optionally weight the distances with a gaussian kernel
    """
    im_height, im_width = img.shape[-2:]
    # x_diff is the image of centrepoints
    x_diff = img[..., window_size:im_height-window_size, window_size:im_width-window_size]

    # Define window weights
    if gaussian:
        kernel = torch.meshgrid(torch.linspace(-1, 1, 2*window_size+1), torch.linspace(-1, 1, 2*window_size+1))
        kernel = torch.sum(torch.stack(kernel, dim=-1)**2, dim=-1)
        kernel = torch.exp(-kernel)
    else:
        kernel = torch.ones((2*window_size+1, 2*window_size+1)).to(img.device)
    
    contrast_window = []
    for i, j in product(range(-window_size, window_size+1), repeat=2):
        # Don't take the distance from the centre point to itself
        if i == 0 and j == 0:
            continue
            
        # Get Euclidean distance from each centre point to its neighbour and weight by kernel
        img_diff = x_diff - img[..., window_size+i:im_height-window_size+i, window_size+j:im_width-window_size+j]
        img_diff = torch.sqrt(torch.sum(torch.pow(img_diff, 2), dim=-3))*kernel[i, j]
        
        contrast_window.append(img_diff)

    contrast_window = torch.stack(contrast_window, dim=-1)

    return contrast_window

def global_contrast(img, num_samples=20000):
    """
    Get global contrast by taking the average of the euclidean distance between random pixels
    """
    bs, c, h, w = img.shape

    img = img.view(bs, c, h*w)

    samples_i = img[:, :, torch.randint(h*w, (20000,))]
    samples_j = img[:, :, torch.randint(h*w, (20000,))]

    return torch.sqrt(torch.pow(samples_i - samples_j, 2).sum(dim=1)).mean()


def get_similarity(output, target):
    """
    Get the co-linear and orthogonal components from the reference line
    """
    target_energy_sq = (target**2).sum(dim=-1, keepdims=True)
    target_energy = torch.sqrt(target_energy_sq)
        
    output_dp = (target*output).sum(dim=-1, keepdims=True) / (target_energy + 1e-9)
    output_proj = output_dp * target / (target_energy + 1e-9)
    output_offset = output - output_proj
    offset_energy = torch.sqrt((output_offset**2).sum(dim=-1, keepdims=True))

    similarity_im = output_dp - (offset_energy / (1+target_energy))**2 - target_energy

    return similarity_im.squeeze(-1), output_dp - target_energy, offset_energy**2 / (1+target_energy)**2

def get_contrast_metric(output, target, window_size=4, gaussian=True, return_components=False):
    # Compute all colour differences in CIELAB space
    output_lab = rgb_to_lab(output)
    target_lab = rgb_to_lab(target)

    # Clamp delta E values from 0 to 3 to account for JND and perceptual non-uniformity
    target_contrast = local_contrast(target_lab, window_size=window_size, gaussian=gaussian).clip(2, 5)-2
    output_contrast = local_contrast(output_lab, window_size=window_size, gaussian=gaussian).clip(2, 5)-2

    if return_components:
        return get_similarity(output_contrast, target_contrast)
    else:
        return get_similarity(output_contrast, target_contrast)[0]
