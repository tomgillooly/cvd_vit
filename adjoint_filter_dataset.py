"""
This file generates the contrast loss per image for a given dataset and CVD type at maximum severity.

These measurements are used to in the dataset sampling procedure to ensure that the evaulated
subsets have a contrast loss distribution that is roughly equivalent across datasets
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as pth_transforms
import tqdm

from collections import defaultdict
from PIL import Image

from adjoint_multires_metrics import get_contrast_metric
from cvd_matrices import CVD_MATRICES

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default='flowers',
                        choices=['abstract_art', 'flowers', 'places', 'wikiart_abstract', 'wikiart_landscape', 'wikiart_still_life'],
                        help="Specify which of the candidate datasets to evaluate over"
                        )
    parser.add_argument('--window_size', type=int, default=4, help="Size of sliding window in local contrast computation")
    parser.add_argument('--im_size', type=int, default=256, help="Size of rescaled images")
    parser.add_argument('--cvd_type', type=str, default='protan', choices=['protan', 'deutan', 'tritan'])
    parser.add_argument('--total_dataset_splits', type=int, default=1000, help="Number of dataset divisions")
    parser.add_argument('--dataset_split_idx', type=int, default=0, help="Dataset division to evalue")
    
    args = parser.parse_args()

    # Find all images from specified dataset
    data_dir = f'data/{args.dataset_type}'

    if args.dataset_type == 'flowers':
        file_glob = os.path.join(data_dir, 'image_*.jpg')
    elif args.dataset_type.startswith('wikiart'):
        file_glob = os.path.join(data_dir, '*/*.jpg')
    elif args.dataset_type == 'abstract_art':
        file_glob = os.path.join(data_dir, '*.jpg')
    elif args.dataset_type == 'places':
        file_glob = os.path.join(data_dir, '*.png')

    filenames = sorted(glob.glob(file_glob))

    # Transforms image from PIL to torch tensor and rescale to specified image size
    transform = pth_transforms.Compose((
        pth_transforms.ToTensor(),
        pth_transforms.Resize((args.im_size, args.im_size))
    ))

    # Load CVD matrix for specified type and severity
    cvd_matrix = torch.from_numpy(CVD_MATRICES[1.0][args.cvd_type]).float()

    # Determine size of dataset divisions and subset filenames
    dataset_split_size = int(np.ceil(len(filenames) / args.total_dataset_splits))
    start_idx = dataset_split_size*args.dataset_split_idx
    stop_idx = start_idx + dataset_split_size

    filenames = filenames[start_idx:stop_idx]

    # Evaluate contrast loss for each image and save in parquet file
    data = defaultdict(list)
    for idx, filename in tqdm.tqdm(enumerate(filenames), total=len(filenames)):
        im = transform(Image.open(filename))[:3]  # Remove transparency channel
        im_cvd = torch.einsum('ij,jkl->ikl', cvd_matrix, im)

        contrast_diff = get_contrast_metric(im_cvd, im).mean()
        
        data['im_idx'].append(idx+start_idx)
        data['cvd'].append(args.cvd_type)
        data['contrast_diff'].append(contrast_diff.item())
        
    parquet_dir = os.path.join(data_dir, f'im_metrics_{args.cvd_type}_ws_{args.window_size}')
    os.makedirs(parquet_dir, exist_ok=True)

    df = pd.DataFrame(data)
    df.to_parquet(os.path.join(parquet_dir, f'part.{args.dataset_split_idx}.parquet'))