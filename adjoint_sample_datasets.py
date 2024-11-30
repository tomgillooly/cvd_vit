"""
This file uses the per-image contrast loss to sample an approximate Gaussian distribution from each dataset

The sampled images are split into subsets of size args.split_size and written to text files which the dataset reads in
during evaluation
"""

import argparse
import glob
import numpy as np
import os
import pandas as pd
import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, default=4, help="Size of sliding window in local contrast computation")
    parser.add_argument('--split_size', type=int, default=100, help="Size of output subsampled dataset splits")
    parser.add_argument('--max_samples', type=int, default=1000, help="Total number of samples to draw from dataset")
    parser.add_argument('--dist_step_size', type=float, default=1e-1, help="Bin width in contrast histogram")
    parser.add_argument('--dist_min_value', type=float, default=-4, help="Approximate minimum value in output distribution")
    parser.add_argument('--dist_max_value', type=float, default=-1, help="Approximate maximum value in output distribution")
    
    args = parser.parse_args()

    datasets = 'abstract_art flowers places wikiart_abstract wikiart_landscape wikiart_still_life'.split()
    cvd_types = 'protan deutan tritan'.split()

    
    for dataset_type in tqdm.tqdm(datasets[1:2]):
        data_dir = f'data/{dataset_type}'
        
        if dataset_type == 'flowers':
            file_glob = os.path.join(data_dir, 'image_*.jpg')
        elif dataset_type.startswith('wikiart'):
            file_glob = os.path.join(data_dir, '*/*.jpg')
        elif dataset_type == 'abstract_art':
            file_glob = os.path.join(data_dir, '*.jpg')
        elif dataset_type == 'places':
            file_glob = os.path.join(data_dir, '*.png')

        filenames = sorted(glob.glob(file_glob))

        for cvd in tqdm.tqdm(cvd_types, leave=False):
            parquet_files = glob.glob(os.path.join(
                data_dir,
                f'im_metrics_{cvd}_ws_{args.window_size}',
                'part.*.parquet'
            ))

            num_samples = min(len(filenames), 1000)

            # Get Gaussian parameters
            mean_value = (args.dist_min_value + args.dist_max_value) / 2
            std_value = (args.dist_max_value - args.dist_min_value) / 6

            # Range of contrast values to sample
            bin_edges = np.arange(args.dist_min_value, args.dist_max_value, args.dist_step_size)
            
            # Calculate number of samples at each contrast value
            samples_per_bin = np.exp(-(bin_edges-mean_value)**2/(std_value*2))

            # Normalise and rescale by total number of samples
            samples_per_bin /= samples_per_bin.sum()
            samples_per_bin *= num_samples

            # Convert to integer value for sampling
            samples_per_bin = np.ceil(samples_per_bin).astype(int)

            # If total samples per bin exceeds target number of samples, reduce the maximum bin samples
            samples_per_bin[samples_per_bin.argmax()] -= samples_per_bin.sum() - num_samples
            
            dataset_df = pd.concat([pd.read_parquet(file) for file in parquet_files])

            # Hard limit by maximum and minimum contrast values
            dataset_df = dataset_df[(dataset_df.contrast_diff > args.dist_min_value) & (dataset_df.contrast_diff < args.dist_max_value)]

            # Find which sample bin each image belongs to and assign category to dataframe
            dataset_df = dataset_df.assign(contrast_bin=np.digitize(dataset_df['contrast_diff'], bin_edges)-1)

            # Group by bin for sampling
            dataset_df_groupby = dataset_df.groupby('contrast_bin')

            sample_dfs = []

            # Track leftover samples from current bin so we can attempt to draw them from next bin instead
            remainder = 0
            # iterate through contrast bins
            for bin_number, group_df in dataset_df_groupby:
                # Draw samples from current bin, capped at bin size
                sample_dfs.append(group_df.sample(n=min(group_df.shape[0], samples_per_bin[bin_number] + remainder)))
                # Update remainder
                remainder = max(samples_per_bin[bin_number] + remainder - group_df.shape[0], 0)

            sample_df = pd.concat(sample_dfs)
            
            # Shuffle sample bins so that subset splits will have images from all contrast bins
            sample_df = sample_df.sample(frac=1.0)

            # Write out dataset filenames in splits defined by args.split_size
            for split_idx in range(0, sample_df.shape[0], args.split_size):
                out_file = os.path.join(data_dir, f'{dataset_type}_{cvd}_ws_{args.window_size}_sample_files_split_{split_idx//args.split_size}.txt')
                with open(out_file, 'w') as file:
                    image_indices = sample_df.im_idx.values[split_idx:split_idx+args.split_size]

                    for im_idx in image_indices:
                        file.write(filenames[im_idx] + '\n')
