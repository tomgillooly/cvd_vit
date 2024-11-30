import argparse
import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

from cvd_matrices import CVD_MATRICES

def render_figure(dataset_type, cvd_type, dataset_split, image_number, window_size, output_folder='example_images',
                  attn_layer=4, multires='all', with_bias=False, **kwargs):
    if dataset_type == 'custom':
        input_filenames = sorted(glob.glob('data/custom/*.png')) + sorted(glob.glob('data/swin_t_sample/*_input.png'))
    else:
        data_dir = f'data/{dataset_type}'

        with open(os.path.join(data_dir, f'{dataset_type}_{cvd_type}_ws_{window_size}_sample_files_split_{dataset_split}.txt')) as file:
            input_filenames = file.read().splitlines()

    im_name = os.path.basename(input_filenames[image_number])

    severities = [0.4, 0.6, 1.0]
    shifts = ['8nm', '12nm', '20nm']

    filenames = []
    parquet_filenames = []
    for sev in severities:
        results_folder = os.path.join('results', dataset_type, cvd_type, f'severity_{sev}')

        # Ablation results stored in separate directory
        if attn_layer != 4:
            results_folder = os.path.join(results_folder, f'layers_{args.attn_layer}')

        if not multires == 'all':
            results_folder = os.path.join(results_folder, f'{multires}_res')

        if with_bias:
            results_folder = os.path.join(results_folder, 'bias')

        filenames.append(os.path.join(results_folder, im_name + '.npz'))
        parquet_filenames.append(os.path.join(results_folder,
                                    f'{dataset_type}_metric_results_{cvd_type}_{sev}_ws_{window_size}_split_{dataset_split}.parquet'))

    dalt_boosts = [pd.read_parquet(filename).dalt_boost[image_number] for filename in parquet_filenames]

    input_image = np.load(filenames[0])['input_image'].transpose(1, 2, 0)

    fig, axes = plt.subplots(3, 5, figsize=(5*3, 3*3))
    axes[0, 0].axis('off')
    axes[1, 0].imshow(input_image)
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    axes[1, 0].set_title('Input Image')
    axes[2, 0].axis('off')


    for ax, filename, sev_idx, dalt_boost in zip(axes[:, 1:], filenames, range(3), dalt_boosts):
        with np.load(filename) as npz_file:
            modified_image = npz_file['image'].transpose(1, 2, 0)
            modified_cvd_image = np.einsum('ijk,lk->ijl', modified_image, CVD_MATRICES[severities[sev_idx]][cvd_type])
            offset_image = npz_file['offset'].transpose(1, 2, 0)/2 + 0.5
            input_cvd_image = np.einsum('ijk,lk->ijl', input_image, CVD_MATRICES[severities[sev_idx]][cvd_type])
            
        ax[0].imshow(modified_image)
        if sev_idx == 0:
            ax[0].set_title('Recoloured Image')
        ax[0].set_ylabel(shifts[sev_idx])
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        ax[1].imshow(offset_image)
        if sev_idx == 0:
            ax[1].set_title('Offset')
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        ax[2].imshow(modified_cvd_image)
        if sev_idx == 0:
            ax[2].set_title('CVD Recoloured Image')
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        ax[2].set_xlabel(f'Contrast boost: {dalt_boost:.04f}')

        ax[3].imshow(input_cvd_image)
        if sev_idx == 0:
            ax[3].set_title('CVD Input Image')
        ax[3].set_xticks([])
        ax[3].set_yticks([])

    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    output_filename = os.path.join(output_folder, f'{dataset_type}_{cvd_type}_{im_name.split(".")[0]}.pdf')
    print('Saved result to ' + output_filename)
    plt.savefig(output_filename)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default='flowers',
                            choices=['abstract_art', 'flowers', 'places', 'wikiart_abstract', 'wikiart_landscape', 'wikiart_still_life'],
                            help="Specify which of the candidate datasets to evaluate over"
                            )
    parser.add_argument('--cvd_type', type=str, default='protan', choices=['protan', 'deutan', 'tritan'])
    parser.add_argument('--with_bias', type=bool, default=False, help='Set to True to retain image bias, False to remove')
    parser.add_argument('--attn_layer', type=int, default=4, help="Number of attention layers to use in loss function")
    parser.add_argument('--multires', type=str, default='all', choices=['all', 'single_32', 'single_64', 'max', 'top_two', 'other'],
                            help='Specify combination of patches to use')
    parser.add_argument('--attn_layer', type=int, default=4, help="Number of attention layers to use in loss function")
    parser.add_argument('--dataset_split', type=int, default=0, help="Specify which split of the dataset to evaluate over")
    parser.add_argument('--window_size', type=int, default=4, help="Size of sliding window in local contrast computation")
    parser.add_argument('--image_number', type=int, default=0, help="Index of image to evaluate")
    args = parser.parse_args()

    render_figure(**vars(args))