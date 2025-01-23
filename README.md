# Image Adaptation for Colour Vision Deficient Viewers Using Vision Transformers

## Datasets

This code runs on the Places365, Oxford Flowers, Abstract Art, WikiArt Abstract, Landscape and Still Life datasets
The code expects to find them under data/[places,flowers,abstract_art,wikiart_abstract,wikiart_landscape,wikiart_still_life]

- [ ] Dataset download

## Dependencies

Dependencies are given in the requirements.txt file. This can be run with 
```
pip install -r requirements.txt
```
Inside a virtual environment

# Submodules
This project has the DINO project as a dependency, with some modifications. This code is not submitted as a git repository for anonymity.
Instead, run the following to clone the DINO repository to the directory the code expects:
```
git clone https://github.com/facebookresearch/dino.git
cd dino
git checkout 7c446df5b9f45747937fb0d72314eb9f7b66930a
git apply ../dino.patch
```

# Running the code

With the above dependencies installed, to prepare data, first run:
```
python adjoint_filter_dataset.py --dataset_type [dataset_name] --cvd_type [cvd_type] --dataset_split_idx [i]
```
Where [i] is the dataset index to be processed. By default the dataset is split into 1000 total subsets, so i should range from 0-999, inclusive. The number of
total subsets can be changed via command line

To sub-sample the dataset splits according to the scheme described in the paper, run:
```
python adjoint_sample_datasets.py 
```
which will create a list of files for the dataset code to read in

Finally, to run the adjoint loop, run:
```
python adjoint_multires_interp.py  --dataset_type [dataset_name] --cvd_type [cvd_type] --severity [cvd_severity] --dataset_split_idx [i]
```

To visualise the results, the command
```
python adjoint_recolour_results.py --dataset_type [dataset_name] --cvd_type [cvd_type] --image_number [i]
```
will output a grid of the style found in the paper for image [i] of [dataset_name] for mild, moderate, and high severities of CVD type [cvd_type]

# Citation

If you use this code, please cite our paper:

```
@InProceedings{Gillooly_2025_WACV,
    author    = {Gillooly, Thomas and Thomas, Jean-Baptiste and Hardeberg, Jon Yngve and Guarnera, Giuseppe Claudio},
    title     = {Image Adaptation for Colour Vision Deficient Viewers Using Vision Transformers},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
}
```
