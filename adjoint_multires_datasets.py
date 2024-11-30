import glob
import os
import torchvision.transforms as pth_transforms

from PIL import Image

class CustomDataset():
    """
    Custom dataset to evaluate on cherry-picked images from prior works
    """
    def __init__(self):
        self.filenames = sorted(glob.glob('data/custom/*.png')) + sorted(glob.glob('data/swin_t_sample/*_input.png'))
        
        self.transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Resize((256, 256), antialias=False),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        return idx, self.transform((Image.open(self.filenames[idx]).convert('RGB')))

class Dataset():
    """
    Generic dataset for all other dataset types

    Expects datasets to be under data/{dataset_name}
    Expects list of image filenames in file output by dataset sampling script
    """
    def __init__(self, dataset_type, cvd_type, dataset_split, *args, **kwargs):
        data_dir = f'data/{dataset_type}'

        with open(os.path.join(data_dir, f'{dataset_type}_{cvd_type}_ws_4_sample_files_split_{dataset_split}.txt'), encoding='utf-8') as file:
            self.filenames = file.read().splitlines()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        return idx, self.transform((Image.open(self.filenames[idx]).convert('RGB')))