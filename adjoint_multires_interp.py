"""
This file contains the main adjoint routine to find recoloured images

Note that this script does not download and prepare input datasets, this must be done separately
"""


import lightning.pytorch as pl
pl.seed_everything(42)
import argparse
import dino.vision_transformer as vits
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms as pth_transforms

from adjoint_multires_datasets import CustomDataset, Dataset
from adjoint_multires_metrics import get_contrast_metric
from cvd_matrices import CVD_MATRICES

class Parser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()

        self.add_argument('--accelerator', type=str, default='gpu')
        self.add_argument('--num_devices', type=int, default=1, help='Number of GPUs on device')
        self.add_argument('--dataset_type', type=str, default='flowers',
                          choices=['abstract_art', 'flowers', 'places', 'wikiart_abstract', 'wikiart_landscape', 'wikiart_still_life'],
                          help="Specify which of the candidate datasets to evaluate over"
                          )
        self.add_argument('--dataset_split', type=int, default=0, help="Specify which split of the dataset to evaluate over")
        self.add_argument('--im_size', type=int, default=256, help="Size of rescaled images")
        self.add_argument('--cvd_type', type=str, default='protan', choices=['protan', 'deutan', 'tritan'])
        self.add_argument('--severity', type=float, default=1.0, choices=[0.4, 0.6, 1.0], help="CVD severity, with 0.4, 0.6, 1.0 corresponding to 8, 12, 20nm")
        self.add_argument('--batch_size', type=int, default=6)
        self.add_argument('--max_epochs', type=int, default=200)
        self.add_argument('--val_interval', type=int, default=10)
        self.add_argument('--with_bias', type=bool, default=False, help='Set to True to retain image bias, False to remove')
        self.add_argument('--attn_layer', type=int, default=4, help="Number of attention layers to use in loss function")
        self.add_argument('--lr_base', type=float, default=8e-3, help="Learning rate")
        self.add_argument('--beta1', type=float, default=0.5, help="Adam beta1 parameter")
        self.add_argument('--beta2', type=float, default=0.5, help="Adam beta2 parameter")
        self.add_argument('--multires', type=str, default='all', choices=['all', 'single_32', 'single_64', 'max', 'top_two', 'other'],
                           help='Specify combination of patches to use')
        self.add_argument('--patches', type=str, default='1,8,16,32,64', help="Resolution of image offsets")
        self.add_argument('--save_images', type=bool, default=True, help="Set to true to save images in results folder")

class Daltoniser(pl.LightningModule):
    def configure_vit_model(self):
        """
        Load the ViT model from the DINO repository and configure it for use in the adjoint routine

        This can be overridden to download different size DINO models
        """
        
        self.arch = 'vit_small'
        self.patch_size = 8

        self.model = vits.__dict__[self.arch](patch_size=self.patch_size, num_classes=0)
        self.model.eval()
        
        url = f'dino_deitsmall{self.patch_size}_pretrain/dino_deitsmall{self.patch_size}_pretrain.pth'
        checkpoint_filename = url.split('/')[-1]

        # Don't redownload if we already have the file locally
        if not os.path.exists(checkpoint_filename):
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            torch.save(state_dict, checkpoint_filename)
        else:
            state_dict = torch.load(checkpoint_filename, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=True)

    def __init__(self, severity, cvd_type, patches, dataset_type, attn_layer, im_size,
                 dataset_split, with_bias, multires,
                 save_images,
                 *args, **kwargs):
        super().__init__()

        self.image_size = (im_size, im_size)
        
        self.patches = [int(e) for e in patches.split(',')]
        
        # Set up output folder for results. Note that this only saves metric results and won't save images by default unless the command line argument is set
        self.results_dict = {}
        self.results_folder = os.path.join('results', dataset_type, cvd_type, f'severity_{severity}')

        # Store ablation results in separate directories
        if attn_layer != 4:
            results_folder = os.path.join(results_folder, f'layers_{args.attn_layer}')

        if not multires == 'all':
            results_folder = os.path.join(results_folder, f'{multires}_res')

        if with_bias:
            results_folder = os.path.join(results_folder, 'bias')

        self.parquet_filename = os.path.join(self.results_folder, f'{dataset_type}_metric_results_{cvd_type}_{severity}_ws_4_split_{dataset_split}.parquet')
        os.makedirs(self.results_folder, exist_ok=True)
        
        if dataset_type == 'custom':
            self.dataset = CustomDataset()
        else:
            self.dataset = Dataset(dataset_type, cvd_type, dataset_split)
                
        # Create model parameters for the offsets per-image at each resolution
        self.offset_images = []
        for patch_size in self.patches:
            parameter_name = f'offset_image_{patch_size}'
            self.register_parameter(parameter_name, nn.Parameter(torch.zeros(len(self.dataset), 3, patch_size, patch_size)))
            self.offset_images.append(self.get_parameter(parameter_name))

        # Save CVD simulation matrix as model property so that it is moved to GPU if necessary
        self.register_buffer('cvd_sim_matrix', torch.from_numpy(CVD_MATRICES[severity][cvd_type]).float(), persistent=False)

        self.configure_vit_model()
        
        # Same normalization parameters used in DINO
        self.normalize_tf = pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        # Same resize factor used when evaluating image metrics per-dataset
        self.transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Resize(self.image_size, antialias=True),
        ])

        # Set the transform on the dataset so it is applied on retrieval
        # This has to be done before collation in the dataloader so that the resulting tensors are of the same size
        # and can be stacked
        self.dataset.transform = self.transform

        self.save_hyperparameters()

    def configure_optimizers(self):
        attn_optimizer = torch.optim.Adam(self.offset_images, lr=self.hparams['lr_base'],
                                          betas=(self.hparams['beta1'], self.hparams['beta2']))

        return attn_optimizer

    def train_dataloader(self):
        return DataLoader(self.dataset, shuffle=False, batch_size=self.hparams['batch_size'], drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.dataset, shuffle=False, batch_size=self.hparams['batch_size'], drop_last=False)

    def get_attn(self, img):
        # Get the attention maps from the first attn_layer layers of the model
        return [a[..., 1:, 1:].log_softmax(dim=-1) for a in self.model.get_first_n_attn(self.normalize_tf(img), n=self.hparams['attn_layer'])]

    def attn_criterion(self, output, target):
        # Attention map loss function
        return (F.mse_loss(target, output, reduction='none')*target.exp()).sum(dim=-1).flatten(1).mean(dim=-1)

    def on_save_checkpoint(self, ckpt):
        # Before saving the model checkpoint, remove the DINO parameters from the checkpoint as
        # we only want to save the offset images - the DINO parameters can be reloaded as they are not changed
        model_keys = [key for key in ckpt['state_dict'].keys() if key.startswith('model.')]
        for key in model_keys:
            ckpt['state_dict'].pop(key)

    def compose_offset(self, image_offsets):
        total_offset = image_offsets[0]
        # Interpolate running offset up to the current resolution and sum them together
        for offset in image_offsets[1:]:
            total_offset = offset + F.interpolate(total_offset, offset.shape[-2:], mode='bilinear', align_corners=True, antialias=False)
            
        # Interpolate the final offset to the full image size
        total_offset = F.interpolate(total_offset, self.image_size, mode='bilinear', align_corners=True, antialias=False)
        
        return total_offset

    def forward(self, image_offsets):
        total_offset = self.compose_offset(image_offsets)

        return total_offset

    def on_after_batch_transfer(self, batch, batch_idx):
        # Calculate the target attention map from the input images on-the-fly
        image_idx, base_image = batch

        with torch.no_grad():
            target_self_attn = self.get_attn(base_image)

        return image_idx.long(), base_image.float(), [attn.float() for attn in target_self_attn]

    def training_step(self, batch):
        image_idx, base_image, target_self_attn = batch

        # Pick out image offsets in current batch. Bias is aggregated down into the lowest resolution offset, so if we are removing
        # bias, don't include this in total offset computation
        image_offsets = [image[image_idx] for image in self.offset_images[(0 if self.hparams['with_bias'] else 1):]]
        
        total_offset = self(image_offsets)
        
        # Rescale and clamp the modified image to the range [0, 1]
        modified_image = F.hardsigmoid((base_image + total_offset)*6 - 3)
        # Simulate CVD on the recoloured image
        daltonised_image = torch.einsum('ijkl,mj->imkl', modified_image, self.cvd_sim_matrix)

        # Get attention maps for recoloured image before and after CVD simulation and calculate loss
        modified_attn = self.get_attn(modified_image)
        daltonised_attn = self.get_attn(daltonised_image)
        
        modified_attn_loss = [self.attn_criterion(m_attn, t_attn) for m_attn, t_attn in zip(modified_attn, target_self_attn)]
        daltonised_attn_loss = [self.attn_criterion(m_attn, t_attn) for m_attn, t_attn in zip(daltonised_attn, target_self_attn)]

        attn_loss = 0
        for idx, (m_attn_loss, d_attn_loss) in enumerate(zip(modified_attn_loss, daltonised_attn_loss)):
            attn_loss += m_attn_loss + d_attn_loss
        
        attn_loss = attn_loss.mean()
        self.log('attn_loss', attn_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                
        return attn_loss

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
    
    def collapse_image_pyramid(self):
        with torch.no_grad():
            # Get mean value of each 2x2 patch in images and shift it down to the next lowest resolution image
            for offset_idx in reversed(range(1, len(self.offset_images))):
                window_size = self.offset_images[offset_idx].shape[-1] // self.offset_images[offset_idx-1].shape[-1]
                bias = F.avg_pool2d(self.offset_images[offset_idx], window_size, stride=window_size)
                self.offset_images[offset_idx-1].data += bias
                self.offset_images[offset_idx].data -= bias.repeat_interleave(window_size, -2).repeat_interleave(window_size, -1)

    def on_train_epoch_end(self):
        super().on_train_epoch_end()

        # Shift bias down to lowest resolution image at the end of each epoch
        self.collapse_image_pyramid()

    def validation_step(self, batch):
        # Generate recoloured images and save results every val_interval epochs
        image_idx, base_image, _ = batch

        # Pick out image offsets in current batch. Bias is aggregated down into the lowest resolution offset, so if we are removing
        # bias, don't include this in total offset computation
        image_offsets = [image[image_idx] for image in self.offset_images[(0 if self.hparams['with_bias'] else 1):]]

        total_offset = self(image_offsets)
        
        # Rescale and clamp the modified image to the range [0, 1]
        modified_image = F.hardsigmoid((base_image + total_offset)*6 - 3)
        # Simulate CVD on the recoloured image
        daltonised_image = torch.einsum('ijkl,mj->imkl', modified_image, self.cvd_sim_matrix)
        
        # Simulate CVD for input image for metric computation
        base_cvd_image = torch.einsum('ijkl,mj->imkl', base_image, self.cvd_sim_matrix)

        # Contrast loss between input image and its CVD simulation
        base_cvd_base_error_im = get_contrast_metric(base_cvd_image, base_image)
        # Contrast loss between recoloured image and its CVD simulation
        daltonised_base_error_im = get_contrast_metric(daltonised_image, base_image)

        # Get mean contrast loss per image for input and recoloured images
        bs, h, w = base_cvd_base_error_im.shape
        base_cvd_base_error = base_cvd_base_error_im.reshape(bs, h*w).mean(dim=-1)
        daltonised_base_error = daltonised_base_error_im.reshape(bs, h*w).mean(dim=-1)

        # Improvement is the difference in contrast loss between input and recoloured images
        dalt_boost = daltonised_base_error - base_cvd_base_error

        self.log(f'base_cvd_base_error', base_cvd_base_error.mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'daltonised_base_error', daltonised_base_error.mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'dalt_boost', dalt_boost.mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'total_offset_energy', (total_offset**2).mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        for i in range(len(image_offsets)):
            self.log(f'offset_energy_{i}', (image_offsets[i]**2).flatten(1).mean(dim=-1).mean(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # For each image, see if metric has improved, if so, log new result and save image
        for batch_idx, im_idx in enumerate(image_idx):
            if im_idx.item() in self.results_dict.keys():
                if dalt_boost[batch_idx] < self.results_dict[im_idx.item()]['dalt_boost']:
                    continue

            self.results_dict[im_idx.item()] = {'base_cvd_base_error': base_cvd_base_error[batch_idx].item(), 'daltonised_base_error': daltonised_base_error[batch_idx].item(), 'total_energy': (total_offset[batch_idx]**2).mean().item(),
                                                'dalt_boost': dalt_boost[batch_idx].item(), 'im_idx': im_idx.item(),
                                                }
            pd.DataFrame.from_dict(self.results_dict, orient='index').to_parquet(self.parquet_filename)

            if self.hparams['save_images']:
                data = {
                    'input_image': base_image[batch_idx].cpu().numpy(),
                    'image': modified_image[batch_idx].cpu().numpy(),
                    'offset': total_offset[batch_idx].cpu().numpy(),
                }
                for idx, offset in enumerate([offsets[batch_idx] for offsets in image_offsets]):
                    data[f'raw_offset_{idx}'] = offset.cpu().numpy()
                np.savez_compressed(os.path.join(self.results_folder, os.path.basename(self.dataset.filenames[im_idx.item()])) + '.npz', **data)

if __name__ == '__main__':
    parser = Parser()
    args = parser.parse_args()

    # Preset patch combinations for use in ablation, if none are specified, use the args.patches command-line argument
    if args.multires == 'all':
        args.patches = '1,8,16,32,64'
    elif args.multires == 'single_64':
        args.patches = '1,64'
    elif args.multires == 'single_32':
        args.patches = '1,32'
    elif args.multires == 'max':
        args.patches = '1,256'
    elif args.multires == 'top_two':
        args.patches = '1,32,64'

    for arg_name, arg in vars(args).items():
        print(f'{arg_name}: {arg}')

    print('Initialising model')
    model = Daltoniser(**vars(args))
    print(f'Saving results to {model.parquet_filename}')
    
    # Accumulate gradients over all batches in an epoch in to avoid any issues with momentum or weight decay
    batches_per_epoch = int(np.ceil(len(model.dataset) / args.batch_size))

    trainer = pl.Trainer(max_epochs=args.max_epochs, accelerator=args.accelerator,
                        devices=args.num_devices, log_every_n_steps=50,
                        precision='32', check_val_every_n_epoch=args.val_interval,
                        accumulate_grad_batches=batches_per_epoch,
                    )

    trainer.fit(model)