"""
daltonise: Algorithms for CVD simulation and daltonisation

Copyright (C) 2020 Ivar Farup

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import argparse
try:
    import cupy as arraylib
except ImportError:
    import numpy as arraylib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
try:
    from cupyx.scipy.signal import correlate2d
except ImportError:
    from scipy.signal import correlate2d
import torch
import tqdm

from kornia.color import rgb_to_lab
from kornia.filters import filter2d
from torch.utils.data import DataLoader

from adjoint_multires_datasets import CustomDataset, Dataset
from adjoint_multires_metrics import local_contrast, get_contrast_metric, get_cvd_swin_metric
from cvd_matrices import CVD_MATRICES

def unit_vectors(im, sim, el=arraylib.array([1.0, 1.0, 1.0])):
    """
    Return principal vectors of the simulation.

    The first vector is taken as the lightness vector, the second is the
    first principal component of the difference between the image
    and the simulation, and the third is orthogonal to both of them.

    Paramters
    ---------
    im : ndarray
        The original image
    sim : ndarray
        The CVD simulation of the original image
    
    Returns
    -------
    el : ndarray
        The lightness vector
    ed : ndarray
        The difference vector
    ec : ndarray
        The chroma vector
    """
    
    el = el / arraylib.linalg.norm(el)

    B, H, W, C = im.shape
    diff = arraylib.reshape(im - sim, (B, H*W, C))
    corr = arraylib.matmul(diff.transpose(0, 2, 1), diff)
    
    eig_val, eig = arraylib.linalg.eigh(corr)
    
    ed = arraylib.take_along_axis(eig, arraylib.argmax(eig_val, axis=-1, keepdims=True)[:, None], axis=-1).squeeze(-1)
    
    ed = ed - arraylib.matmul(ed, el)[..., None] * el[None]
    ed = ed / arraylib.linalg.norm(ed, axis=-1, keepdims=True)
    
    ec = arraylib.cross(ed, el)
    ec = ec / arraylib.linalg.norm(ec, axis=-1, keepdims=True)

    return el, ed, ec


def diff_filters(diff):
    """
    Compute different forward and backward FDM correlation filters.

    Parameters
    ----------
    diff : str
        finite difference method (FB, cent, Sobel, SobelFB, Feldman, FeldmanFB)

    Returns
    -------
    F_x : ndarray
    F_y : ndarray
    B_x : ndarray
    B_y : ndarray
    """

    if diff == 'FB':
        F_x = arraylib.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
        F_y = arraylib.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])
        B_x = arraylib.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
        B_y = arraylib.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]])
    elif diff == 'cent':
        F_x = .5 * arraylib.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
        F_y = .5 * arraylib.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
        B_x = F_x
        B_y = F_y
    elif diff == 'Sobel':
        F_x = .125 * arraylib.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        F_y = .125 * arraylib.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        B_x = F_x
        B_y = F_y
    elif diff == 'SobelFB':
        F_x = .25 * arraylib.array([[0, -1, 1], [0, -2, 2], [0, -1, 1]])
        F_y = .25 * arraylib.array([[1, 2, 1], [-1, -2, -1], [0, 0, 0]])
        B_x = .25 * arraylib.array([[-1, 1, 0], [-2, 2, 0], [-1, 1, 0]])
        B_y = .25 * arraylib.array([[0, 0, 0], [1, 2, 1], [-1, -2, -1]])
    elif diff == 'Feldman':
        F_x = 1 / 32 * arraylib.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
        F_y = 1 / 32 * arraylib.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])
        B_x = F_x
        B_y = F_y
    elif diff == 'FeldmanFB':
        F_x = 1 / 16 * arraylib.array([[0, -3, 3], [0, -10, 10], [0, -3, 3]])
        F_y = 1 / 16 * arraylib.array([[3, 10, 3], [-3, -10, -3], [0, 0, 0]])
        B_x = 1 / 16 * arraylib.array([[-3, 3, 0], [-10, 10, 0], [-3, 3, 0]])
        B_y = 1 / 16 * arraylib.array([[0, 0, 0], [3, 10, 3], [-3, -10, -3]])
    elif diff == 'circFB':
        x = (arraylib.sqrt(2) - 1) / 2
        F_x = (arraylib.array([[0, -x, x], [0, -1, 1], [0, -x, x]]) /
               (2 * x + 1))
        F_y = (arraylib.array([[x, 1, x], [-x, -1, -x], [0, 0, 0]]) /
               (2 * x + 1))
        B_x = (arraylib.array([[-x, x, 0], [-1, 1, 0], [-x, x, 0]]) /
               (2 * x + 1))
        B_y = (arraylib.array([[0, 0, 0], [x, 1, x], [-x, -1, -x]]) /
               (2 * x + 1))
    return F_x, F_y, B_x, B_y


def diffusion_tensor(im, fx, fy, kappa, isotropic=False):
    """
    Compute the diffusion tensor for the given image.

    Parameters
    ----------
    im : ndarray
        The image
    fx : ndarray
        Convolution filter for the x component of the gradient
    fy : ndarray
        Convolution filter for the y component of the gradient
    kappa : float
        The diffusion parameter
    isotropic : bool
        Use isotropic instead of anisotropic diffusion

    Returns
    -------
    D11, D22, D12 : ndarray
        The components of the diffusion tensor
    """
    
    im = im.copy()

    gx = arraylib.zeros(im.shape)
    gy = arraylib.zeros(im.shape)

    if isotropic:

        for c in range(3):
            gx[..., c] = correlate2d(im.sum(2), fx, 'same', 'symm')
            gy[..., c] = correlate2d(im.sum(2), fy, 'same', 'symm')

        gradsq = (gx**2 + gy**2).sum(2)

        D11 = 1 / (1 + kappa * gradsq**2)
        D22 = D11.copy()
        D12 = arraylib.zeros(D11.shape)
                    
    else:

        for c in range(3):
            gx[..., c] = arraylib.asarray(filter2d(torch.as_tensor(im[..., c][:, None]), torch.as_tensor(fx)[None], padding='same', border_type='reflect')).squeeze(1)
            gy[..., c] = arraylib.asarray(filter2d(torch.as_tensor(im[..., c][:, None]), torch.as_tensor(fy)[None], padding='same', border_type='reflect')).squeeze(1)

            S11 = (gx**2).sum(-1)
            S12 = (gx * gy).sum(-1)
            S22 = (gy**2).sum(-1)

        # Eigenvalues and eigenvectors of the structure tensor

        lambda1 = .5 * (S11 + S22 + arraylib.sqrt((S11 - S22)**2 + 4 * S12**2))
        lambda2 = .5 * (S11 + S22 - arraylib.sqrt((S11 - S22)**2 + 4 * S12**2))

        theta1 = .5 * arraylib.arctan2(2 * S12, S11 - S22)
        theta2 = theta1 + arraylib.pi / 2

        v1x = arraylib.cos(theta1)
        v1y = arraylib.sin(theta1)
        v2x = arraylib.cos(theta2)
        v2y = arraylib.sin(theta2)

        # Diffusion tensor

        Dlambda1 = 1 / (1 + kappa * lambda1**2)
        Dlambda2 = 1 / (1 + kappa * lambda2**2)

        D11 = Dlambda1 * v1x**2 + Dlambda2 * v2x**2
        D22 = Dlambda1 * v1y**2 + Dlambda2 * v2y**2
        D12 = Dlambda1 * v1x * v1y + Dlambda2 * v2x * v2y

    return D11, D22, D12


def daltonise_simple(im, sim_function):
    """
    Simple baseline daltonisation algorithm. Mainly for use as initial value.

    Parameters
    ----------
    im : ndarray
        The input image
    sim_function : func
        The CVD simulation function

    Returns
    -------
    sdalt : ndarray
        The daltonised image
    """
    
    sim = sim_function(im)
    _, ed, ec = unit_vectors(im, sim)
    
    d = ((im - sim) * ed[:, None, None]).sum(axis=-1)

    dalt = im.copy()
    
    for i in range(3):
        dalt[..., i] += d * ec[:, None, None, i]
    
    dalt[dalt < 0] = 0
    dalt[dalt > 1] = 1

    return dalt


def construct_gradient(im, sim, fx, fy, simple=True):
    """
    Construct the gradient field for the daltonised image

    Parameters
    ----------
    im : ndarray
        The input image
    sim : ndarray
        The CVD simulation of the input image
    fx, fy : ndarray
        Convolution filters for the x and y components of the gradient
    simple : bool
        Use a simplified version of the gradient
    
    Returns
    -------
    Gx, Gy : ndarray
        The components of the constructed gradient field
    """
    
    _, ed, ec = unit_vectors(im, sim)

    gx = arraylib.zeros(im.shape)
    gy = arraylib.zeros(im.shape)

    for c in range(im.shape[-1]):

        gx[..., c] = arraylib.asarray(filter2d(torch.as_tensor(im[..., c][:, None]), torch.as_tensor(fx)[None], padding='same', border_type='reflect')).squeeze(1)
        gy[..., c] = arraylib.asarray(filter2d(torch.as_tensor(im[..., c][:, None]), torch.as_tensor(fy)[None], padding='same', border_type='reflect')).squeeze(1)

    if simple: # as described in Farup, 2020

        Gx = gx + arraylib.einsum('nijk,nk,nl->nijl', gx, ed, ec)
        Gy = gy + arraylib.einsum('nijk,nk,nl->nijl', gy, ed, ec)

    else: # as described in Simon, J. Percept. Imag., 2018

        gsimx = arraylib.zeros(im.shape)
        gsimy = arraylib.zeros(im.shape)

        for c in range(im.shape[2]):

            gsimx[..., c] = correlate2d(sim[..., c], fx, 'same', 'symm')
            gsimy[..., c] = correlate2d(sim[..., c], fy, 'same', 'symm')

        ax = arraylib.einsum('ijk,k,l', gx, ed, ec)
        ay = arraylib.einsum('ijk,k,l', gy, ed, ec)
        a = arraylib.einsum('ijk,ijk->ij', ax, ax) + arraylib.einsum('ijk,ijk->ij', ay, ay)

        b = 2 * (arraylib.einsum('ijk,ijk->ij', ax, gsimx) + arraylib.einsum('ijk,ijk->ij', ay, gsimy))

        c = (arraylib.einsum('ijk,ijk->ij', gsimx, gsimx) + arraylib.einsum('ijk,ijk->ij', gsimy, gsimy) -
             arraylib.einsum('ijk,ijk->ij', gx, gx) - arraylib.einsum('ijk,ijk->ij', gy, gy))

        b2m4ac = b**2 - 4 * a * c
        b2m4ac[b2m4ac < 0] = 0
        a[a <= 0] == 1e-15

        chi_p = -b + arraylib.sqrt(b2m4ac) / 2 * a
        chi_n = -b - arraylib.sqrt(b2m4ac) / 2 * a
        print(chi_p.sum(), chi_n.sum())
        if abs(chi_p.sum()) < abs(chi_n.sum()):
            chi = chi_p
            print('chi_p')
        else:
            chi = chi_n
            print('chi_n')

        chi_stack = arraylib.stack((chi, chi, chi), 2)

        Gx = gx + chi_stack * arraylib.einsum('ijk,k,l', gx, ed, ec)
        Gy = gy + chi_stack * arraylib.einsum('ijk,k,l', gy, ed, ec)
        
    return Gx, Gy


def daltonise_poisson(im, sim_function, nit=501, diff='FB', save=None, save_every=100):
    """
    Daltonise with the Poisson method of Simon and Farup, J. Percept. Imag., 2018

    Parameters
    ----------
    im : ndarray
        The input image to be daltonised
    sim_function : func
        The CVD simulation function
    nit : int
        Number of iterations
    diff : str
        The type of difference convolution filters (see diff_filters)
    save : str
        Filenamebase (e.g., 'im-%03d.png') or None
    save_evry : int
        Save every n iterations

    Returns
    -------
    pdalt : ndarray
        The daltonised image
    """
    
    sim = sim_function(im)
    sdalt = daltonise_simple(im, sim_function)
    fx, fy, bx, by = diff_filters(diff)
    Gx, Gy = construct_gradient(im, sim, fx, fy)
    gx = arraylib.zeros(im.shape)
    gy = arraylib.zeros(im.shape)
    pdalt = sdalt.copy()

    if save:
        plt.imsave(save % 0, pdalt)

    for i in range(nit):
        for c in range(im.shape[2]):
            gx[..., c] = correlate2d(pdalt[..., c], fx, 'same', 'symm')
            gy[..., c] = correlate2d(pdalt[..., c], fy, 'same', 'symm')
        
            pdalt[..., c] += .24 * (correlate2d(gx[..., c] - Gx[..., c], bx, 'same', 'symm') + 
                                    correlate2d(gy[..., c] - Gy[..., c], by, 'same', 'symm'))
        
        pdalt[pdalt < 0] = 0
        pdalt[pdalt > 1] = 1

        if save and i % save_every == 0:
            plt.imsave(save % i, pdalt)

    return pdalt

def kornia_correlate(im, f):
    return arraylib.asarray(filter2d(torch.as_tensor(im[:, None]), torch.as_tensor(f), padding='same', border_type='reflect')).squeeze(1)


def daltonise_anisotropic(im, sim_function, nit=501, kappa=1e4, diff='FB',
                          save_dir=None, save_every=100,
                          isotropic=False, debug=False,
                          *args, **kwargs):
    """
    Map the image to the spatial gamut defined by wp and bp.

    Parameters
    ----------
    im : ndarray (M x N x 3)
        The original image
    sim_function: func
        The CVD simulation function
    nit : int
        Number of iterations
    kappa : float
        anisotropy parameter
    diff : str
       finite difference method (FB, cent, Sobel, SobelFB, Feldman,
       FeldmanFB)
    isotropic : bool
       isotropic instead of anisotropi
    debug : bool
       print number of iterations every 10

    Returns
    -------
    adalt : ndarray
        The daltonised image
    """

    # Initialize

    sim = sim_function(im)
    adalt = daltonise_simple(im, sim_function)

    fx, fy, bx, by = diff_filters(diff)
    gx = arraylib.zeros(im.shape)
    gy = arraylib.zeros(im.shape)
    gxx = arraylib.zeros(im.shape)
    gyy = arraylib.zeros(im.shape)
    Gx, Gy = construct_gradient(im, sim, fx, fy)

    D11, D22, D12 = diffusion_tensor(im, fx, fy, kappa, isotropic)

    # Iterate

    for i in tqdm.trange(nit):

        if (i % 10 == 0) and debug: print(i)

        # Anisotropic diffusion

        for c in range(3):
            gx[..., c] = kornia_correlate(adalt[..., c], fx[None])
            gy[..., c] = kornia_correlate(adalt[..., c], fy[None])
            gxx[..., c] = kornia_correlate(D11 * (gx[..., c] - Gx[..., c]) +
                                      D12 * (gy[..., c] - Gy[..., c]),
                                      bx[None])
            gyy[..., c] = kornia_correlate(D12 * (gx[..., c] - Gx[..., c]) +
                                      D22 * (gy[..., c] - Gy[..., c]),
                                      by[None])

        adalt += .24 * (gxx + gyy)

        adalt[adalt < 0] = 0
        adalt[adalt > 1] = 1
            
    offset = adalt-im

    return adalt, sim, sim_function(adalt), offset

if __name__ == '__main__':
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default='places',
                    choices=['abstract_art', 'custom', 'flowers', 'places', 'wikiart_abstract', 'wikiart_landscape', 'wikiart_still_life'],
                    help="Specify which of the candidate datasets to evaluate over"
                    )
    parser.add_argument('--dataset_split', type=int, default=0, help="Specify which split of the dataset to evaluate over")
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--cvd_type', type=str, default='protan', choices=['protan', 'deutan', 'tritan'])
    parser.add_argument('--severity', type=float, default=1.0, choices=[0.4, 0.6, 1.0], help="CVD severity, with 0.4, 0.6, 1.0 corresponding to 8, 12, 20nm")

    parser.add_argument('--diff', type=str, default='FB', choices='FB,cent,Sobel,SobelFB,Feldman,FeldmanFB'.split(','))
    parser.add_argument('--nit', type=int, default=501)

    args = parser.parse_args()

    kwargs = vars(args)

    if args.dataset_type == 'custom':
        dataset = CustomDataset()
    else:
        dataset = Dataset(**vars(args))

    im_size = (256, 256)
    
    def im_transform(im):
        return np.array(im.resize(im_size))

    dataset.transform = im_transform

    num_cpus = os.cpu_count()
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_cpus-1,
        drop_last=False,
    )

    os.makedirs('farup_results', exist_ok=True)

    cvd_matrix = arraylib.array(CVD_MATRICES[args.severity][args.cvd_type])
    def simulate(im):
        return arraylib.einsum('ijkl,ml->ijkm', im, cvd_matrix)

    image_indices = []
    input_images = []
    recoloured_images = []
    cvd_input_images = []
    cvd_recoloured_images = []
    offsets = []

    for image_idx, input_image in iter(dataloader):
        input_image = arraylib.array(input_image).astype(float) / 255.0
        result = daltonise_anisotropic(input_image, simulate, save_dir='farup_results', **kwargs)
        recoloured_image, cvd_input_image, cvd_recoloured_image, offset = result

        image_indices.append(arraylib.array(image_idx))
        input_images.append(input_image)
        recoloured_images.append(recoloured_image)
        cvd_input_images.append(cvd_input_image)
        cvd_recoloured_images.append(cvd_recoloured_image)
        offsets.append(offset)
        
    image_indices = arraylib.concatenate(image_indices)
    input_images = arraylib.vstack(input_images)
    recoloured_images = arraylib.vstack(recoloured_images)
    cvd_images = arraylib.vstack(cvd_input_images)
    cvd_recoloured_images = arraylib.vstack(cvd_recoloured_images)
    offsets = arraylib.vstack(offsets)

    cvd_images_t = torch.as_tensor(cvd_images).permute(0, 3, 1, 2)
    input_images_t = torch.as_tensor(input_images).permute(0, 3, 1, 2)
    cvd_recoloured_images_t = torch.as_tensor(cvd_recoloured_images).permute(0, 3, 1, 2)
    recoloured_images_t = torch.as_tensor(recoloured_images).permute(0, 3, 1, 2)

    data_zip = zip(cvd_images_t, input_images_t, cvd_recoloured_images_t, recoloured_images_t)

    base_cvd_base_error = []
    daltonised_base_error = []
    dalt_boost = []
    offset_energy = []
    lc_base = []
    lc_recolour = []

    for data in tqdm.tqdm(data_zip, total=cvd_images_t.shape[0]):
        cvd_image, input_image, cvd_recoloured_image, recoloured_image = data
        base_cvd_base_error_im = get_contrast_metric(cvd_image[None].to('cuda'), input_image[None].to('cuda')).mean().item()
        daltonised_base_error_im = get_contrast_metric(cvd_recoloured_image[None].to('cuda'), input_image[None].to('cuda')).mean().item()

        offset_image = cvd_recoloured_image - input_image
        lc_im, _, offset_e = get_cvd_swin_metric(input_image[None], cvd_recoloured_image[None], offset_image[None])

        base_cvd_base_error.append(base_cvd_base_error_im)
        daltonised_base_error.append(daltonised_base_error_im)
        dalt_boost.append(daltonised_base_error_im - base_cvd_base_error_im)
        offset_energy.append(offset_e.item())
        lc_recolour.append(lc_im.item())

        lc_im, _, _ = get_cvd_swin_metric(input_image[None], cvd_image[None], offset_image[None])
        lc_base.append(lc_im.item())


    output_folder = os.path.join('farup_results')
    os.makedirs(output_folder, exist_ok=True)
    output_filename = os.path.join(output_folder, f'{args.dataset_type}_metric_results_{args.cvd_type}_{args.severity}_ws_4_split_{args.dataset_split}.parquet')
    pd.DataFrame({
        'base_cvd_base_error': base_cvd_base_error,
        'daltonised_base_error': daltonised_base_error,
        'dalt_boost': dalt_boost,
        'im_idx': image_indices.get(),
        'offset_energy': offset_energy,
        'lc_recolour': lc_recolour,
        'lc_base': lc_base,
        }).to_parquet(output_filename)
    
    results_folder = os.path.join('farup_results', f'{args.dataset_type}_{args.cvd_type}_{args.severity}_split_{args.dataset_split}', 'images')
    os.makedirs(results_folder, exist_ok=True)

    for im_idx, input_image, modified_image, total_offset in zip(image_indices, input_images, recoloured_images, offsets):
        data = {
            'input_image': input_image.get(),
            'image': modified_image.get(),
            'offset': total_offset.get(),
        }
        output_filename = os.path.basename(dataset.filenames[im_idx.item()]) + '.npz'
        np.savez_compressed(os.path.join(results_folder, output_filename), **data)

    if args.save_images:

        num_im = input_images.shape[0]
        num_im_j = int(np.ceil(np.sqrt(num_im)))
        num_im_i = int(np.ceil(num_im / num_im_j))

        axis_spacing = 6
        fig = plt.figure(figsize=(axis_spacing*4*num_im_j, axis_spacing*2*num_im_i))
        gs = fig.add_gridspec(num_im_i, num_im_j)

        for idx, data in enumerate(zip(input_images, cvd_images, recoloured_images, cvd_recoloured_images, offsets, offset_energy, dalt_boost)):
            input_image, cvd_image, recoloured_image, cvd_recoloured_image, offset_image, offset_e, dalt_boost_im = data
            gs_i = idx // num_im_j
            gs_j = idx % num_im_j

            sub_gs = gs[gs_i, gs_j].subgridspec(2, 5)
            ax = [
                fig.add_subplot(sub_gs[0, 1]),
                fig.add_subplot(sub_gs[0, 3]),
                fig.add_subplot(sub_gs[0, 2]),
                fig.add_subplot(sub_gs[0, 4]),
                fig.add_subplot(sub_gs[1, 1]),
                fig.add_subplot(sub_gs[1, 3]),
                fig.add_subplot(sub_gs[1, 2]),
                fig.add_subplot(sub_gs[1, 4]),
                fig.add_subplot(sub_gs[1, 0]),
            ]

            input_im_local_contrast = local_contrast(rgb_to_lab(torch.as_tensor(input_image).permute(2, 0, 1))).mean(dim=-1)
            output_im_local_contrast = local_contrast(rgb_to_lab(torch.as_tensor(recoloured_image).permute(2, 0, 1))).mean(dim=-1)

            input_im_cvd_local_contrast = local_contrast(rgb_to_lab(torch.as_tensor(cvd_image).permute(2, 0, 1))).mean(dim=-1)
            output_im_cvd_local_contrast = local_contrast(rgb_to_lab(torch.as_tensor(cvd_recoloured_image).permute(2, 0, 1))).mean(dim=-1)

            cvd_images_min = cvd_images.min()
            cvd_images_max = cvd_images.max()
            cvd_images = (cvd_images - cvd_images_min) / (cvd_images_max - cvd_images_min)

            cvd_recoloured_image_min = cvd_recoloured_image.min()
            cvd_recoloured_image_max = cvd_recoloured_image.max()
            cvd_recoloured_image = (cvd_recoloured_image - cvd_recoloured_image_min) / (cvd_recoloured_image_max - cvd_recoloured_image_min)

            ax[0].imshow(input_image.get())
            ax[0].set_title(f'Input', loc='left')
            ax[0].set_xlabel(f'Image {idx}')
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            
            ax[1].imshow(input_im_local_contrast)
            ax[1].set_title(f'Input Local Contrast - {input_im_local_contrast.mean():.04f}', loc='left')
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            
            ax[2].imshow(cvd_image.get())
            ax[2].set_title(f'Input CVD', loc='left')
            ax[2].set_xticks([])
            ax[2].set_yticks([])
            
            ax[3].imshow(input_im_cvd_local_contrast)
            ax[3].set_title(f'Input CVD Local Contrast - {input_im_cvd_local_contrast.mean():.04f}', loc='left')
            ax[3].set_xticks([])
            ax[3].set_yticks([])
            
            ax[4].imshow(recoloured_image.get())
            ax[4].set_title(f'Output - {dalt_boost_im:.04f}', loc='right')
            ax[4].set_xticks([])
            ax[4].set_yticks([])
            
            ax[5].imshow(output_im_local_contrast)
            ax[5].set_title(f'Output Local Contrast - {output_im_local_contrast.mean():.04f}', loc='left')
            ax[5].set_xticks([])
            ax[5].set_yticks([])
                
            ax[6].imshow(cvd_recoloured_image.get())
            ax[6].set_title(f'Output CVD', loc='right')
            ax[6].set_xticks([])
            ax[6].set_yticks([])
            
            ax[7].imshow(output_im_cvd_local_contrast)
            ax[7].set_title(f'Output CVD Local Contrast - {output_im_cvd_local_contrast.mean():.04f}', loc='left')
            ax[7].set_xticks([])
            ax[7].set_yticks([])
            
            offset_min = offset_image.min()
            offset_max = offset_image.max()
            scaled_offset = (offset_image - offset_min) / (offset_max - offset_min)
            output_energy = arraylib.sqrt(arraylib.power(recoloured_image, 2).sum(axis=-1)).mean()
            ax[8].imshow(scaled_offset.get(), vmin=0, vmax=1)
            ax[8].set_xlabel(f'Offset Energy - {offset_e:.04f} - {output_energy.item():.04f} - {offset_e/output_energy.item():.04f}', loc='left')
            ax[8].set_xticks([])
            ax[8].set_yticks([])

        plt.tight_layout()
        
        os.makedirs(os.path.join('farup_results', 'images'), exist_ok=True)
        output_filename = os.path.join(output_folder, 'images', f'{args.dataset_type}_results_{args.cvd_type}_{args.severity}_split_{args.dataset_split}.png')
        plt.savefig(output_filename)
        output_filename = os.path.join(output_folder, 'images', f'{args.dataset_type}_results_{args.cvd_type}_{args.severity}_split_{args.dataset_split}.pdf')
        plt.savefig(output_filename)  
