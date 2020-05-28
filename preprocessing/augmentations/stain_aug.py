# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:29:01 2020

@author: Stephan
"""

from histomicstk.saliency.tissue_detection import get_tissue_mask
from skimage.transform import resize
from histomicstk.preprocessing.augmentation.color_augmentation \
    import rgb_perturb_stain_concentration
from albumentations import Lambda

def stain_aug(img, **kwargs):
    """ Stain color augmentation

    Parameters
    ----------
    img : uint8 image
        A [w,h,c] uint8 image
    **kwargs : dict
        optional kwargs for albumentations

    Returns
    -------
    augmented_rgb : uint8 image
        A [w,h,c] uint8 image with augmentations

    """

    # Get tissue masks, no background should be selected
    mask_out, _ = get_tissue_mask(
        img, 
        deconvolve_first=False,
        n_thresholding_steps=1, 
        sigma=5, min_size=30)

    # gather into one consistent mask
    mask_out = resize(
        mask_out == 0, 
        output_shape=img.shape[:2],
        order=0, 
        preserve_range=True) == 1

    # Augment the image
    augmented_rgb = rgb_perturb_stain_concentration(img, mask_out=mask_out)

    return augmented_rgb

def identity_transform(img, **kwargs):
    """ Identity transform for masks, when using stain augmentation
    
    Parameters
    ----------
    img : uint8 image
        A [w,h,c] uint8 image
    **kwargs : dict
        optional kwargs for albumentations

    Returns
    -------
    augmented_rgb : uint8 image
        A [w,h,c] uint8 image with augmentations

    """
    return img


def StainAugment(always_apply=False, p=1.0, name=None):
    return Lambda(stain_aug, 
                  identity_transform, 
                  always_apply=always_apply, 
                  p=p,
                  name=None)