# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:14:51 2020.

@author: Stephan
"""

import tensorflow as tf
import numpy as np
import random
import os
import tensorflow as tf

# TODO: set_seed function
# TO

def seed_all(SEED=2020):
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(SEED)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(SEED)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(SEED)
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(SEED)
    
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def set_gpu_memory(num_gpu=1, device_type='GPU'):
    """
    Configure Tensorflow Memory behaviour.
    
    Only use one GPU and do not reserve all memory
    Returns
    -------
    None.
    """
    gpus = tf.config.experimental.list_physical_devices(device_type)
    tf.config.experimental.set_visible_devices(gpus[0:num_gpu], device_type)
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices(device_type)
            print(len(gpus), "Physical GPUs,",
                  len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


# TODO: test this function
def tile(img, mask, sz, N, pad_val):
    """ Tiles a Tiff image in tiles in ascending order of background existence.

    Parameters
    ----------
    img : skimage.io.MultiImage
        The tiff file, read at level i.
        
    mask : skimage.io.MultiImage
        The tiff file mask
        
    sz : tuple of ints
        Desired size of the image, specified in (W, H) of ints
        
    N : int
        Number of tiles to return
        
    pad_val : int
        The values to pad the image with, should be 255 for images, 0 for
        mask images.
        
    Returns
    -------
    result : dict
        A list of dictionaries, each dictionary contains an image,
        and an index
    """
    result = []
    
    # [W, H, C]
    shape = img.shape
    
    # Padding is only needed if the image cannot be tiled exactly into N tiles
    # in this case, the image is padded with the size it would be missing
    pad0, pad1 = (sz - shape[0] % sz) % sz, (sz - shape[1] % sz) % sz
    
    # Pad the image into size [W + pad0, H + pad1, C]
    # from now on: W = W + pad0, H = H + pad1
    img = np.pad(img,
                 [[pad0 // 2, pad0 - pad0 // 2],
                  [pad1 // 2, pad1 - pad1 // 2],
                  [0, 0]],
                 constant_values=pad_val)
    
    # Resize to [W // sz, sz, H // sz, sz, 3], here W//sz, H//sz is the total
    # number of tiles, and is effectively a grid
    img = img.reshape(img.shape[0] // sz, sz, img.shape[1] // sz, sz, 3)
    
    # Transpose to [W // sz, H // sz, sz, sz, 3]
    # reshape to [(W // sz) * (H // sz), sz, sz, 3],
    # tot_tiles = (W // sz) * (H // sz)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)
    
    # If the total number of tiles is smaller than the desired number of tiles
    if len(img) < N:
        # Add background tiles
        img = np.pad(img, [[0, N - len(img)],
                           [0, 0],
                           [0, 0],
                           [0, 0]], constant_values=pad_val)
    
    # Idea: background values have (255, 255, 255) values, which is the max
    # By summing all pixel values in each tile, we can distil background from
    # non-background by total pixel values
    # Reshape to [tot_tiles, sz * sz * 3]
    # Sum the last dimension, so shape is [tot_tiles, 1]
    # Get the indices of the sorted values, and reverse them
    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:N]
    
    # Sort the images in ascending order
    img = img[idxs]
    
    # Append the result to the list
    for i in range(len(img)):
        result.append({'img': img[i], 'idx': i})
    return result
