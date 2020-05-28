# -*- coding: utf-8 -*-
"""
Created on Sat May 23 19:15:19 2020

@author: Stephan
"""
import numpy as np
import tensorflow as tf


def tf_tile(img, shape, sz=128, N=16, pad_val=255):
    """ Tensorflow version of the tiling version for on the fly generation

    Parameters
    ----------
    img : skimage.io.MultiImage
        The tiff file, read at level i.
        
    shape : The dimensions of the image
        A [W,H,C] tensor that contains the size of the image
        Tiff files have to be imported through tf.numpy_function or
        tf.py_function (tf.deocde_tiff does not work) and these functions 
        lose the image shape. See the tiff generator for more information
        
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
    
    # Padding is only needed if the image cannot be tiled exactly into N tiles
    # in this case, the image is padded with the size it would be missing
    pad0, pad1 = (sz - shape[0] % sz) % sz, (sz - shape[1] % sz) % sz
    
    w = shape[0] + pad0
    h = shape[1] + pad1
    
    # Pad the image into size [W + pad0, H + pad1, C]
    # from now on: W = W + pad0, H = H + pad1
    img = tf.pad(img,
                 [[pad0 // 2, pad0 - pad0 // 2],
                  [pad1 // 2, pad1 - pad1 // 2],
                  [0, 0]],
                 constant_values=pad_val)


    # Resize to [W // sz, sz, H // sz, sz, 3], here W//sz, H//sz is the total
    # number of tiles, and is effectively a grid
    img = tf.reshape(img, (w // sz, sz, h // sz, sz, 3))
    
    # Transpose to [W // sz, H // sz, sz, sz, 3]
    # reshape to [(W // sz) * (H // sz), sz, sz, 3],
    # tot_tiles = (W // sz) * (H // sz)
    img = tf.transpose(img, (0, 2, 1, 3, 4))
    img = tf.reshape(img, (-1, sz, sz, 3))
    
    # If the total number of tiles is smaller than the desired number of tiles
    if len(img) < N:
        # Add background tiles
        img = tf.pad(img, [[0, N - len(img)],
                           [0, 0],
                           [0, 0],
                           [0, 0]], constant_values=pad_val)
    
    # Sort indices in descending order, get tiles which are mostly dark colours.
    idxs = tf.math.reduce_sum(tf.cast(tf.reshape(img, (len(img), -1)), tf.int32),-1)
    idxs = tf.argsort(idxs, direction='ASCENDING')[:N]
     
    # Sort the images in ascending order
    img = tf.gather(img, idxs)
    img.set_shape((N,sz,sz,3))
    return img


# TODO: test this function
def tile(img,sz, N, pad_val):
    """ Tiles a Tiff image in tiles in ascending order of background existence.

    Parameters
    ----------
    img : skimage.io.MultiImage
        The tiff file, read at level i.
        
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
    
    # Sort indices in descending order, get tiles which are mostly dark colours.
    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:N]
    
    # Sort the images in ascending order
    img = img[idxs]
    
    # Append the result to the list
    for i in range(len(img)):
        result.append({'img': img[i], 'idx': i})
    return result
