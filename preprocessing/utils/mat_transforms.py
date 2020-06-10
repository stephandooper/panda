# -*- coding: utf-8 -*-
"""
Created on Tue May 26 01:28:32 2020

@author: Stephan
"""

import numpy as np
import tensorflow as tf


def tf_tile2mat(image, row, col):
    """ A tensorflow version of tile2mat. Converts tiles into a single matrix

    Parameters
    ----------
    image : Tensorflow Array
        A [NT, H, W, C] array of tile images
    row : int, 
        The number of tiles to place in the first dimension (rows/height)
    col : TYPE
        The number of tiles to place in the second dimension (columns/ width)

    Returns
    -------
    image : Tensorflow Array
        A [H * row, W * col, C] Tensorflow array  containing the tiled images
        as a single big image.

    """
    # TODO: refactor this into a separate function
    nt, w ,h ,c = image.shape
    # square root of num tiles

    # reshape to [row, col, w, h, c]
    image = tf.reshape(image, (row, col, w, h, c))
    
    # transpose to [w, row, h, col, c]
    image = tf.transpose(image,(0, 2, 1, 3, 4))
    
    # reshape to [w * row, h * col, c]
    # The image is now all tiles stacked together into a row*col grid
    image = tf.reshape(image,(w*row, h*col, c))
    
    return image

def tf_tile2square(image):
    """ A square version of the tile2mat. Only works if sqrt(num_tiles) = int
    
    This method only works when the square root of num_tiles is a natural 
    number

    Parameters
    ----------
    image : Tensorflow Array
        A [NT, H, W, C] array of tile images

    Returns
    -------
    image : Tensorflow Array
        A [H * sqnt, W * sqnt, C] Tensorflow array containing the tiled images
        as a single big image.

    """
    num_tiles = tf.cast(image.shape[0], tf.int32)
    sqnt = tf.cast(tf.math.sqrt(num_tiles), dtype=tf.int32)
    return tf_tile2mat(image, sqnt, sqnt)
    


def tf_tile_concat(image):
    """Concatenates tiles into a single rectangular image in the row dimension

    Parameters
    ----------
    image : Tensorflow array
        A [NT, H, W, C] Tensorflow array containing the image tiles

    Returns
    -------
    image : A [W * NT, H, C] image concatening the tile along the first 
    dimension (height, or number of rows)
        DESCRIPTION.

    """
    # TODO: refactor this into a separate function
    dims = tf.shape(image)
    nt, w, h, c = dims[0], dims[1], dims[2], dims[3]
    
    # square root of num tiles
    #row = tf.cast(tf.math.sqrt(tf.cast(nt, tf.float32)), tf.int32)
    #col = tf.cast((nt / row), dtype=tf.int32)
    
    # ensure that the ordering is the same through transpose
    # in numpy this is fixed through Fortran indexing in reshape
    image = tf.transpose(image, (0, 2, 1, 3))
    
    # reshape to [nt, w, h, c]
    image = tf.reshape(image, (nt, w, h, c))

    # transpose to [nt, h, w, c]
    image = tf.transpose(image, (0, 2, 1, 3))
    
    # reshape to [nt*w, h, c]
    # The image is now all tiles stacked together into a row*col grid
    image = tf.reshape(image, (nt* w, h, c))

    return image


def tile2mat(image, row, col):
    """  Converts tiles into a single matrix

    Parameters
    ----------
    image : Array
        A [NT, H, W, C] array of tile images
    row : int, 
        The number of tiles to place in the first dimension (rows/height)
    col : TYPE
        The number of tiles to place in the second dimension (columns/ width)

    Returns
    -------
    image : Array
        A [H * row, W * col, C] containing the tiled images
        as a single big image.

    """
    # TODO: refactor this into a separate function
    nt, w ,h ,c = image.shape
    # square root of num tiles

    # reshape to [row, col, w, h, c]
    image = image.reshape((row, col, w, h, c))
    
    # transpose to [w, row, h, col, c]
    image = image.transpose((0, 2, 1, 3, 4))
    
    # reshape to [w * row, h * col, c]
    # The image is now all tiles stacked together into a row*col grid
    image = image.reshape((w*row, h*col, c))
    
    return image


# DO NOT USE WHEN SHAPES CHANGE THROUGH AUGMENTATION (ROTATIONS)
def tf_mat2tile(image, num_splits):
    """ DEPRECATED

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    num_splits : TYPE
        DESCRIPTION.

    Returns
    -------
    image : TYPE
        DESCRIPTION.
    row : TYPE
        DESCRIPTION.
    col : TYPE
        DESCRIPTION.

    """
    
    dims = image.shape
    row, col = num_splits
    split_size_row = tf.cast(dims[0] / row, tf.int32)
    split_size_col = tf.cast(dims[1] / col, tf.int32)  

    image = tf.expand_dims(image, 0)
    image = tf.concat(tf.split(image, tf.repeat(split_size_row, row), axis=1), axis=0)
    image = tf.concat(tf.split(image, tf.repeat(split_size_col, col), axis=2), axis=0)
    return image, row, col

def mat2tile(image, num_splits, old_shape=None):
    """ DEPRECATED

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    num_splits : TYPE
        DESCRIPTION.
    old_shape : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    image : TYPE
        DESCRIPTION.

    """
    
    if (image.shape[0] != old_shape[0]) and old_shape is not None:
        col, row = num_splits
    else:
        row, col = num_splits
    
    image = np.expand_dims(image, 0)
    image = np.concatenate(np.split(image, row, axis=1), axis=0)
    image = np.concatenate(np.split(image, col, axis=2), axis=0)
    return image