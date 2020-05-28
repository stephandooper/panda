# -*- coding: utf-8 -*-
"""
Created on Tue May 26 01:28:32 2020

@author: Stephan
"""

import numpy as np
import tensorflow as tf

def tile2mat(image):
    # TODO: refactor this into a separate function
    nt, w ,h ,c = image.shape
    # square root of num tiles
    row = int(np.sqrt(nt))
    col = int(nt / row)

    # reshape to [row, col, w, h, c]
    image = image.reshape((row, col, w, h, c), order='F')
    
    # transpose to [w, row, h, col, c]
    image = image.transpose((0, 2, 1, 3, 4))
    
    # reshape to [w * row, h * col, c]
    # The image is now all tiles stacked together into a row*col grid
    image = image.reshape((w*row, h*col, c))
    
    return image, (row, col)

def mat2tile(image, num_splits, old_shape=None):
    
    if (image.shape[0] != old_shape[0]) and old_shape is not None:
        col, row = num_splits
    else:
        row, col = num_splits
    
    image = np.expand_dims(image, 0)
    image = np.concatenate(np.split(image, row, axis=1), axis=0)
    image = np.concatenate(np.split(image, col, axis=2), axis=0)
    return image

def tf_tile2mat(image):
    # TODO: refactor this into a separate function
    dims = tf.shape(image)
    nt, w, h, c = dims[0], dims[1], dims[2], dims[3]
    
    # square root of num tiles
    row = tf.cast(tf.math.sqrt(tf.cast(nt, tf.float32)), tf.int32)
    col = tf.cast((nt / row), dtype=tf.int32)
    
    # ensure that the ordering is the same through transpose
    # in numpy this is fixed through Fortran indexing in reshape
    image = tf.transpose(image, (0, 2, 1, 3))
    
    # reshape to [col, row, w, h, c]
    image = tf.reshape(image, (col, row, w, h, c))

    # transpose to [h, col, w, row, c]
    image = tf.transpose(image, (1, 3, 0, 2, 4))
    
    # reshape to [w * row, h * col, c]
    # The image is now all tiles stacked together into a row*col grid
    image = tf.reshape(image, (w * row, h * col, c))

    return image, (row, col)

# DO NOT USE WHEN SHAPES CHANGE THROUGH AUGMENTATION (ROTATIONS)
def tf_mat2tile(image, num_splits):
    
    dims = image.shape
    row, col = num_splits
    split_size_row = tf.cast(dims[0] / row, tf.int32)
    split_size_col = tf.cast(dims[1] / col, tf.int32)  

    image = tf.expand_dims(image, 0)
    image = tf.concat(tf.split(image, tf.repeat(split_size_row, row), axis=1), axis=0)
    image = tf.concat(tf.split(image, tf.repeat(split_size_col, col), axis=2), axis=0)
    return image, row, col