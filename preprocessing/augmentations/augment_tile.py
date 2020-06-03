# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:52:55 2020

@author: Stephan
"""

import tensorflow as tf
from preprocessing.utils.mat_transforms import mat2tile, tf_tile2mat


def augment_tile(image, label, aug_func):
    
    original_shape = image.shape
    image, num_splits = tf_tile2mat(image)
    # augmentation operations
    image = tf.image.convert_image_dtype(image, tf.uint8)
    shape = image.shape
    image = tf.numpy_function(aug_func, [image], tf.uint8, name='aug')
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.numpy_function(mat2tile, 
                              [image, num_splits, shape], 
                              tf.float32, 
                              name='mat2tile')
    image.set_shape(original_shape)
    return image, label