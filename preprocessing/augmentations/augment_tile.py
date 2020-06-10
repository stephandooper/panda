# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:52:55 2020

@author: Stephan
"""

import tensorflow as tf
from preprocessing.utils.mat_transforms import tf_tile_concat


def augment_tile(image, label, aug_func):
    
    nt, h, w, c = image.shape
    image = tf_tile_concat(image)
    # augmentation operations
    image = tf.image.convert_image_dtype(image, tf.uint8)
    #shape = image.shape
    image = tf.numpy_function(aug_func, [image], tf.uint8, name='aug')
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.reshape(image, (nt, h, w, c))
    image.set_shape((nt, h, w, c))
    return image, label