# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:14:51 2020

@author: Stephan
"""

import tensorflow as tf

def set_gpu_memory(device_type='GPU'):
    """
    Configure Tensorflow Memory behaviour.
    
    Only use one GPU and do not reserve all memory
    Returns
    -------
    None.
    """
    gpus = tf.config.experimental.list_physical_devices(device_type)
    tf.config.experimental.set_visible_devices(gpus[0], device_type)
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