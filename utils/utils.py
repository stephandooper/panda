# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:14:51 2020.

@author: Stephan
"""

import tensorflow as tf
import numpy as np
import random
import os

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


