# -*- coding: utf-8 -*-.
"""
Created on Thu May 14 16:41:22 2020.

@author: Stephan

This file contains a convenience class that does most of the standard
processing regarding the Keras pipeline, e.g. 
    1. compiling the model
    2. Loading weights
    2. Declaring callbacks
    3. fitting
    4. creating predictions (making submissions).
    
that takes as its arguments:
    1. A keras Model instance
    2. A tf dataset, tailored to the input/output of the Model
    3. optional args, such as callbacks or loss functions. These must adhere 
        To the constraints of the model/dataset.
"""

import tensorflow as tf
from tensorflow.keras.callbacks import 


class BaseModelWrapper(object):
    
    def __init__(self, model, *args, **kwargs):
        
        self._model = model
        self._ds = ds
    
    @property
    def model(self):
        return self._model
    
    @property 
    def ds(self):
        return self._ds
    
    def load_weights(self, path, by_name=False, skip_mismatch=False):
        self._model.load_weights(path, by_name=False, skip_mismatch=False)
        
    
    
