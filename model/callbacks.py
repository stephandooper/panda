# -*- coding: utf-8 -*-
"""
Created on Fri May 15 00:36:32 2020

@author: Stephan
"""

from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import resource


# A custom callback that 
class MemoryCallback(Callback):
    def on_epoch_end(self, epoch, log={}):
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
