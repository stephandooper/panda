# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 12:27:55 2020

@author: Stephan

a slightly different version that works for probabilities
The docs in tensorflow assume labels (integer), but model outputs are often
probabilities [0,1] -> R, so adjust this

For mmore information on tensorflow iou metric
https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanIoU

For the specific SO post with the proposed fix
https://stackoverflow.com/questions/60507120/how-to-correctly-use-the-tensorflow-meaniou-metric


"""

import tensorflow as tf

class MeanIoU_prob(tf.keras.metrics.MeanIoU):
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), 
                                    tf.argmax(y_pred, axis=-1), 
                                    sample_weight)
