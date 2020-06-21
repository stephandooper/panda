# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 12:22:04 2020

@author: Stephan

Unet loss functions. 
# forked from :
https://github.com/JielongZ/full-reimplementation-of-unet/tree/master/unet

NOTE: of these losses are from tensorflow v1, and are not compatible
      with tensorflow keras version >=2.


A few of these losses work with the latest version of tf keras
# these are:
    1. the standard 'categorical_crossentropy' loss from tf keras
    2. The generalized dice loss, if the last conv layer has a softmax activ
    
The other functions have not been properly tested yet
    
"""

import tensorflow as tf

def dice_coefficient(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    y_pred_f = tf.cast(tf.greater(tf.reshape(y_pred, [-1]), 0.5), dtype=tf.float32)
    intersection = y_true_f * y_pred_f
    dice_score = 2. * (tf.reduce_sum(intersection) + 1e-9) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-9)
    return dice_score


def dice_loss(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1. - (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)

# works if activ is linear
def generalized_dice_loss(labels, logits):
    smooth = 1e-15
    #logits = tf.nn.softmax(logits)
    weights = 1. / (tf.reduce_mean(labels, axis=[0, 1, 2]) ** 2 + smooth)

    numerator = tf.reduce_sum(labels * logits, axis=[0, 1, 2])
    numerator = tf.reduce_sum(weights * numerator)

    denominator = tf.reduce_sum(labels + logits, axis=[0, 1, 2])
    denominator = tf.reduce_sum(weights * denominator)

    loss = 1. - 2. * (numerator + smooth) / (denominator + smooth)
    return loss


def pixel_wise_softmax(output_featmap):
    with tf.name_scope("pixel_wise_softmax"):
        max_val = tf.reduce_max(output_featmap, axis=3, keepdims=True)
        exponential_featmap = tf.math.exp(output_featmap - max_val)
        normalize_factor = tf.reduce_sum(exponential_featmap, axis=3, keepdims=True)
        return exponential_featmap / normalize_factor


def cross_entropy(labels, logits):
    return -tf.reduce_mean(labels * tf.math.log(tf.clip_by_value(logits, 1e-10, 1.0)), name="cross_entropy")


def multi_bce_dice_loss(y_true, y_pred):
    return 0.5 * tf.compat.v1.losses.softmax_cross_entropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def binary_bce_dice_loss(y_true, y_pred):
    return 0.5 * tf.compat.v1.losses.sigmoid_cross_entropy(y_true, y_pred) + dice_loss(y_true, y_pred)