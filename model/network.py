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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        TensorBoard)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow_addons.metrics import CohenKappa

import pandas as pd
import os

import sys
sys.path.insert(0, '..')


class Network(object):
    """ A basic wrapper for a Keras Model object.
    
    This is a convenience class meant to keep all of the most basic and
    common operations in one place, to avoid unnecessary code repetition
    
    A training routine often involves defining a loss function, optimizers,
    compilations, adding commonly used callbacks, and finally training.
    This can be done in one place, without losing flexibility: with function
    arguments, custom losses, optimizers can be defined, overriding the default
    as well ass adding extra callbacks
    such as callbacks, optimizers, losses, compiling, and fitting.
    There is a also a function
    
    """
    
    def __init__(self, model, *args, **kwargs):
        
        self._model = model
        # TODO: root files for weights and tensorboard?
        
    
    @property
    def model(self):
        """ Return the inference model.

        Returns
        -------
        Keras Model object
            The Keras model

        """
        return self._model
    
    def load_weights(self, path, by_name=False, skip_mismatch=False):
        """ Load weights of the Keras Model.

        Parameters
        ----------
        path : str or Pathlib Path
            A path to the Keras model weights (.h5 file)
        by_name : bool, optional
            Whether to load the weights by layer names. All non-matches are
            skipped. The default is False.
        skip_mismatch : bool, optional
            Skip loading weights for layers where the size of layers in the
            weights file do no match that of the model. The default is False.

        Returns
        -------
        None.

        """
        self._model.load_weights(path, by_name=False, skip_mismatch=False)
        
    def train(self, dataset,
              val_dataset,
              epochs,
              learning_rate,
              num_classes=6,
              sparse_labels=False,
              regression=False,
              custom_callbacks=[],
              custom_metrics=[],
              custom_loss=None,
              custom_optimizer=None,
              save_weights_name='best_weights.h5',
              tb_logdir="./",
              **kwargs):
        """ Train the model.

        Parameters
        ----------
        dataset : tf Dataset
            A tf dataset that returns tuples (x, y) or (x, y, sample_weight).
        val_dataset : tf Dataset
            A tf dataset that returns tuples (x, y) or (x, y, sample_weight).
            This dataset is used exclusively for validation
        epochs : int
            The number of epochs.
        learning_rate : float
            The learning rate for the optimizer
        num_classes : int, optional
            The number of classes for the cohen kappa metric. The default is 6.
        sparse_labels : bool, optional
            Whether to use sparse label representation. The default is False.
        regression : bool, optional
            If the neural network is a regression framework. If True, then the
            final layer can only have 1 unit. The default is False.
        custom_callbacks : list, optional
            A list of keras callbacks, custom written callbacks are also
            supported, as long as they inherit from Callback.
            The default is [].
        custom_metrics : list, optional
            A list of metrics that can be passed to Keras. The default is [].
        custom_loss : Loss object, optional
            A Keras (Tensorflow) loss object. The default is None.
        custom_optimizer : A (Tensorflow) Keras optimizer, optional
            Custom optimizer, must be supported by Keras. The default is None.
        save_weights_name : string, optional
            The name of the weights file. The default is 'weights.h5'.
        tb_logdir : string
            The location of the tensorboard logs
        **kwargs : kwargs
            other keyworded arguments, currently unused.

        Returns
        -------
        None.

        """
        # ---------------------------
        # Metrics
        # ---------------------------
        
        # QWK: the target metric
        # TODO: num classes dependent on output shape of the model
        qwk = CohenKappa(num_classes=num_classes,
                         name='cohen_kappa',
                         weightage='quadratic',
                         sparse_labels=False,
                         regression=False)
        
        if not isinstance(custom_metrics, list):
            custom_metrics = [custom_metrics]
        
        metrics = [qwk] + custom_metrics
        # ---------------------------
        # Callbacks
        # ---------------------------
        
        # TODO: file name splitting and adding
        
        # Save based on loss
        # TODO: fill in
        root, ext = os.path.splitext(save_weights_name)
        ckp_best_loss = ModelCheckpoint(filepath=root + "_bestLoss" + ext,
                                        monitor='val_loss',
                                        verbose=0,
                                        save_best_only=True,
                                        save_weights_only=True)
        
        # Save based on metric
        # TODO: fill in
        ckp_best_metric = ModelCheckpoint(filepath=root + "_bestQWK" + ext,
                                        monitor='val_cohen_kappa',
                                        verbose=1,
                                        save_best_only=True,
                                        mode='max',
                                        save_weights_only=True)
        
        # TODO: add early stopping?
        
        # Tensorboard instance for tracking metrics and progress
        # TODO: configure
        tensorboard = TensorBoard(log_dir=tb_logdir,
                                     histogram_freq=0,
                                     # write_batch_performance=True,
                                     write_graph=True,
                                     write_images=False)
        
        if not isinstance(custom_callbacks, list):
            custom_callbacks = [custom_callbacks]
        
        callbacks = [ckp_best_loss, ckp_best_metric, tensorboard] + \
            custom_callbacks
            
        # ---------------------------
        # Compile model
        # ---------------------------
        
        # TODO: lr args
        # TODO: control flow
        optimizer = Adam(lr=1e-4)
        
        # TODO: fill in
        # TODO: control flow
        loss = CategoricalCrossentropy()

        self._model.compile(loss=loss,
                            optimizer=optimizer,
                            metrics=metrics)
        
        # ---------------------------
        # fit
        # ---------------------------
        # Note: some args are not supported with tf data (processing differs)
        
        # TODO: check performance without repeat on dataset
        # TODO: this way, steps per epoch args no longer needed
        
        self._model.fit(x=dataset,
                        epochs=epochs,
                        validation_data=val_dataset,
                        #validation_steps=validation_steps, 
                        #steps_per_epoch=steps_per_epoch,
                        callbacks=callbacks)
        
    
    def make_test_submission(dataset, **kwargs):
        """ Create test submissions for Kaggle PANDA.
        
        Parameters
        ----------
        dataset : tf Dataset
            A Tensorflow Dataset that returns (a batch) of [W,H,C] images
        **kwargs : Other keyword arguments
            Other keyword arguments, not used yet.

        Returns
        -------
        None.

        """
        pass
