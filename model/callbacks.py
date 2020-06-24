# -*- coding: utf-8 -*-
"""
Created on Fri May 15 00:36:32 2020.

@author: Stephan
"""

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import resource
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools
import io

import sys
sys.path.insert(0, '..')
from model.qwk_optimizer import KappaOptimizer
from sklearn.metrics import cohen_kappa_score
import copy
import os

# A custom callback that logs memory usage each epoch
# TODO: case for regression (i.e. predictions are already one dimensional)
class MemoryCallback(Callback):
    """ Memory callback for debugging and checking memory leaks
    During training with Keras
    """
    def on_epoch_end(self, epoch, log={}):
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


class CMCallback(Callback):
    """ Confusion matrix callback
    
    The confusion matrix callback should be used in combination with
    Tensorboard
    """
    def __init__(self, ds, 
                 file_writer=None, 
                 normalize=None, 
                 regression=False, 
                 qwk_threshold=[0.5, 1.5, 2.5, 3.5, 4.5]):
        """ Class initializer

        Parameters
        ----------
        ds : Tf Dataset
            The validation dataset.
        file_writer : TF summary object, optional
            The tf summary object to write the CM to. The default is None.
        normalize : ('all', 'pred', 'True'), optional
            Whether to normalize the confusion matrix. The default is None

        Returns
        -------
        None.

        """
        super().__init__()
        self.normalize = normalize
        self.file_writer = file_writer
        self.ds = ds
        self.regression = regression
        self.qwk_threshold = qwk_threshold
        
    def on_epoch_end(self, epoch, logs={}):
        """ Declare confusion matrix on epoch end

        Parameters
        ----------
        epoch : int
            The epoch, filled in by Keras.fit automatically
        logs : dict, optional
            the logs from Keras. The default is {}.

        Returns
        -------
        None.

        """

        # Log the confusion matrix as an image summary.
        cm_image = self.log_confusion_matrix(self.model, self.ds)
        
        with self.file_writer.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)
              
    def log_confusion_matrix(self, model, val_ds):
        """ Calculate and figurize the confusion matrix

        Parameters
        ----------
        model : TYPE
            DESCRIPTION.
        val_ds : TYPE
            DESCRIPTION.

        Returns
        -------
        cm_image : TYPE
            DESCRIPTION.

        """
        
        preds = []
        labels = []
        for img, label in val_ds:
            preds.append(model.predict(img))
            labels.append(label)
        
        
        if self.regression:
            # Flatten to [num_samples] np arrays (1 D)
            preds = np.concatenate( preds, axis=0 )
            labels = np.concatenate( labels, axis=0 )
            # The predictions are not thresholded, do this
            preds = KappaOptimizer().predict(preds)
        
        else:
            preds = np.argmax(np.concatenate( preds, axis=0 ), axis=1)
            labels = np.argmax(np.concatenate(labels, axis=0), axis=1)
        
        # Calculate the confusion matrix.
        cm = confusion_matrix(labels, preds)
        # Log the confusion matrix as an image summary.
        class_names = ['0', '1', '2', '3', '4', '5']
        figure = self.plot_confusion_matrix(cm,
                                            class_names=class_names)
        
        cm_image = self.plot_to_image(figure)
        
        return cm_image
        
    def plot_confusion_matrix(self, cm, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.
          
        Args:
          cm (array, shape = [n, n]): a confusion matrix of integer classes
          class_names (array, shape = [n]): String names of the integer classes
        """
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
          
        # Normalize the confusion matrix.
        # cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
          
        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
          
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call.
        """
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image
    
    

class CohenKappa(tf.keras.callbacks.Callback):
    '''
    Computes Cohen's Kappa score with quadratic weighting for one hot encoded data.
    data is the data set to calculate scores.
    labels is the set of labels corresponding to the data provided.
    num_classes is the number of classes in the data.
    file_path is the path to save model.
    '''
    def __init__(self,  data, labels, num_classes=6, file_path='best_weights.h5', **kwargs):
        super(CohenKappa, self).__init__(**kwargs)
        self.data = data
        
        
        root, ext = os.path.splitext(file_path)
        self.file_path = root + "_bestQWK" + ext
        self.num_classes = num_classes
        self.scores = []
        self.y_true = labels

    def on_train_begin(self, logs=None):
        self.best = np.NINF

    def on_epoch_end(self, epoch, logs=None):
        thresholds = [0.5, 1.5, 2.5, 3.5, 4.5]
        y_pred = self.model.predict(self.data, verbose=1)
        y_pred = self.thresh2int(thresholds, y_pred)

        QWK = cohen_kappa_score(self.y_true, y_pred, weights='quadratic')
        if np.greater(QWK, self.best):
            print(f'\nModel improved from {self.best} to {QWK}. Saving model to {self.file_path}')
            self.best = QWK
            self.model.save_weights(self.file_path)
        else:
            print(f'\nModel did not improve from {self.best}. Kappa score {QWK}')
        self.scores.append(QWK)
        
    def thresh2int(self,thresholds, preds_raw):
        y_hat=copy.deepcopy(preds_raw)

        for i,pred in enumerate(y_hat):
            if   pred < thresholds[0]: y_hat[i] = 0
            elif pred < thresholds[1]: y_hat[i] = 1
            elif pred < thresholds[2]: y_hat[i] = 2
            elif pred < thresholds[3]: y_hat[i] = 3
            elif pred < thresholds[4]: y_hat[i] = 4
            else: y_hat[i] = 5
        return y_hat.astype('int')    
