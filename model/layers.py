# -*- coding: utf-8 -*-
"""
Created on Sun May 31 01:09:51 2020

@author: Stephan
"""


from tensorflow.keras import layers
import tensorflow as tf
import tensorflow.keras.backend as K
   

class GeneralizedMeanPooling2D(layers.Layer):
    """ Generalized mean pooling layer for 2D images
        This also works for Timedistributed tiled images (PANDA challenge)
        
    
        References
        ----------
        https://www.tensorflow.org/guide/keras/custom_layers_and_models
        https://arxiv.org/pdf/1711.02512.pdf
        https://github.com/filipradenovic/cnnimageretrieval
    
    
    """
    def __init__(self):
        """

        Parameters
        ----------
        p : float, optional
            The exponent (power). The default is 3.
        eps : float, optional
            Values lower than eps will be clipped. The default is 1e-6.

        Returns
        -------
        None.

        """
        p=3
        eps=1e-6
        super(GeneralizedMeanPooling2D, self).__init__()
        self.eps = eps
        shape_init = tf.ones(1, name="ones")
        self.p = tf.Variable(initial_value=p * shape_init, trainable=True)
        
    def call(self, inputs):
        return self.gem2d(inputs, p=self.p, eps=self.eps)
    
    
    def gem2d(self, x, p=3, eps=1e-6):
        """ A generalized mean pooling  function for 2d images

        Parameters
        ----------
        x : A tensorflow Tensor
            A 4D [D, BS, H, W, C] tensor containing the images
        p : float, optional
            A learnable exponent parameter. The default is 3.
        eps : float, optional
            an epsilon to cut low values. The default is 1e-6.

        Returns
        -------
        pooled : Tensorflow Tensor.
            The generalized average pooled images 4D
            Returns a [BS, 1, 1, C] array
        """
        x_max = tf.reduce_max(x)
        x = tf.clip_by_value(x, eps, x_max, name='clip')
        
        # a [Nt, H, W, C] array
        shapes = x.get_shape()
        
        # for fp16 support
        p = tf.cast(p, x.dtype)
        pooled = tf.nn.avg_pool2d(tf.math.pow(x, p), 
                                  ksize=[shapes[-2], shapes[-3]],
                                  strides=[shapes[-2], shapes[-3]],
                                  padding='SAME',
                                  )
        pooled = tf.math.pow(pooled, 1./p)
        
        return pooled
    
    
    def compute_output_shape(self, input_shape):
        print("SHAPE IS", input_shape)
        new_shape = (input_shape[0], 1, 1, input_shape[3])
        return new_shape
    
    
def qwk_act(x):
    """ Bounded linear activation

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    x : TYPE
        DESCRIPTION.

    """
    x = K.switch(x>=0, x, 0)
    x = K.switch(x <=5, x, 5)
    return x