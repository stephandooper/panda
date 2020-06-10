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
    def __init__(self, p=3):
        """ Class initializer
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
            Returns a [BS, C] array
        """
        x_max = tf.reduce_max(x)
        x = tf.clip_by_value(x, eps, x_max, name='clip')
        
        # a [Nt, H, W, C] array
        shapes = x.get_shape()
        
        # for fp16 support
        p = tf.cast(p, x.dtype)
        pooled = tf.nn.avg_pool2d(tf.math.pow(x, p), 
                                  ksize=[shapes[-3], shapes[-2]],
                                  strides=[shapes[-3], shapes[-2]],
                                  padding='SAME',
                                  )
        pooled = tf.math.pow(pooled, 1./p)
        shape = pooled.shape
        
        # squeeze from [BS, 1, 1, C] to [BS, C]
        pooled = tf.squeeze(pooled)
        
        # for some reason, pooled shape is <unknown> after tf squeeze
        pooled.set_shape((shape[0],shape[3]))
        return pooled
    
    def compute_output_shape(self, input_shape):
        new_shape = (input_shape[0],input_shape[3])
        return new_shape
        
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'p': self.p,
        })
        return config
    
class MPGeneralizedMeanPooling2D(layers.Layer):
    """ Generalized mean pooling layer for 2D images
        This also works for Timedistributed tiled images (PANDA challenge)
        The layer learns a different parameter for each feature map
        
        # THIS LAYER IS NOT PROPERLY DEBUGGED/TESTED, USE AT OWN RISK
        
    
        References
        ----------
        https://www.tensorflow.org/guide/keras/custom_layers_and_models
        https://arxiv.org/pdf/1711.02512.pdf
        https://github.com/filipradenovic/cnnimageretrieval
    
    
    """
    def __init__(self, dim, p=2):
        """

        Parameters
        ----------
        p : float, optional
            The exponent (power). The default is 3.
        dim : int
            The feature map dimension of the previous layer
        eps : float, optional
            Values lower than eps will be clipped. The default is 1e-6.

        Returns
        -------
        None.

        """
        self.dim = dim
        eps=1e-6
        super(MPGeneralizedMeanPooling2D, self).__init__()
        self.eps = eps
        shape_init = tf.ones(tf.cast(self.dim,tf.int32), name="ones")
        self.p = tf.Variable(initial_value=p * shape_init, trainable=True)
        
    def call(self, inputs):
        return self.gem2d(inputs, p=self.p[tf.newaxis, tf.newaxis, tf.newaxis, :], eps=self.eps)
    
    
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
        print("SHAPE", x.shape)
        x = tf.cast(x,tf.float32)
        x_max = tf.reduce_max(x)
        x = tf.clip_by_value(x, eps, x_max, name='clip')
        
        # a [Nt, H, W, C] array
        shapes = x.get_shape()
        
        # for fp16 support
        p = tf.cast(p, x.dtype)
        pooled = tf.nn.avg_pool2d(tf.math.pow(x, p), 
                                  ksize=[shapes[-2], shapes[-3]],
                                  strides=[shapes[-2], shapes[-3]],
                                  padding='SAME')
        pooled = tf.math.pow(pooled, 1./p)
        shape = pooled.shape
        
        pooled = tf.squeeze(pooled)
        
        # for some reason, pooled shape is <unknown> after tf squeeze
        pooled.set_shape((shape[0],shape[3]))
        return pooled
    
    
    def compute_output_shape(self, input_shape):
        new_shape = (input_shape[0],input_shape[3])
        return new_shape
    
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'dim': self.dim,
            'p': self.p,
        })
        return config
        
        
class GeM(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GeM, self).__init__(**kwargs)
        self.p = tf.Variable(3.)
        self.eps = 1e-6
    def call(self, inputs):
        x = tf.math.pow(tf.math.maximum(self.eps, inputs), self.p)
        size = (x.shape[1], x.shape[2])
        return tf.math.pow(tf.nn.avg_pool2d(x, size, 1, padding='VALID'), 1./self.p)        


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