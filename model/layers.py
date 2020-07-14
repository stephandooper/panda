# -*- coding: utf-8 -*-
"""
Created on Sun May 31 01:09:51 2020

@author: Stephan
"""


from tensorflow.keras import layers
import tensorflow as tf
import tensorflow.keras.backend as K
## Import Necessary Modules

from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects

class Mish(Activation):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    '''

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

get_custom_objects().update({'Mish': Mish(mish)})        

def customsigmoid(inputs):
    return 5 * tf.math.sigmoid(inputs)
 
class GeM(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GeM, self).__init__(**kwargs)
        self.p = tf.Variable(3.)
        self.eps = 1e-6
    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        x = tf.math.pow(tf.math.maximum(self.eps, inputs), self.p)
        size = (x.shape[1], x.shape[2])
        return tf.math.pow(tf.nn.avg_pool2d(x, size, 1, padding='VALID'), 1./self.p)        


def qwk_act(x):
    """ Bounded linear activation

    Parameters
    ----------
    x : Tf tensor
        A tf tensor of size 1.

    Returns
    -------
    x : Tf tensor
        A Tf tensor of size 1, clipped to the range [0,5]

    """
    x = K.switch(x>=0, x, 0)
    x = K.switch(x <=5, x, 5)
    return x


class WeightLayer(tf.keras.layers.Layer):
    """ A weightlayer to be applied after Unet 
        The output probability map of the Unet is thresholded according to
        threshold, and then each prediction map (background, benign, malignent)
        is weighed according to [lambda1, lambda2, lambda3]
        
        The default is that background is removed (0), benign is weighed by
        1, and malignent is weighed by 3.
    """
    
    def __init__(self, lambdas = [0.,1.,3.], threshold=0.8):
        """ class initializer

        Parameters
        ----------
        lambdas : list, optional
            a list of floats to weigh each output map with.
            The default is [0.,1.,3.].
        threshold : float, optional
            the value to threshold each output map with. The default is 0.8.

        Returns
        -------
        None.

        """
        super().__init__()
        
        self.lambdas = tf.Variable(lambdas, trainable=False, dtype='float32')
        self.lambdas = self.lambdas*-1
        self.thresh = threshold
    
    def call(self, inputs):
        """ the call function that returns a weighted output map for further
        tiling

        Parameters
        ----------
        inputs : list of tensorflow tensors
            A 1-d list with a [BS,H,W,3] tensor

        Returns
        -------
        inputs : TYPE
            A [BS, H, W, 3] thresholded and weighted according to thresh, and
            lambdas

        """
        inputs = tf.cast(inputs[0], tf.float32)
        inputs = tf.greater_equal(inputs, self.thresh) #will return boolean values
        inputs = tf.cast(inputs, dtype=tf.float32) #will convert bool to 0 and 1    
        inputs = tf.multiply(inputs, self.lambdas)
        
        return inputs
        
class WeightedSoftMax(tf.keras.layers.Layer):
    def __init__(self):
        super(WeightedSoftMax, self).__init__()
        
        self.fixed_weights = tf.Variable(tf.ones(6), trainable=True)
        self.fixed_weights = self.fixed_weights[tf.newaxis,...]
        #print("fixed weights shape", self.fixed_weights.shape)
        
    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        #print("inputs shape",inputs.shape)
        return 5 * tf.math.sigmoid(tf.matmul(inputs, tf.transpose(self.fixed_weights)))
        
        
class Sum(tf.keras.layers.Layer):
    def __init__(self):
        super(Sum, self).__init__()
        
    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        return tf.math.reduce_sum(inputs, axis=1)