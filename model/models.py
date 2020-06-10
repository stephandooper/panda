# -*- coding: utf-8 -*-
"""
Created on Sun May 31 18:29:10 2020

@author: Stephan
"""

from tensorflow.keras import Sequential
import tensorflow.keras.layers as KL
from tensorflow.keras.models import Model
from model.layers import qwk_act
import efficientnet.tfkeras as efn
from classification_models.tfkeras import Classifiers
from model.layers import GeneralizedMeanPooling2D
import sys
sys.path.insert(0,'..')
import efficientnet_gn.tfkeras as efn_gn

class BaseModel(object):
    
    # to be defined in each subclass
    def __init__(self, input_size):
        raise NotImplementedError("error message")

    # to be defined in each subclass
    def normalize(self, image):
        raise NotImplementedError("error message")

    def get_output_shape(self):
        return self.model.get_output_shape_at(-1)[1:3]

    def extract(self):
        return self.model
    
    
class ResNext50(BaseModel):
    
    def __init__(self, NUM_TILES, SZ):
        seed = 1

        self.SZ = SZ
        self.NUM_TILES = NUM_TILES
        
        ResNext50, _ = Classifiers.get('resnext50')
        bottleneck = ResNext50(input_shape=(SZ, SZ, 3),
                               weights='imagenet', 
                               include_top=False)
        
        
        bottleneck = Model(inputs=bottleneck.inputs, 
                           outputs=bottleneck.layers[-2].output)
        self.model = Sequential()
        self.model.add(KL.TimeDistributed(bottleneck, 
                                          input_shape=(self.NUM_TILES, 
                                                              self.SZ, 
                                                              self.SZ, 
                                                              3)))
        self.model.add(KL.TimeDistributed(KL.BatchNormalization()))
        self.model.add(KL.TimeDistributed(KL.GlobalMaxPooling2D()))
        self.model.add(KL.Flatten())
        self.model.add(KL.BatchNormalization())
        self.model.add(KL.Dropout(.25, seed=seed))
        self.model.add(KL.Dense(512, activation='elu'))
        self.model.add(KL.BatchNormalization())
        self.model.add(KL.Dropout(.2, seed=seed))
        self.model.add(KL.Dense(1, activation=qwk_act, dtype='float32'))
        
    
    
def EfficientNetB1(NUM_TILES, SZ):
    seed = 1
    bottleneck = efn.EfficientNetB1( 
        include_top=False, 
        pooling='avg',
        weights='imagenet' # or 'imagenet'
    )
    
    
    bottleneck = Model(inputs=bottleneck.inputs, outputs=bottleneck.layers[-2].output)
    model = Sequential()
    model.add(KL.TimeDistributed(bottleneck, input_shape=(NUM_TILES, SZ, SZ, 3)))
    model.add(KL.TimeDistributed(KL.BatchNormalization()))
    model.add(KL.TimeDistributed(KL.GlobalMaxPooling2D()))
    model.add(KL.Flatten())
    model.add(KL.BatchNormalization())
    model.add(KL.Dropout(.25, seed=seed))
    model.add(KL.Dense(512, activation='elu'))
    model.add(KL.BatchNormalization())
    model.add(KL.Dropout(.25, seed=seed))
    model.add(KL.Dense(1, activation=qwk_act, dtype='float32'))
    return model
    
def EfficientNetB1_bigimg(row=1440, col=1440):
    bottleneck = efn.EfficientNetB1( 
        include_top=False, 
        pooling='avg',
        weights='imagenet',
        input_shape = (row, col, 3)
    )

    #ResNext50, _ = Classifiers.get('resnext50')
    #bottleneck = ResNext50(input_shape=(SZ, SZ, 3),
    #                       weights='imagenet', include_top=False)


    from tensorflow.keras import Sequential
    bottleneck = Model(inputs=bottleneck.inputs, outputs=bottleneck.layers[-2].output)
    model = Sequential()
    model.add(bottleneck)
    model.add(KL.BatchNormalization())
    model.add(KL.GlobalMaxPooling2D())
    model.add(KL.Flatten())
    model.add(KL.BatchNormalization())
    model.add(KL.Dropout(.25))
    model.add(KL.Dense(512, activation='elu'))
    model.add(KL.BatchNormalization())
    model.add(KL.Dropout(.25))
    model.add(KL.Dense(1, activation=qwk_act, dtype='float32'))
    return model

    
    