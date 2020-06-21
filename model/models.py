# -*- coding: utf-8 -*-
"""
Created on Sun May 31 18:29:10 2020

@author: Stephan
"""

from tensorflow.keras import Sequential
import tensorflow.keras.layers as KL
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from model.layers import qwk_act
import efficientnet.tfkeras as efn
from classification_models.tfkeras import Classifiers
from model.layers import GeM, Mish
import model.efficientnet_bn.tfkeras as efn_bn

import sys
sys.path.insert(0,'..')


# ===================================
# RESNEXTS
# ===================================
    
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
        
def ResNext50_bigimg(row=1536, col=1536):

    seed = 1
    ResNext50, _ = Classifiers.get('resnext50')
    bottleneck = ResNext50(input_shape=(row, col, 3),
                           weights='imagenet', 
                           include_top=False)
    bottleneck.summary()
    bottleneck = Model(inputs=bottleneck.inputs, 
                       outputs=bottleneck.layers[-2].output)
    
    model = Sequential()
    model.add(bottleneck)
    model.add(KL.ReLU())
    model.add(GeM())
    model.add(KL.Flatten())
    model.add(KL.Dense(1, activation=qwk_act, dtype='float32')) 
    
    return model

# ===================================
# RESNETS
# ===================================
def ResNet34_bigimg(row=1440, col=1440):

    seed = 1
    ResNet34, _ = Classifiers.get('resnet34')
    bottleneck = ResNet34(input_shape=(row, col, 3),
                           weights='imagenet', 
                           include_top=False)
    
    bottleneck = Model(inputs=bottleneck.inputs, 
                       outputs=bottleneck.layers[-2].output)
    bottleneck.summary()
    model = Sequential()
    model.add(bottleneck)
    model.add(KL.ReLU())
    model.add(KL.GlobalAveragePooling2D())
    model.add(KL.Flatten())
    model.add(KL.Dense(1, activation=qwk_act, dtype='float32')) 
    
    return model

def ResNet34_bigimg_GeM(row=1440, col=1440):

    seed = 1
    ResNet34, _ = Classifiers.get('resnet34')
    bottleneck = ResNet34(input_shape=(row, col, 3),
                           weights='imagenet', 
                           include_top=False)
    
    bottleneck = Model(inputs=bottleneck.inputs, 
                       outputs=bottleneck.layers[-2].output)
    bottleneck.summary()
    model = Sequential()
    model.add(bottleneck)
    model.add(KL.ReLU())
    model.add(GeM())
    model.add(KL.Flatten())
    model.add(KL.Dense(1, activation=qwk_act, dtype='float32')) 
    
    return model
    
    
def ResNet34_bigimg_GeM_v2(row=1440, col=1440):

    seed = 1
    ResNet34, _ = Classifiers.get('resnet34')
    bottleneck = ResNet34(input_shape=(row, col, 3),
                           weights='imagenet', 
                           include_top=False)
    
    bottleneck = Model(inputs=bottleneck.inputs, 
                       outputs=bottleneck.layers[-2].output)
    bottleneck.summary()
    model = Sequential()
    model.add(bottleneck)
    model.add(KL.ReLU())
    model.add(GeM())
    model.add(KL.Flatten())
    model.add(KL.Dense(512, activation='linear'))
    model.add(Activation('Mish', name="conv1_act"))
    model.add(KL.BatchNormalization())
    model.add(KL.Dropout(0.25))
    model.add(KL.Dense(1, activation=qwk_act, dtype='float32')) 
    
    return model

# ===================================
# EFFICIENTNETS
# ===================================

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
    
def EfficientNetB0_bigimg(row=1440, col=1440):
    bottleneck = efn.EfficientNetB0( 
        include_top=False, 
        pooling='avg',
        weights='imagenet',
        input_shape = (row, col, 3)
    )

    #ResNext50, _ = Classifiers.get('resnext50')
    #bottleneck = ResNext50(input_shape=(SZ, SZ, 3),
    #                       weights='imagenet', include_top=False)


    from tensorflow.keras import Sequential
    bottleneck = Model(inputs=bottleneck.inputs, outputs=bottleneck.layers[-1].output)
    model = Sequential()
    model.add(bottleneck)
    model.add(KL.Flatten())
    model.add(KL.Dense(1, activation=qwk_act, dtype='float32'))
    return model

def EfficientNetB0_bigimg_batchrenorm(row=1440, col=1440):
    bottleneck = efn_bn.EfficientNetB0( 
        include_top=False, 
        pooling='avg',
        weights='imagenet',
        input_shape = (row, col, 3)
    )

    #ResNext50, _ = Classifiers.get('resnext50')
    #bottleneck = ResNext50(input_shape=(SZ, SZ, 3),
    #                       weights='imagenet', include_top=False)


    from tensorflow.keras import Sequential
    bottleneck = Model(inputs=bottleneck.inputs, outputs=bottleneck.layers[-1].output)
    model = Sequential()
    model.add(bottleneck)
    model.add(KL.Flatten())
    model.add(KL.Dense(1, activation=qwk_act, dtype='float32'))
    return model
    
    