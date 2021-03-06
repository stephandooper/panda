# -*- coding: utf-8 -*-
"""
Created on Sun May 31 18:29:10 2020

@author: Stephan
"""
import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
#from tensorflow_addons.activations import mish


from model.layers import qwk_act
import efficientnet.tfkeras as efn
from classification_models.tfkeras import Classifiers
from model.layers import GeM, Mish, mish, WeightedSoftMax, customsigmoid, Sum


import sys
sys.path.insert(0,'..')


# ======================================
# Model architectures
# ======================================

# Resnets: without pooling, they have relu
# resnext: last layer without pooling is activation
# Efficientnets: without pooling, activation is its last layer 


# ======================================
# basic tile model classes
# ======================================

class TileModel_resnets(tf.keras.Model):

    def __init__(self, engine, input_shape, weights, num_tiles):
        super(TileModel_resnets, self).__init__()
        self.input_shapes = input_shape
        self.num_tiles = num_tiles        
        self.engine = engine(
            include_top=False, input_shape=input_shape, weights=weights)
        #self.avg_pool2d = tf.keras.layers.GlobalAveragePooling2D()
        
        self.gem = GeM()
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(512, activation=mish)
        self.batchnorm_1 = tf.keras.layers.BatchNormalization()
        self.dropout_1 = tf.keras.layers.Dropout(0.5)
        
        self.dense_2 = tf.keras.layers.Dense(256, activation='linear')
        self.batchnorm_2 = tf.keras.layers.BatchNormalization()
        self.dropout_2 = tf.keras.layers.Dropout(0.5)
        self.dense_3 = tf.keras.layers.Dense(5, activation='sigmoid')
        self.sum_1 = Sum()
        
        self.out_shapes = self.engine.outputs[0].shape
           
        
    def call(self, inputs):
        inputs = tf.reshape(inputs, (-1, self.input_shapes[0], self.input_shapes[1], self.input_shapes[2]))
        x = self.engine(inputs)
        x = tf.reshape(x, (-1, 
                           self.num_tiles * self.out_shapes[1], 
                           self.out_shapes[2], 
                           self.out_shapes[3]))
        x = self.gem(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.batchnorm_1(x)
        x = self.dropout_1(x)
        
        x = self.dense_2(x)
        x = self.batchnorm_2(x)
        x = self.dropout_2(x)
        x = self.dense_3(x)
        x = self.sum_1(x)
        return x
    
    
class TileModel_efficientnets(tf.keras.Model):

    def __init__(self, engine, input_shape, weights, num_tiles):
        super(TileModel_efficientnets, self).__init__()
        self.input_shapes = input_shape
        self.num_tiles = num_tiles        
        self.engine = engine(
            include_top=False, 
            pooling=None, 
            input_shape=input_shape, 
            weights=weights)
        
        #self.avg_pool2d = tf.keras.layers.GlobalAveragePooling2D()
        self.gem = GeM()
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense_1 = tf.keras.layers.Dense(512)        
        self.dense_2 = tf.keras.layers.Dense(1, activation='linear')
        self.out_shapes = self.engine.outputs[0].shape
            
    def call(self, inputs):    
        
        # From [Bs, num_tiles, h, w ,c] -> [BS* num_tiles, h, w, c]
        inputs = tf.reshape(inputs, (-1, self.input_shapes[0], self.input_shapes[1], self.input_shapes[2]))
        
        # pass through backend model
        x = self.engine(inputs)
        
        # reshape to [bs, num_tiles*H, W, c]
        x = tf.reshape(x, (-1, 
                           self.num_tiles * self.out_shapes[1], 
                           self.out_shapes[2], 
                           self.out_shapes[3]))
        # pass through pooling + dense layers
        x = self.gem(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense_1(x)
        x = mish(x)
        return self.dense_2(x)
    
    def build_graph(self, input_shape): 
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        
        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")
        
        _ = self.call(inputs)
           
    
class TileModel_efficientnetsV2(tf.keras.Model):
    def __init__(self, engine, input_shape, weights, num_tiles):
        super(TileModel_efficientnetsV2, self).__init__()
        self.input_shapes = input_shape
        self.num_tiles = num_tiles        
        self.engine = engine(
            include_top=False, 
            pooling=None, 
            input_shape=input_shape, 
            weights=weights)
        
        #self.avg_pool2d = tf.keras.layers.GlobalAveragePooling2D()
        self.gem = GeM()
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.01)
        self.dense_1 = tf.keras.layers.Dense(512)        
        self.dense_2 = tf.keras.layers.Dense(6, activation='softmax')
        self.output_layer = tf.keras.layers.Dense(1, activation=customsigmoid)
        self.out_shapes = self.engine.outputs[0].shape
            
    def call(self, inputs):    
        
        # From [Bs, num_tiles, h, w ,c] -> [BS* num_tiles, h, w, c]
        inputs = tf.reshape(inputs, (-1, self.input_shapes[0], self.input_shapes[1], self.input_shapes[2]))
        
        # pass through backend model
        x = self.engine(inputs)
        
        # reshape to [bs, num_tiles*H, W, c]
        x = tf.reshape(x, (-1, 
                           self.num_tiles * self.out_shapes[1], 
                           self.out_shapes[2], 
                           self.out_shapes[3]))
        # pass through pooling + dense layers
        x = self.gem(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense_1(x)
        x = mish(x)
        x = self.dense_2(x)
        return self.output_layer(x)


# ========================
# Resnext
# ========================
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
    model.add(GeM())
    model.add(KL.Flatten())
    model.add(KL.Dense(512, activation='linear'))
    model.add(Activation('Mish', name="conv1_act"))
    model.add(KL.BatchNormalization())
    model.add(KL.Dropout(0.25))
    model.add(KL.Dense(1, activation=qwk_act, dtype='float32')) 
    
    return model

def ResNext50_tile(NUM_TILES, SZ):

    engine, _ = Classifiers.get('resnext50')
    
    model = TileModel_resnets(engine, (SZ, SZ, 3), 'imagenet', NUM_TILES)
    model.build((None,SZ,SZ,3))
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

def ResNet34_bigimg_softmax(row=1024, col=1024):

    seed = 1
    ResNet34, _ = Classifiers.get('resnet34')
    bottleneck = ResNet34(input_shape=(row, col, 3),
                           weights='imagenet', 
                           include_top=False)
    
    bottleneck = Model(inputs=bottleneck.inputs, 
                       outputs=bottleneck.layers[-2].output)
    model = Sequential()
    model.add(bottleneck)
    model.add(KL.ReLU())
    model.add(GeM())
    model.add(KL.Flatten())
    model.add(KL.Dense(512, activation=mish, dtype='float32'))
    model.add(KL.BatchNormalization())
    model.add(KL.Dropout(0.5))
    model.add(KL.Dense(6, activation='linear', dtype='float32'))
    model.add(KL.Dense(1, activation=customsigmoid, dtype='float32'))
    
    return model

def ResNet34_bigimg_softmaxV2(row=1024, col=1024):

    seed = 1
    ResNet34, _ = Classifiers.get('resnet34')
    bottleneck = ResNet34(input_shape=(row, col, 3),
                           weights='imagenet', 
                           include_top=False)
                           
    bottleneck = Model(inputs=bottleneck.inputs, 
                       outputs=bottleneck.layers[-2].output)
    model = Sequential()
    model.add(bottleneck)
    model.add(GeM())
    model.add(KL.Flatten())
    model.add(KL.Dense(512, activation=mish, dtype='float32'))
    model.add(KL.BatchNormalization())
    model.add(KL.Dropout(0.5))
    model.add(KL.Dense(256, activation='linear', dtype='float32'))
    model.add(KL.BatchNormalization())
    model.add(KL.Dropout(0.5))
    model.add(KL.Dense(5, activation='sigmoid', dtype='float32'))
    model.add(Sum())
    
    return model
    


def ResNet34_tile(NUM_TILES, SZ):

    engine, _ = Classifiers.get('resnet34')
    
    model = TileModel_resnets(engine, (SZ, SZ, 3), 'imagenet', NUM_TILES)
    model.build((None,SZ,SZ,3))
    return model
    



def ResNet34_bigimg_GeM_v2(row=1440, col=1440):
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


def ResNet50_bigimg(row=1536, col=1536):
    ResNet50, _ = Classifiers.get('resnet50')
    bottleneck = ResNet50(input_shape=(row, col, 3),
                           weights='imagenet', 
                           include_top=False)
    bottleneck.summary()
    bottleneck = Model(inputs=bottleneck.inputs, 
                       outputs=bottleneck.layers[-2].output)
    
    model = Sequential()
    model.add(bottleneck)
    model.add(GeM())
    model.add(KL.Flatten())
    model.add(KL.Dense(512, activation='linear'))
    model.add(Activation('Mish', name="conv1_act"))
    model.add(KL.Dropout(0.25))
    model.add(KL.Dense(1, activation=qwk_act, dtype='float32')) 
    
    return model

def ResNet50_tile(NUM_TILES, SZ):
    engine, _ = Classifiers.get('resnet50')
    
    model = TileModel_resnets(engine, (SZ, SZ, 3), 'imagenet', NUM_TILES)
    model.build((None,SZ,SZ,3))
    return model

# ===================================
# EFFICIENTNETS
# ===================================

def EfficientNetB0_tile(NUM_TILES, SZ):
    bottleneck = efn.EfficientNetB0
    
    model = TileModel_efficientnets(bottleneck, (SZ, SZ, 3), 'imagenet', NUM_TILES)
    model.build((None,SZ,SZ,3))
    return model
    
def EfficientNetB0_tileV2(NUM_TILES, SZ):
    bottleneck = efn.EfficientNetB0
    
    model = TileModel_efficientnetsV2(bottleneck, (SZ, SZ, 3), 'imagenet', NUM_TILES)
    model.build((None,SZ,SZ,3))
    return model
    
def EfficientNetB1_tile(NUM_TILES, SZ):
    bottleneck = efn.EfficientNetB1
    
    model = TileModel_efficientnets(bottleneck, (SZ, SZ, 3), 'imagenet', NUM_TILES)
    model.build((None,SZ,SZ,3))
    return model    
    

def EfficientNetB0_bigimg(row=1440, col=1440):
    bottleneck = efn.EfficientNetB0( 
        include_top=False, 
        pooling='avg',
        weights='imagenet',
        input_shape = (row, col, 3)
    )

    bottleneck = Model(inputs=bottleneck.inputs, outputs=bottleneck.layers[-1].output)
    model = Sequential()
    model.add(bottleneck)
    model.add(KL.Flatten())
    model.add(KL.Dense(1, activation=qwk_act, dtype='float32'))
    return model    
    
def EfficientNetB0_bigimg_softmax(row=1024, col=1024):
    bottleneck = efn.EfficientNetB0( 
        include_top=False, 
        pooling=None,
        weights='imagenet',
        input_shape = (row, col, 3)
    )
    
    bottleneck = Model(inputs=bottleneck.inputs, outputs=bottleneck.layers[-1].output)
    model = Sequential()
    model.add(bottleneck)
    model.add(GeM())
    model.add(KL.Flatten())
    model.add(KL.Dense(512, activation=mish, dtype='float32'))
    model.add(KL.BatchNormalization())
    model.add(KL.Dropout(0.5))
    model.add(KL.Dense(6, activation='softmax', dtype='float32'))
    model.add(KL.Dense(1, activation=customsigmoid, dtype='float32'))
    return model     