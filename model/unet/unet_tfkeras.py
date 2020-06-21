# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 12:15:24 2020

@author: Stephan

A variation of the Unet described in
https://github.com/JielongZ/full-reimplementation-of-unet

This alternative has batchnormalization after every convolution, 
before activation.


"""

from tensorflow.keras.layers import (Conv2D, 
                                     Conv2DTranspose, 
                                     ELU, 
                                     BatchNormalization,
                                     MaxPooling2D,
                                     Input,
                                     concatenate,
                                     Dropout)

from tensorflow.keras.models import Model

def unet():
    inputs = Input(shape=(None, None, 3))
    
    c1 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(inputs)
    c1 = ELU()(c1)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(c1)
    c1 = ELU()(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(p1)
    c2 = ELU()(c2)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(c2)
    c2 = ELU()(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(p2)
    c3 = ELU()(c3)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(c3)
    c3 = ELU()(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(p3)
    c4 = ELU()(c4)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(c4)
    c4 = ELU()(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    
    c5 = Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(p4)
    c5 = ELU()(c5)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(c5)
    c5 = ELU()(c5)
    c5 = BatchNormalization()(c5)
    
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(u6)
    c6 = ELU()(c6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(c6)
    c6 = ELU()(c6)
    c6 = BatchNormalization()(c6)
    
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(u7)
    c7 = ELU()(c7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(c7)
    c7 = ELU()(c7)
    c7 = BatchNormalization()(c7)
    
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(u8)
    c8 = ELU()(c8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(c8)
    c8 = ELU()(c8)
    c8 = BatchNormalization()(c8)
    
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(u9)
    c9 = ELU()(c9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(c9)
    c9 = ELU()(c9)
    c9 = BatchNormalization()(c9)
    
    outputs = Conv2D(3, (1, 1), activation='softmax', dtype='float32')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model