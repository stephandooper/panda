# -*- coding: utf-8 -*-
"""
Created on Sun May 31 15:36:03 2020

@author: Stephan
"""

# load tensorflow dependencies
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers import RectifiedAdam
from tensorflow.keras.callbacks import ReduceLROnPlateau
# 16 bit precision computing
from tensorflow.keras.mixed_precision import experimental as mixed_precision
print(tf.__version__)
print(tf.config.experimental.list_physical_devices())

from pathlib import Path
import pandas as pd
import sys
import datetime
import numpy as np
from functools import partial

sys.path.insert(0,'..')

# custom packages
from preprocessing.utils.data_loader import PandasDataLoader
from preprocessing.generators import TiffGenerator, TiffFromCoords
from preprocessing.utils.mat_transforms import tf_tile2mat
from utils.utils import set_gpu_memory, seed_all

from model.network import Network
# Augmentation packages

# custom stain augmentation
from preprocessing.augmentations import StainAugment
import albumentations as A
from albumentations import (
    Flip, ShiftScaleRotate, RandomRotate90,
    Blur, RandomBrightnessContrast, 
    Compose, RandomScale, ElasticTransform, GaussNoise, GaussianBlur,
    RandomBrightness, RandomContrast
)

from model.models import ResNext50, EfficientNetB1, EfficientNetB1_bigimg
import os

#this isnt urgent
os.nice(19)


def aug_routine(image):
    #R = Compose([RandomRotate90(p=0.5), Flip(p=0.5)])
    #S = Compose([RandomScale(scale_limit=0.1, interpolation=1, p=0.5)])
    C = Compose([StainAugment(p=0.9)])
    #E = Compose([ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4, p=0.5)])

    BRIGHT = Compose([RandomBrightness(limit=0.08, p=0.3)])
    CONTR = Compose([RandomContrast(limit=0.08, p=0.3)])    
    
    B = Compose([GaussianBlur(blur_limit=3, p=0.1)])
    #G = Compose([GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.5)])
    augment = Compose([C, BRIGHT, CONTR, B], p=0.9)
    aug_op = augment(image=image)
    image = aug_op['image']
    return image


def flip(x: tf.Tensor) -> tf.Tensor:
    x = tf.image.random_flip_left_right(x, seed=1)
    x = tf.image.random_flip_up_down(x, seed=1)

    return x

def rotate(x: tf.Tensor) -> tf.Tensor:
    # Rotate 0, 90, 180, 270 degrees
    return tf.image.rot90(x, 
                          tf.random.uniform(shape=[], 
                                               minval=0, 
                                               maxval=4, 
                                               dtype=tf.int32, 
                                               seed=1))

augments = [flip, rotate]
'''
class Config(object):
       
    # Dataset parameters
    DATA_DIR = 'data'
    IMG_DIR = 'data/train_images'
    TRAIN_MASKS_DIR = 'data/masks'
    SKIP_LIST = [] # What slides to not filter out (see PandasDataLoader)
    
    # Image parameters
    IMG_SIZE = 256  # width and heigth of the image
    
    # TIFF parameters
    TIFF_LEVEL = 1
    NUM_CLASSES = 6 
    
    # Method specific parameters
    METHOD = 'tile'
    NUM_TILES = 16
    
    # TRAINING PARAMETERS
    NFOLDS = 4    # number of folds to use for (cross validation)
    SEED=5        # the seed TODO: REPLACE THIS WITH A FUNCTION THAT SEEDS EVERYTHING WITH SEED
    TRAIN_FOLD=0  # select the first fold for training/validation
    BATCH_SIZE = 6
    NUM_EPOCHS = 15  
    LEARNING_RATE = 1e-3
    
    
    MODEL = EfficientNetB1
    MODEL_NAME = MODEL.__name__
    MODEL = staticmethod(MODEL)
'''    
    
class CoordsConfig(object):
    
    # in case tiffFromCoords is used instead of regular tiff generator
    
    # Dataset parameters
    DATA_DIR = 'data'
    IMG_DIR = 'data/train_images'
    TRAIN_MASKS_DIR = 'data/masks'
    SKIP_LIST = [] # What slides to not filter out (see PandasDataLoader)
    
    # TIFF parameters
    NUM_CLASSES = 6 
    
    # TRAINING PARAMETERS
    NFOLDS = 4    # number of folds to use for (cross validation)
    SEED=5        # the seed TODO: REPLACE THIS WITH A FUNCTION THAT SEEDS EVERYTHING WITH SEED
    TRAIN_FOLD=0  # select the first fold for training/validation
    BATCH_SIZE = 3
    NUM_EPOCHS = 40 
    LEARNING_RATE = 3e-4
    
    MODEL = EfficientNetB1_bigimg
    MODEL_NAME = MODEL.__name__
    MODEL = staticmethod(MODEL)    
    
    # transform list of tiles to a single big image of 6x6 tiles
    IMG_TRANSFORM_FUNC = staticmethod(partial(tf_tile2mat, row=6, col=6))
    
    COORDS = np.load('coordinates/1-36-240-255.npy',allow_pickle=True)

def setup(seed):
    
    # Reproducibility, but mind individual tensorflow OPS (layers)

    # control for the gpu memory, and number of used gpu's
    #set_gpu_memory(device_type='GPU')
    
    # see the utils functions which seeds are set
    # tensorflow ops still have to be seeded manually...
    seed_all(seed)   
    
    # fp 16 training
    policy = mixed_precision.Policy('mixed_float16')
    
    mixed_precision.set_policy(policy)
    
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)
    
    
def create_split(config):
    # an example: loading the skip dataframe and listing the possible reasons
    skip_df = (pd.read_csv(Path(config.DATA_DIR) / 
                           Path('PANDA_Suspicious_Slides_15_05_2020.csv')))
    
    
    print("possible faulty slide reasons", skip_df['reason'].unique())
    
    fold_df = PandasDataLoader(images_csv_path=Path(config.DATA_DIR) /
                                   Path('train.csv'),
                               skip_csv=Path(config.DATA_DIR) / 
                                   Path('PANDA_Suspicious_Slides_15_05_2020.csv'), 
                               skip_list=[])
    
    # we create a possible stratification here, the options are by isup grade,
    # or further distilled by isup grade and data provider
    # stratified_isup_sample or stratified_isup_dp_sample, we use the latter.
    
    fold_df = fold_df.stratified_isup_sample(config.NFOLDS, config.SEED)
    
    return fold_df



if __name__ =='__main__':
    
    config = CoordsConfig()
    
    # seed, fp16 training, 
    # Reproducibility, but mind individual tensorflow OPS (layers)

    # control for the gpu memory, and number of used gpu's
    set_gpu_memory(num_gpu=1, device_type='GPU')
    
    # see the utils functions which seeds are set
    # tensorflow ops still have to be seeded manually...
    seed_all(config.SEED)   
        
    # fp 16 training
    policy = mixed_precision.Policy('mixed_float16')
    
    mixed_precision.set_policy(policy)
    
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)
    
    model = config.MODEL
    network = Network(model())
    
    network.summary()
    # get initial weights and use them at the start of each fold
    # Clearing TF/keras graph still has a lot of issues (freeing VRAM)
    init_weights = network.get_weights()
    
    #optimizer = Adam(lr=config.LEARNING_RATE)
    optimizer = RectifiedAdam(lr=config.LEARNING_RATE)

    lrreducer = ReduceLROnPlateau(
        monitor='val_loss',
        factor=.5,
        patience=5,
        verbose=1,
        min_lr=1e-7
    )
    
    # custom callbacks
    custom_callbacks = [lrreducer]
    
    # qwk is added by default in network class
    custom_metrics = [tf.keras.metrics.RootMeanSquaredError()]
    custom_loss = tf.keras.losses.MeanSquaredError()
    
    fold_df = create_split(config)
    
    for train_fold in range(0, config.NFOLDS):
        #only train a single fold
        print("preparing training fold", train_fold)
        if train_fold == 1:
            break
        # model 
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # set the weights to start with
        network.set_weights(init_weights)
        
        # create the folds
        # we can create training/validation splits from the fold column
        train_df = fold_df[fold_df['split'] != train_fold]
        valid_df = fold_df[fold_df['split'] == train_fold]
        
        # create training data
        print("creating training set")
        data = TiffFromCoords(coords=config.COORDS,
                                     df=train_df, 
                                     img_dir=config.IMG_DIR, 
                                     batch_size=config.BATCH_SIZE, 
                                     aug_func=aug_routine,
                                     tf_aug_list=augments,
                                     one_hot=False,
                                     img_transform_func=config.IMG_TRANSFORM_FUNC)
        
        # do not use augmentation for the val_data, if aug_func is empty 
        # (None default), then no augmentation is used.
        # Create validation data
        
        print("Creating validation set")
        val_data = TiffFromCoords(coords=config.COORDS,
                                 df=valid_df, 
                                 img_dir=config.IMG_DIR, 
                                 batch_size=config.BATCH_SIZE, 
                                 aug_func=None,
                                 one_hot=False,
                                 img_transform_func=config.IMG_TRANSFORM_FUNC)
        
        lbl_value_counts = train_df['isup_grade'].value_counts()
        class_weights = {i: max(lbl_value_counts) / v for i, v in lbl_value_counts.items()}
        print('classes weigths:', class_weights)
        
        weights_fname = f'{config.MODEL_NAME}_{data.tiff_level}_{now}_{train_fold}.h5'
        
        # loading network weights
        network.load_weights('EfficientNetB1_bigimg_1_20200608-221404_0_bestQWK.h5')
        
        # Train the network
        network.train(dataset=data(),
                  val_dataset=val_data(mode='validation'),
                  epochs=config.NUM_EPOCHS,
                  learning_rate=config.LEARNING_RATE,
                  num_classes=config.NUM_CLASSES,
                  sparse_labels=True,
                  regression=True,
                  class_weights=None,
                  custom_callbacks=custom_callbacks,
                  custom_metrics=custom_metrics,
                  custom_loss=custom_loss,
                  custom_optimizer=optimizer,
                  save_weights_name=weights_fname,
                  tb_logdir=f"./{config.MODEL_NAME}")
