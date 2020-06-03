# -*- coding: utf-8 -*-
"""
Created on Sun May 31 15:45:27 2020

@author: Stephan
"""

class Config(object):
    
    BATCH_SIZE = 16
    
    
    # Dataset parameters
    DATA_DIR = '../data'
    IMG_DIR = '../data/train_images'
    TRAIN_MASKS_DIR = '../data/masks'
    SKIP_LIST = [] # What slides to not filter out (see PandasDataLoader)
    
    # Image parameters
    IMG_SIZE = 128  # width and heigth of the image

    
    # TIFF parameters
    TIFF_LEVEL = 1
    NUM_CLASSES = 6 
    
    # Method specific parameters
    METHOD = 'tile'
    NUM_TILES = 32
    
    # TRAINING PARAMETERS
    NFOLDS = 4    # number of folds to use for (cross validation)
    SEED=5        # the seed TODO: REPLACE THIS WITH A FUNCTION THAT SEEDS EVERYTHING WITH SEED
    TRAIN_FOLD=0  # select the first fold for training/validation
    BATCH_SIZE = 8
    NUM_EPOCHS = 15  
    LEARNING_RATE = 1e-4
    
    
