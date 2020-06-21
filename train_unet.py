# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 02:26:04 2020

@author: Stephan
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# load tensorflow dependencies
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from model.unet.unet_tfkeras import unet
from model.unet.loss import generalized_dice_loss
from model.unet.metric import MeanIoU_prob


# 16 bit precision computing
from tensorflow.keras.mixed_precision import experimental as mixed_precision
print(tf.__version__)
print(tf.config.experimental.list_physical_devices())

from pathlib import Path
import pandas as pd
import sys
import datetime

sys.path.insert(0,'..')

# custom packages
from preprocessing.utils.data_loader import PandasDataLoader
from preprocessing.generators import UNETGenerator
from utils.utils import set_gpu_memory, seed_all


# control for the gpu memory, and number of used gpu's
set_gpu_memory(device_type='GPU')
# see the utils functions which seeds are set
# tensorflow ops still have to be seeded manually...

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)


# ------------------
# Directories
# ------------------


class Config(object):
    # constants
    DATA_DIR = 'data'  # General path to the data dir
    IMG_DIR = 'data/train_images'  # Path to the TILED images
    TRAIN_MASKS_DIR = 'data/train_label_masks'  # Path to the masks
    NFOLDS=4
    SEED=1
    TRAIN_FOLD=0
    
    # model
    MODEL_NAME = 'UNET'
    
    # training parameters
    
    EPOCHS = 10
    LR = 1e-3
    

# we don't use CV for this, so this function is probably a bit sloppy (probably??)
def load_data(img_dir, mask_dir, data_dir, nfolds, train_fold, seed):
    # an example: loading the skip dataframe and listing the possible reasons
    mask_paths = list(Path(mask_dir).rglob('*.tiff'))
    image_paths = list(Path(img_dir).rglob('*.tiff'))

    skip_df = pd.read_csv(Path(data_dir) / Path('PANDA_Suspicious_Slides_15_05_2020.csv'))
    print("possible faulty slide reasons", skip_df['reason'].unique())
    
    fold_df = PandasDataLoader(images_csv_path=Path(data_dir) / Path('train.csv'),
                               skip_csv=Path(data_dir) / Path('PANDA_Suspicious_Slides_15_05_2020.csv'), 
                               skip_list=['No Mask'])
    
    # we create a possible stratification here, the options are by isup grade, or further distilled by isup grade and data provider
    # stratified_isup_sample or stratified_isup_dp_sample, we use the former.
    
    fold_df = fold_df.stratified_isup_sample(nfolds, seed)
    
    # we can create training/validation splits from the fold column
    train_df = fold_df[fold_df['split'] != train_fold]
    valid_df = fold_df[fold_df['split'] == train_fold]
    
    img_ids = [x.stem for x in image_paths]
    img_df = pd.DataFrame({'paths': image_paths, 'image_id': img_ids})
    
    mask_ids = [x.stem.split('_')[0] for x in mask_paths]
    mask_df = pd.DataFrame({'paths': mask_paths, 'image_id': mask_ids})
    
    train_df = train_df.merge(img_df, on='image_id')
    train_df = train_df.merge(mask_df, on='image_id')
    valid_df = valid_df.merge(img_df, on='image_id')
    valid_df = valid_df.merge(mask_df, on='image_id')
    
    return train_df, valid_df


def main():
    print("[!] Instantiation")
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    seed_all(20)
    config = Config()
    train_df, valid_df = load_data(img_dir= config.IMG_DIR, 
                                   mask_dir= config.TRAIN_MASKS_DIR,
                                   data_dir=config.DATA_DIR,
                                   nfolds=config.NFOLDS,
                                   train_fold=config.TRAIN_FOLD,
                                   seed=config.SEED)
    
    filtered_img_paths_train = [str(x) for x in train_df['paths_x'].tolist()]
    filtered_mask_paths_train = [str(x) for x in train_df['paths_y'].tolist()]
    filtered_data_providers_train = train_df['data_provider'].to_list()
    
    filtered_img_paths_valid = [str(x) for x in valid_df['paths_x'].tolist()]
    filtered_mask_paths_valid = [str(x) for x in valid_df['paths_y'].tolist()]
    filtered_data_providers_valid = valid_df['data_provider'].to_list()
    
    # defining generators
    print("[!] Creating generators")

    train_gen = UNETGenerator(filtered_img_paths_train, 
                      filtered_mask_paths_train, 
                      filtered_data_providers_train,
                      ksize=(256, 256),
                      mode='training')
    
    train_gen = train_gen.load_process()
    
    
    val_gen = UNETGenerator(filtered_img_paths_valid, 
                      filtered_mask_paths_valid, 
                      filtered_data_providers_valid,
                      ksize=(256, 256),
                      mode='validation') 

    val_gen = val_gen.load_process()    
    # Defining model and training
    print("[!] Creating model")

    model = unet()
    root, ext = os.path.splitext(f"unet_weights/{config.MODEL_NAME}_{now}.h5")
    # callbacks
    ckp_best_loss = ModelCheckpoint(filepath=root + "_bestLoss" + ext,
                                monitor='val_loss',
                                verbose=0,
                                save_best_only=True,
                                save_weights_only=True)
    
    # Save based on metric
    ckp_best_metric = ModelCheckpoint(filepath=root + "_bestQWK" + ext,
                                      monitor='val_mean_iou',
                                      verbose=1,
                                      save_best_only=True,
                                      mode='max',
                                      save_weights_only=True)
    
    # save epoch
    ckp_epoch = ModelCheckpoint(filepath=root + "_epoch" + ext,
                              monitor='val_mean_iou',
                              verbose=1,
                              save_best_only=False,
                              mode='max',
                              save_weights_only=True,
                              save_freq='epoch')
    
    # metrics
    callbacks = [ckp_best_loss, ckp_best_metric, ckp_epoch]
    IOU_metric = MeanIoU_prob(3, name='mean_iou')
        
    model.compile(optimizer=Adam(lr=config.LR), 
                  loss=[ generalized_dice_loss], 
                  metrics=[IOU_metric, 'accuracy'])
    
    
    print("[!] fitting!")

    model.fit(train_gen,
              validation_data=val_gen, 
              epochs=config.EPOCHS,
              steps_per_epoch=3566,
              callbacks=callbacks)
    
if __name__ == '__main__':
    main()
