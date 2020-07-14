# -*- coding: utf-8 -*-
"""
Created on Sun May 31 15:36:03 2020

@author: Stephan
"""

# load tensorflow dependencies
import os
# this isnt urgent
os.nice(19)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0" # second gpu
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

from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import ConfusionMatrixDisplay

sys.path.insert(0, "..")

# custom packages
from preprocessing.utils.data_loader import PandasDataLoader
from preprocessing.generators import TiffFromCoords
from preprocessing.utils.mat_transforms import tf_tile2mat
from preprocessing.utils.thresh2int import thresh2int
from utils.utils import set_gpu_memory, seed_all

from model.network import Network
from model.callbacks import CohenKappa

# Augmentation packages

# custom stain augmentation
from preprocessing.augmentations import StainAugment
import albumentations as A

from albumentations import (
    Flip,
    ShiftScaleRotate,
    RandomRotate90,
    Blur,
    RandomBrightnessContrast,
    Compose,
    RandomScale,
    ElasticTransform,
    GaussNoise,
    GaussianBlur,
    RandomBrightness,
    RandomContrast,
)

from model.models import (
    ResNet34_tile,
    EfficientNetB0_tile,
    EfficientNetB1_tile,
    EfficientNetB0_tileV2,
    ResNext50_tile,
    EfficientNetB0_bigimg_softmax,
    ResNet34_bigimg_softmax,
    ResNet34_bigimg_softmaxV2,
    ResNet34_bigimg_softmaxV2_bnfalse
)

def aug_routine(image):
    # R = Compose([RandomRotate90(p=0.5), Flip(p=0.5)])
    # S = Compose([RandomScale(scale_limit=0.1, interpolation=1, p=0.5)])
    C = Compose([StainAugment(p=0.7)])
    # E = Compose([ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4, p=0.5)])

    BRIGHT = Compose([RandomBrightness(limit=0.03, p=0.5)])
    CONTR = Compose([RandomContrast(limit=0.03, p=0.5)])

    B = Compose([GaussianBlur(blur_limit=3, p=0.1)])
    G = Compose([GaussNoise(var_limit=(10.0, 20.0), mean=0, p=0.1)])
    augment = Compose([C, BRIGHT, CONTR], p=0.75)
    aug_op = augment(image=image)
    image = aug_op["image"]
    return image


def flip(x: tf.Tensor) -> tf.Tensor:
    x = tf.image.random_flip_left_right(x, seed=1)
    x = tf.image.random_flip_up_down(x, seed=1)

    return x

def rotate(x: tf.Tensor) -> tf.Tensor:
    # Rotate 0, 90, 180, 270 degrees
    return tf.image.rot90(
        x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32, seed=1)
    )

augments = [flip, rotate]

class CoordsConfig(object):

    # in case tiffFromCoords is used instead of regular tiff generator

    # Dataset parameters
    DATA_DIR = "data"
    IMG_DIR = "data/train_images"
    TRAIN_MASKS_DIR = "data/masks"
    SKIP_LIST = []  # What slides to not filter out (see PandasDataLoader)

    # TIFF parameters
    NUM_CLASSES = 6

    # TRAINING PARAMETERS
    NFOLDS = 5  # number of folds to use for (cross validation)
    SEED = (
        0  # the seed TODO: REPLACE THIS WITH A FUNCTION THAT SEEDS EVERYTHING WITH SEED
    )
    TRAIN_FOLD = 0  # select the first fold for training/validation
    BATCH_SIZE = 8
    NUM_EPOCHS = 40
    LEARNING_RATE = 1e-4
    MODEL = ResNet34_bigimg_softmaxV2
    MODEL_NAME = MODEL.__name__
    MODEL = staticmethod(MODEL)
    
    # transform list of tiles to a single big image of 6x6 tiles
    # adjust the row and col parameters such that row*col = MAX_TILES
    # or num_tiles in the coordinates file
    IMG_TRANSFORM_FUNC = staticmethod(partial(tf_tile2mat, row=6, col=6))
    
    AUG_ROUTINE= None #staticmethod(aug_routine)
    TILE_AUGMENTS=augments

    COORDS = np.load("coordinates/1-36-256-255.npy", allow_pickle=True)
    
    # oversampling coefficients. If both coefficients are 0, then no sampling
    # is used
    # The degree of undersampling, only applies to the training generator
    UNDERSAMPLING_COEF = 0.1
    
    # The degree of oversampling, only applies to the training generator
    OVERSAMPLING_COEF = 0.9
    
    # the maximum number of tiles to use, has to be 0<= MAX_TILES <= num_tiles
    # where num_tiles is specified in the coordinate file
    MAX_TILES= 36
    
    # set this number to the desired image size (CANNOT BE CHOSEN FREELY
    # WHEN USING COORDINATE FILES)
    SZ  = 256
    
    # The target tiff level: will up or downsample from the coordinate tiff
    # level towards the target. The following will be sampled up or down:
    # coordinates, image size
    TARGET_TIFF_LEVEL=1
    
    # Shuffle the coordinates as an extra form of augmentation, only applies
    # to the training generator
    COORD_SHUFFLE=False


def setup(seed):

    # Reproducibility, but mind individual tensorflow OPS (layers)

    # control for the gpu memory, and number of used gpu's
    # set_gpu_memory(device_type='GPU')

    # see the utils functions which seeds are set
    # tensorflow ops still have to be seeded manually...
    seed_all(seed)

    # fp 16 training
    policy = mixed_precision.Policy("mixed_float16")

    mixed_precision.set_policy(policy)

    print("Compute dtype: %s" % policy.compute_dtype)
    print("Variable dtype: %s" % policy.variable_dtype)


def create_split(config):
    # an example: loading the skip dataframe and listing the possible reasons
    skip_df = pd.read_csv(
        Path(config.DATA_DIR) / Path("PANDA_Suspicious_Slides_15_05_2020.csv")
    )

    print("possible faulty slide reasons", skip_df["reason"].unique())

    fold_df = PandasDataLoader(
        images_csv_path=Path(config.DATA_DIR) / Path("train.csv"),
        skip_csv=Path(config.DATA_DIR) / Path("PANDA_Suspicious_Slides_15_05_2020.csv"),
        skip_list= ["Background only", "tiss", "blank",],
    )

    #skip_list = ["marks", "Background only",     "tiss",    "blank",]
    # we create a possible stratification here, the options are by isup grade,
    # or further distilled by isup grade and data provider
    # stratified_isup_sample or stratified_isup_dp_sample, we use the latter.

    fold_df = fold_df.stratified_isup_sample(config.NFOLDS, config.SEED)

    return fold_df

if __name__ == "__main__":

    
    config = CoordsConfig()
    LR_START = 1e-5
    LR_MAX = config.LEARNING_RATE
    LR_MIN = 0.000001
    LR_RAMPUP_EPOCHS = 2
    LR_SUSTAIN_EPOCHS = 1
    LR_EXP_DECAY = .92

    def lrfn(epoch):
        if epoch < LR_RAMPUP_EPOCHS:
            lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
        elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
            lr = LR_MAX
        else:
            lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
        return lr
        
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)   
    # seed, fp16 training,
    # Reproducibility, but mind individual tensorflow OPS (layers)

    # control for the gpu memory, and number of used gpu's
    #set_gpu_memory(num_gpu=1, device_type="GPU")

    # see the utils functions which seeds are set
    # tensorflow ops still have to be seeded manually...
    seed_all(config.SEED)

    # fp 16 training
    policy = mixed_precision.Policy("mixed_float16")

    mixed_precision.set_policy(policy)

    print("Compute dtype: %s" % policy.compute_dtype)
    print("Variable dtype: %s" % policy.variable_dtype)
    print("using model", config.MODEL_NAME)

    model = config.MODEL
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(config.LEARNING_RATE, 1.0, 0.85)
    optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE)
    #optimizer = RectifiedAdam(lr=config.LEARNING_RATE)

    lrreducer = ReduceLROnPlateau(
        monitor="val_loss", factor=0.90, patience=1, verbose=1, min_lr=1e-7
    )
    
    
    # qwk is added by default in network class
    custom_metrics = [tf.keras.metrics.RootMeanSquaredError()]
                      
    custom_loss = tf.keras.losses.MeanSquaredError()
    
    
    # create model instance with all needed args for compile
    # i.e. model, metric, optimizer, loss
    network = Network(model(1536,1536), 
                      num_classes=config.NUM_CLASSES,
                      sparse_labels=True,
                      regression=True,
                      custom_metrics=custom_metrics,
                      custom_optimizer=optimizer,
                      custom_loss=custom_loss,
                      use_tf_kappa=False)
    
    network.summary()
    # get initial weights and use them at the start of each fold
    # Clearing TF/keras graph still has a lot of issues (freeing VRAM)
    init_weights = network.get_weights()
    
    fold_df = create_split(config)
    
    for train_fold in range(0, config.NFOLDS):
        # only train a single fold
        print("preparing training fold", train_fold)
        if train_fold == 1:
            break
        # model
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
        # set the weights to start with
        network.set_weights(init_weights)
    
        # create the folds
        # we can create training/validation splits from the fold column
        train_df = fold_df[fold_df["split"] != train_fold]
        valid_df = fold_df[fold_df["split"] == train_fold]
    
        # create training data
        print("creating training set")
        data = TiffFromCoords(
            coords=config.COORDS,
            df=train_df,
            tiff_level=config.TARGET_TIFF_LEVEL,
            img_dir=config.IMG_DIR,
            batch_size=config.BATCH_SIZE,
            max_num_tiles=config.MAX_TILES,
            aug_func=config.AUG_ROUTINE,
            tf_aug_list=config.TILE_AUGMENTS,
            one_hot=False,
            img_transform_func=config.IMG_TRANSFORM_FUNC,
            base_sample_factor=4,
            coord_shuffle=config.COORD_SHUFFLE,
            undersampling_coef=config.UNDERSAMPLING_COEF,
            oversampling_coef=config.OVERSAMPLING_COEF
        )
    
        # do not use augmentation for the val_data, if aug_func is empty
        # (None default), then no augmentation is used.
        # Create validation data
    
        print("Creating validation set")
        val_data = TiffFromCoords(
            coords=config.COORDS,
            df=valid_df,
            tiff_level=config.TARGET_TIFF_LEVEL,
            img_dir=config.IMG_DIR,
            batch_size=config.BATCH_SIZE,
            max_num_tiles=config.MAX_TILES,
            aug_func=None,
            one_hot=False,
            img_transform_func=config.IMG_TRANSFORM_FUNC,
            base_sample_factor=4,
            coord_shuffle=None,
            undersampling_coef=0,
            oversampling_coef=0
        )

        
        '''
        lbl_value_counts = train_df["isup_grade"].value_counts()
        class_weights = {
            i: max(lbl_value_counts) / v for i, v in lbl_value_counts.items()
        }
        print("classes weigths:", class_weights)
        '''
        weights_fname = f"{config.MODEL_NAME}_{data.tiff_level}_{now}_{train_fold}.h5"
        # custom callbacks
        cohen_kappa = CohenKappa(val_data(mode='validation'),
                                 val_data.df['isup_grade'],
                                 file_path=weights_fname
                                 )
        custom_callbacks = [cohen_kappa, lr_callback]
        
        #network.load_weights('EfficientNetB0_bigimg_softmax_1_20200711-215824_0_epoch.h5')
        
        # Train the network
        network.train(
            dataset=data(),
            val_dataset=val_data(mode="validation"),
            epochs=config.NUM_EPOCHS,
            class_weights=None,
            custom_callbacks=custom_callbacks,
            save_weights_name=weights_fname,
            tb_logdir=f"./logs/{config.MODEL_NAME}",
        )
        

        # =====================
        # validation
        # ======================
        thresholds = [0.5, 1.5, 2.5, 3.5, 4.5]

        print("predicting cohen kappa score")
        radboud_valid_df = valid_df[valid_df['data_provider'] == 'radboud']
        
        karolinska_valid_df = valid_df[valid_df['data_provider'] == 'karolinska']

        val_data_radboud = TiffFromCoords(  coords=config.COORDS,
                                            df=radboud_valid_df,
                                            tiff_level=config.TARGET_TIFF_LEVEL,
                                            img_dir=config.IMG_DIR,
                                            batch_size=config.BATCH_SIZE,
                                            max_num_tiles=config.MAX_TILES,
                                            aug_func=None,
                                            one_hot=False,
                                            img_transform_func=config.IMG_TRANSFORM_FUNC,
                                            base_sample_factor=4,
                                            coord_shuffle=None,
                                            undersampling_coef=0,
                                            oversampling_coef=0
                                        )
        
        val_data_karolinska = TiffFromCoords(coords=config.COORDS,
                                            df=karolinska_valid_df,
                                            tiff_level=config.TARGET_TIFF_LEVEL,
                                            img_dir=config.IMG_DIR,
                                            batch_size=config.BATCH_SIZE,
                                            max_num_tiles=config.MAX_TILES,
                                            aug_func=None,
                                            one_hot=False,
                                            img_transform_func=config.IMG_TRANSFORM_FUNC,
                                            base_sample_factor=4,
                                            coord_shuffle=None,
                                            undersampling_coef=0,
                                            oversampling_coef=0
                                        )

        preds_radboud = network.predict(val_data_radboud(mode='validation'),verbose=1)
        preds_karolinska = network.predict(val_data_karolinska(mode='validation'),verbose=1)
        
        preds_isup_radboud = thresh2int(thresholds, preds_radboud)
        isups_radboud = [int(x) for x in val_data_radboud.df['isup_grade'].to_list()]
        tuples_radboud = val_data_radboud.image_ids


        preds_isup_karolinska = thresh2int(thresholds, preds_karolinska)
        isups_karolinska = [int(x) for x in val_data_karolinska.df['isup_grade'].to_list()]
        tuples_karolinska = val_data_karolinska.image_ids

        all_isups = np.concatenate((np.array(isups_radboud), np.array(isups_karolinska)))
        all_preds = np.concatenate((preds_isup_radboud, preds_isup_karolinska))

        cm_radboud = confusion_matrix(isups_radboud,preds_isup_radboud ,normalize='true')
        print("cm radboud", cm_radboud)

        cm_karolinska = confusion_matrix(isups_karolinska,preds_isup_karolinska ,normalize='true')
        print("cm karolinska", cm_karolinska)

        cm_all = confusion_matrix(all_isups, all_preds ,normalize='true')
        print("cm all", cm_all)
        
        print("karolinska", cohen_kappa_score(isups_karolinska, preds_isup_karolinska, weights='quadratic'))
        print("radboud", cohen_kappa_score(isups_radboud, preds_isup_radboud, weights='quadratic'))
        print("total", cohen_kappa_score(all_isups,
                                         all_preds,
                                         weights='quadratic'))
