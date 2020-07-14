import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io

from pathlib import Path
import sys

sys.path.insert(0,'..')

from preprocessing.utils.tiling import tile
from preprocessing.utils.mat_transforms import tile2mat
from preprocessing.utils.data_loader import PandasDataLoader

from utils.utils import set_gpu_memory, seed_all


# directories

IMG_DIR = Path('../data/train_images')
DATA_DIR = '../data/'
train_csv = pd.read_csv('../data/train.csv')
train_csv_path = '../data/train.csv'
NFOLDS=5
SEED=5

TIFF_LEVEL = 1
N = 16
SZ = 512
PAD_VAL=255
PROVIDER='PANDA'

seed_all(SEED)

# an example: loading the skip dataframe and listing the possible reasons
skip_df = pd.read_csv(Path(DATA_DIR) / Path('PANDA_Suspicious_Slides_15_05_2020.csv'))
print("possible faulty slide reasons", skip_df['reason'].unique())

fold_df = PandasDataLoader(images_csv_path=train_csv_path,
                           skip_csv=None) #Path(DATA_DIR) / Path('PANDA_Suspicious_Slides_15_05_2020.csv'), 
                           #skip_list=["Background only", "tiss", "blank"]) #["marks", "Background only", "tiss", "blank"]

# we create a possible stratification here, the options are by isup grade, or further distilled by isup grade and data provider
# stratified_isup_sample or stratified_isup_dp_sample, we use the former.

fold_df = fold_df.stratified_isup_sample(NFOLDS, SEED)

# class probabilities for sampling procedures
num_classes = fold_df['isup_grade'].nunique()

lbl_value_counts = fold_df["isup_grade"].value_counts()
# Get the distinct isup grades
isups = lbl_value_counts.index.to_numpy().astype('int')

# frequency per isup grade
lbl_probs = lbl_value_counts / sum(lbl_value_counts)

# turn it into a dataframe
data_probs = pd.DataFrame({'class_prob': lbl_probs, 'isup_grade': isups})

# merge with the original dataframe
fold_df = fold_df.merge(data_probs, on='isup_grade')

# the desired probabilities: equal sampling
fold_df['class_target_prob'] = 1 / num_classes

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _string_feature(value):
    """Returns a bytes_list from a string / byte."""
    value = str.encode(value)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(image_string, row_df):
    image_shape = tf.image.decode_jpeg(image_string).shape
    
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(row_df['isup_grade']),
        'class_prob': _float_feature(row_df['class_prob']),
        'class_target_prob': _float_feature(row_df['class_target_prob']),
        'data_provider':_string_feature(row_df['data_provider']),
        'image_id': _string_feature(row_df['image_id']),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))
    
for i in range(0,NFOLDS):
    part_df = fold_df[fold_df['split']==i]
    print(len(part_df))

for i in range(0,NFOLDS):
    print("writing fold", i)
    part_df = fold_df[fold_df['split']==i]
    record_file=f'fold_{i}_{PROVIDER}_{SZ}_{N}_{SEED}_{NFOLDS}.tfr'
    with tf.io.TFRecordWriter(record_file) as writer:
        for index, row in part_df.iterrows():
            label = row['isup_grade']
            path = IMG_DIR / Path(row['image_id'] + '.tiff')
            img = skimage.io.MultiImage(str(path))[TIFF_LEVEL]
            tiles = tile(img, SZ, N, PAD_VAL)
            tiles = [x['img'] for x in tiles]
            bigimg = tf.io.encode_jpeg(tile2mat(np.array(tiles), 4, 4), quality=100)
            example = image_example(bigimg, row)
            writer.write(example.SerializeToString())
        print("done writing file")
