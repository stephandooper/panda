# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 17:03:28 2020

@author: Stephan
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from preprocessing.utils.tiling import get_tile_coords
import numpy as np
import skimage.io
from tqdm import tqdm
from pathlib import Path
from preprocessing.generators.unet_generator import UNETGenerator


import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
from tensorflow.keras.models import Model
from model.unet.unet_tfkeras import unet
from model.layers import WeightLayer

os.nice(19)


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate tile coordinates')

    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/images/dataset/",
                        help='Directory of the Panda train images')
    parser.add_argument('--img_size', required=True,
                        metavar="size of square image",
                        help='size of square image',
                        type=int)
    parser.add_argument('--num_tiles', required=True,
                        metavar="tile number",
                        help='number of tiles',
                        type=int)
    parser.add_argument('--pad_val', required=True,
                    metavar="pad values",
                    help='the value to pad tiles with',
                    type=int)
    args = parser.parse_args()
    
    
    
    DATA_DIR = list(Path(args.dataset).glob('*.tiff'))
    data_list = [str(x) for x in DATA_DIR]
    # generate inference 
    inference_generator = UNETGenerator(data_list, mode='inference')
    inference_generator = inference_generator.load_process()
    
    
    unet = unet()
    print("loading unet weights")
    unet.load_weights('unet_weights/UNET_20200622-003518_bestIoU.h5')
    weight_layer = WeightLayer()(unet.outputs)
    model = Model(inputs=unet.inputs, outputs=weight_layer)
    print("done compiling model")
    
    # level is always 2 for unet model generation
    level = 2
    sz = args.img_size
    N = args.num_tiles
    pad_val = args.pad_val
    all_coords = []
    for i,x in enumerate(tqdm(inference_generator)):
        crops = tf.squeeze(x[1])
        preds = model.predict(x[0])
        result = UNETGenerator.crop(preds, crops)
        img_id = x[2].numpy()[0].decode('utf8').split('/')[-1].split('.')[0]
        
        coords = get_tile_coords(result[0,...], sz, N, pad_val)
        all_coords.append(np.array([img_id, coords]))
        #print("img id in gen", img_id)
        #print("path in gen", x[2])
        #print(x[0].shape)
        #print(coords)
        
    
    all_coords = np.array(all_coords)
    all_coords = {'NUM_TILES': N, 'IMG_SIZE': sz, 'PAD_VAL': pad_val, 'TIFF_LEVEL': level, 'DATA': all_coords}
    np.save(f'UNET-{level}-{N}-{sz}-{pad_val}.npy', all_coords)