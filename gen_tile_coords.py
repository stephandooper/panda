# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 17:03:28 2020

@author: Stephan
"""

from preprocessing.utils.tiling import get_tile_coords
import numpy as np
import skimage.io
from tqdm import tqdm
from pathlib import Path


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate tile coordinates')

    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/images/dataset/",
                        help='Directory of the Panda train images')
    parser.add_argument('--tiff_level', required=True,
                        metavar="tiff level (0,1,2)",
                        help="the tiff level, either (0,1,2)",
                        type=int)
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
    
    
    
    DATA_DIR = Path(args.dataset).glob('*.tiff')
    level = args.tiff_level
    sz = args.img_size
    N = args.num_tiles
    pad_val = args.pad_val
    all_coords = []
    for i,x in enumerate(tqdm(list(DATA_DIR))):
        img_id = x.stem
        image = skimage.io.MultiImage(str(x))[level]
        coords = get_tile_coords(image, sz, N, pad_val)
        all_coords.append(np.array([img_id, coords]))
        
    
    all_coords = np.array(all_coords)
    all_coords = {'NUM_TILES': N, 'IMG_SIZE': sz, 'PAD_VAL': pad_val, 'TIFF_LEVEL': level, 'DATA': all_coords}
    np.save(f'{level}-{N}-{sz}-{pad_val}.npy', all_coords)