# -*- coding: utf-8 -*-
"""
Created on Sat May 23 21:51:58 2020

@author: Stephan
"""

import tensorflow as tf
import numpy as np
from preprocessing.augmentations import augment_tile
from preprocessing.utils.tiling import tf_tile
import skimage.io
import functools
import matplotlib.pyplot as plt

class TiffGenerator(object):
    """ Generate tiles from .tiff files.
    
    Generate tiff tiles on the fly using this generator.
    Additionally, it offers parameters for tiling/tissue detect preprocessing
    algorithms, which can be passed through the tile_params and td_params
    dictionaries
    """
    
    def __init__(self, 
                 df, 
                 img_dir, 
                 tiff_level=2, 
                 batch_size=8, 
                 preproc_method='tile', 
                 aug_func=None, 
                 tile_params=None,
                 td_params=None):
        """ Class initializer

        Parameters
        ----------
        df : Pandas Dataframe
            Pandas Dataframe containing image_id and isup_grade columns
        img_dir : Pathlib path or string
            The path to the directory, containing TIFF images
        tiff_level : int, optional
            The level to which to load the tiff in (0,1,2). 0 is the original
            super resolution, 1 is 4x downsamples, 2 is 16x downsampled.
            The default is 2.
        batch_size : int, optional
            The batch size for training. The dataset will return arrays of
            [batch_size, img, label] . The default is 8.
        preproc_method : string, optional
            The preprocessing method, either 'tile' or 'tissue_detect'. 
            The default is 'tile'.
        aug_func : func, optional
            An augmentation routine function, containing Albumentation 
            augmentations. The default is None.
        tile_params : dict, optional
            A dictionary with optional parameters for tiling. The options are:
                sz: int 
                    containing the size of an individual tile, default is 128
                N: int 
                    The number of tiles to generate. The default is 16
                pad_val: int
                    The value for the paddings. The default is 255
                    This should be 255 for white backgrounds (tissue samples)
                    and 0 for black backgrounds (masks)
        td_params : dict, optional
            Optional kwargs for tissue detect algorithm. Currently not 
            implemented!

        Yields
        ------
        None.

        """
        
        self.df = df
        self.img_dir = img_dir
        self.tiff_level=tiff_level
        self.batch_size = batch_size
        self.preproc_method = preproc_method
        self.aug_func = aug_func
        
        self.tile_params = {} if tile_params is None else tile_params
        self.td_params = {} if td_params is None else td_params
        

        
        # convert image dirs and classes to tensor formats
        self.img_dir = tf.convert_to_tensor(str(img_dir))
        self.num_classes = df['isup_grade'].nunique()
        
        # Generate a dataset that yields (image_id, isup_grade) tuples
        self.isup_str = [str(x) for x in df['isup_grade'].tolist()]
        
        # Zip the image_id and isup dataset to yield tuple pairs (img_id, isup)
        image_ids = [self.img_dir +'/' + x + '.tiff' for x in self.df['image_id'].tolist()]
        new_ds = [*zip(image_ids, self.isup_str)]
        
        # Create the dataset from the slices
        self.image_ids = tf.data.Dataset.from_tensor_slices(new_ds)
        
        # create a iter dataset for get_batch and display_batch methods
        self._ds_iter = iter(self.load_process(shuffle_buffer_size=1))     
        

    def _read_img(self, path):
        """ Read an image from a path
        
        Parameters
        ----------
        path : Tensorflow byte string
            A tensorflow bytestring containing the path to the image

        Returns
        -------
        img : Tensorflow array
            A [W, H, 3] array containing the image with tiles
        TYPE
            DESCRIPTION.

        """
        
        def read_skim(path):
            
            # decode byte string
            path = path.decode('utf8')
            img = skimage.io.MultiImage(path)[self.tiff_level]
            return img, img.shape

        img = tf.numpy_function(read_skim, 
                                [path], [tf.uint8, tf.int64], 
                                name='tiff_read')
        img, dims = img[0], img[1]

        return img, dims
    
    def _load_label(self, label):
        """ Get the label from the string.

        Parameters
        ----------
        label : Tensor
            Contains label as byte string, can be an array of strings
            [1] array of byte strings

        Returns
        -------
        label
            A [num_classes] int32 tensor, containing a
            one hot representation of the labels
        """            
        label = tf.strings.to_number(label, out_type=tf.int64)
        return tf.one_hot(tf.cast(label, dtype=tf.int32),
                          self.num_classes, dtype=tf.int32)
    
    
    def _parse_img_label(self, preproc_method):
        ''' Obtain image and labels from a path
        
        Parameters
        ----------
        preproc_method : string
            The preprocessing method, can be 'tile', or 'tissue_detect'

        Raises
        ------
        NotImplementedError
            When tissue_detect is chosen, since it is not implemented yet
        ValueError
            When the specified method is not 'tile' or 'tissue_detect'

        Returns
        -------
        fun
            returns the wrapper function with the given preprocessing method
            initialized.

        '''
        
        # set the preprocessing functions
        if preproc_method =='tile':
            preproc_func = tf_tile
            preproc_kwargs = self.tile_params
            
        elif preproc_method == 'tissue_detect':
            raise NotImplementedError("not implemented yet, only tiling available")
        else:
            raise ValueError("Unknown preprocessing method, must be either 'tile' or 'tissue_detect'")   
        
        def wrapper(path_label):
            path, label = path_label[0], path_label[1]

            label = self._load_label(label)
            
            img, shape = self._read_img(path)
            
            # preprocess the image according to tiling or tissue detect
            img = preproc_func(img, shape, **preproc_kwargs)

            # cast to tf float 32
            img = tf.image.convert_image_dtype(img, tf.float32)
            return img, label
        
        return wrapper    
        
    
    def load_process(self,
                     mode='training',
                     shuffle_buffer_size=1000):
        """ The dataset loading process
        
        Parameters
        ----------
        mode : string, optional
            Either 'validation' or 'training'. Validation does not shuffle 
            the dataset. The default is 'training'.
        shuffle_buffer_size : int, optional
            The size of the shuffle buffer. A higher number is computationally 
            heavy, but has better pseudo randomness, a lower number is less
            less random. The default is 1000.

        Returns
        -------
        ds : Tf Dataset Adapter
            returns an iterable dataset that yields a batch_size of tuples
            (img, label)

        """
        
        assert mode in ('training', 'validation')

        AUTOTUNE = tf.data.experimental.AUTOTUNE
        
        # shuffle before map (list of filenames is more efficient)
        if mode == 'training':
            ds = self.image_ids.shuffle(shuffle_buffer_size, seed=1)
        else:
            ds = self.image_ids
        
        ds = ds.map(self._parse_img_label(self.preproc_method), num_parallel_calls=AUTOTUNE)
        #ds = ds.map(tile_aug(), num_parallel_calls=AUTOTUNE)
        # map before batch here: we do not want to load giant images in batch before preprocessing into tiles
        
        if self.aug_func is not None:
            ds = ds.map(self._aug(self.preproc_method, self.aug_func), num_parallel_calls=AUTOTUNE)
            
        # Gather images into a batch
        ds = ds.batch(self.batch_size)
        
        # `prefetch` lets the dataset fetch batches in the background
        # while the model is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        
        return ds
    
    def _aug(self, preproc_method, aug_func):
        """ Augmentation function

        Parameters
        ----------
        preproc_method : string
            The preprocessing method, either 'tile' or 'tissue_detect'
        aug_func : func, optional
            An augmentation routine function, containing Albumentation 
            augmentations. The default is None.

        Raises
        ------
        NotImplementedError
            Tissue detect is not implemented yet.

        Returns
        -------
        aug_process : func
            a function with the pre-loaded aug_func routine

        """
        if preproc_method == 'tile':
            aug_process = functools.partial(augment_tile, aug_func=aug_func)
        elif preproc_method =='tissue_detect':
            raise NotImplementedError("This is not implemented yet!") 
        
        return aug_process
    
    def get_batch(self):
        """ Retrieve a batch from the dataset.
        
        Returns
        -------
        TF Tensor
            A [batch_size, num_tiles, W, H, C] array
            of images in Tensor format
        """
        return next(self._ds_iter)
   
    # TODO: refactor for tissue detect
    def display_batch(self, batch=None, plot_grid = (4,4)):
        """ Plot a batch of images.
        
        Parameters
        ----------
        batch : TF Tensor or numpy array, optional
            A [batch_size, num_tiles, M, N, C] array of images.
            The default is None.
        plot_grid: tuple of ints
            A tuple of ints displaying the grid for the plot
            
        Returns
        -------
        None.
        """
        if batch is None:
            batch = self.get_batch()
        # The batch is [BS, NT, W, H, C], this is not a supported format
        # Loop over the batch (rows) and num_tiles (columns) and display
        img_batch, label_batch = batch
        img_batch = img_batch.numpy()
        label_batch = label_batch.numpy()
        label_batch = np.argmax(label_batch, axis=1)
        
        bs, nt, w, h, c = img_batch.shape
        rows = int(np.sqrt(nt))
        cols = int(nt / rows)
        print("shape of batch, done batch creating", img_batch.shape)
        img_batch = np.reshape(img_batch, (bs,
                                           rows,
                                           cols,
                                           w, h, c))
        
        img_batch = np.transpose(img_batch, (0, 1, 3, 2, 4, 5))
        img_batch = np.reshape(img_batch, (bs, w * rows, h * cols, c))
        img_batch = np.uint8(img_batch * 255)
        fig, ax = plt.subplots(plot_grid[0], plot_grid[1], figsize=(16, 18))
        print("done reshaping image", img_batch.shape)
        # Plot each image in the dataframe
        tot = 0
        for row in range(plot_grid[0]):
            for col in range(plot_grid[1]):
                ax[row, col].imshow(img_batch[tot, ...])
                ax[row, col].axis('off')
                ax[row, col].set_title(f'label: {label_batch[tot]}')
                tot +=1
        print("done plotting")
                
            
    def display_augmentation(self):
        '''Display augmentations of a single image of tiles

        Returns
        -------
        None.

        '''
        # Temporarily change the batch size
        temp = self.batch_size
        self.batch_size = 1
        
        # Shuffle buffer to 1 eliminates shuffling
        ds = self.load_process(shuffle_buffer_size=1)
        
        # Create a batch to parse to display_batch
        ds = next(iter(ds.unbatch().take(1).repeat(16).batch(16)))
        
        # display
        self.display_batch(ds, plot_grid=(4,4))
        
        # reset the batch size to the original value
        self.batch_size = temp
        
    def __call__(self, mode='training'):
        """Call method, that returns the dataset defined in self.load_process.

        Parameters
        ----------
        mode : string, optional
            either 'training' or 'validation'. The default is 'training'.
            validation will not repeat or shuffle

        Returns
        -------
        Tensorflow Dataset
            can be directly fed into Keras pipeline, or eagerly executed
            in for loops. Will yield batches of batch_size containing
            image/label pairs of size ([num_tiles, W, H, C], num_classes)
        """
        assert mode in ('training', 'validation')
        
        # return a dataset
        return self.load_process(mode)