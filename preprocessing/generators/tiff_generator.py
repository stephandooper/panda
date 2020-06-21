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
                 aug_func=None, 
                 tile_params=None,
                 one_hot=True,
                 tf_aug_list=[],
                 img_transform_func=None):
        
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
        aug_func : func, optional
            An augmentation routine function, containing Albumentation 
            augmentations. The default is None. These augmentations are done
            on an image wide level (i.e on all NUM_TILES simultaneously). This
            is useful for costly computations such as stain augmentation
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
        one_hot : bool, optional
            The choice whether to use one hot encoding for the labels (True)
            or sparse encodings (False). The default is True
        tf_aug_list : list, optional
            A list of tensorlfow augmentation functions. The functions must
            have as input a tensorflow Tensor, and also output a Tensorflow
            Tensor. The augmentations will be done per tile element i.e not on
            an entire image like aug_func. An example of this is
            [t_rot_func, tf_flip_func] which is a list of 2 augmentations which
            will be applied sequentially. The default is []
        img_transform_func : func, None
            An optional image transformation function. This can be useful
            for when the list of tiles needs to be changed into e.g. one big
            image. This can be done by making a function that trasnforms
            the [NT, H, W, C] tensorflow array into a [Hnew, Wnew, C] array
            and passing it to img_transform func. The default is None, so no
            function is passed.
        Yields
        ------
        None.

        """
        
        self.df = df
        self.img_dir = img_dir
        self.tiff_level=tiff_level
        self.batch_size = batch_size
        self.aug_func = aug_func
        self.tile_params = {} if tile_params is None else tile_params
        self.one_hot=one_hot
        self.tf_aug_list = tf_aug_list
        self.img_transform_func = img_transform_func
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
        label = tf.strings.to_number(label, out_type=tf.int32)
        if self.one_hot:
            label = tf.one_hot(label, self.num_classes, dtype=tf.int32)
            
        return label
    
    
    
    def _parse_img_label(self, path_label):
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
        path, label = path_label[0], path_label[1]

        label = self._load_label(label)

        img, shape = self._read_img(path)

        # preprocess the image according to tiling or tissue detect
        img = tf_tile(img, shape, **self.tile_params)

        # cast to tf float 32
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img, label
        
    def load_process(self,
                     mode='training',
                     shuffle_buffer_size=10000):
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
        
        ds = ds.map(self._parse_img_label, num_parallel_calls=AUTOTUNE)
        #ds = ds.map(tile_aug(), num_parallel_calls=AUTOTUNE)
        # map before batch here: we do not want to load giant images in batch before preprocessing into tiles
        
        if self.aug_func is not None:
            print("USING AN AUGMENTATION ROUTINE")
            ds = ds.map(self._aug(self.aug_func), num_parallel_calls=AUTOTUNE)
        
        
        for f in self.tf_aug_list:
            ds = ds.map(lambda x,y: (f(x),y), num_parallel_calls=AUTOTUNE)
            
            
        if self.img_transform_func is not None:
            ds = ds.map(lambda x,y: (self.img_transform_func(x),y), num_parallel_calls=AUTOTUNE)
        
        # black background for zero pooling
        ds = ds.map(lambda x,y: ((1 - x), y), num_parallel_calls=AUTOTUNE)
        
        # Gather images into a batch, drop remainder for cohen kappa batches (cannot be 1)
        ds = ds.batch(self.batch_size, drop_remainder=True)
        
        # `prefetch` lets the dataset fetch batches in the background
        # while the model is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        
        return ds
    
    def _aug(self, aug_func):
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
        aug_process = functools.partial(augment_tile, aug_func=aug_func)

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
    def display_batch(self, batch=None, plot_grid = (4,4), rows=None, cols=None):
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
        # The batch is [BS, NT, H, W, C], this is not a supported format
        # Loop over the batch (rows) and num_tiles (columns) and display
        img_batch, label_batch = batch
        img_batch = img_batch.numpy()
        label_batch = label_batch.numpy()
        
        if self.one_hot:
            label_batch = np.argmax(label_batch, axis=1)
        
        if len(img_batch.shape) == 5:
            bs, nt, h, w, c = img_batch.shape
            
            if rows is None:
                rows = int(np.sqrt(nt))
            if cols  is None:
                cols = int(nt / rows)
                
            print("shape of batch, done batch creating", img_batch.shape)
            img_batch = np.reshape(img_batch, (bs,
                                               rows,
                                               cols,
                                               h, w, c))
            
            img_batch = np.transpose(img_batch, (0, 1, 3, 2, 4, 5))
            img_batch = np.reshape(img_batch, (bs, h * rows, w * cols, c))
            img_batch = np.uint8(img_batch * 255)
        else:
            bs, h, w, c = img_batch.shape
        
        
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
                
            
    def display_augmentation(self, plot_grid=(4,4), rows=None, cols=None):
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
        self.display_batch(ds, plot_grid, rows, cols)
        
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
    
    
class TiffFromCoords(TiffGenerator):
    """ Generates samples from tiff files according to a coordinate file
    the coordinates can be generated for a specific number of tiles, image size
    and pad_val
    
    The coordinates are supposed to be in a npy file containing a dictionary
    containng the following keyword:
        NUM_TILES: an int given the number of tiles
        TIFF_LEVEL: the tiff level (int)
        PAD_VAL: the padding values used during padding
        IMG_SIZE: and int with the image size (square)
        DATA: a np array containing the image_id, and [NUM_TILES, 2] coordinates
            per image id. The coordinates are supposed to be unnormalized
            and uncentered (i.e regular pixel coordinates)
    """
    
    def __init__(self, 
                 coords,
                 df, 
                 img_dir,  
                 batch_size=8, 
                 aug_func=None, 
                 one_hot=True,
                 tf_aug_list=[],
                 img_transform_func=None):
        """Class initializer

        Parameters
        ----------
        coords : dict
            A dictionary containing the information posted above.
        df : Pandas DataFrame
            A dataframe containing the image ids and isup grades
        img_dir : string or pathlib object
            The path to the image directory
        batch_size : int, optional
            The batch size for training. The default is 8.
        aug_func : function, optional
            DESCRIPTION. An augmentation routine function.
        one_hot : bool, optional
            Whether or not to use sparse or one hot labels. The default is True.
        tf_aug_list : list of functions, optional
            A list of tf augmentation functions. For example, 
            [tf_rotate, tf_flip] where tf_rotate takes a (batch of) images
            and returns the augmentated images. The default is [].

        Yields
        ------
        None.

        """
        
        # Dictionary with keys: NUM_TILES, IMG_SIZE, PAD_VAL, DATA, TIFF_LEVEL
        coords = coords.flatten()[0]
               
        self.df = df
        self.img_dir = img_dir
        self.tiff_level=coords['TIFF_LEVEL']
        self.img_size = coords['IMG_SIZE']
        self.pad_val = coords['PAD_VAL']
        self.num_tiles = coords['NUM_TILES']
        
        # get array with image ids
        self.coord_ids = coords['DATA'][:,0]
        
        # get the arrays of corresponding coordinates
        self.coords = coords['DATA'][:,1]
        
        self.batch_size = batch_size
        self.aug_func = aug_func
        self.one_hot=one_hot
        self.tf_aug_list=tf_aug_list
        self.img_transform_func = img_transform_func

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
        """ Read the image from a tensor byte string path

        Parameters
        ----------
        path : A tf Tensor
            Tf tensor containing a byte string with the path to the image

        Returns
        -------
        tf tensor
            A [NUM_TILES, H,W,C] tensor containing the image patch
        """
        
        def read_skim_regions(path):
            """ Read image regions

            Parameters
            ----------
            path : A tf Tensor
                Tf tensor containing a byte string with the path to the image

            Returns
            -------
            NP array
                A [NUM_TILES, H, W, C] numpy array containing the image patch

            """
            
            result = []
            # decode byte string
            path = path.decode('utf8')
            img = skimage.io.MultiImage(path)[self.tiff_level]
            shape = img.shape
            
            pad0, pad1 = ((self.img_size - shape[0] % self.img_size) % self.img_size, 
                         (self.img_size - shape[1] % self.img_size) % self.img_size)

            img = np.pad(img,
                 [[pad0 // 2, pad0 - pad0 // 2],
                  [pad1 // 2, pad1 - pad1 // 2],
                  [0, 0]],
                 constant_values=self.pad_val)
            
            ids = path.split('/')[-1].split('.')[0]
            
            idx = np.where(self.coord_ids == ids)
            coords = self.coords[idx][0]
            
            for x,y in coords:
                result.append(img[x:x + self.img_size, y:y+self.img_size, :])
            
            result = np.array(result, dtype=np.uint8)
            
            if len(result) < self.num_tiles:
                
                result = np.pad(result, [[0, self.num_tiles - len(result)],
                   [0, 0],
                   [0, 0],
                   [0, 0]], constant_values=self.pad_val)
  
                
            return result

        img = tf.numpy_function(read_skim_regions, 
                                [path], tf.uint8, 
                                name='tiff_read')
        img.set_shape([self.num_tiles, self.img_size, self.img_size, 3])

        return img
    
    def _parse_img_label(self, path_label):
        """ Parse image and labels from (path, label) tuples

        Parameters
        ----------
        path_label : Tuple of (Tf byte string, tf string)
            A tuple of strings containing the path to the image and label
            
        Returns
        -------
        img : Tensorflow Array
            A [NUM_TILES, H, W, C] array containing the list of tiled images
        label : TF array
            A Tensorflow arra containing the label. It is either a single
            integer (sparse), or 1D array of size NUM_CLASSES in case
            of one hot encodings.

        """
        path, label = path_label[0], path_label[1]

        label = self._load_label(label)

        #img, shape = self._read_img(path)

        # preprocess the image according to tiling or tissue detect
        img = self._read_img(path)

        # cast to tf float 32
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img, label
