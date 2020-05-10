# -*- coding: utf-8 -*-
"""
Created on Fri May  8 20:36:15 2020.

@author: Stephan
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path


class BaseDataLoader(object):
    """ The parent class for loading datasets.

    The input is assumed to be a tf TensorSliceDataset.
    In case of BaseDataLoaderPNG, see that class for more info
    """

    def __init__(self, ds_list, df, num_tiles, batch_size):
        """ Class initializer.
        
        Parameters
        ----------
        ds_list : Tensorflow TensorSliceDataset
            A dataset containing the paths to the images
        df : Pandas DataFrame
            Pandas dataframe containing the image_id, ISUP grade,
            and Gleason scores as columns.
        num_tiles : int
            The number of tiles. In case of PNG loader: the amount of tiles
            that were created, in case of TIFF loader: the amount of tiles
            to be created.
        batch_size : int
            The batch size that the processed Tensorflow dataset will yield.
            This batch size is also used during training networks

        Returns
        -------
        None.

        """
        # dataset for the images
        self._ds_list = ds_list
        
        self._batch_size = batch_size
        self._num_tiles = num_tiles
        
        # get the image ID in tensor format
        self._df = df
        self._image_id = tf.convert_to_tensor(self._df['image_id'])
        self._isup = tf.convert_to_tensor(self._df['isup_grade'])
        
        # dataset for labels
        self._label_ds = tf.data.Dataset.from_tensor_slices(self._isup)

    @property
    def ds_list(self):
        """
        Returns the original list.

        Returns
        -------
        TensorSlice Dataset
            returns the original dataset,a  generator that yields tensors
            of byte strings containing the paths to the image(s)
        """
        return self._ds_list
    
    @property
    def df(self):
        """Ni."""
        return self._df

    def __call__(self):
        """
        Return the dataset for Keras and Tensorflow compatibility.

        Returns
        -------
        Tensorflow processed dataset
            Returns a processed Tensorflow dataset that yield a batch of
            batch_size img/label pairs. This can then be used as input for
            training/validating models.

        """
        return self._ds


class DataLoaderPNG(BaseDataLoader):
    """ Class that handles the loading and processing of concat tile pooling.

    The input is assumed to be a dataset from list_files() function.
    The images in the data folder are expected to contain PNG images,
    and represent shards of a larger image according to concat tile pooling
    i.e. a single image is represented by a batch of multiple images
    which is expressed in the filename according to the pattern
    <fname>_<X>.png where X represents the total number of tiles per image
    """

    def __init__(self, ds_list, df, num_tiles, batch_size):
        """ Class initializer.
        
        Parameters
        ----------
        ds_list : Tensorflow TensorSliceDataset
            A dataset containing the paths to the images
        df : Pandas DataFrame
            Pandas dataframe containing the image_id, ISUP grade,
            and Gleason scores as columns.
        num_tiles : int
            The number of tiles. In case of PNG loader: the amount of tiles
            that were created, in case of TIFF loader: the amount of tiles
            to be created.
        batch_size : int
            The batch size that the processed Tensorflow dataset will yield.
            This batch size is also used during training networks

        Returns
        -------
        None.

        """
        super().__init__(ds_list, df, num_tiles, batch_size)
        
        # Initialize a TF dataset that returns (img/label) pairs
        self._ds = self.load_process(self._batch_size)
        
        self._ds_iter = iter(self._ds)

    def _load_image(self, path):
        """
        Process an image from a path.

        Parameters
        ----------
        path : Tensor of byte strings
            The path to the image

        Returns
        -------
        image : Tensorflow image
            [M, N, 3] image in Tensor format

        """
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image
    
    def _load_label(self, isup):
        
        return tf.one_hot(tf.cast(isup, dtype=tf.int32), 6, dtype=tf.int32)
    
    def load_process(self, batch_size, shuffle_buffer_size=1000, cache=True):
        """
        Create the Tensorflow dataset pipeline.

        Parameters
        ----------
        batch_size : int
            The batch size the dataset will yield
        shuffle_buffer_size : int, optional
            The size of the shuffle buffer. Shuffle buffers have to be filled.
            Larger means better randomness, but more memory consumption.
            Lower is faster and uses less memory, but less randomness.
            The default is 1000.
        cache : bool or string, optional
            Whether to use a cache or not. The cache improves data loading for
            large datasets (unable to hold in memory). In case of a string
            the file must be named according to <path_to_folder/fname.tfcache>.
            In case the file does not exist, it will be created.
            The default is True.

        Returns
        -------
        Tensorflow dataset
            A tensorflow dataset that can be executed eagerly, and will yield
            batches of self._batch_size. Shuffle buffer sizes and cache
            behaviour are dependent on user input.
            The dataset is inexhaustible (repeats forever)

        """
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        
        # map the parsing functions to gather (img, label) pairs
        self._ds = self._ds_list.map(self._load_image,
                                     num_parallel_calls=AUTOTUNE)
        
        # use a batching process to acquire concat tile pooling batches
        self._ds = self._ds.batch(self._num_tiles)
        
        self._labels = self._label_ds.map(self._load_label,
                                          num_parallel_calls=AUTOTUNE)

        # label stuff
        self._ds = tf.data.Dataset.zip((self._ds, self._labels))
        
        # use cache(filename) to cache preprocessing work for datasets
        # that do not fit in memory
        if cache:
            if isinstance(cache, str):
                self._ds = self._ds.cache(cache)
            else:
                self._ds = self._ds.cache()

        self._ds = self._ds.shuffle(buffer_size=shuffle_buffer_size)
        
        # Repeat forever
        self._ds = self._ds.repeat()
        
        # Make an actual batch of examples
        self._ds = self._ds.batch(self._batch_size)

        # `prefetch` lets the dataset fetch batches in the background
        # while the model is training.
        self._ds = self._ds.prefetch(buffer_size=AUTOTUNE)
        return self._ds
    
    def get_batch(self):
        """ Retrieve a batch from the dataset.
        
        Returns
        -------
        TF Tensor
            A [batch_size, num_tiles, M, N, C] array
            of images in Tensor format
        """
        return next(self._ds_iter)
    
    def display_batch(self, batch=None, max_batch_size=5, max_tile_size=10):
        """ Plot a batch of images.
        
        Parameters
        ----------
        batch : TF Tensor or numpy array, optional
            A [batch_size, num_tiles, M, N, C] array of images.
            The default is None.
        max_batch_size : int, optional
            The amount of the batch size to plot.
            Should be <= self._batch_size.The default is 5.
        max_tile_size : int, optional
            The number of tiles to plot, should be <= num_tiles.
            The default is 10.

        Returns
        -------
        None.

        """
        if batch is None:
            batch = self.get_batch()
         
        # The batch is [BS, NT, W, H, C], this is not a supported format
        # Loop over the batch (rows) and num_tiles (columns) and display
        img_batch, label_batch = batch
        
        fig, ax = plt.subplots(max_batch_size, max_tile_size, figsize=(16, 18))
            
        # Plot each image in the dataframe
        for batch in range(max_batch_size):
            for tile in range(max_tile_size):
                ax[batch, tile].imshow(img_batch[batch, tile, ...])
                ax[batch, tile].axis('off')
                ax[batch, tile].set_title(f'batch: {batch}, tile: {tile}')
                
    def print_fname_order(self, head=10):
        """Prints the fname order of ds_list, image_id, and df."""
        check_ds = self._ds_list.filter(lambda x: tf.strings.regex_full_match(x, ".*_0.png"))
        
        print("lists order as processed by tensorflow environment")
        for x, y, isup in zip(check_ds.take(head), self._image_id, self._isup):
            print("images:", Path(x.numpy().decode('utf8')).stem,
                  "labels:", y.numpy().decode('utf8'),
                  "isup grade", isup.numpy())

        print("original pandas dataframe")
        print(self._df[0:head])


def set_gpu_memory():
    """
    Configure Tensorflow Memory behaviour.
    
    Only use one GPU and do not reserve all memory

    Returns
    -------
    None.

    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,",
                  len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def seed_all():
    pass
