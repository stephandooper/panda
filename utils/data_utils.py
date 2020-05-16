# -*- coding: utf-8 -*-
"""
Created on Fri May  8 20:36:15 2020.

@author: Stephan
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

# TODO: augmentation
# TODO: preprocess (1- img values for black background, (img-mean)/std)
# TODO: TiffTileGenerator for test sets and immediate image generation
# TODO: PandaDataLoader: load split possibilities from train csv:        DONE
# TODO: add references to ideas


class PandasDataLoader(object):
    """ Not to be confused with the PANDAS challenge.
    
    Author: Zac Dannelly
    link: https://www.kaggle.com/dannellyz/collection-of-600-suspicious-slides-data-loader
        
    This class created a train/validation split from a pandas df in multiple
    ways (I found several ways people do it).
    
    There is also an increasing amount of faulty slides, such as :
        1. Background only images
        2. Images tainted by pen marks
        3. Images without masks
        4. Images without cancerous tissue, but ISUP grades > 3
           (mask/image comparison)
           
    An option should be provided to filter these faulty image categories out
    """
    
    def __init__(self, images_csv_path, skip_csv=None, skip_list=[]):
        """ Initializer for panda dataset loader.

        Parameters
        ----------
        images_csv_path : pathlib path or string
            A path of the train_csv for the PANDA challenge
        skip_csv : pathlib path or string, optional
            A path or . The default is None.
        skip_list : list, optional
            A list with the columns to be used for filtering in skip_csv.
            All other columns that are not present in this list, are removed
            from the skip_csv within the class. The default is [], which means
            that all columns are used for filtering.

        Raises
        ------
        ValueError
            You cannot pass a skip list without a skip csv file path

        Returns
        -------
        None.

        """
        assert isinstance(skip_list, list)
        
        if skip_csv is None and skip_list:
            raise ValueError("skip csv cannot be empty if skip list is not empty")

        # Load images and masks dataframes
        self.image_df = pd.read_csv(images_csv_path)
        shape_img = self.image_df.shape
        
        # image ids without masks are included in skip csv
        if skip_csv is not None:
            skip_df = pd.read_csv(skip_csv)
            shape_skip = skip_df.shape
            
            # all unique values for reason column
            skip_list_all = skip_df['reason'].unique().tolist()

            # If the argument is the default, filter using all columns
            if not skip_list:
                skip_list = skip_list_all
            
            # Filter out the desired columns (ones not in skip list)
            skip_df = skip_df[skip_df['reason'].isin(skip_list)]
                        
            # Filter image_df with the values in skip_df (filterd by skip_list)
            self.image_df = (self.image_df.merge(skip_df,
                                                 on='image_id',
                                                 how='outer',
                                                 indicator=True)
                             .query('_merge != "both"')
                             .drop(columns=['reason', '_merge'])
                             )
            
            # Print summary statistics
            print('*' * 20)
            print(f"The training dataframe shape before filtering:{shape_img}")
            print(f"The skip dataframe has shape: {shape_skip}, with reasons {skip_list_all}")
            print(f"Filtering based on the following columns: {skip_list}")
            print(f"number of duplicates in the skip df: {skip_df[skip_df['image_id'].duplicated()].shape}")
            print(f"Training dataframe after filtering: {self.image_df.shape}")
            print(f"Number of rows removed by filter: {shape_img[0] - self.image_df.shape[0]}")
            print('*' * 20)

        else:
            print("No skip file given or found, not filtering images/masks")
    
    def _create_folds(self, df, nfolds, SEED):
        """ An internal function for creating folds.

        Parameters
        ----------
        df : Pandas Dataframe
            The pandas dataframe for creating splits for
        nfolds : int
            The number of folds for cross validation.
        SEED : int
            A seed for rng.

        Returns
        -------
        df : Pandas DataFrame
         The dataframe with an added 'splits' columns
        """
        # Stratified splits
        splits = StratifiedKFold(n_splits=nfolds,
                                 random_state=SEED,
                                 shuffle=True)
        splits = list(splits.split(df, df.isup_grade))
        folds_splits = np.zeros(len(df)).astype(np.int)
        for i in range(nfolds):
            folds_splits[splits[i][1]] = i
        df['split'] = folds_splits
        return df
    
    
    def stratified_isup_sample(self, nfolds, SEED):
        """ Stratified split on isup categories.
        
        Parameters
        ----------
        nfolds : int
            The number of folds to split the data into
        SEED : int
            The seed.

        Returns
        -------
        df : Pandas DataFrame
            The pandas Dataframe, with an added split column
        """
        # The way training/test was split in lafoss' notebook
        # https://www.kaggle.com/iafoss/panda-concat-tile-pooling-starter-0-79-lb
        df = self.image_df
        df = self._create_folds(df, nfolds, SEED)
        
        return df
        
    def stratified_isup_dp_sample(self, nfolds, SEED):
        """ Stratified split on the individual data providers by isup grade.

        Parameters
        ----------
        nfolds : int
            The number of folds to split the data into
        SEED : int
            The seed.

        Returns
        -------
        df : Pandas DataFrame
            The pandas Dataframe, with an added split column
        """
        # the way training/validation was split in xie29 notebook
        # https://www.kaggle.com/xiejialun/panda-tiles-training-on-tensorflow-0-7-cv
        radboud_df = self.image_df[self.image_df['data_provider'] == 'radboud']
        karolinska_df = self.image_df[self.image_df['data_provider'] != 'radboud']
                
        radboud_df = self._create_folds(radboud_df, nfolds, SEED)

        karolinska_df = self._create_folds(karolinska_df, nfolds, SEED)

        self.train_df = pd.concat([radboud_df, karolinska_df])
        return self.train_df


class TiffTileGenerator(object):
    """ Generate tiles from .tiff files.
    
    Generate tiff tiles on the fly using this generator. This class has to be
    utilized during evaluation of the test set.
    """
    
    def __init__(self):
        pass


class PNGTileGenerator(object):
    """ Class that handles the loading and processing of concat tile pooling.

    The input is assumed to be a pandas dataframe function, and a path to
    the image folder.
    The images in the data folder are expected to contain PNG images,
    and represent shards of a larger image according to concat tile pooling
    i.e. a single image is represented by a batch of multiple images
    which is expressed in the filename according to the pattern
    <fname>_<X>.png where X represents the index of tiles per image, starting
    from 0.
    
    The dataframe must have the following columns: image_id, isup_grade
    """

    def __init__(self, df, img_dir, num_tiles=16, batch_size=8):
        """ Initializer to the DataLoaderPNG class.

        Parameters
        ----------
        df : pandas dataframe
            The pandas dataframe with the image ids <image_id> and ISUP
            grades <isup_grade>
        img_dir : Pathlib Path
            Path to the image folder, must be a Pathlib Path object
        num_tiles : int, optional
            the number of tiles per image. The default is 16.
        batch_size : int, optional
            The batch size to be used during network training.
            The default is 8.

        Returns
        -------
        None.

        """
        self.df = df
        self.img_dir = tf.convert_to_tensor(str(img_dir))
        self.num_tiles = num_tiles
        self.num_classes = df['isup_grade'].nunique()
        self.batch_size = batch_size
        
        # Tensorflow Dataset from the pandas df
        # Generate a dataset that yields (image_id, isup_grade) tuples
        self.isup_str = [str(x) for x in df['isup_grade'].tolist()]
        
        # Zip the image_id and isup dataset to yield tuple pairs (img_id, isup)
        new_ds = [*zip(self.df['image_id'].tolist(), self.isup_str)]
        
        # Create the dataset from the slices
        self.image_ids = tf.data.Dataset.from_tensor_slices(new_ds)
        
        # create a iter dataset for get_batch and display_batch methods
        self._ds_iter = iter(self.load_process())
        
    def _load_images(self, img_ids):
        """ Vectorized function for loading images.

        Parameters
        ----------
        path : Tensorflow tensor containing byte strings
            A [batch_size,1] Tensor byte string containing image_id's'

        Returns
        -------
        A [batch_size, num_tiles, W, H, C] tensor
            The images in a [batch_size, num_tiles, W, H, C] tensor

        """
        def get_image_ids(path):
            """ Get all the image paths from the id.
                    
            Parameters
            ----------
            path : Tensor of byte strings
                A [batch_size,1] tensor of byte strings

            Returns
            -------
            image_ids : Tensor of byte strings
                A [batch_size * num_tiles] tensor of byte strings, containing
                the full paths to the corresponding image files
            """
            # Infer the shape from the batch itself
            batch_shape = tf.shape(path)
            
            # Append the image path to the id's: <img_dir>/<img_id>
            # size: [batch_size]
            image_ids = tf.strings.join([self.img_dir, path], separator='/')
            
            # There are num_tiles tile images sharing the same id
            # [batch_size * num_tiles] e.g.: [0, 1, 2] -> [0, 0, 1, 1, 2, 2]
            image_ids = tf.repeat(image_ids, self.num_tiles)
            
            # Create a list of indices [0:num_tiles]
            indices = tf.constant(list(range(0, self.num_tiles)))
            
            # [num_tiles * batch_size] -> [0:num_tiles 0:num_tiles]
            indices = tf.tile(indices, [batch_shape[0]])
            
            # Convert the numbers to strings for joining
            indices = tf.strings.as_string(indices)
            
            # Add indices to the filenames with tile indices
            image_ids = tf.strings.join([image_ids, indices], separator='_')
            
            # Add png extension
            image_ids = tf.strings.join([image_ids, 'png'], separator='.')
            return image_ids
        
        def read_img(path):
            """ Reads the image from a path.

            Parameters
            ----------
            image_id : Tensor
                A tensor with the paths

            Returns
            -------
            image : Tensor
                A [W,H,C] Tensor containing the image
            """
            image = tf.io.read_file(path)
            image = tf.io.decode_png(tf.squeeze(image), channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.strings.as_string(image)
            return image
        batch_shape = tf.shape(img_ids)
        # Get all the images in the dir, and their full path from the id
        image_paths = get_image_ids(img_ids)
        
        # [batch_size * num_tiles, W, H, C]
        # Vectorized map, since read_file and decode are not vectorizes funs
        images = tf.map_fn(read_img, image_paths)
        images = tf.strings.to_number(images, out_type=tf.dtypes.float32)
        
        dims = tf.shape(images)
        
        # Reshape to [batch_size, num_tiles, W, H, C]
        return tf.reshape(images, (batch_shape[0], self.num_tiles,
                                   dims[1], dims[2], dims[3]))
    
    def _load_label(self, label):
        """ Get the label from the string.

        Parameters
        ----------
        label : Tensor
            Contains label as byte string, can be an array of strings
            [batch_size,1] array of byte strings

        Returns
        -------
        label
            A [batch_size, num_classes] int32 tensor, containing a
            one hot representation of the labels
        """
        label = tf.strings.to_number(label, out_type=tf.int64)
        return tf.one_hot(tf.cast(label, dtype=tf.int32),
                          self.num_classes, dtype=tf.int32)
    
    def parse_img_label(self, img_label):
        """ Parse images and label from (image_id, label) pairs.

        Parameters
        ----------
        img_label : tuple
            A tuple of image_id and label

        Returns
        -------
        images : Tensor
            A [batch_size, num_tiles, W, H, C] Tensor in float32, containing
            the image
        label : Tensor
            A [batch_size, num_classes] Tensor containing a one hot
            representation of the labels
        """
        path, label = img_label[:, 0], img_label[:, 1]
        
        label = self._load_label(label)
        images = self._load_images(path)
        
        return images, label
    
    def load_process(self,
                     mode='training',
                     cache=False,
                     shuffle_buffer_size=10000):
        """ Create a Tensorflow dataset that returns (image, label) pairs.
        
        Configuring the best pipeline is hard to do. We used the following
        guidelines to create a pipeline that is at least performing decently
        https://www.tensorflow.org/guide/data#randomly_shuffling_input_data
        https://www.tensorflow.org/guide/data_performance#vectorizing_mapping
        https://www.tensorflow.org/datasets/performances

        Parameters
        ----------
        mode : string, optional
            either 'validation' or 'training. The default is 'training'.
        cache : bool or string, optional
            A boolean to use cache, or a string to the tfcache location e.g.
            <path-to-cache.tfache> . The default is False.
        shuffle_buffer_size : TYPE, optional
            DESCRIPTION. The default is 10000.

        Returns
        -------
        ds : Tensorfloat Dataset
            can be directly fed into Keras pipeline, or eagerly executed
            in for loops. Will yield batches of batch_size containing
            image/label pairs of size ([num_tiles, W, H, C], num_classes)
        """
        assert mode in ('training', 'validation')
        
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        
        # shuffle before map (list of filenames is more efficient)
        if mode == 'training':
            ds = self.image_ids.shuffle(shuffle_buffer_size, seed=1)
        else:
            print("skipping shuffling operations in validation generator")
            ds = self.image_ids
        
        # batch before map: vectorize operations
        ds = ds.batch(self.batch_size, drop_remainder=False)
        
        # TODO: cache before map? idk if we should use it.
        # use cache(filename) to cache preprocessing work for datasets
        # that do not fit in memory
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        
        # map 1: transform to images
        ds = ds.map(self.parse_img_label, num_parallel_calls=AUTOTUNE)
        
        # TODO: map2: augmentations and other operations (random)
        
        # Repeat forever (only in training)
        if mode == 'training':
            ds = ds.repeat()

        # `prefetch` lets the dataset fetch batches in the background
        # while the model is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds
    
    def get_batch(self):
        """ Retrieve a batch from the dataset.
        
        Returns
        -------
        TF Tensor
            A [batch_size, num_tiles, W, H, C] array
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
        label_batch = tf.math.argmax(label_batch, axis=1)
        fig, ax = plt.subplots(max_batch_size, max_tile_size, figsize=(16, 18))
            
        # Plot each image in the dataframe
        for i, batch in enumerate(range(max_batch_size)):
            for tile in range(max_tile_size):
                ax[batch, tile].imshow(img_batch[batch, tile, ...])
                ax[batch, tile].axis('off')
                ax[batch, tile].set_title(f'label: {label_batch[i]}')
                
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
