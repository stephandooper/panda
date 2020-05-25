# -*- coding: utf-8 -*-
"""
Created on Sat May 23 21:49:15 2020

@author: Stephan
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold



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