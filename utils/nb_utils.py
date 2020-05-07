# -*- coding: utf-8 -*-
"""
Created on Thu May  7 12:33:40 2020

@author: Stephan

This file contains utility functions to be run in notebooks.
Many of the function are taken from kernels in the panda kaggle challenge
https://www.kaggle.com/c/prostate-cancer-grade-assessment/overview

The functions are stored here to maintain readability within the notebook
as plotting functions do not add to understanding or readability at first glance
"""


import seaborn as sns
import matplotlib.pyplot as plt
import os
import openslide
from matplotlib import colors
import numpy as np
import skimage.io
import cv2
import PIL



def plot_count(df, feature, title='', size=2):
    ''' Plot the counts of a df wrt a column as stratification
    
    Args:
        df (pandas dataframe): The pandas dataframe
        feature (string): the column (feature) within the pandas dataframe
        title (string, optional): title to add to the plot. Defaults to ''.
        size (int, optional): figure size. Defaults to 2.

    Returns:
        None.

    '''
    
    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))
    total = float(len(df))
    sns.countplot(df[feature],order = df[feature].value_counts().index, palette='Set2')
    plt.title(title)
    
    # formatting the % text over each individual bar
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()



def plot_relative_distribution(df, feature, hue, title='', size=2):
    ''' Plots the count of a feature column, stratified by the feature in hue
    

    Args:
        df (pandas dataframe): The pandas dataframe
        feature (string): the column (feature) within the pandas dataframe
        hue (string): The stratification variable, i.e. the categories
        title (string, optional): title to add to the plot. Defaults to ''.
        size (int, optional): figure size. Defaults to 2.

    Returns:
        None.

    '''
    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))
    total = float(len(df))
    sns.countplot(x=feature, hue=hue, data=df, palette='Set2')
    plt.title(title)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()

    
def print_slide_details(slide, show_thumbnail=True, max_size=(600,400)):
    ''' Print some basic information about a slide
    
    Args:
        slide (openslide object): the openslide image
        show_thumbnail (bool, optional): shows the thumbnail image. Defaults to True.
        max_size (tuple of ints, optional): size of the thumbnail image. Defaults to (600,400).

    Returns:
        None.

    '''
    # Generate a small image thumbnail
    if show_thumbnail:
        display(slide.get_thumbnail(size=max_size))

    # Here we compute the "pixel spacing": the physical size of a pixel in the image.
    # OpenSlide gives the resolution in centimeters so we convert this to microns.
    spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)
    
    print(f"File id: {slide}")
    print(f"Dimensions: {slide.dimensions}")
    print(f"Microns per pixel / pixel spacing: {spacing:.3f}")
    print(f"Number of levels in the image: {slide.level_count}")
    print(f"Downsample factor per level: {slide.level_downsamples}")
    print(f"Dimensions of levels: {slide.level_dimensions}")


def display_patches(df, path, read_region=(1780,1950)):
    '''

    Args:
        df (pandas dataframe): The pandas dataframe
        path (string): the path to the image.
        read_region (tuple, optional): the center coordinates of the 
        region to extract from. Defaults to (1780,1950).

    Returns:
        None.

    '''
    
    # create figure and axes objects
    f, ax = plt.subplots(3,3, figsize=(16,18))
    
    # Plot each image in the dataframe
    for i,data_row in enumerate(df.iterrows()):
        # the image id is in the index, add tiff to it
        image = str(data_row[0])+'.tiff'
        
        # get the path
        image_path = os.path.join(path, image)
        image = openslide.OpenSlide(image_path)
        patch = image.read_region(read_region, 0, (256, 256))
        ax[i//3, i%3].imshow(patch) 
        
        #close the object to save memory and unwanted behaviour
        image.close()       
        ax[i//3, i%3].axis('off')
        ax[i//3, i%3].set_title(f'ID: {data_row[0]}\nSource: {data_row[1][0]} ISUP: {data_row[1][1]} Gleason: {data_row[1][2]}')

    plt.show()
    
    
def pairwise_plot(df1, df2, path, max_size=(640,400)):
    ''' Pairwise plot for karolinska and radboud images

    Args:
        df1 (pandas dataframe): The pandas dataframe for radboud or karolinska
        df1 (pandas dataframe): The pandas dataframe for radboud or karolinska
        path (TYPE): the path to the image directory
        max_size (tuple of ints, optional): size of the images. Defaults to (640,400).

    Returns:
        None.

    '''
    
    # only works with equal length dataframes for now
    assert df1.shape == df2.shape
    
    rows = df1.shape[0]
    
    f, ax = plt.subplots(rows,2, figsize=(16,18))
    
    for i, (kar_data, rad_data) in enumerate(zip(df1.iterrows(), df2.iterrows())):
        # get the image IDs
        image_id_kar = str(kar_data[0]) + '.tiff'
        image_id_rad = str(rad_data[0]) + '.tiff'

        # get the images = from the path
        image_path_kar = os.path.join(path, image_id_kar)
        image_path_rad = os.path.join(path, image_id_rad)

        # get the actual images
        image_kar = openslide.OpenSlide(image_path_kar)
        image_rad = openslide.OpenSlide(image_path_rad)

        # display the images
        ax[i, 0].imshow(image_kar.get_thumbnail(size=max_size))
        ax[i, 1].imshow(image_rad.get_thumbnail(size=max_size))

        # close
        image_rad.close()
        image_kar.close()

        ax[i,0].axis('off')
        ax[i,0].set_title(f'ID: {kar_data[0]}\nSource: {kar_data[1][0]} ISUP: {kar_data[1][1]} Gleason: {kar_data[1][2]}')

        ax[i,1].axis('off')
        ax[i,1].set_title(f'ID: {rad_data[0]}\nSource: {rad_data[1][0]} ISUP: {rad_data[1][1]} Gleason: {rad_data[1][2]}')
    plt.show()
    
    
def get_hist(df, level=2, remove_white=True):
    ''' Compute the cumulative histogram of images
    

    Parameters
    ----------
    df : pandas dataframe 
        The pandas dataframe with image ID.
    level : int, optional
        int in [0,1,2], controls the zoom level. The default is 2.
    remove_white : bool, optional
        whether to remove the white background. 
        This is done by removing all values equal to 255 The default is True.

    Returns
    -------
    None.

    '''
    for row in df.iterrows():
        
        # bin edges for the histogram
        bin_edges = np.linspace(1, 256, 256)
        
        # Keep track of everything in a bincount
        cum_hist = np.zeros([bin_edges.shape[0] -1, 3])
        
        # count the total number of pixel in the image
        pixel_count = np.zeros([1,3])

        # load file
        file = row[0] +'.tiff'
        path = os.path.join(IMG_DIR, file)
        biopsy = skimage.io.MultiImage(path)
        
        # load the image level, and optionally remove all max values (alot of white values)
        # incidentally, this can also mean that other values from white are also removed
        img = biopsy[level]


        for channel in [0,1,2]:
            channel_img = img[...,channel]
            
            if remove_white is True:
                channel_img = channel_img[channel_img !=255]
            
            hist, _ = np.histogram(channel_img.ravel(), bins = bin_edges)
            
            # add channel histogram information
            cum_hist[:,channel] += hist
            pixel_count[:,channel] += channel_img.size
        
        # clear memory
        del biopsy, img
        
    return cum_hist, bin_edges, pixel_count


def display_masks(slides, train_df, mask_dir): 
    f, ax = plt.subplots(5,3, figsize=(18,22))
    for i, slide in enumerate(slides):
        
        mask = openslide.OpenSlide(os.path.join(mask_dir, f'{slide}_mask.tiff'))
        mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])
        cmap = colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])

        ax[i//3, i%3].imshow(np.asarray(mask_data)[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5) 
        mask.close()       
        ax[i//3, i%3].axis('off')
        
        image_id = slide
        data_provider = train_df.loc[slide, 'data_provider']
        isup_grade = train_df.loc[slide, 'isup_grade']
        gleason_score = train_df.loc[slide, 'gleason_score']
        ax[i//3, i%3].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")
        f.tight_layout()
        
    plt.show()
    
    
def load_and_resize_image(img_id, img_dir):
    """
    Edited from https://www.kaggle.com/xhlulu/panda-resize-and-save-train-data
    """
    biopsy = skimage.io.MultiImage(os.path.join(img_dir, f'{img_id}.tiff'))
    return cv2.resize(biopsy[-1], (512, 512))

def load_and_resize_mask(img_id, mask_dir):
    """
    Edited from https://www.kaggle.com/xhlulu/panda-resize-and-save-train-data
    """
    biopsy = skimage.io.MultiImage(os.path.join(mask_dir, f'{img_id}_mask.tiff'))
    return cv2.resize(biopsy[-1], (512, 512))[:,:,0]


def plot_distribution_grouped(feature, feature_group, hist_flag=True):
    fig, ax = plt.subplots(nrows=1,figsize=(12,6)) 
    for f in train_df[feature_group].unique():
        df = train_df.loc[train_df[feature_group] == f]
        sns.distplot(df[feature], hist=hist_flag, label=f)
    plt.title(f'Images {feature} distribution, grouped by {feature_group}')
    plt.legend()
    plt.show()
    
    
def overlay_mask_on_slide(images, img_dir, mask_dir, center='radboud', alpha=0.8, max_size=(800, 800)):
    """Show a mask overlayed on a slide."""
    f, ax = plt.subplots(5,3, figsize=(18,22))
    
    
    for i, image_id in enumerate(images):
        slide = openslide.OpenSlide(os.path.join(img_dir, f'{image_id}.tiff'))
        mask = openslide.OpenSlide(os.path.join(mask_dir, f'{image_id}_mask.tiff'))
        slide_data = slide.read_region((0,0), slide.level_count - 1, slide.level_dimensions[-1])
        mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])
        mask_data = mask_data.split()[0]
        
        
        # Create alpha mask
        alpha_int = int(round(255*alpha))
        if center == 'radboud':
            alpha_content = np.less(mask_data.split()[0], 2).astype('uint8') * alpha_int + (255 - alpha_int)
        elif center == 'karolinska':
            alpha_content = np.less(mask_data.split()[0], 1).astype('uint8') * alpha_int + (255 - alpha_int)

        alpha_content = PIL.Image.fromarray(alpha_content)
        preview_palette = np.zeros(shape=768, dtype=int)

        if center == 'radboud':
            # Mapping: {0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}
            preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)
        elif center == 'karolinska':
            # Mapping: {0: background, 1: benign, 2: cancer}
            preview_palette[0:9] = (np.array([0, 0, 0, 0, 1, 0, 1, 0, 0]) * 255).astype(int)

        mask_data.putpalette(data=preview_palette.tolist())
        mask_rgb = mask_data.convert(mode='RGB')
        overlayed_image = PIL.Image.composite(image1=slide_data, image2=mask_rgb, mask=alpha_content)
        overlayed_image.thumbnail(size=max_size, resample=0)

        
        ax[i//3, i%3].imshow(overlayed_image) 
        slide.close()
        mask.close()       
        ax[i//3, i%3].axis('off')
                
        data_provider = train_df.loc[image_id, 'data_provider']
        isup_grade = train_df.loc[image_id, 'isup_grade']
        gleason_score = train_df.loc[image_id, 'gleason_score']
        ax[i//3, i%3].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")