# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 23:59:19 2020

@author: Stephan

Yet another generator :)
This time for Unet training and inference. 
It can be used for images of any sizes (or should be) for this particular use 
case (PANDA challenge). 

"""
import tensorflow as tf
from tensorflow.keras.layers import Cropping2D
import skimage.io
import numpy as np


# TODO: docs


class UNETGenerator(object):
    """ Generate image/mask pairs"""
    
    def __init__(self, 
                 img_paths,
                 mask_paths=None,
                 data_providers=None,
                 tiff_level=2,
                 ksize=(512, 512),
                 mode = 'training',
                 aug_func_list=[]):
        
        assert mode in ('training', 'validation', 'inference')
        
        self.mode=mode
        # only grab common ids 
        if mask_paths is not None and data_providers is not None and mode != 'inference':
            self.paths = list(zip(img_paths, mask_paths, data_providers))
        else:
            self.paths = list(img_paths)
    
        self.paths = tf.data.Dataset.from_tensor_slices(self.paths)
        self.tiff_level = tiff_level
        self.ksize = ksize
        self.aug_func_list = aug_func_list
    
    def _read_img(self, path):
        # decode byte string
        path = path.decode('utf8')
        img = skimage.io.MultiImage(path)[self.tiff_level]
        return img
        
        
    def load_img(self, path):
        img = tf.numpy_function(self._read_img, 
                                [path], tf.uint8, 
                                name='tiff_read')
        return img
    
    def adjust_label(self, label):
        label = label.astype('int32')
        label = label[:, :, :, 0] if (len(label.shape) == 4) else label[:, :, 0]
        
        new_label = np.zeros(label.shape + (3,)).astype('int32')
        for i in range(3):
            new_label[label == i, i] = 1
        label = new_label
        return label.astype('float32')
    
    def load_mask(self, path, dp):
        
        def map_labels(img):
            # radboud to karolinska
            
            # radboud score >=3 becomes 2
            c_1 = tf.greater_equal(img, tf.ones(tf.shape(img)) *3)
            
            # radboud scores in [1,2] become 1
            c_2 =tf.greater_equal(img, tf.ones(tf.shape(img))*1)
            c_3 = tf.less_equal(img, tf.ones(tf.shape(img))*2)
            
            # combine to an interval 1 < img < 2
            c_f = tf.logical_and(c_2, c_3)

            # first condition all socres >=3 -> 2
            mask = tf.where(c_1, 2., img)
            # Second condition: [1,2] -> 1
            mask = tf.where(c_f, 1., mask)

            return mask
        
        img = self.load_img(path)
        img = tf.cast(img, dtype=tf.float32)
        img = tf.cond(tf.equal(dp, tf.constant(['radboud'], dtype=tf.string)), 
                         lambda:map_labels(img), 
                         lambda:img)

        return img
    
    
    def parse_img_mask(self, img_mask_dp):
        img_path, mask_path, dp = (img_mask_dp[0], 
                                   img_mask_dp[1], 
                                   img_mask_dp[2])
        
        # tf uint 8 
        image = self.load_img(img_path)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # tf float 32
        mask = self.load_mask(mask_path, dp)
        
        # return 1 - image, since we use patch extracting with padding
        # in create_patches, which uses zero padding
        return 1 - image, mask
    
    def create_patches(self, img, mask):        
        # do not convert image dtype before separation
        img, mask = tf.expand_dims(img, 0), tf.expand_dims(mask, 0)
        
        batch = tf.concat((img, mask), axis=0)

        # [(img, mask), patches_per_row, patches_per_column, num_pixels_per_patch]
        image_patches=tf.image.extract_patches(batch, 
                                               [1, self.ksize[0], self.ksize[1], 1], 
                                               [1, self.ksize[0], self.ksize[1], 1], 
                                               [1,1,1,1], 
                                               padding='SAME', 
                                               name=None)

        def construct(image_patches):
            # extract image and mask from the input
            images = image_patches[0,...]
            masks = image_patches[1,...]
            
            # shape, should be the same for mask and image
            row, col, pixels = masks.shape
            
            # reshape the last dimension back to an image format
            images = images.reshape((-1, self.ksize[0], self.ksize[1], 3))
            
            # A dummy is created for summing later on
            dummy_mask = masks.reshape((-1, self.ksize[0], self.ksize[1], 3))
            dummy_mask = dummy_mask[...,0]

            # tf image extract introduces extra parts due to padding
            # reshape it back into a 'big' image, and then adjust labels
            # otherwise, the masks will be wrong
            masks = masks.reshape(row, col, self.ksize[0], self.ksize[1], 3)
            masks = masks.transpose((0, 2, 1, 3, 4))
            masks = masks.reshape((row * self.ksize[0], col * self.ksize[1], 3))
            masks = self.adjust_label(masks)
            
            # reshape the mask back to tiles (patches)
            masks = masks.reshape((row,  self.ksize[0], col,self.ksize[1], 3))
            masks = masks.transpose((0, 2, 1, 3, 4))
            masks = masks.reshape(-1, self.ksize[0], self.ksize[1], 3)
            
            # sum the values in the dummies, filter out black (no information) areas
            sums = np.sum(dummy_mask, axis=(1,2))
            
            # get the indices with at least some biopsies (>0)
            # there are also some patches with very small amount of tissue,
            # filter these out by setting a higher number
            idxs = np.squeeze(np.where(sums > 5000.0))
            # filter out non-biopsy areas
            masks = masks[idxs]
            
            # Return the image to its original representation (we did 1 - image before)
            images = 1 - images[idxs]
            return images, masks

        result = tf.numpy_function(construct, 
                                  [image_patches], [tf.float32, tf.float32], 
                                   name='construct')

        return result[0], result[1]
    
    def set_shape(self, images, masks):
        images.set_shape([None, self.ksize[0], self.ksize[1], 3])
        masks.set_shape([None, self.ksize[0], self.ksize[1], 3])
        return images, masks
    
    
    def pad_image(self, image):
        shape = image.shape
        # 256 is just a number to get dimensions through Unet
        # to have correct shapes throughout the model
        # it was found by 'guessing'
        pad0, pad1 = ((256 - shape[0] % 256) % 256, 
                      (256 - shape[1] % 256) % 256)
        
        # pad to the right and on the bottom
        img = np.pad(image,
                 [[0, pad0],
                  [0, pad1],
                  [0, 0]],
                  constant_values=1.0)
        
        crops  = (0, pad0,
                  0, pad1)

        return img, crops
    
    def inference(self, path):
        image = self.load_img(path)
        image = tf.image.convert_image_dtype(image,tf.float32)
        image = tf.numpy_function(self.pad_image, [image],
                                  [tf.float32, tf.int64])

        image, crops = image[0], image[1]
        return image, crops, path
    
    @staticmethod
    def crop(image, crop_coords):
        return Cropping2D(((crop_coords[0], crop_coords[1]), 
                           (crop_coords[2], crop_coords[3])))(image)

    
    def load_process(self, batch_size=16):
        # shuffling array of strings: not all that intensive
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        if self.mode=='training' or self.mode=='validation':
            ds = self.paths.shuffle(buffer_size=5000)

            # parse image and masks
            ds = ds.map(self.parse_img_mask, num_parallel_calls=AUTOTUNE)

            # create patches from full images
            ds = ds.map(self.create_patches, num_parallel_calls=AUTOTUNE)

            # filter out empty images/masks, this can probably be done better
            ds = ds.filter(lambda x,y: tf.rank(x) > 3)

            # set the shape: losses/metrics don't like undefined shapes
            ds = ds.map(self.set_shape, num_parallel_calls=AUTOTUNE)

            # decouple the image in smaller non related batches
            ds = ds.unbatch()
            
            for f in self.aug_func_list:
                ds = ds.map(lambda x,y: (f(x,y)), num_parallel_calls=AUTOTUNE)
            
            
            # shuffle images, not many
            ds = ds.shuffle(buffer_size=500)

            # set custom batch size
            ds = ds.batch(batch_size)
            
            #repeat
            if self.mode=='training':
                ds = ds.repeat()
            
        elif self.mode=='inference':
            ds = self.paths.map(self.inference, num_parallel_calls=AUTOTUNE)          
            ds = ds.batch(1)
        return ds