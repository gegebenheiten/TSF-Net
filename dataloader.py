# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 03:19:04 2021

@author: biren
"""

import os
import pywt
import numpy as np
from scipy.fftpack import dct, idct
from skimage.util import view_as_blocks
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torchvision.utils as utils
import torchvision.transforms as tforms

#------------------------------------------------------------------------------

def img_to_patches(img, patch_size):
    
    h, w = img.shape[:2]
    x, y = np.mgrid[0:h:patch_size, 0:w:patch_size]
    x[-1, :] = (h-patch_size) if (x[-1, -1] + patch_size) > h else x[-1, :]
    y[:, -1] = (w-patch_size) if (y[-1, -1] + patch_size) > w else y[:, -1]
    
    x, y = x.flatten(), y.flatten()
    patches = [ img[i:i+patch_size, j:j+patch_size]  for i, j in zip(x, y) ]
    patches = np.stack(patches)
    return patches

#------------------------------------------------------------------------------

def patches_to_image(patches, image_size, patch_size):
    
    h, w, c = image_size
    image = np.zeros((h, w, c))
    
    x, y = np.mgrid[0:h:patch_size, 0:w:patch_size]
    x[-1, :] = (h-patch_size) if (x[-1, -1] + patch_size) > h else x[-1, :]
    y[:, -1] = (w-patch_size) if (y[-1, -1] + patch_size) > w else y[:, -1]
    x, y = x.flatten(), y.flatten()
    
    for  c, (i, j) in enumerate(zip(x, y)):
        image[i:i+patch_size, j:j+patch_size] = patches[c]
        
    return image

#------------------------------------------------------------------------------

# # Encoded Image Loader

class ImagePatchLoader(torch.utils.data.Dataset):
    
    def __init__(self, args, frame_list, nbit, transforms=False, train=True):
        self.args = args
        self.patch_size = args.patch_size
        self.block_size = 4
        self.img_range = 2**nbit -1
        self.data_path = args.div2kdata
        
        self.patch_coord = [ self.get_patchcoordinate(frame, args.num_patches_per_frame, train=train) for frame in frame_list]
        
        self.image_patches = [ self.get_patches(frame, patch_coord, frame_type='inloopdisabled')  for (frame, patch_coord) in zip(frame_list, self.patch_coord) ]
        self.partition_patches = [ self.get_patches(frame, patch_coord, frame_type='partition')  for (frame, patch_coord) in zip(frame_list, self.patch_coord) ]
        self.prediction_patches = [ self.get_patches(frame, patch_coord, frame_type='prediction')  for (frame, patch_coord) in zip(frame_list, self.patch_coord) ]
        self.uncompressed_patches = [ self.get_patches(frame, patch_coord, frame_type='uncompressed')  for (frame, patch_coord) in zip(frame_list, self.patch_coord) ]
        self.inloopenabled_patches = [ self.get_patches(frame, patch_coord, frame_type='inloopenabled')  for (frame, patch_coord) in zip(frame_list, self.patch_coord) ]
            
        self.image_patches = np.concatenate(self.image_patches)
        self.partition_patches = np.concatenate(self.partition_patches)
        self.prediction_patches = np.concatenate(self.prediction_patches)
        self.uncompressed_patches = np.concatenate(self.uncompressed_patches)
        self.inloopenabled_patches = np.concatenate(self.inloopenabled_patches)
        
        self.transforms = transforms
        self.train = train
    
    def get_patches(self, frame, patch_coord, frame_type='inloopdisabled'):
        if frame_type == 'inloopdisabled':
            image = self.deInterleaveYwithUV(np.load(os.path.join(self.data_path, 'InLoopFilterDisabled', f'qp-{self.args.qp}', frame[0])))
        elif frame_type == 'partition':
            image = self.deInterleaveYwithUV(np.load(os.path.join(self.data_path, 'Partition', f'qp-{self.args.qp}', frame[0])))
        elif frame_type == 'prediction':
            image = self.deInterleaveYwithUV(np.load(os.path.join(self.data_path, 'Prediction', f'qp-{self.args.qp}', frame[0])))
        elif frame_type == 'inloopenabled':
            image = self.deInterleaveYwithUV(np.load(os.path.join(self.data_path, 'InLoopFilterEnabled', f'qp-{self.args.qp}', frame[0])))
        elif frame_type == 'uncompressed':
            image = self.deInterleaveYwithUV(np.load(os.path.join(self.data_path, 'Uncompressed', frame[0])))
        patches = []
        for xy in patch_coord:
            x, y = xy[0], xy[1]
            patches.append(image[x:x+self.patch_size, y:y+self.patch_size])
        return patches
        
    def get_patchcoordinate(self, frame, no_patches, train=True):
        h, w = frame[1]    # shape info
        if train:
            x, y = np.mgrid[0:h-self.patch_size:25, 0:w-self.patch_size:25]
            x, y = x.flatten(), y.flatten()
            idx = np.random.choice(np.arange(len(x)), size=no_patches, replace=False)
        else:
            x, y = np.mgrid[0:h-self.patch_size:250, 0:w-self.patch_size:250]
            x, y = x.flatten(), y.flatten()
            idx = np.arange(len(x))
        return [[x[i], y[i]] for i in idx]
        
    
    def transform(self, *args):
        no_args = len(args)
        aug_data = [arg.copy() for arg in args]
        if np.random.random() > 0.5:
            # horizontal flip
            for i,data in enumerate(aug_data):
                aug_data[i] = np.fliplr(data).copy()
            
        if np.random.random() > 0.5:
            # vertical flip
            for i, data in enumerate(aug_data):
                aug_data[i] = np.flipud(data).copy()
            
        return aug_data
    
    def deInterleaveYwithUV(self, interleaved_yuv):
        h, w = np.array(interleaved_yuv[0].shape) * 2
        y = np.zeros((h, w), dtype=interleaved_yuv.dtype)
        y[0::2, 0::2] = interleaved_yuv[0]
        y[0::2, 1::2] = interleaved_yuv[1]
        y[1::2, 0::2] = interleaved_yuv[2]
        y[1::2, 1::2] = interleaved_yuv[3]
        
        #u = interleaved_yuv[4]
        #v = interleaved_yuv[5]
        return y
    
    def __len__(self):
        return len(self.uncompressed_patches)

    def __getitem__(self, idx):
        image = self.image_patches[idx]/self.img_range
        partition = self.partition_patches[idx]
        prediction = self.prediction_patches[idx]/self.img_range
        groundtruth = self.uncompressed_patches[idx]/self.img_range
        image_vvc = self.inloopenabled_patches[idx]/self.img_range
        
        if self.train: 
            if self.transforms:
                image_vvc, image, partition, prediction, groundtruth = self.transform(image_vvc, image, partition, prediction, groundtruth)
        
        image = torch.from_numpy(image[np.newaxis, :, :]).float()
        partition = torch.from_numpy(partition[np.newaxis, :, :].astype(np.float32)).float()
        prediction = torch.from_numpy(prediction[np.newaxis, :, :]).float()
        
        groundtruth = torch.from_numpy(groundtruth[np.newaxis, :, :]).float()
        return image, partition, prediction, groundtruth, image_vvc