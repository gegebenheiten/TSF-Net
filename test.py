#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 14:31:20 2021

@author: biren
"""

import os
import shutil
import time
import copy
import pywt
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from skimage.util import view_as_blocks

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as tforms

from options import args_parser
from models.EDSR import EDSR
from utility import get_psnr, get_PSNR, saveas_image #, interleaveYwithUV, deInterleaveYwithUV
from seq_info import seq_info_sdr, read_seq_config

from img_op import blockify, unblockify
from torch_dct import dct_2d

best_psnr = 0
nbit = 10

#------------------------------------------------------------------------------

def pad_image(image, patch_size, padding='center'):
    
    image_shape = np.array(image.shape)
    image_padded_shape = np.ceil(image_shape/patch_size).astype(int) * patch_size
    image_padded = np.zeros(image_padded_shape)
    
    if padding == 'topleft':
        image_padded[:image_shape[0], :image_shape[1]] = image   #top-left
    
    if padding == 'center':
        offset_topleft = (image_padded_shape - image_shape)//2
        offset_bottomright = offset_topleft + image_shape
        image_padded[offset_topleft[0]:offset_bottomright[0], offset_topleft[1]:offset_bottomright[1]] = image   #centered
    
    return image_padded
    
#------------------------------------------------------------------------------

def image_to_patches(image, patch_size, mode='no_padding', padding='topleft'):
    
    if mode == 'padded':
        padded_image = pad_image(image, patch_size, padding=padding)
        patches = view_as_blocks(padded_image, (patch_size, patch_size))
        block_size = patches.shape
        patches = patches.reshape(block_size[0]*block_size[1], patch_size, patch_size)
        #patches = patches.reshape((-1, patch_size, patch_size))
        
    if mode == 'no_padding':
        h, w = image.shape
        x, y = np.mgrid[0:h:patch_size, 0:w:patch_size]
        x[-1, :] = (h-patch_size) if (x[-1, -1] + patch_size) > h else x[-1, :]
        y[:, -1] = (w-patch_size) if (y[-1, -1] + patch_size) > w else y[:, -1]
        
        x, y = x.flatten(), y.flatten()
        patches = [ image[i:i+patch_size, j:j+patch_size]  for i, j in zip(x, y) ]
        patches = np.stack(patches)
    return patches

#------------------------------------------------------------------------------

def patches_to_image(patches, image_size, patch_size, mode='no_padding', padding='topleft'):
    image_size = np.array(image_size)
    
    if mode == 'padded':
        image_padded_size = np.ceil(image_size/patch_size).astype(int) * patch_size
        block_size = image_padded_size//patch_size
        patches = patches.reshape((block_size[0], block_size[1], patch_size, patch_size))
        # https://forum.omz-software.com/topic/6266/help-inverting-scikimage-utils-view_as_blocks
        image_padded = patches.transpose(0,2,1,3).reshape(image_padded_size) 
        
        if padding == 'topleft':
             image = image_padded[:image_size[0], :image_size[1]]
             
        if padding == 'center':
            offset_topleft = (image_padded_size - image_size)//2
            offset_bottomright = offset_topleft + image_size
            image = image_padded[offset_topleft[0]:offset_bottomright[0], offset_topleft[1]:offset_bottomright[1]]
            
    if mode == 'no_padding':
        patches = patches[:, 0]
        h, w = image_size
        patch_size = patches[0].shape[0]
        image = np.zeros((h, w))
        
        x, y = np.mgrid[0:h:patch_size, 0:w:patch_size]
        x[-1, :] = (h-patch_size) if (x[-1, -1] + patch_size) > h else x[-1, :]
        y[:, -1] = (w-patch_size) if (y[-1, -1] + patch_size) > w else y[:, -1]
        x, y = x.flatten(), y.flatten()
        
        for  c, (i, j) in enumerate(zip(x, y)):
            image[i:i+patch_size, j:j+patch_size] = patches[c]
        
    return image

#------------------------------------------------------------------------------

def deInterleaveYwithUV(interleaved_yuv):
    h, w = np.array(interleaved_yuv[0].shape) * 2
    y = np.zeros((h, w), dtype=interleaved_yuv.dtype)
    y[0::2, 0::2] = interleaved_yuv[0]
    y[0::2, 1::2] = interleaved_yuv[1]
    y[1::2, 0::2] = interleaved_yuv[2]
    y[1::2, 1::2] = interleaved_yuv[3]
    
    #u = interleaved_yuv[4]
    #v = interleaved_yuv[5]
    return y #, u, v

#------------------------------------------------------------------------------

# # Encoded Image Loader

class ImageLoader(torch.utils.data.Dataset):
    
    def __init__(self, y_image, y_parition, y_prediction, patch_size, nbit):
        
        self.yimage_patches = image_to_patches(y_image, patch_size)
        self.ypartition_patch = image_to_patches(y_parition, patch_size)
        self.yprediction_patch = image_to_patches(y_prediction, patch_size)
        self.img_range = 2**nbit-1
        self.patch_size = patch_size
    
    def __len__(self):
        return len(self.yimage_patches)

    def __getitem__(self, idx):
       
        image = self.yimage_patches[idx]/self.img_range
        partition = self.ypartition_patch[idx]
        prediction = self.yprediction_patch[idx]/self.img_range
        
        image = torch.from_numpy(image[np.newaxis, :, :]).float()
        partition = torch.from_numpy(partition[np.newaxis, :, :].astype(np.float32)).float()
        prediction = torch.from_numpy(prediction[np.newaxis, :, :]).float()
        
        return image, partition, prediction

#------------------------------------------------------------------------------

def validate(args, val_loader, stats, model):
    """
    Run evaluation on Test Images
    """
    
    n_blocks = (args.patch_size//args.block_size)**2
     
    dct_min = torch.from_numpy(stats['dct_input']['min'][None,:, None, None]).float().to(args.device) 
    dct_max = torch.from_numpy(stats['dct_input']['max'][None,:, None, None]).float().to(args.device) 

    outputs = []
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (image, partition, prediction) in enumerate(val_loader):

            image = image.to(args.device)
            partition = partition.to(args.device) 
            prediction = prediction.to(args.device)
            
            # compute output
            output = model(args, image, partition, prediction, n_blocks, dct_max, dct_min)
    
            outputs.append(output.float().cpu().numpy())
        
        outputs = np.concatenate(outputs)

    return outputs

#------------------------------------------------------------------------------

if __name__ == "__main__":
    
    args = args_parser()
    args.device = torch.device("cuda:0")
    
    #--------------------------------------------------------------------------
    
    args.block_size = 4
    patch_size = 256
    
    #-----------load model-----------------------------------------------------
    
    qp = 42
    epoch = 149
    
    args.save_dir = 'pretrained_model'
    chkpt_path = args.save_dir + f'/qp-{qp}/epoch_{epoch}/checkpoint_{epoch}.pth.tar'
    model = EDSR()
    device_ids = [0]
    model = nn.DataParallel(model, device_ids=device_ids).to(args.device)
    model.load_state_dict(torch.load(chkpt_path, map_location='cuda:0')['model'])
    
    #--------------------------------------------------------------------------
    
    with open(f'data_stats/div2k/stats_qp{qp}.pkl','rb') as f:
        stats = pickle.load(f)
        
    #-----------load input data------------------------------------------------
    
    content_type = 'sdr'
    seq_config_path = 'cfg/per-sequence'
    ctc_dataset = 'test_dataset/ctc-test-seq/SDRExtractedFrames'
    out_basepath = f'TSFNet_Reconstructed/qp-{qp}'
    
    sdr_seqs = ['Campfire']
    
    no_encodedframes_intra = {'Campfire': 1}
    
    for i, seq in enumerate(sdr_seqs):
        
        seq_fname = seq_info_sdr[seq]['file-name']
        cfg_fname = seq_info_sdr[seq]['cfg-name']
        file_name, input_bitdepth, chroma_format, frame_rate, frame_skip, frame_width, frame_height, n_frames, level = read_seq_config(seq_config_path + f'/{cfg_fname}')
       
        frame_count = no_encodedframes_intra[seq]
        inputY_psnr = [None] * frame_count
        predY_psnr = [None] * frame_count
        vvcY_psnr = [None] * frame_count
        for frame_no in range(frame_count): 
            
            # read data svaed as numpy file (y,u,v in compact form)
            yuv_ilf_disabled = np.load( os.path.join(ctc_dataset, 'InLoopFilterDisabled', f'qp-{qp}', seq, f'frame-{frame_no:03}.npy') )
            partition = np.load( os.path.join(ctc_dataset, 'Partition', f'qp-{qp}', seq, f'frame-{frame_no:03}.npy') )
            prediction = np.load( os.path.join(ctc_dataset, 'Prediction', f'qp-{qp}', seq, f'frame-{frame_no:03}.npy') )
            yuv_groundtuth = np.load( os.path.join(ctc_dataset, 'Uncompressed', seq, f'frame-{frame_no:03}.npy') )
            
            # deinterleave to output only 'luma' channel
            y_ilf_disabled = deInterleaveYwithUV(yuv_ilf_disabled)  
            y_partition = deInterleaveYwithUV(partition)  
            y_prediction = deInterleaveYwithUV(prediction)  
            y_groundtruth = deInterleaveYwithUV(yuv_groundtuth)  
            
            args.patch_size = patch_size
            h,w = y_ilf_disabled.shape
            if min(h,w) < patch_size:
                args.patch_size = patch_size//2
            if min(h,w) < args.patch_size:
                args.patch_size = args.patch_size//2
            
            #-----------test model---------------------------------------------
            
            test_loader = torch.utils.data.DataLoader(
                ImageLoader(y_ilf_disabled, y_partition, y_prediction, args.patch_size, nbit),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            
            predY_patches = validate(args, test_loader, stats, model)
            y_image_pred = patches_to_image(predY_patches, y_groundtruth.shape, args.patch_size)
            
            y_image_pred = np.round(y_image_pred*(2**input_bitdepth-1))
             
            inputY_psnr[frame_no] = get_psnr(y_groundtruth/(2**input_bitdepth-1), y_ilf_disabled/(2**input_bitdepth-1))
            predY_psnr[frame_no] = get_psnr(y_groundtruth/(2**input_bitdepth-1), y_image_pred/(2**input_bitdepth-1))
        
            print(f'Sequence: {seq}, frame-no: {frame_no},  input psnr: {inputY_psnr[frame_no]:.2f}, pred-psnr: {predY_psnr[frame_no]:.2f}')
        
    
