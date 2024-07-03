#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 14:35:34 2021

@author: biren
"""

import os
import pywt
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------

def get_psnr(origY, predY):
    mse = np.mean((origY - predY)**2)
    if mse == 0:
        return 100
    else:
        psnr = 10*np.log10(1./mse)
    return psnr

#------------------------------------------------------------------------------

def get_PSNR(original, compressed, max_pixel=255):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):
        return 100
    max_pixel = max_pixel
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

#------------------------------------------------------------------------------

def save_fig(dir_path, figs, psnrs, epoch):
    titles = ['pred', 'vvc', 'gt']
    fig, axes = plt.subplots(nrows=1, ncols=3)
    for i, (fig, psnr, title) in enumerate(zip(figs, psnrs, titles)):
        axes[i].imshow(fig)
        axes[i].set_axis_off()
        axes[i].set_title(title + f'({psnr: .2f})')
    
    save_dir = os.path.join(dir_path, 'epoch_{}'.format(epoch))
    if not os.path.isdir(save_dir): 
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, 'sample-image.png'), bbox_inches='tight', dpi=200)
    
#------------------------------------------------------------------------------

def interleaveYwithUV(y, u, v):
    y1 = y[0::2, 0::2]
    y2 = y[0::2, 1::2]
    y3 = y[1::2, 0::2]
    y4 = y[1::2, 1::2]
    
    return  np.stack((y1, y2, y3, y4, u, v))

#------------------------------------------------------------------------------

def deInterleaveYwithUV(interleaved_yuv):
    h, w = np.array(interleaved_yuv[0].shape) * 2
    y = np.zeros((h, w), dtype=interleaved_yuv.dtype)
    y[0::2, 0::2] = interleaved_yuv[0]
    y[0::2, 1::2] = interleaved_yuv[1]
    y[1::2, 0::2] = interleaved_yuv[2]
    y[1::2, 1::2] = interleaved_yuv[3]
    
    u = interleaved_yuv[4]
    v = interleaved_yuv[5]
    return y, u, v

#------------------------------------------------------------------------------

def interleaveY(y):
    y1 = y[0::2, 0::2]
    y2 = y[0::2, 1::2]
    y3 = y[1::2, 0::2]
    y4 = y[1::2, 1::2]
    
    return  np.stack((y1, y2, y3, y4))

#------------------------------------------------------------------------------

def deInterleaveY(interleaved_y):
    h, w = np.array(interleaved_y[0].shape) * 2
    y = np.zeros((h, w), dtype=interleaved_y.dtype)
    y[0::2, 0::2] = interleaved_y[0]
    y[0::2, 1::2] = interleaved_y[1]
    y[1::2, 0::2] = interleaved_y[2]
    y[1::2, 1::2] = interleaved_y[3]
    
    return y

#------------------------------------------------------------------------------

def interleaveYwithUorV(y, u):
    y1 = y[0::2, 0::2]
    y2 = y[0::2, 1::2]
    y3 = y[1::2, 0::2]
    y4 = y[1::2, 1::2]
    
    return  np.stack((y1, y2, y3, y4, u))

#------------------------------------------------------------------------------

def deInterleaveYwithUorV(interleaved_yu):
    h, w = np.array(interleaved_yu[0].shape) * 2
    y = np.zeros((h, w), dtype=interleaved_yu.dtype)
    y[0::2, 0::2] = interleaved_yu[0]
    y[0::2, 1::2] = interleaved_yu[1]
    y[1::2, 0::2] = interleaved_yu[2]
    y[1::2, 1::2] = interleaved_yu[3]
    
    u = interleaved_yu[4]
    
    return y, u

#------------------------------------------------------------------------------

def saveas_image(image, image_name):
    import matplotlib
    #matplotlib.image.imsave(image_name, image, cmap='gray')
    matplotlib.image.imsave(image_name, image)
    
#------------------------------------------------------------------------------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)