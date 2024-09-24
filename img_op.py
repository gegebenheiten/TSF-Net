#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 00:16:45 2022

@author: biren
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#------------------------------------------------------------------------------

def blockify(image, n_blocks, block_size):
    '''image: BxCxHxW'''
    return F.unfold(image, kernel_size=block_size, stride=block_size).permute(0,2,1).reshape(-1, n_blocks, block_size, block_size)

#------------------------------------------------------------------------------

# def unblockify(image_block, img_size, n_blocks, block_size):
#     return F.fold(image_block.reshape(-1, n_blocks, block_size**2).permute(0, 2, 1), output_size=(img_size[0], img_size[1]), kernel_size=block_size, stride=block_size)


def unblockify(image_block, img_size, n_blocks, block_size):
    '''image_block: BxCxNxKxK, img_size: [H, W]'''
    b, c, h, w = img_size

    blocks = image_block.reshape(b, c, n_blocks, block_size * block_size)
    blocks = blocks.permute(0, 1, 3, 2)
    blocks = blocks.reshape(b, c * block_size * block_size, n_blocks)
    folded = F.fold(blocks, output_size=(h, w), kernel_size=block_size, stride=block_size)

    return folded
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------