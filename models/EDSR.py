from . import common 
from . import MST

import torch
import torch.nn as nn
import torch.nn.functional as F
from img_op import blockify, unblockify
from torch_dct import dct_2d

def default_conv(in_channels, out_channels, kernel_size, bias=True, groups=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, groups=groups)

class MSTFusionBlock(nn.Module):
    def __init__(self, dim_imgfeat, dim_dctfeat, kernel_size=3, conv=default_conv):
        super(MSTFusionBlock, self).__init__()
            
        self.conv_img = nn.Sequential(conv(dim_imgfeat, dim_imgfeat, kernel_size=kernel_size),
                                      nn.ReLU(True),
                                      conv(dim_imgfeat, dim_imgfeat, kernel_size=kernel_size))
        
        self.conv_dct = nn.Sequential(conv(dim_dctfeat, dim_dctfeat, kernel_size=kernel_size),
                                      nn.ReLU(True),
                                      conv(dim_dctfeat, dim_dctfeat, kernel_size=kernel_size))
        
        self.stage_tconv = nn.ConvTranspose2d(dim_dctfeat, dim_dctfeat, kernel_size=kernel_size, stride=2, padding=(kernel_size//2))
        self.msab = MST.MSAB(dim_in=(dim_imgfeat+dim_dctfeat), dim_head=(dim_imgfeat+dim_dctfeat), dim_out=dim_imgfeat, heads=4, num_blocks=1)
    
    def forward(self, in_pix, in_dct):    
        out_pix = self.conv_img(in_pix)
        out_dct = self.conv_dct(in_dct)
        out_pix = self.msab(out_pix, self.stage_tconv(out_dct, output_size=in_pix.shape[2:]))
        #out_pix = self.msab(out_pix, out_dct)
        return out_pix+in_pix, out_dct+in_dct
        

class EDSR(nn.Module):
    def __init__(self, conv=default_conv):
        super(EDSR, self).__init__()

        in_channel_img = 4
        in_channel_dct = 16
        out_channel = 4
        
        dim_imgfeat_rec = 64  # 48
        dim_imgfeat_prt = 8
        dim_imgfeat_res = 24  # 8
        dim_imgfeat = dim_imgfeat_rec + dim_imgfeat_prt + dim_imgfeat_res
        
        dim_dctfeat_rec = 16
        dim_dctfeat_res = 16
        dim_dctfeat = dim_dctfeat_rec + dim_dctfeat_res
        
        kernel_size = 3
        n_basicblock = 20
        
        # define head module for pixel input
        self.head_pix_rec = nn.Sequential(nn.Conv2d(in_channels=in_channel_img, out_channels=dim_imgfeat_rec//2, kernel_size=kernel_size, padding=(kernel_size//2), stride=1),
                                nn.PReLU(dim_imgfeat_rec//2),
                                nn.Conv2d(in_channels=dim_imgfeat_rec//2, out_channels=dim_imgfeat_rec, kernel_size=kernel_size, padding=(kernel_size//2), stride=1)
                                )
        
        self.head_pix_prt = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=dim_imgfeat_prt//2, kernel_size=kernel_size, padding=(kernel_size//2), stride=1),
                                nn.PReLU(dim_imgfeat_prt//2),
                                nn.AvgPool2d(kernel_size=2, stride=2),
                                nn.Conv2d(in_channels=dim_imgfeat_prt//2, out_channels=dim_imgfeat_prt, kernel_size=kernel_size, padding=(kernel_size//2), stride=1)
                                )
        
        self.head_pix_res = nn.Sequential(nn.Conv2d(in_channels=in_channel_img, out_channels=dim_imgfeat_res//2, kernel_size=kernel_size, padding=(kernel_size//2), stride=1),
                                nn.PReLU(dim_imgfeat_res//2),
                                nn.Conv2d(in_channels=dim_imgfeat_res//2, out_channels=dim_imgfeat_res, kernel_size=kernel_size, padding=(kernel_size//2), stride=1)
                                )
        
        # define head module for dct input
        self.head_dct_rec = nn.Sequential(nn.Conv2d(in_channels=in_channel_dct, out_channels=dim_dctfeat_rec, kernel_size=kernel_size, padding=(kernel_size//2), stride=1),
                               nn.PReLU(dim_dctfeat_rec),
                               nn.Conv2d(in_channels=dim_dctfeat_rec, out_channels=dim_dctfeat_rec, kernel_size=kernel_size, padding=(kernel_size//2), stride=1)
                               )
        
        self.head_dct_res = nn.Sequential(nn.Conv2d(in_channels=in_channel_dct, out_channels=dim_dctfeat_res, kernel_size=kernel_size, padding=(kernel_size//2), stride=1),
                               nn.PReLU(dim_dctfeat_res),
                               nn.Conv2d(in_channels=dim_dctfeat_res, out_channels=dim_dctfeat_res, kernel_size=kernel_size, padding=(kernel_size//2), stride=1)
                               )
        
        self.body = nn.ModuleList([ MSTFusionBlock(dim_imgfeat, dim_dctfeat, kernel_size) for _ in range(n_basicblock) ])
        
        
        # define tail module
        self.tail = conv(dim_imgfeat, out_channel, kernel_size)
        self.pix_shuffle = nn.PixelShuffle(2)
        
      
    def forward(self, args, image, partition, prediction, n_blocks, dct_max, dct_min):
        
        b, c, h, w = image.shape
        #----------------------------------------------------------------------
        
        img_ds = F.pixel_unshuffle(image, args.block_size//2)
        #prt_ds = F.pixel_unshuffle(partition, args.block_size//2)
        res_ds = F.pixel_unshuffle(prediction, args.block_size//2)
    
        # create blocks (say, bxcx4x4) to be applied by DCT 
        img_block = torch.cat((blockify(image, n_blocks, args.block_size), 
                                blockify(prediction, n_blocks, args.block_size)), dim=1)
        
        dct_block = dct_2d(img_block, norm='ortho')
        
        img_dct = unblockify(dct_block[:, 0:n_blocks], [h, w], n_blocks, args.block_size)
        res_dct = unblockify(dct_block[:, n_blocks:2*n_blocks], [h, w], n_blocks, args.block_size)
        
        img_dct = F.pixel_unshuffle(img_dct, args.block_size)
        res_dct = F.pixel_unshuffle(res_dct, args.block_size)
        
        img_dct = (img_dct -  dct_min[:,:16,:,:])/(dct_max[:,:16,:,:] - dct_min[:,:16,:,:]) 
        res_dct = (res_dct -  dct_min[:,32:,:,:])/(dct_max[:,32:,:,:] - dct_min[:,32:,:,:]) 
        
        #----------------------------------------------------------------------
        
        x_pix_rec = self.head_pix_rec(img_ds)
        x_pix_prt = self.head_pix_prt(partition)
        x_pix_res = self.head_pix_res(res_ds)
        
        x_dct_rec = self.head_dct_rec(img_dct)
        x_dct_res = self.head_dct_res(res_dct)
        
        x_pix = torch.cat((x_pix_rec, x_pix_prt, x_pix_res), dim=1)
        x_dct = torch.cat((x_dct_rec, x_dct_res), dim=1)
        
        for i, layer in enumerate(self.body):
            if i == 0:
                res_pix, x_dct = layer(x_pix, x_dct)
            else:
                res_pix, x_dct = layer(res_pix, x_dct)
            
        res_pix += x_pix
        x_pix = self.tail(res_pix)
        x_pix = self.pix_shuffle(x_pix)
        
        x_pix += image
        return x_pix

#------------------------------------------------------------------------------
