import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from . import common
from . import MST

import torch
import torch.nn as nn
import torch.nn.functional as F
from .img_op import blockify, unblockify
from .torch_dct import dct_2d

from .options import args_parser
import pickle

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

        # in_channel_img = 4
        in_channel_img = 12 # (4x3)
        # in_channel_dct = 16
        in_channel_dct = 48 # (16*3)
        in_channel_event = 5
        # out_channel = 4
        out_channel = 12

        dim_imgfeat_left = 64  # 48
        dim_imgfeat_event = 8
        dim_imgfeat_right = 24  # 8
        dim_imgfeat = dim_imgfeat_left + dim_imgfeat_event * 2 + dim_imgfeat_right

        dim_dctfeat_left = 16
        dim_dctfeat_right = 16
        dim_dctfeat = dim_dctfeat_left + dim_dctfeat_right

        kernel_size = 3
        n_basicblock = 10 # 20

        # define head module for pixel input
        self.head_pix_left = nn.Sequential(nn.Conv2d(in_channels=in_channel_img, out_channels=dim_imgfeat_left//2, kernel_size=kernel_size, padding=(kernel_size//2), stride=1),
                                nn.PReLU(dim_imgfeat_left//2),
                                nn.Conv2d(in_channels=dim_imgfeat_left//2, out_channels=dim_imgfeat_left, kernel_size=kernel_size, padding=(kernel_size//2), stride=1)
                                )

        self.head_event_left = nn.Sequential(nn.Conv2d(in_channels=in_channel_event, out_channels=dim_imgfeat_event//2, kernel_size=kernel_size, padding=(kernel_size//2), stride=1),
                                nn.PReLU(dim_imgfeat_event//2),
                                nn.AvgPool2d(kernel_size=2, stride=2),
                                nn.Conv2d(in_channels=dim_imgfeat_event//2, out_channels=dim_imgfeat_event, kernel_size=kernel_size, padding=(kernel_size//2), stride=1)
                                )

        self.head_pix_right = nn.Sequential(nn.Conv2d(in_channels=in_channel_img, out_channels=dim_imgfeat_right//2, kernel_size=kernel_size, padding=(kernel_size//2), stride=1),
                                nn.PReLU(dim_imgfeat_right//2),
                                nn.Conv2d(in_channels=dim_imgfeat_right//2, out_channels=dim_imgfeat_right, kernel_size=kernel_size, padding=(kernel_size//2), stride=1)
                                )

        self.head_event_right = nn.Sequential(nn.Conv2d(in_channels=in_channel_event, out_channels=dim_imgfeat_event//2, kernel_size=kernel_size, padding=(kernel_size//2), stride=1),
                                nn.PReLU(dim_imgfeat_event//2),
                                nn.AvgPool2d(kernel_size=2, stride=2),
                                nn.Conv2d(in_channels=dim_imgfeat_event//2, out_channels=dim_imgfeat_event, kernel_size=kernel_size, padding=(kernel_size//2), stride=1)
                                )

        # define head module for dct input
        self.head_dct_left = nn.Sequential(nn.Conv2d(in_channels=in_channel_dct, out_channels=dim_dctfeat_left, kernel_size=kernel_size, padding=(kernel_size//2), stride=1),
                               nn.PReLU(dim_dctfeat_left),
                               nn.Conv2d(in_channels=dim_dctfeat_left, out_channels=dim_dctfeat_left, kernel_size=kernel_size, padding=(kernel_size//2), stride=1)
                               )

        self.head_dct_right = nn.Sequential(nn.Conv2d(in_channels=in_channel_dct, out_channels=dim_dctfeat_right, kernel_size=kernel_size, padding=(kernel_size//2), stride=1),
                               nn.PReLU(dim_dctfeat_right),
                               nn.Conv2d(in_channels=dim_dctfeat_right, out_channels=dim_dctfeat_right, kernel_size=kernel_size, padding=(kernel_size//2), stride=1)
                               )

        self.body = nn.ModuleList([ MSTFusionBlock(dim_imgfeat, dim_dctfeat, kernel_size) for _ in range(n_basicblock) ])


        # define tail module
        self.tail = conv(dim_imgfeat, out_channel, kernel_size)
        self.pix_shuffle = nn.PixelShuffle(2)

    def forward(self, args, left_image, left_events, right_image, right_events, n_blocks, dct_max, dct_min):
        b, c, h, w = left_image.shape
        #----------------------------------------------------------------------

        img_left = F.pixel_unshuffle(left_image, args.block_size//2)
        #prt_ds = F.pixel_unshuffle(partition, args.block_size//2)
        img_right = F.pixel_unshuffle(right_image, args.block_size//2)

        # create blocks (say, bxcx4x4) to be applied by DCT 
        img_block = torch.cat((blockify(left_image, n_blocks, args.block_size),
                                blockify(right_image, n_blocks, args.block_size)), dim=1)

        dct_block = dct_2d(img_block, norm='ortho')

        img_left_dct = unblockify(dct_block[:, 0:n_blocks], [b, c, h, w], n_blocks, args.block_size)
        img_right_dct = unblockify(dct_block[:, n_blocks:2*n_blocks], [b, c, h, w], n_blocks, args.block_size)

        img_left_dct = F.pixel_unshuffle(img_left_dct, args.block_size)
        img_right_dct = F.pixel_unshuffle(img_right_dct, args.block_size)

        # img_dct = (img_dct -  dct_min[:,:16,:,:])/(dct_max[:,:16,:,:] - dct_min[:,:16,:,:])
        # res_dct = (res_dct -  dct_min[:,32:,:,:])/(dct_max[:,32:,:,:] - dct_min[:,32:,:,:])

        img_left_dct = (img_left_dct - dct_min[:, :16, :, :].repeat(1, 3, 1, 1)) / (dct_max[:, :16, :, :].repeat(1, 3, 1, 1) - dct_min[:, :16, :, :].repeat(1, 3, 1, 1))
        img_right_dct = (img_right_dct - dct_min[:, 32:, :, :].repeat(1, 3, 1, 1)) / (dct_max[:, 32:, :, :].repeat(1, 3, 1, 1) - dct_min[:, 32:, :, :].repeat(1, 3, 1, 1))

        #----------------------------------------------------------------------
        x_pix_left = self.head_pix_left(img_left)
        x_event_left = self.head_event_left(left_events)
        x_pix_right = self.head_pix_right(img_right)
        x_event_right = self.head_event_right(right_events)

        x_dct_left = self.head_dct_left(img_left_dct)
        x_dct_right = self.head_dct_right(img_right_dct)

        x_pix = torch.cat((x_pix_left, x_event_left, x_pix_right, x_event_right), dim=1)
        x_dct = torch.cat((x_dct_left, x_dct_right), dim=1)

        for i, layer in enumerate(self.body):
            if i == 0:
                res_pix, x_dct = layer(x_pix, x_dct)
            else:
                res_pix, x_dct = layer(res_pix, x_dct)

        res_pix += x_pix
        x_pix = self.tail(res_pix)
        x_pix = self.pix_shuffle(x_pix)

        x_pix += left_image
        return x_pix


#------------------------------------------------------------------------------
if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device("cuda:0")
    qp = 42
    args.epochs = 150
    args.qp = qp
    args.patch_size = 256
    args.block_size = 4
    args.num_patches_per_frame = 50
    # n_blocks = (args.patch_size // args.block_size) ** 2
    n_blocks = (856 // args.block_size) * (960 // args.block_size)

    with open(f'../data_stats/div2k/stats_qp{qp}.pkl', 'rb') as f:
        stats = pickle.load(f)

    dct_min = torch.from_numpy(stats['dct_input']['min'][None, :, None, None]).float().to(args.device)
    dct_max = torch.from_numpy(stats['dct_input']['max'][None, :, None, None]).float().to(args.device)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EDSR().to(device)
    left_image = torch.randn(1, 3, 856, 960).cuda()
    right_image = torch.randn(1, 3, 856, 960).cuda()
    left_events = torch.randn(1, 5, 856, 960).cuda()
    right_events = torch.randn(1, 5, 856, 960).cuda()
    # input = {
    #     "before": {"rgb_image": left_image, "events": left_events},
    #     "middle": {"weight": right_weight},
    #     "after": {"rgb_image": right_image, "events": right_events},
    # }

    output = model(args, left_image, left_events, right_image, right_events, n_blocks, dct_max, dct_min)
    print(output[0].shape)
