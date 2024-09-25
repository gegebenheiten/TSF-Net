import argparse
import torch
class SimpleOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Simple Options for Training')
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        self.parser.add_argument('--initial_lr', type=float, default=1e-4, help='initial learning rate')
        self.parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train')
        self.parser.add_argument('--gpu_ids', type=str, default='cuda:2', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for cpu')
        self.parser.add_argument('--senarios', type=str, nargs='+', default=['ball_00', 'ball_02', 'basket_05', 'eggs_01', 'eggs_03', 'eggs_05', 'horse_03', 'may29_handheld_02', 'may29_handheld_04'], help='list of scenarios')
        self.parser.add_argument('--data_root_dir', type=str, default='data/EXP1_dataset', help='root directory for data')
        self.parser.add_argument('--skip_number', default=7, type=int, metavar='N', help='number of interpolation frames')
        self.parser.add_argument('--qp', default=42, type=int, metavar='N', help='number of interpolation frames')
        self.parser.add_argument('--patch_size', default=256, type=int, metavar='N', help='number of interpolation frames')
        self.parser.add_argument('--block_size', default=4, type=int, metavar='N', help='number of interpolation frames')
        self.parser.add_argument('--num_patches_per_frame', default=50, type=int, metavar='N', help='number of interpolation frames')
       
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = str_ids[0]
        
        args = vars(self.opt)
        
        
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print(f'{k}: {v}')
        print('-------------- End ----------------')

        return self.opt