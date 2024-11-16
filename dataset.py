import sys
from torchvision import transforms
import cv2
import os
from tools import representation, event
from PIL import Image
from util import  randomcrop, centercrop, padding
import numpy as np
TIMESTAMP_COLUMN = 2
X_COLUMN = 0
Y_COLUMN = 1
POLARITY_COLUMN = 3

class HSERGBDataset:
    def __init__(self, data_path='/home/lisiqi/data/HSERGB', mode='train', folder='', number_of_frames_to_skip=15, nb_of_time_bin=20):
        if mode not in ['train', 'test','val']:
            raise ValueError

        self.folder = os.path.join(data_path, mode, folder)
        self.number_of_frames_to_skip = number_of_frames_to_skip
        print('skip %d frame' % self.number_of_frames_to_skip)
        self.mode = mode
        self.nb_of_time_bin = nb_of_time_bin
        self.generate_data()
        self.folder = folder
        self.osize = 128

    def __len__(self):
        return len(self.idx)

    def generate_data(self):
        self.left_image = []
        self.right_image = []
        self.gt_image = []
        self.event = []
        self.gt_timestamp = []
        self.idx = []
        self.lr_timestamp = []
        with open(os.path.join(self.folder,'images', 'timestamp.txt'), 'r') as f:
            ts = [float(l.strip('\n')) for l in f.readlines()]
        N = len(ts)

        for k in range(int(N/(self.number_of_frames_to_skip+1))-2):
            rand = 0
            start = k*(self.number_of_frames_to_skip+1) + rand
            end = (k+1)*(self.number_of_frames_to_skip+1) + rand

            self.left_image.append(os.path.join(self.folder,'images',  f'{start:06}.png'))
            self.right_image.append(os.path.join(self.folder,'images',  f'{end:06}.png'))
            self.event.append([os.path.join(self.folder, 'events', f'{k:06}.npz') for k in range(start, end)])
            self.gt_image.append([os.path.join(self.folder, 'images', f'{k:06}.png') for k in range(start+1, end)])
            self.gt_timestamp.append([ts[k] for k in range(start+1, end)])
            self.lr_timestamp.append([ts[start], ts[end]])
        for k in range(len(self.left_image)):
            self.idx += [k] * len(self.gt_image[0])
        self.start_idx = [k * len(self.gt_image[0]) for k in range(len(self.left_image))]
    def __getitem__(self, idx):
            seq_idx = self.idx[idx]
            sample_idx = idx - self.start_idx[seq_idx]
            left_image = transforms.ToTensor()(Image.open(self.left_image[seq_idx]))
            right_image = transforms.ToTensor()(Image.open(self.right_image[seq_idx]))

            w, h = left_image.shape[2], left_image.shape[1]
            gt_image = transforms.ToTensor()(Image.open(self.gt_image[seq_idx][sample_idx]))

            events = event.EventSequence.from_npz_files(self.event[seq_idx], h, w)
            ts = self.gt_timestamp[seq_idx][sample_idx]

            duration_left = ts - self.lr_timestamp[seq_idx][0]
            duration_right = self.lr_timestamp[seq_idx][1] - ts
            weight = duration_left / (duration_left+duration_right)
            e_left = events.filter_by_timestamp(events.start_time(), duration_left)
            e_right = events.filter_by_timestamp(ts, duration_right)

            event_left_forward = representation.to_count_map(e_left, self.nb_of_time_bin).clone()
            event_right_forward = representation.to_count_map(e_right, self.nb_of_time_bin).clone()
            
            left_voxel_grid = representation.to_voxel_grid(e_left, nb_of_time_bins=self.nb_of_time_bin)
            right_voxel_grid = representation.to_voxel_grid(e_right, nb_of_time_bins=self.nb_of_time_bin)

            e_right.reverse()
            e_left.reverse()
            event_left_backward = representation.to_count_map(e_left, self.nb_of_time_bin)
            event_right_backward = representation.to_count_map(e_right, self.nb_of_time_bin)
            events_forward = np.concatenate((event_left_forward, event_right_forward), axis=-1)
            events_backward = np.concatenate((event_right_backward, event_left_backward), axis=-1)

            

            surface = events.filter_by_timestamp(ts-200, 400)
            surface = representation.to_count_map(surface)

            if self.mode == 'train' or self.mode == 'val':
                left_image, gt_image, right_image, left_voxel_grid, right_voxel_grid = randomcrop(left_image, gt_image, right_image, left_voxel_grid, right_voxel_grid, self.osize, self.osize)
            elif self.mode == 'test':
                left_image, gt_image, right_image, left_voxel_grid, right_voxel_grid = padding(left_image, gt_image, right_image, left_voxel_grid, right_voxel_grid, 640, 1024)
            
            return events_forward, events_backward, left_image, right_image, gt_image, weight, \
                 [self.nb_of_time_bin, self.nb_of_time_bin], surface, left_voxel_grid, right_voxel_grid, self.gt_image[seq_idx][sample_idx]


