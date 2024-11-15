import torch
import torch.nn as nn
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import cv2
from event import EventSequence, events_to_voxel_grid
from util import get_all_file_paths, group_image_data, group_event_data, randomcrop, centercrop, padding

TIMESTAMP_COLUMN = 2
X_COLUMN = 0
Y_COLUMN = 1
POLARITY_COLUMN = 3

class demoDataset(torch.utils.data.Dataset):
    def __init__(self, opt, se_idx=None):
        self.skip_number = opt.skip_number
        self.opt = opt
        self.num_bins = 5
        if se_idx is not None:
            self.opt.se_idx = se_idx
        elif not hasattr(self.opt, 'se_idx'):
            self.opt.se_idx = 0
        self.img_path_list = []
        self.event_path_list = []
        
        for senario in self.opt.senarios:
            if self.opt.isTrain:
                one_sinario_img = sorted(get_all_file_paths(os.path.join(self.opt.data_root_dir, '3_TRAINING', senario, 'images')))
                one_sinario_eve = sorted(get_all_file_paths(os.path.join(self.opt.data_root_dir, '3_TRAINING', senario, 'events')))
            elif self.opt.isValidate:
                one_sinario_img = sorted(get_all_file_paths(os.path.join(self.opt.data_root_dir, '2_VALIDATION', senario, 'images')))
                one_sinario_eve = sorted(get_all_file_paths(os.path.join(self.opt.data_root_dir, '2_VALIDATION', senario, 'events')))
            else:
                one_sinario_img = sorted(get_all_file_paths(os.path.join(self.opt.data_root_dir, '1_TEST', senario, 'images')))
                one_sinario_eve = sorted(get_all_file_paths(os.path.join(self.opt.data_root_dir, '1_TEST', senario, 'events')))
            
            group_image_path = group_image_data(one_sinario_img, self.skip_number + 2)
            group_event_path = group_event_data(one_sinario_eve, self.skip_number + 1)
            self.img_path_list.append(group_image_path)
            self.event_path_list.append(group_event_path)

        self.osize = (opt.image_height, opt.image_width)  # (972, 628)
        self.device = torch.device(self.opt.gpu_ids)

    def __len__(self):
        return len(self.img_path_list[self.opt.se_idx]) * self.skip_number

    def __getitem__(self, idx):
        self.I0 = torch.zeros(3, self.osize[0], self.osize[1]).to(self.device)
        self.I1 = torch.zeros(3, self.osize[0], self.osize[1]).to(self.device)
        self.voxel_eve_0_t = torch.zeros(self.num_bins, self.osize[0], self.osize[1]).to(self.device)
        self.voxel_eve_1_t = torch.zeros(self.num_bins, self.osize[0], self.osize[1]).to(self.device)
        self.label = torch.zeros(3, self.osize[0], self.osize[1]).to(self.device)

        t = idx % self.skip_number
        group_image_path = self.img_path_list[self.opt.se_idx][idx // (self.skip_number + 2)]
        group_event_path = self.event_path_list[self.opt.se_idx][idx // (self.skip_number + 1)]

        eve_0_t_paths = group_event_path[:t + 1]
        eve_t_1_paths = group_event_path[t + 1:]
        I0_path = group_image_path[0]
        I1_path = group_image_path[-1]
        I0 = Image.open(I0_path)
        I1 = Image.open(I1_path)
        label = Image.open(group_image_path[t + 1])

        # Load timelens
        eve_0_t = EventSequence.from_npz_files(list_of_filenames=eve_0_t_paths, image_height=970,
                                               image_width=625)._features
        eve_t_1 = EventSequence.from_npz_files(list_of_filenames=eve_t_1_paths, image_height=970,
                                               image_width=625)
        eve_1_t = eve_t_1._features[eve_t_1._features[:, TIMESTAMP_COLUMN].argsort()[::-1]]

        # Convert events to voxel grid
        voxel_eve_0_t = events_to_voxel_grid(eve_0_t, num_bins=self.num_bins, width=970, height=625)
        voxel_eve_1_t = events_to_voxel_grid(eve_1_t, num_bins=self.num_bins, width=970, height=625)

        # Define transformations
        self.transforms_toTensor = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.transforms_normalize = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,)),
        ])

        # Normalize and convert to tensor
        I0 = self.transforms_toTensor(I0)
        I1 = self.transforms_toTensor(I1)
        label = self.transforms_toTensor(label)
        voxel_eve_0_t = torch.from_numpy(voxel_eve_0_t)
        voxel_eve_1_t = torch.from_numpy(voxel_eve_1_t)

        if self.opt.isTrain == True or self.opt.isValidate == True:
            I0, label, I1, voxel_eve_0_t, voxel_eve_1_t = randomcrop(I0, label, I1, voxel_eve_0_t, voxel_eve_1_t, self.osize[1], self.osize[0])
        elif self.opt.isTest == True:
            I0, label, I1, voxel_eve_0_t, voxel_eve_1_t = padding(I0, label, I1, voxel_eve_0_t, voxel_eve_1_t, 640, 1024)
            # I0, label, I1, voxel_eve_0_t, voxel_eve_1_t = centercrop(I0, label, I1, voxel_eve_0_t, voxel_eve_1_t, 512, 512)

        if voxel_eve_0_t.max() != 0:
            voxel_eve_0_t = voxel_eve_0_t / voxel_eve_0_t.max()
        if voxel_eve_1_t.max() != 0:
            voxel_eve_1_t = voxel_eve_1_t / voxel_eve_1_t.max()

        I0 = self.transforms_normalize(I0) 
        I1 = self.transforms_normalize(I1)
        label = self.transforms_normalize(label)

        self.I0.copy_(I0)
        self.I1.copy_(I1)
        self.voxel_eve_0_t.copy_(voxel_eve_0_t)
        self.voxel_eve_1_t.copy_(voxel_eve_1_t)
        self.label.copy_(label)

        return (self.I0, self.I1, self.voxel_eve_0_t, self.voxel_eve_1_t), self.label
