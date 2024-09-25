import torch
import torch.nn as nn
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import cv2
from event import EventSequence, events_to_voxel_grid
from util import get_all_file_paths, group_data

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
            if self.opt.isTrain == True:
                one_sinario_img = sorted(
                    get_all_file_paths(os.path.join(self.opt.data_root_dir, '3_TRAINING', senario, 'images')))
                one_sinario_eve = sorted(
                    get_all_file_paths(os.path.join(self.opt.data_root_dir, '3_TRAINING', senario, 'events')))

            else:
                one_sinario_img = sorted(
                    get_all_file_paths(os.path.join(self.opt.data_root_dir, '1_TEST', senario, 'images')))
                one_sinario_eve = sorted(
                    get_all_file_paths(os.path.join(self.opt.data_root_dir, '1_TEST', senario, 'events')))
            group_image_path = group_data(one_sinario_img, self.skip_number + 1)
            group_event_path = group_data(one_sinario_eve, self.skip_number + 1)
            self.img_path_list.append(group_image_path)
            self.event_path_list.append(group_event_path)
        self.osize = (256, 256)
        self.device = torch.device(self.opt.gpu_ids)

        

    def __len__(self):

        return len(self.img_path_list[self.opt.se_idx] * self.skip_number)

    def __getitem__(self, idx):
        self.I0 = torch.zeros(3, self.osize[0], self.osize[1]).to(self.device)
        self.I1 = torch.zeros(3, self.osize[0], self.osize[1]).to(self.device)
        self.voxel_eve_0_t = torch.zeros(self.num_bins, self.osize[0], self.osize[1]).to(self.device)
        self.voxel_eve_1_t = torch.zeros(self.num_bins, self.osize[0], self.osize[1]).to(self.device)
        self.label = torch.zeros(3, self.osize[0], self.osize[1]).to(self.device)

        t = idx % (self.skip_number + 1)
        if t == 0:
            return (self.I0, self.I1, self.voxel_eve_0_t, self.voxel_eve_1_t), self.label
        # idx / group = x ...... t

        group_image_path = self.img_path_list[self.opt.se_idx][idx // (self.skip_number + 1)]
        group_event_path = self.event_path_list[self.opt.se_idx][idx // (self.skip_number + 1)]
=======
        self.I0 = torch.zeros(3,self.osize[0],self.osize[1]).to(self.device)
        self.I1 = torch.zeros(3,self.osize[0],self.osize[1]).to(self.device)
        self.voxel_eve_0_t = torch.zeros(self.num_bins,self.osize[0],self.osize[1]).to(self.device)
        self.voxel_eve_1_t = torch.zeros(self.num_bins,self.osize[0],self.osize[1]).to(self.device)
        self.label = torch.zeros(3,self.osize[0],self.osize[1]).to(self.device)
        
        t = idx % (self.skip_number+1)
        if t == 0 :
            return (self.I0, self.I1, self.voxel_eve_0_t, self.voxel_eve_1_t), self.label
        #idx / group = x ...... t 
        
        group_image_path = self.img_path_list[self.opt.se_idx][idx // (self.skip_number+1)]
        group_event_path = self.event_path_list[self.opt.se_idx][idx // (self.skip_number+1)]
>>>>>>> bdca369b89db6ee1642fe0976b288ba7c332952c
        eve_0_t_paths = group_event_path[:t]
        eve_t_1_paths = group_event_path[t:]
        I0_path = group_image_path[0]
        I1_path = group_image_path[-1]
        label_paths = group_image_path[1:self.skip_number + 1]
        I0 = Image.open(I0_path)
        I1 = Image.open(I1_path)
        label = Image.open(label_paths[t - 1])
        label = np.array(label)
        label = np.array([cv2.resize(label[:, :, i], (256, 256)) for i in range(label.shape[2])])
        image_height, image_width = self.osize

        # load timelens
        eve_0_t = EventSequence.from_npz_files(list_of_filenames=eve_0_t_paths, image_height=image_height,
                                               image_width=image_width)._features
        eve_t_1 = EventSequence.from_npz_files(list_of_filenames=eve_t_1_paths, image_height=image_height,
                                               image_width=image_width)
        eve_1_t = eve_t_1._features[eve_t_1._features[:, TIMESTAMP_COLUMN].argsort()[::-1]]

        # to voxel  e2vid
        voxel_eve_0_t = events_to_voxel_grid(eve_0_t, num_bins=self.num_bins, width=image_width, height=image_height)
        voxel_eve_1_t = events_to_voxel_grid(eve_1_t, num_bins=self.num_bins, width=image_width, height=image_height)

        # torch transformer  i0 i1 -> normalized
        transform_list = []
        transform_list.append(transforms.ToTensor())
        self.transforms_toTensor = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Normalize((0.5,),
                                                   (0.5,)))
        self.transforms_normalize = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Resize(self.osize, interpolation=Image.BICUBIC))
        self.transforms_scale = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Normalize((0.5,),
                                                   (0.5,)))
        self.transforms_eve_normalize = transforms.Compose(transform_list)

        I0 = self.transforms_scale(I0)
        I1 = self.transforms_scale(I1)

        voxel_eve_0_t = np.array(
            [cv2.resize(voxel_eve_0_t[i, :, :], (256, 256)) for i in range(voxel_eve_0_t.shape[0])])
        voxel_eve_1_t = np.array(
            [cv2.resize(voxel_eve_1_t[i, :, :], (256, 256)) for i in range(voxel_eve_1_t.shape[0])])

        I0 = self.transforms_toTensor(I0)
        I1 = self.transforms_toTensor(I1)
        label = torch.from_numpy(label).float()
        voxel_eve_0_t = voxel_eve_0_t / voxel_eve_0_t.max()
        voxel_eve_1_t = voxel_eve_1_t / voxel_eve_1_t.max()
        voxel_eve_0_t = torch.from_numpy(voxel_eve_0_t)
        voxel_eve_1_t = torch.from_numpy(voxel_eve_1_t)
        I0 = self.transforms_normalize(I0)
        I1 = self.transforms_normalize(I1)
        label = self.transforms_normalize(label)
        self.I0.copy_(I0)
        self.I1.copy_(I1)
        self.voxel_eve_0_t.copy_(voxel_eve_0_t)
        self.voxel_eve_1_t.copy_(voxel_eve_1_t)
        self.label.copy_(label)

        # to tensor
        return (self.I0, self.I1, self.voxel_eve_0_t, self.voxel_eve_1_t), self.label