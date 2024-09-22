import torch
import torch.nn as nn
import os
from itertools import islice
from PIL import Image
import numpy as np

'''



reverse t-1 ->voxel 
        现阶段 I0和I1还未PIL.image 格式 需要转化为 tensor 
        label 为np格式 需要转化为tensor 
        event 没有读取 还差读取和转化为voxel 
        t-1需要reverse成1-t 
        
        
        
        
        '''
#  Assuming you have a dataset, using dummy data here for illustration
def get_all_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith((".png", ".npz")):
                file_paths.append(os.path.join(root, file))
    return file_paths
def group_data(data, group_size):
    # 初始化空列表来存储分组结果
    grouped_data = []
    # 按照步长为 group_size 分组
    ends = (len(data)//group_size)*group_size
    for i in range(0, ends, group_size):
        start = max(0,i-1)
        grouped_data.append(data[start:i+group_size])
    return grouped_data
class demoDataset(torch.utils.data.Dataset):
    def __init__(self,opt,skip_number,se_idx=None):
        
        self.skip_number = skip_number
        self.opt = opt 
        if se_idx is not None:
            self.opt.se_idx = se_idx
        elif not hasattr(self.opt, 'se_idx'):
            self.opt.se_idx = 0
        self.img_path_list = []
        self.event_path_list = []
        for senario in self.opt.senarios:
            if self.opt.isTrain == True: 
                one_sinario_img = sorted(get_all_file_paths(os.path.join(self.opt.data_root_dir,'3_TRAINING',senario,'images')))
                one_sinario_eve = sorted(get_all_file_paths(os.path.join(self.opt.data_root_dir,'3_TRAINING',senario,'events')))
                
            else:
                one_sinario_img = sorted(get_all_file_paths(os.path.join(self.opt.data_root_dir,'1_TEST',senario,'images')))
                one_sinario_eve = sorted(get_all_file_paths(os.path.join(self.opt.data_root_dir,'1_TEST',senario,'events')))
            group_image_path = group_data(one_sinario_img,skip_number+1)
            group_event_path = group_data(one_sinario_eve,skip_number+1)
            self.img_path_list.append(group_image_path)
            self.event_path_list.append(group_event_path)


    def __len__(self):
        
        return len(self.img_path_list[self.opt.se_idx]*self.skip_number)

    def __getitem__(self, idx):
        group_image_path = self.img_path_list[self.opt.se_idx][idx//self.skip_number]
        group_event_path = self.event_path_list[self.opt.se_idx][idx//self.skip_number]
        I0_path = group_image_path[0]
        I1_path = group_image_path[-1]
        label_paths = group_image_path[1:self.skip_number+1]
        self.I0 = Image.open(I0_path)
        self.I1 = Image.open(I1_path)
        labels = []
        for path in label_paths:
            img = Image.open(path)  
            img_array = np.array(img)  
            labels.append(img_array)
        self.labels = np.array(labels)
        eve_0_t_paths = group_event_path[:idx%self.skip_number]
        eve_t_1_paths = group_event_path[idx%self.skip_number:]

        
        return  self.I0, self.I1, self.labels