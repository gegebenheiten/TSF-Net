import torch
import torch.nn as nn
import os
from itertools import islice
from PIL import Image
import numpy as np
import tqdm
import torchvision.transforms as transforms
import cv2

TIMESTAMP_COLUMN = 2
X_COLUMN = 0
Y_COLUMN = 1
POLARITY_COLUMN = 3

'''

tensor transformer 
        
        
        
        '''


class EventSequence(object):
    """Stores events in oldes-first order."""

    def __init__(
        self, features, image_height, image_width, start_time=None, end_time=None
    ):
        """Returns object of EventSequence class.

        Args:
            features: numpy array with events softed in oldest-first order. Inside,
                      rows correspond to individual events and columns to event
                      features (x, y, timestamp, polarity)

            image_height, image_width: widht and height of the event sensor.
                                       Note, that it can not be inferred
                                       directly from the events, because
                                       events are spares.
            start_time, end_time: start and end times of the event sequence.
                                  If they are not provided, this function inferrs
                                  them from the events. Note, that it can not be
                                  inferred from the events when there is no motion.
        """
        self._features = features
        self._image_width = image_width
        self._image_height = image_height
        self._start_time = (
            start_time if start_time is not None else features[0, TIMESTAMP_COLUMN]
        )
        self._end_time = (
            end_time if end_time is not None else features[-1, TIMESTAMP_COLUMN]
        )

    def __len__(self):
        return self._features.shape[0]

   
    def duration(self):
        return self.end_time() - self.start_time()

    def start_time(self):
        return self._start_time

    def end_time(self):
        return self._end_time

    def min_timestamp(self):
        return self._features[:, TIMESTAMP_COLUMN].min()

    def max_timestamp(self):
        return self._features[:, TIMESTAMP_COLUMN].max()
    
    def reverse(self):
        """Reverse temporal direction of the event stream.

        Polarities of the events reversed.

                          (-)       (+)
        --------|----------|---------|------------|----> time
           t_start        t_1       t_2        t_end

                          (+)       (-)
        --------|----------|---------|------------|----> time
                0    (t_end-t_2) (t_end-t_1) (t_end-t_start)

        """
        if len(self) == 0:
            return
        self._features[:, TIMESTAMP_COLUMN] = (
            self._end_time - self._features[:, TIMESTAMP_COLUMN]
        )
        self._features[:, POLARITY_COLUMN] = -self._features[:, POLARITY_COLUMN]
        self._start_time, self._end_time = 0, self._end_time - self._start_time
        # Flip rows of the 'features' matrix, since it is sorted in oldest first.
        self._features = np.copy(np.flipud(self._features))
    @classmethod
    def from_npz_files(
        cls,
        list_of_filenames,
        image_height,
        image_width,
        start_time=None,
        end_time=None,
    ):
        """Reads event sequence from numpy file list."""
        if len(list_of_filenames) > 1:
            features_list = []
            for f in tqdm.tqdm(list_of_filenames):
                features_list += [load_events(f)]# for filename in list_of_filenames]
            features = np.concatenate(features_list)
        else:
            features = load_events(list_of_filenames[0])

        return EventSequence(features, image_height, image_width, start_time, end_time)

def events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(np.int16)
    ys = events[:, 2].astype(np.int16)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int16)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid


def load_events(file):
    """Load events to ".npz" file.

    See "save_events" function description.
    """
    tmp = np.load(file, allow_pickle=True)
    (x, y, timestamp, polarity) = (
        tmp["x"].astype(np.float64).reshape((-1,)),
        tmp["y"].astype(np.float64).reshape((-1,)),
        tmp["timestamp"].astype(np.float64).reshape((-1,)),
        tmp["polarity"].astype(np.float32).reshape((-1,)) * 2 - 1,
    )
    events = np.stack((x, y, timestamp, polarity), axis=-1)
    return events

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
            group_image_path = group_data(one_sinario_img,skip_number+2)
            group_event_path = group_data(one_sinario_eve,skip_number+2)
            self.img_path_list.append(group_image_path)
            self.event_path_list.append(group_event_path)
            self.osize = (256,256)

        self.device = torch.device('cuda:'+ self.opt.gpu_ids if torch.cuda.is_available() else "cpu")
        


    def __len__(self):
        
        return len(self.img_path_list[self.opt.se_idx]*self.skip_number)

    def __getitem__(self, idx):
        group_image_path = self.img_path_list[self.opt.se_idx][idx//self.skip_number]
        group_event_path = self.event_path_list[self.opt.se_idx][idx//self.skip_number]
        t = idx%self.skip_number
        eve_0_t_paths = group_event_path[1:t+1]
        eve_t_1_paths = group_event_path[t+1:8]
        I0_path = group_image_path[0]
        I1_path = group_image_path[-1]
        label_paths = group_image_path[1:self.skip_number+1]
        self.I0 = Image.open(I0_path)
        self.I1 = Image.open(I1_path)
        self.label = Image.open(label_paths[t])
        self.label = np.array(self.label)
        self.label  = np.array([cv2.resize(self.label[:,:,i],(256,256)) for i in range(self.label.shape[2])])
        image_height,image_width,_ = np.array(self.I0).shape
        
        # load timelens 
        eve_0_t = EventSequence.from_npz_files(list_of_filenames=eve_0_t_paths, image_height=image_height,image_width=image_width)._features
        eve_t_1 =EventSequence.from_npz_files(list_of_filenames=eve_t_1_paths, image_height=image_height,image_width=image_width)
        eve_1_t = eve_t_1._features[eve_t_1._features[:, TIMESTAMP_COLUMN].argsort()[::-1]]
        
        # to voxel  e2vid 
        voxel_eve_0_t = events_to_voxel_grid(eve_0_t,num_bins=5,width=image_width,height=image_height)
        voxel_eve_1_t = events_to_voxel_grid(eve_1_t,num_bins=5,width=image_width,height=image_height)

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


        I0 = self.transforms_scale(self.I0)
        I1 = self.transforms_scale(self.I1)


        voxel_eve_0_t =  np.array([cv2.resize(voxel_eve_0_t[i,:,:],(256,256)) for i in range(voxel_eve_0_t.shape[0]) ])
        voxel_eve_1_t = np.array([cv2.resize(voxel_eve_1_t[i,:,:],(256,256)) for i in range(voxel_eve_1_t.shape[0]) ])
        


        I0 =  self.transforms_toTensor(I0).to(self.device)
        I1 =  self.transforms_toTensor(I1).to(self.device)
        label = torch.from_numpy(self.label).float().to(self.device)
        voxel_eve_0_t =voxel_eve_0_t/ voxel_eve_0_t.max() 
        voxel_eve_1_t =voxel_eve_1_t / voxel_eve_1_t.max()
        voxel_eve_0_t = torch.from_numpy(voxel_eve_0_t).to(self.device)
        voxel_eve_1_t = torch.from_numpy(voxel_eve_1_t).to(self.device)
        I0 = self.transforms_normalize(I0)
        I1 = self.transforms_normalize(I1)

        

        
        # to tensor 
        return I0, I1, label, voxel_eve_0_t, voxel_eve_1_t
    