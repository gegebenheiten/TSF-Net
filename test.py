import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.EDSR import EDSR
from dataset import demoDataset
import numpy  as np
import os
from basic_option import SimpleOptions
from util import save_output_as_png, denormalize_output, images_to_video, ssim, psnr, mse, mae
import pickle
import time
import math
import cv2
from PIL import Image
import re
import copy
import pdb


'''def images_to_video(image_folder, output_video, repeat =None ,fps=30):
    
    # 获取所有 PNG 文件的路径，并按文件名排序
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
    sorted_files = sorted(images, key=lambda x: int(re.search(r'\d+', x).group()))
    
    # 检查文件夹是否有图片
    if len(sorted_files) == 0:
        print("No PNG images found in the folder.")
        return
    if repeat is not None:
        sorted_files = [img for img in sorted_files for _ in range(repeat)]

    # 读取第一张图片，获取帧的尺寸
    first_image_path = os.path.join(image_folder, sorted_files[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Error reading the first image: {first_image_path}")
        return
    height, width, layers = frame.shape

    # 初始化视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image_file in sorted_files:
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)  # 读取每张图片
        if frame is None:
            print(f"Error reading image: {image_path}")
            continue
        
        # 将图片写入视频
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video.release()
    print(f"Video saved as {output_video}")
'''

# initialize parser
option = SimpleOptions()
opt = option.parse()
opt.isTrain = False
opt.isValidate = False
opt.isTest = True
opt.device = torch.device(opt.gpu_ids)
# senarios = [f.name for f in os.scandir(os.path.join(opt.data_root_dir,'1_TEST')) if f.is_dir()]
# opt.senarios = senarios
opt.senarios = ['ball_05']

n_blocks = (opt.image_height // opt.block_size) * (opt.image_width // opt.block_size)
new_size = (opt.image_height, opt.image_width)
# Prepare data
test_dataset = demoDataset(opt)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

with open(f'data_stats/div2k/stats_qp{opt.qp}.pkl', 'rb') as f:
    stats = pickle.load(f)

dct_min = torch.from_numpy(stats['dct_input']['min'][None, :, None, None]).float().to(opt.device)
dct_max = torch.from_numpy(stats['dct_input']['max'][None, :, None, None]).float().to(opt.device)
# Initialize model, criterion, and optimizer
model = EDSR(n_blocks, dct_max, dct_min).to(opt.device)

model_path = 'save_models/model_epoch_100_batch_16_insert_1.pth'
state_dict = torch.load(model_path)
new_state_dict = {}
for key in state_dict:
    if key.startswith('module.'):
        new_state_dict[key[7:]] = state_dict[key]  # 去掉 'module.' 前缀
    else:
        new_state_dict[key] = state_dict[key]

model.load_state_dict(new_state_dict)
model.eval() 
result_mse = []
result_mae = []
result_psnr = []
result_ssim = []

with torch.no_grad():  # 不计算梯度
    for se_idx in range(len(opt.senarios)):
        idx = 0
        for i, (inputs, labels)  in enumerate(test_loader):
            opt.se_idx = se_idx  # 假设输入数据键为 'input'
            left_image, right_image, left_events, right_events = inputs
            dir =os.path.join('./output', opt.senarios[opt.se_idx])
            if not os.path.exists(dir):
                os.makedirs(dir,exist_ok=True)
            output = model(opt, left_image, left_events, right_image, right_events) # 生成预测
            output = denormalize_output(output=output.squeeze(0).cpu().numpy())
            labels = denormalize_output(labels.squeeze(0).cpu().numpy())
            result_mse.append(mse(labels, output))
            result_mae.append(mae(labels, output))
            result_ssim.append(ssim(labels, output))
            result_psnr.append(psnr(labels, output))
            # output = np.array([cv2.resize(output[j, :, :], new_size) for j in range(output.shape[0])])
            output = np.transpose(output, (1, 2, 0))  # 转换维度为 (H, W, 3)
            if idx% (opt.skip_number+1) == 0:
                left_image_path = os.path.join(dir, f'{idx}.png')
                left_image = left_image.squeeze(0).cpu().numpy()
                # left_image = np.array([cv2.resize(left_image[j, :, :], new_size) for j in range(left_image.shape[0])])
                left_image = np.transpose(left_image, (1, 2, 0))  # 转换维度为 (H, W, 3)
                save_output_as_png(denormalize_output(output=left_image), left_image_path)
                idx +=1
            name = f'{idx}.png'
            idx +=1
            path = os.path.join(dir, name)
            save_output_as_png(output, path)
        right_image_path = os.path.join(dir, f'{idx}.png')
        right_image = right_image.squeeze(0).cpu().numpy()
        right_image = np.array([cv2.resize(right_image[i, :, :], new_size) for i in range(right_image.shape[0])])
        right_image = np.transpose(right_image, (1, 2, 0))  # 转换维度为 (H, W, 3)
        save_output_as_png(denormalize_output(output=right_image), right_image_path)
    print(np.mean(result_mse), np.mean(result_mae), np.mean(result_psnr), np.mean(result_ssim)) 


image_folder = "output/ball_05"  # 替换为你的 PNG 文件夹路径
output_video = f"./ball_05_large_insert_{opt.skip_number}.mp4"  # 输出 MP4 文件名
fps = 30 # 设置帧率

images_to_video(image_folder, output_video, fps = 30)

# input_folder = "data/bs_ergb/1_TEST/ball_05/images"  
# input_video = "./input_ball_05.mp4"  
# fps = 30 # 设置帧率

# images_to_video(input_folder, input_video, 1, fps)

# GT_video = "./GT_ball_05.mp4"  
# images_to_video(input_folder, GT_video)
