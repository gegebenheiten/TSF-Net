import os
import re
import numpy  as np
from PIL import Image
import cv2
import copy
from skimage.metrics import structural_similarity as compare_ssim
import math
import torch

#  Assuming you have a dataset, using dummy data here for illustration
def get_all_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith((".png", ".npz")):
                file_paths.append(os.path.join(root, file))
    return file_paths


def group_image_data(data, group_size):
    # 初始化空列表来存储分组结果
    grouped_data = []
    # 按照步长为 group_size 分组
    #ends = (len(data) // (group_size-1)) * (group_size-1)
    
    for i in range(0, len(data) , group_size-1):
        group = data[i: i + group_size]
        if len(group)==group_size:
            grouped_data.append(group)
    return grouped_data

def group_event_data(data, group_size):
    # 初始化空列表来存储分组结果
    grouped_data = []
    # 按照步长为 group_size 分组
    
    for i in range(0, len(data) , group_size):
        group = data[i: i + group_size]
        if len(group)==group_size:
            grouped_data.append(group)
    return grouped_data

def save_output_as_png(output, file_name):
    output_image = Image.fromarray(output)
    output_image.save(file_name)

def denormalize_output(output):
    output = (output + 1.0) * 127.5  # 将 [-1, 1] 转换到 [0, 255]
    output = output.clip(0, 255)  # 确保数值在 [0, 255] 范围内
    return output.astype(np.uint8)

def images_to_video(image_folder, output_video, skip=None ,fps=30):
    
    # 获取所有 PNG 文件的路径，并按文件名排序
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
    sorted_files = sorted(images, key=lambda x: int(re.search(r'\d+', x).group()))
    
    # 检查文件夹是否有图片
    if len(sorted_files) == 0:
        print("No PNG images found in the folder.")
        return
    files = copy.deepcopy(sorted_files)
    if skip is not None:

        '''sorted_files = [img for img in range(0,len(sorted_files),(skip+1))]
        sorted_files = [img for img in sorted_files for _ in range(skip)]'''

        files =[]
        for i in range(0, len(sorted_files), (skip + 1)):
            for _ in range(skip+1):
                files.append(sorted_files[i])
    
    # 读取第一张图片，获取帧的尺寸
    first_image_path = os.path.join(image_folder, files[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Error reading the first image: {first_image_path}")
        return
    height, width, layers = frame.shape

    # 初始化视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image_file in files:
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)  # 读取每张图片
        if frame is None:
            print(f"Error reading image: {image_path}")
            continue
        
        # 将图片写入视频
        video.write(frame)

    video.release()
    print(f"Video saved as {output_video}")

def ssim(img_org, img1):
    ssim_value, _ = compare_ssim(img_org, img1, win_size=7, channel_axis=0, full=True)
    return ssim_value

def psnr(img1, img2, const=1):
    mse = np.mean( (img1/const - img2/const) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def mse(img1, img2, const=255):
    mse = np.mean( (img1/const - img2/const) ** 2 )
    return mse
    
def mae(img1, img2, const=255): 
    mae = np.mean( abs(img1/const - img2/const)  )
    return mae   

def randomcrop(img0, gt, img1, l_event, r_event, h, w):
    _, iw, ih = img0.shape
    y = np.random.randint(0, ih - h + 1)
    x = np.random.randint(0, iw - w + 1)
    img0 = img0[:,x:x+h, y:y+w]
    img1 = img1[:,x:x+h, y:y+w]
    l_event = l_event[:, x:x+h, y:y+w]
    r_event = r_event[:, x:x+h, y:y+w]
    gt = gt[:, x:x+h, y:y+w]
    return img0, gt, img1, l_event, r_event

def centercrop(img0, gt, img1, l_event, r_event, h, w):
    _, iw, ih = img0.shape

    center_x = iw // 2
    center_y = ih // 2
    x1 = max(center_x - w // 2, 0)
    y1 = max(center_y - h // 2, 0)
    x1 = min(x1, iw - w)
    y1 = min(y1, ih - h)

    img0 = img0[:, x1:x1 + w, y1:y1 + h]
    img1 = img1[:, x1:x1 + w, y1:y1 + h]
    l_event = l_event[:, x1:x1 + w, y1:y1 + h]
    r_event = r_event[:, x1:x1 + w, y1:y1 + h]
    gt = gt[:, x1:x1 + w, y1:y1 + h]

    return img0, gt, img1, l_event, r_event

def padding(img0, gt, img1, l_event, r_event, h, w):
    # 目标大小
    target_height = h
    target_width = w

    # 获取当前图像的大小
    _, current_height, current_width = img0.shape

    # 计算需要填充的高度和宽度
    pad_height = target_height - current_height
    pad_width = target_width - current_width

    # 在图像的上下左右进行填充
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # 对每个图像和事件进行填充
    img0 = torch.nn.functional.pad(img0, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    img1 = torch.nn.functional.pad(img1, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    gt = torch.nn.functional.pad(gt, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    l_event = torch.nn.functional.pad(l_event, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    r_event = torch.nn.functional.pad(r_event, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    return img0, gt, img1, l_event, r_event