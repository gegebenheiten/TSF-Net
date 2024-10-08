import numpy as np

import numpy as np

# 定义一个函数来读取并输出npz文件内容
def load_npz_file(file_path):
    # 加载npz文件
    data = np.load(file_path)
    
    # 输出文件中的键和数据的形状与类型
    print(f"NpzFile '{file_path}' with keys: {list(data.keys())}")
    
    for key in data:
        print(f"Key: {key}, Shape: {data[key].shape}, Data type: {data[key].dtype}")
        print(f"Data (first 10 elements): {data[key][:10]}\n")  # 输出前10个元素
    
# 示例：调用函数并加载指定路径的npz文件
file_path = 'EXP1_dataset/3_TRAINING/ball_00/events/000000.npz'
load_npz_file(file_path)
