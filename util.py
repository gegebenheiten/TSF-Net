import os

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
    ends = (len(data) // group_size) * group_size
    
    for i in range(0, ends, group_size-1):
        grouped_data.append(data[i: i + group_size])
    return grouped_data

