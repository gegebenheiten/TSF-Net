import os

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

