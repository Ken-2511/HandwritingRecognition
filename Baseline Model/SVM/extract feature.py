import os
from skimage import io
from skimage.feature import hog
from skimage.color import rgb2gray
import numpy as np
import torch

# def convert_png_to_hog_tensor(png_directory, output_pt_file):
#     hog_features = []

#     # 遍历 .png 文件
#     for png_file in os.listdir(png_directory):
#         if png_file.endswith('.png'):
#             png_path = os.path.join(png_directory, png_file)
#             image = io.imread(png_path)
            
#             # 将图像转换为灰度图像
#             gray_image = rgb2gray(image)
            
#             # 提取 HOG 特征
#             feature_vector, hog_image = hog(gray_image, pixels_per_cell=(8, 8),
#                                             cells_per_block=(2, 2), visualize=False, feature_vector=True)
            
#             hog_features.append(feature_vector)  

#     # 将 HOG 特征转换为张量
#     hog_tensor = torch.tensor(hog_features)
#     print(hog_tensor.shape)

import os
import torch
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import hog
from concurrent.futures import ThreadPoolExecutor,as_completed
import time

start_time = time.time()
def process_image(png_path):
    try:
        image = io.imread(png_path)
        gray_image = rgb2gray(image)
        feature_vector = hog(
            gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
            visualize=False, feature_vector=True
        )
        return feature_vector
    except Exception as e:
        print(f'Error processing {png_path}: {e}')
        return None

def convert_png_to_hog_tensor(png_directory):
    png_files = [os.path.join(png_directory, file) for file in os.listdir(png_directory) if file.endswith('.png')]
    
    # 创建一个空列表来收集特征向量
    hog_features_list = []
    
    # 使用 ThreadPoolExecutor 并行处理图像
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, file): file for file in png_files}
        for future in as_completed(futures):
            file = futures[future]
            try:
                feature_vector = future.result()
                if feature_vector is not None:
                    hog_features_list.append(feature_vector)  # 收集特征向量
            except Exception as e:
                print(f'{file} generated an exception: {e}')
    
    # 将特征向量列表转换为 NumPy 数组
    hog_features_array = np.array(hog_features_list)
    
    # 将 NumPy 数组转换为张量
    hog_tensor = torch.tensor(hog_features_array)  # 使用 torch.tensor 转换
    print('finish converting')
    
    return hog_tensor

# 示例使用
png_directory_CVL = '/root/autodl-tmp/APS360_Project/Baseline Model/SVM/CVL_dataset_png'
hog_tensor = convert_png_to_hog_tensor(png_directory_CVL)
print(f"HOG Tensor shape: {hog_tensor.shape}")
end_time = time.time()

print('time used is ',end_time-start_time)

import csv
import numpy as np

# CSV文件路径
csv_file_path = '/root/autodl-tmp/CVL_indices.csv'

# 初始化一个空列表，用于存储CSV文件中的每一行
label_list_CVL = []

# 打开CSV文件
with open(csv_file_path, newline='') as csvfile:
    # 创建一个csv阅读器
    reader = csv.reader(csvfile)
    
    # 遍历csv阅读器中的每行
    for row in reader:
        # 将每行的数据添加到列表中
        label_list_CVL.append(row)  
y_CVL = np.array(label_list_CVL).reshape(-1,1)
print(y_CVL.shape )

    # 保存张量为 .pt 文件
    # torch.save(hog_tensor, output_pt_file)
    # print(f'Successfully saved HOG features to {output_pt_file}')

# 定义输入和输出目录
png_directory_CVL = '/root/autodl-tmp/APS360_Project/Baseline Model/SVM/CVL_dataset_png'
output_pt_file_CVL = '/root/autodl-tmp/APS360_Project/Baseline Model/SVM/CVL_HOG_pt'
png_directory_IAM = '/root/autodl-tmp/APS360_Project/Baseline Model/SVM/IAM_dataset_png'
output_pt_file_IAM = '/root/autodl-tmp/APS360_Project/Baseline Model/SVM/IAM_HOG_pt'


# 执行转换
# convert_png_to_hog_tensor(png_directory_CVL)
# convert_png_to_hog_tensor(png_directory_IAM)

