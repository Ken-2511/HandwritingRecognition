import os
from skimage import io
from skimage.feature import hog
from skimage.color import rgb2gray
import numpy as np
import torch

def convert_png_to_hog_tensor(png_directory, output_pt_file):
    hog_features = []

    # 遍历 .png 文件
    for png_file in os.listdir(png_directory):
        if png_file.endswith('.png'):
            png_path = os.path.join(png_directory, png_file)
            image = io.imread(png_path)
            
            # 将图像转换为灰度图像
            gray_image = rgb2gray(image)
            
            # 提取 HOG 特征
            feature_vector, hog_image = hog(gray_image, pixels_per_cell=(8, 8),
                                            cells_per_block=(2, 2), visualize=True, feature_vector=True)
            
            hog_features.append(feature_vector)  
            print(f'Processed {png_file}')

    # 将 HOG 特征转换为张量
    hog_tensor = torch.tensor(hog_features)
    
    # 保存张量为 .pt 文件
    torch.save(hog_tensor, output_pt_file)
    print(f'Successfully saved HOG features to {output_pt_file}')

# 定义输入和输出目录
png_directory_CVL = '/root/autodl-tmp/APS360_Project/Machine_Learning_Output/SVM/CVL_dataset_png'
output_pt_file_CVL = '/root/autodl-tmp/APS360_Project/Machine_Learning_Output/SVM/CVL_HOG_pt'
png_directory_IAM = '/root/autodl-tmp/APS360_Project/Machine_Learning_Output/SVM/IAM_dataset_png'
output_pt_file_IAM = '/root/autodl-tmp/APS360_Project/Machine_Learning_Output/SVM/IAM_HOG_pt'


# 执行转换
convert_png_to_hog_tensor(png_directory_CVL, output_pt_file_CVL)
convert_png_to_hog_tensor(png_directory_IAM, output_pt_file_IAM)

