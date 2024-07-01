import os
import torch
from PIL import Image
import numpy as np

def convert_pt_to_png(pt_directory, png_directory):
    # 确保输出目录存在
    if not os.path.exists(png_directory):
        os.makedirs(png_directory)
    
    # 遍历 .pt 文件
    for pt_file in os.listdir(pt_directory):
        if pt_file.endswith('.pt'):
            pt_path = os.path.join(pt_directory, pt_file)
            tensor = torch.load(pt_path)
            

         # 检查张量的形状
            if tensor.ndimension() == 4 and tensor.shape[0] > 90000:
                # 如果张量的形状是 (N, C, H, W)，拆开成每张图片的张量
                tensor_list = []
                for i in range(tensor.shape[0]):
                    tensor_list.append(tensor[i])
            else:
                print(f'Skipping {pt_file}: unexpected tensor shape {tensor.shape}')
                continue
            
            file_num = 0
            for tensor in tensor_list:
            # 如果张量的形状是 (H, W, C)，无需转换
                if tensor.shape[0] in [1, 3]:
                    # 如果张量是单通道，扩展到三通道
                    if tensor.shape[0] == 1:
                        tensor = tensor.repeat(3, 1, 1)
                
                    # 如果值在 [0, 1] 之间，将其缩放到 [0, 255]
                    if tensor.max() <= 1.0:
                        tensor = tensor * 255

                # 将张量转换为 numpy 数组并确保类型为 uint8
                array = tensor.numpy().astype(np.uint8)

                # 将 (C, H, W) 转换为 (H, W, C)
                array = np.transpose(array, (1, 2, 0))

                # 创建 PIL 图像对象
                image = Image.fromarray(array)

                # 构造输出 .png 文件路径
                png_file_name = os.path.splitext(pt_file)[0]+ str(file_num) + '.png'
                png_path = os.path.join(png_directory, png_file_name)
                file_num += 1
                # 保存图像
                image.save(png_path)
                print(f'Successfully saved {png_path}')
            else:
                print(f'Skipping {pt_file}: unexpected channel count {tensor.shape[0]}')

# 定义输入和输出目录
pt_directory_CVL = '/root/autodl-tmp/APS360_Project/Datasets/CVL_Processed'
png_directory_CVL = '/root/autodl-tmp/APS360_Project/Machine_Learning_Output/SVM/CVL_dataset_png'
pt_directory_IAM = '/root/autodl-tmp/APS360_Project/Datasets/IAM_Processed'
png_directory_IAM = '/root/autodl-tmp/APS360_Project/Machine_Learning_Output/SVM/IAM_dataset_png'
# 执行转换
convert_pt_to_png(pt_directory_CVL, png_directory_CVL)
convert_pt_to_png(pt_directory_IAM, png_directory_IAM)
print('convert finished')

