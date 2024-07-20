

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import hog
import os
import torch
import csv
import numpy as np
from concurrent.futures import ThreadPoolExecutor,as_completed
import time


# CSV文件路径
csv_CVL_path = '/root/autodl-tmp/CVL_indices.csv'
csv_IAM_path = '/root/autodl-tmp/IAM_indices.csv'

# 初始化一个空列表，用于存储CSV文件中的每一行
label_list_CVL = []
#label_list_IAM = []

# 打开CSV文件
with open(csv_CVL_path, newline='') as csvfile:
    # 创建一个csv阅读器
    reader = csv.reader(csvfile)
    
    # 遍历csv阅读器中的每行
    for row in reader:
        # 将每行的数据添加到列表中
        label_list_CVL.append(row)
label_list_check =label_list_CVL
y_CVL = np.array(label_list_check).reshape(-1,1)


# with open(csv_IAM_path, newline='') as csvfile:
#     # 创建一个csv阅读器
#     reader = csv.reader(csvfile)
    
#     # 遍历csv阅读器中的每行
#     for row in reader:
#         # 将每行的数据添加到列表中
#         label_list_CVL.append(row)
# y_IAM = np.array(label_list_CVL).reshape(-1,1)
def convert_png_to_hog_tensor(png_directory):
    hog_features = []
    count = 0
    limit = 20000

    # 遍历 .png 文件
    for png_file in os.listdir(png_directory):
        if png_file.endswith('.png'):
            if count >= limit:  # 如果已经处理了limit张图片，退出循环
                break
            png_path = os.path.join(png_directory, png_file)
            try:
                image = io.imread(png_path)
                
                # 将图像转换为灰度图像
                gray_image = rgb2gray(image)
            
                # 提取 HOG 特征
                feature_vector = hog(gray_image, pixels_per_cell=(8, 8),
                                                cells_per_block=(2, 2), visualize=False, feature_vector=True)
                
                hog_features.append(feature_vector)  
                
                count += 1
                print(count)
            except Exception as e:
                print(f"Error processing {png_path}: {e}")


    # 将 HOG 特征转换为张量
    hog_tensor = torch.tensor(hog_features)
    return hog_tensor

# def process_image(png_path):
#     try:
#         image = io.imread(png_path)
#         gray_image = rgb2gray(image)
#         feature_vector = hog(
#             gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
#             visualize=False, feature_vector=True
#         )
#         return feature_vector
#     except Exception as e:
#         print(f'Error processing {png_path}: {e}')
#         return None

# def convert_png_to_hog_tensor(png_directory):
#     png_files = [os.path.join(png_directory, file) for file in os.listdir(png_directory) if file.endswith('.png')]
    
#     # 创建一个空列表来收集特征向量
#     hog_features_list = []
    
#     # 使用 ThreadPoolExecutor 并行处理图像
#     with ThreadPoolExecutor() as executor:
#         futures = {executor.submit(process_image, file): file for file in png_files}
#         for future in as_completed(futures):
#             file = futures[future]
#             try:
#                 feature_vector = future.result()
#                 if feature_vector is not None:
#                     hog_features_list.append(feature_vector)  # 收集特征向量
#             except Exception as e:
#                 print(f'{file} generated an exception: {e}')
    
#     # 将特征向量列表转换为 NumPy 数组
#     hog_features_array = np.array(hog_features_list)
    
#     # 将 NumPy 数组转换为张量
#     hog_tensor = torch.tensor(hog_features_array)  # 使用 torch.tensor 转换
#     print('finish converting')
    
#     return hog_tensor

# 加载数据集
pt_directory_CVL = '/root/autodl-tmp/APS360_Project/Baseline Model/SVM/CVL_dataset_png'
pt_directory_IAM = '/root/autodl-tmp/APS360_Project/Baseline Model/SVM/IAM_dataset_png'


X_CVL = convert_png_to_hog_tensor(pt_directory_CVL)
print(y_CVL.shape)
#X_IAM = convert_png_to_hog_tensor(pt_directory_IAM)

#y_IAM = []


# 划分训练集和测试集
X_CVL_train, X_CVL_test, y_CVL_train, y_CVL_test = train_test_split(X_CVL, y_CVL, test_size=0.5, random_state=42)
#X_IAM_train, X_IAM_test, y_IAM_train, y_IAM_test = train_test_split(X_IAM, y_IAM, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_CVL_train = scaler.fit_transform(X_CVL_train)
X_CVL_test = scaler.transform(X_CVL_test)
# X_IAM_train = scaler.fit_transform(X_IAM_train)
# X_IAM_test = scaler.transform(X_IAM_test)
# 建立 SVM 模型
svm_CVL = SVC(kernel='linear', decision_function_shape='ovo')  # 使用一对一方法
# svm_IAM = SVC(kernel='linear', decision_function_shape='ovo')
# 训练模型
svm_CVL.fit(X_CVL_train, y_CVL_train)
# svm_IAM.fit(X_IAM_train, y_IAM_train)

# 预测
y_pred_CVL = svm_CVL.predict(X_CVL_test)
# y_pred_IAM = svm_IAM.predict(X_IAM_test)

# 评估模型
def evaluate(y_test,y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

print(evaluate(y_CVL_test,y_pred_CVL))


def predict_word():
    # 输入图片路径
    input_image_path = input('the image path is ',)
    image = io.imread(input_image_path)
                
    # 将图像转换为灰度图像
    gray_image = rgb2gray(image)
            
    # 提取 HOG 特征
    feature_vector = hog(gray_image, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=False, feature_vector=True)

    # 使用训练好的 SVM 模型进行预测
    predicted_class = svm_CVL.predict(feature_vector)

    # 输出预测结果
    print("Predicted class:", predicted_class)


