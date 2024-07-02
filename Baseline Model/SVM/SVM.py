

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

# CSV文件路径
csv_file_path = '/root/autodl-tmp/CVL_indices.csv'

# 初始化一个空列表，用于存储CSV文件中的每一行
y_CVL = []

# 打开CSV文件
with open(csv_file_path, newline='') as csvfile:
    # 创建一个csv阅读器
    reader = csv.reader(csvfile)
    
    # 遍历csv阅读器中的每行
    for row in reader:
        # 将每行的数据添加到列表中
        y_CVL.append(row)

def convert_png_to_hog_tensor(png_directory):
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

    # 将 HOG 特征转换为张量
    hog_tensor = torch.tensor(hog_features)
    return hog_tensor

# 加载数据集
pt_directory_CVL = '/root/autodl-tmp/APS360_Project/Baseline Model/SVM/CVL_HOG_pt'
pt_directory_IAM = '/root/autodl-tmp/APS360_Project/Baseline Model/SVM/IAM_HOG_pt'


X_CVL = convert_png_to_hog_tensor(pt_directory_CVL)
X_IAM = convert_png_to_hog_tensor(pt_directory_IAM)

y_IAM = []


# 划分训练集和测试集
X_CVL_train, X_CVL_test, y_CVL_train, y_CVL_test = train_test_split(X_CVL, y_CVL, test_size=0.3, random_state=42)
X_IAM_train, X_IAM_test, y_IAM_train, y_IAM_test = train_test_split(X_IAM, y_IAM, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_CVL_train = scaler.fit_transform(X_CVL_train)
X_CVL_test = scaler.transform(X_CVL_test)
X_IAM_train = scaler.fit_transform(X_IAM_train)
X_IAM_test = scaler.transform(X_IAM_test)
# 建立 SVM 模型
svm_CVL = SVC(kernel='linear', decision_function_shape='ovo')  # 使用一对一方法
svm_IAM = SVC(kernel='linear', decision_function_shape='ovo')
# 训练模型
svm_CVL.fit(X_CVL_train, y_CVL_train)
svm_IAM.fit(X_IAM_train, y_IAM_train)

# 预测
y_pred_CVL = svm_CVL.predict(X_CVL_test)
y_pred_IAM = svm_IAM.predict(X_IAM_test)

# 评估模型
def evaluate(y_test,y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

print(evaluate(y_CVL_test,y_pred_CVL))


# # 假设 input_features 是一个 2D 数组，包含一行特征向量
# input_features = [[6.3, 3.3, 6.0, 2.5]]
# # 使用训练好的 SVM 模型进行预测
# predicted_class = svm_CVL.predict(input_features)

# # 输出预测结果
# print("Predicted class:", predicted_class)




