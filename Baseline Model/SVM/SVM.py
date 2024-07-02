

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import os
import torch

# 加载数据集
pt_directory_CVL = '/root/autodl-tmp/APS360_Project/Baseline Model/SVM/CVL_HOG_pt'
pt_directory_IAM = '/root/autodl-tmp/APS360_Project/Baseline Model/SVM/IAM_HOG_pt'

def loading_data(pt_directory):
    for pt_file in pt_directory:
        pt_path = os.path.join(pt_directory, pt_file)
        data = torch.load(pt_path)

X_CVL = loading_data(pt_directory_CVL)
X_IAM = loading_data(pt_directory_IAM)
y_CVL = []
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


# 假设 input_features 是一个 2D 数组，包含一行特征向量
input_features = [[6.3, 3.3, 6.0, 2.5]]
# 使用训练好的 SVM 模型进行预测
predicted_class = svm.predict(input_features)

# 输出预测结果
print("Predicted class:", predicted_class)




