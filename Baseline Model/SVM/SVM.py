

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 加载数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target
print(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 建立 SVM 模型
svm = SVC(kernel='linear', decision_function_shape='ovo')  # 使用一对一方法

# 训练模型
svm.fit(X_train, y_train)

# 预测
y_pred = svm.predict(X_test)

# 评估模型
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# 假设 input_features 是一个 2D 数组，包含一行特征向量
input_features = [[6.3, 3.3, 6.0, 2.5]]
# 使用训练好的 SVM 模型进行预测
predicted_class = svm.predict(input_features)

# 输出预测结果
print("Predicted class:", predicted_class)




