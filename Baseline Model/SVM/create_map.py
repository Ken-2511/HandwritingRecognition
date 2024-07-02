from sklearn.preprocessing import OneHotEncoder
import csv
import numpy as np

# 打开CSV文件
with open('/root/autodl-tmp/词频.csv', newline='') as csvfile:
    # 创建CSV阅读器
    reader = csv.reader(csvfile)
    
    # 遍历CSV文件中的每一行
    classes = []
    for row in reader:
        word = []
        word.append(row[0])
        classes.append(word)
        

encoder = OneHotEncoder()

# 拟合并转换数据
classes_encoded = encoder.fit_transform(classes)








