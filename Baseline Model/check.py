import csv

word_map_path ='/root/autodl-tmp/APS360_Project/word_map.csv'

word_map = dict()
with open(word_map_path, mode='r', newline='', encoding='utf-8') as file:
    # 创建 csv.DictReader 对象
    reader = csv.DictReader(file)
    
    # 遍历CSV文件中的每一行
    for row in reader:
        # 将每行数据作为一个字典项添加到data_dict中
        # 假设CSV文件中的列名是 'Index' 和 'Key'
        index = int(row['Index'])  # 将Index转换为整数
        value = row['Value']
        word_map[index] = value  # 用 'Key' 作为键，'Index' 作为值

# 打印结果
print(word_map)