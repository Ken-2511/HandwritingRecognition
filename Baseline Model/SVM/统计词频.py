import os
import torch
import numpy as np


def extract_tensor(pt_directory):
    # 遍历 .pt 文件
    pt_file = 'rec_label.pt'
    pt_path = os.path.join(pt_directory, pt_file)
    tensor = torch.load(pt_path)
    tensor_list = list(tensor)

    return tensor_list


def create_word_list(tensor_list):
    total_word = []
    for sub_tensor in tensor_list:
        # 生成单词
        word = ''
        for j in range(1, 64):
            o = np.argmax(sub_tensor[j])
            if o == 3:
                break
            c = chr(o)
            word += c
            
        total_word.append(word)
    return total_word


def count_word_num(total_word):
    word_num = dict()
    for word in total_word:
        current_word = word_num.get(word, 0)
        word_num[word] = current_word + 1

    return word_num

pt_directory_CVL = '/root/autodl-tmp/APS360_Project/Datasets/CVL_Processed'
pt_directory_IAM = '/root/autodl-tmp/APS360_Project/Datasets/IAM_Processed'

# print(count_word_num(create_word_list(extract_tensor(pt_directory_CVL))))
# print(count_word_num(create_word_list(extract_tensor(pt_directory_IAM))))

result = count_word_num(create_word_list(extract_tensor(pt_directory_CVL)))
sorted_result = sorted(result.items(), key=lambda x: x[1], reverse=True)

classified = 0
threshold = 300
for i in sorted_result[:threshold]:
    classified += i[1]
    print(i)
others = 0
for i in sorted_result[threshold:]:
    others += i[1]
print(('classified', classified))
print(('others', others))
print(f"total classes: {len(sorted_result)}")