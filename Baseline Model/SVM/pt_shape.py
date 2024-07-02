import os
import torch
from PIL import Image
import numpy as np

pt_directory = '/root/autodl-tmp/APS360_Project/Datasets/IAM_Processed'
for pt_file in os.listdir(pt_directory):
        if pt_file.endswith('.pt'):
            pt_path = os.path.join(pt_directory, pt_file)
            tensor = torch.load(pt_path)
            print(pt_file)
            print(tensor.shape)



