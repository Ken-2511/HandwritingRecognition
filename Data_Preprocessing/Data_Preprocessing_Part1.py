# This file is for processing the IAM database
# written by Ken and Michael

import os
import torch
import tarfile
import logging
from torch.utils.data import Dataset, DataLoader

IAM_path = '/root/APS360_Project/Datasets/IAM'
IAM_processed_path = '/root/APS360_Project/Datasets/IAM_Processed'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create the processed directory if it does not exist
if not os.path.exists(IAM_processed_path):
    os.mkdir(IAM_processed_path)

os.mkdir(IAM_processed_path)

## unzip all .tgz files
#for file in os.listdir(IAM_path):
#     if file.endswith('.tgz'):
#         os.system('tar -xvzf ' + IAM_path + '/' + file + ' -C ' + IAM_processed_path)

# Unzip all .tgz files
for file in IAM_path.glob('*.tgz'):
    try:
        with tarfile.open(file, 'r:gz') as tar:
            tar.extractall(path=IAM_processed_path)
        logging.info(f'Extracted {file}')
    except Exception as e:
        logging.error(f'Failed to extract {file}: {e}')

#for file in os.listdir(IAM_processed_path):
#    if not file.endswith(".png"):
#       os.system("mv " + IAM_processed_path + "/" + file + " " + IAM_path + "/" + file)

# Move non-image files back to the original directory
for file in IAM_processed_path.iterdir():
    if not file.suffix == '.png':
        try:
            file.rename(IAM_path / file.name)
            logging.info(f'Moved {file.name} back to {IAM_path}')
        except Exception as e:
            logging.error(f'Failed to move {file.name}: {e}')